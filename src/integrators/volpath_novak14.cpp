#include <random>
#include <tuple>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>


NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-volpath:

Volumetric path tracer (:monosp:`volpath`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1 corresponds to
     :math:`\infty`). A value of 1 will only render directly visible light sources. 2 will lead
     to single-bounce (direct-only) illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start to use the
     *russian roulette* path termination criterion. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This plugin provides a volumetric path tracer that can be used to compute approximate solutions
of the radiative transfer equation. Its implementation makes use of multiple importance sampling
to combine BSDF and phase function sampling with direct illumination sampling strategies. On
surfaces, it behaves exactly like the standard path tracer.

This integrator has special support for index-matched transmission events (i.e. surface scattering
events that do not change the direction of light). As a consequence, participating media enclosed by
a stencil shape are rendered considerably more efficiently when this shape
has a :ref:`null <bsdf-null>` or :ref:`thin dielectric <bsdf-thindielectric>` BSDF assigned
to it (as compared to, say, a :ref:`dielectric <bsdf-dielectric>` or
:ref:`roughdielectric <bsdf-roughdielectric>` BSDF).

.. note:: This integrator does not implement good sampling strategies to render
    participating media with a spectrally varying extinction coefficient. For these cases,
    it is better to use the more advanced :ref:`volumetric path tracer with
    spectral MIS <integrator-volpathmis>`, which will produce in a significantly less noisy
    rendered image.

.. warning:: This integrator does not support forward-mode differentiation.

.. tabs::
    .. code-tab::  xml

        <integrator type="volpath">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'volpath',
        'max_depth': 8

*/
template <typename Float, typename Spectrum>
class ResidualRatioTrackingIntegrator : public MonteCarloIntegrator<Float, Spectrum> {

public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters, m_sampler, m_est)
    MI_IMPORT_TYPES(Scene, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr,
                     Medium, MediumPtr, PhaseFunctionContext)

    ResidualRatioTrackingIntegrator(const Properties &props) : Base(props) {
        std::string estimator = props.string("transmittance_estimator", "rrt");
        std::string sampler = props.string("distance_sampler", "ff_local");

        if (estimator == "rt") {
            m_est = 0;
        } else if (estimator == "rrt") {
            m_est = 1;
        } else if (estimator == "rt_local") {
            m_est = 2;
        } else if (estimator == "rrt_local") {
            m_est = 3;
        } else if (estimator == "nf") {
            m_est = 4;
        } else if (estimator == "rm") {
            m_est = 5;
        } else if (estimator == "ps_cum") {
            m_est = 6;
        } else if (estimator == "ps_cmf") {
            m_est = 7;
        } else if (estimator == "nf_local") {
            m_est = 8;
        } else {
            NotImplementedError("Unsupported estimator. Select one of the following: rt, rrt, rt_local, rrt_local, rm, nf, ps_cum, ps_cmf, nf_local");
        }

        if (sampler == "ff") {
            m_sampler = 0;
        } else if (sampler == "ff_local") {
            m_sampler = 1;
        } else if (sampler == "ff_weighted_local") {
            m_sampler = 2;
        } else {
            NotImplementedError("Unsupported sampler. Select one of the following: ff, ff_local, ff_weighted_local");
        }

    }

    MI_INLINE
    Float index_spectrum(const UnpolarizedSpectrum &spec, const UInt32 &idx) const {
        Float m = spec[0];
        if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
            dr::masked(m, dr::eq(idx, 1u)) = spec[1];
            dr::masked(m, dr::eq(idx, 2u)) = spec[2];
        } else {
            DRJIT_MARK_USED(idx);
        }
        return m;
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium *initial_medium,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        // If there is an environment emitter and emitters are visible: all rays will be valid
        // Otherwise, it will depend on whether a valid interaction is sampled
        Mask valid_ray = !m_hide_emitters && dr::neq(scene->environment(), nullptr);

        // For now, don't use ray differentials
        Ray3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        Spectrum throughput(1.f), result(0.f);
        MediumPtr medium = initial_medium;
        MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
        Mask specular_chain = active && !m_hide_emitters;
        UInt32 depth = 0;

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Mask needs_intersection = true;
        Interaction3f last_scatter_event = dr::zeros<Interaction3f>();
        Float last_scatter_direction_pdf = 1.f;

        /* Set up a Dr.Jit loop (optimizes away to a normal loop in scalar mode,
           generates wavefront or megakernel renderer based on configuration).
           Register everything that changes as part of the loop here */
        dr::Loop<Mask> loop("Volpath integrator",
                            /* loop state: */ active, depth, ray, throughput,
                            result, si, mei, medium, eta, last_scatter_event,
                            last_scatter_direction_pdf, needs_intersection,
                            specular_chain, valid_ray, sampler);

        while (loop(active)) {
            // ----------------- Handle termination of paths ------------------
            // Russian roulette: try to keep path weights equal to one, while accounting for the
            // solid angle compression at refractive index boundaries. Stop with at least some
            // probability to avoid  getting stuck (e.g. due to total internal reflection)
            active &= dr::any(dr::neq(unpolarized_spectrum(throughput), 0.f));
            Float q = dr::minimum(dr::max(unpolarized_spectrum(throughput)) * dr::sqr(eta), .95f);
            Mask perform_rr = (depth > (uint32_t) m_rr_depth);
            active &= sampler->next_1d(active) < q || !perform_rr;
            dr::masked(throughput, perform_rr) *= dr::rcp(dr::detach(q));

            // std::cout<< "check 1"<<std::endl;
            active &= depth < (uint32_t) m_max_depth;   
            if (dr::none_or<false>(active))
                break;
            // ----------------------- Sampling the RTE -----------------------
            Mask active_medium  = active && dr::neq(medium, nullptr);
            Mask active_surface = active && !active_medium;
            Mask absorptive = active_medium && medium->is_absorptive();
            Mask escaped_medium = false;

            if (dr::any_or<true>(active_medium)) {
                UInt32 v0(0), v1(0);
                Spectrum integral_tr(1.f);
                if constexpr (dr::is_array_v<Float>) {
                    UInt32 seed = UInt32(sampler->next_1d(active_medium) * UINT32_MAX);
                    UInt32 idx = dr::arange<UInt32>(sampler->wavefront_size());
                    std::tie(v0, v1) = sample_tea_32(seed, idx);
                } else {
                    v0 = UInt32(sampler->next_1d(active_medium) * UINT32_MAX);
                }
                if (m_sampler == 0){
                    dr::masked(mei, active_medium && !absorptive) = medium->sample_interaction_real(ray, v0, v1, active_medium && !absorptive); //free flight sampling with global majorant
                    // dr::masked(throughput, active_medium) *= weight;
                } else if (m_sampler == 1){
                    dr::masked(mei, active_medium && !absorptive) = medium->sample_interaction_real_super(ray, v0, v1, active_medium && !absorptive); //free flight sampling with supervoxel majorants
                } else if (m_sampler == 2){
                    dr::masked(mei, active_medium && !absorptive) = medium->sample_interaction_super_weighted_dt(ray, v0, v1, active_medium && !absorptive); //free flight sampling with supervoxel majorants
                } else {
                    std::cout<<"Sampler: "<<m_sampler<<std::endl;
                    NotImplementedError("select one of the following distance samplers: ff, ff_local, ff_weighted_local");
                }
                std::tie(dr::masked(mei, absorptive), dr::masked(integral_tr, absorptive)) = medium->integrate_tr(ray, v0, v1, m_est, sampler->wavefront_size(), absorptive);
                dr::masked(throughput, absorptive) *= (integral_tr);
                
                dr::masked(ray.maxt, active_medium && medium->is_homogeneous() && mei.is_valid()) = mei.t;
                dr::masked(ray.maxt, active_medium && medium->is_absorptive() && mei.is_valid()) = mei.t;     
                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);
                needs_intersection &= !active_medium;

                dr::masked(mei.t, active_medium && (si.t < mei.t)) = dr::Infinity<Float>;

                escaped_medium = active_medium && !mei.is_valid();
                active_medium &= mei.is_valid();

                dr::masked(depth, active_medium) += 1;
                dr::masked(last_scatter_event, active_medium) = mei;
                dr::masked(throughput, active_medium && !absorptive) *= mei.weight;//(mei.sigma_s / mei.sigma_t);
            }
            
            // Dont estimate lighting if we exceeded number of bounces
            active &= depth < (uint32_t) m_max_depth;
            active_medium &= active;

            if (dr::any_or<true>(active_medium)) {
                // dr::masked(throughput, active_medium) *= (mei.sigma_s);
                PhaseFunctionContext phase_ctx(sampler);
                auto phase = mei.medium->phase_function();

                // --------------------- Emitter sampling ---------------------
                Mask sample_emitters = mei.medium->use_emitter_sampling();
                valid_ray |= active_medium;
                specular_chain &= !active_medium;
                specular_chain |= active_medium && !sample_emitters;

                Mask active_e = active_medium && sample_emitters;
                if (dr::any_or<true>(active_e)) {
                    auto [emitted, ds] = sample_emitter(mei, scene, sampler, medium, active_e);
                    auto [phase_val, phase_pdf] = phase->eval_pdf(phase_ctx, mei, ds.d, active_e);
                    dr::masked(result, active_e) += throughput * phase_val * emitted *
                                                    mis_weight(ds.pdf, dr::select(ds.delta, 0.f, phase_pdf));
                }
                // ------------------ Phase function sampling -----------------
                dr::masked(phase, !active_medium) = nullptr;
                auto [wo, phase_weight, phase_pdf] = phase->sample(phase_ctx, mei,
                    sampler->next_1d(active_medium),
                    sampler->next_2d(active_medium),
                    active_medium);
                active_medium &= phase_pdf > 0.f;
                Ray3f new_ray  = mei.spawn_ray(wo);
                dr::masked(ray, active_medium) = new_ray;
                needs_intersection |= active_medium;
                dr::masked(last_scatter_direction_pdf, active_medium) = phase_pdf;
                dr::masked(throughput, active_medium) *= phase_weight;
            }
            // Log(Info, "Distance sampled exit medium used 2");        
            // --------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium;
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);

            if (dr::any_or<true>(active_surface)) {
                // ---------------- Intersection with emitters ----------------
                Mask ray_from_camera = active_surface && dr::eq(depth, 0u);
                Mask count_direct = ray_from_camera || specular_chain;
                EmitterPtr emitter = si.emitter(scene);
                Mask active_e = active_surface && dr::neq(emitter, nullptr)
                                && !(dr::eq(depth, 0u) && m_hide_emitters);
                if (dr::any_or<true>(active_e)) {
                    Float emitter_pdf = 1.0f;
                    if (dr::any_or<true>(active_e && !count_direct)) {
                        // Get the PDF of sampling this emitter using next event estimation
                        DirectionSample3f ds(scene, si, last_scatter_event);
                        emitter_pdf = scene->pdf_emitter_direction(last_scatter_event, ds, active_e);
                    }
                    Spectrum emitted = emitter->eval(si, active_e);
                    Spectrum contrib = dr::select(count_direct, throughput * emitted,
                                                  throughput * mis_weight(last_scatter_direction_pdf, emitter_pdf) * emitted);
                    dr::masked(result, active_e) += contrib;
                }
            }
            active_surface &= si.is_valid();
            if (dr::any_or<true>(active_surface)) {
                // --------------------- Emitter sampling ---------------------
                BSDFContext ctx;
                BSDFPtr bsdf  = si.bsdf(ray);
                Mask active_e = active_surface && has_flag(bsdf->flags(), BSDFFlags::Smooth) && (depth + 1 < (uint32_t) m_max_depth);

                if (likely(dr::any_or<true>(active_e))) {
                    auto [emitted, ds] = sample_emitter(si, scene, sampler, medium, active_e);

                    // Query the BSDF for that emitter-sampled direction
                    Vector3f wo       = si.to_local(ds.d);
                    Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                    bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                    // Determine probability of having sampled that same
                    // direction using BSDF sampling.
                    Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);
                    dr::masked(result, active_e) += throughput * bsdf_val * mis_weight(ds.pdf, dr::select(ds.delta, 0.f, bsdf_pdf)) * emitted;
                }

                // ----------------------- BSDF sampling ----------------------
                auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active_surface),
                                                   sampler->next_2d(active_surface), active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

                dr::masked(throughput, active_surface) *= bsdf_val;
                dr::masked(eta, active_surface) *= bs.eta;

                Ray3f bsdf_ray                  = si.spawn_ray(si.to_world(bs.wo));
                dr::masked(ray, active_surface) = bsdf_ray;
                needs_intersection |= active_surface;

                Mask non_null_bsdf = active_surface && !has_flag(bs.sampled_type, BSDFFlags::Null);
                dr::masked(depth, non_null_bsdf) += 1;

                // update the last scatter PDF event if we encountered a non-null scatter event
                dr::masked(last_scatter_event, non_null_bsdf) = si;
                dr::masked(last_scatter_direction_pdf, non_null_bsdf) = bs.pdf;

                valid_ray |= non_null_bsdf;
                specular_chain |= non_null_bsdf && has_flag(bs.sampled_type, BSDFFlags::Delta);
                specular_chain &= !(active_surface && has_flag(bs.sampled_type, BSDFFlags::Smooth));
                Mask has_medium_trans                = active_surface && si.is_medium_transition();
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
            active &= (active_surface | active_medium);
  
        }
        return { result, valid_ray };
    }

    /// Samples an emitter in the scene and evaluates its attenuated contribution
    
    template <typename Interaction>
    std::tuple<Spectrum, DirectionSample3f>
    sample_emitter(const Interaction &ref_interaction, const Scene *scene,
                   Sampler *sampler, MediumPtr medium,
                   Mask active) const {
        Spectrum transmittance(1.0f);

        auto [ds, emitter_val] = scene->sample_emitter_direction(ref_interaction, sampler->next_2d(active), false, active);
        dr::masked(emitter_val, dr::eq(ds.pdf, 0.f)) = 0.f;
        active &= dr::neq(ds.pdf, 0.f);

        if (dr::none_or<false>(active)) {
            return { emitter_val, ds };
        }

        Ray3f ray = ref_interaction.spawn_ray_to(ds.p);
        Float max_dist = ray.maxt;

        // Potentially escaping the medium if this is the current medium's boundary
        if constexpr (std::is_convertible_v<Interaction, SurfaceInteraction3f>)
            dr::masked(medium, ref_interaction.is_medium_transition()) =
                ref_interaction.target_medium(ray.d);

        Float total_dist = 0.f;
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        Mask needs_intersection = true;

        dr::Loop<Mask> loop("Volpath integrator emitter sampling");
        loop.put(active, ray, total_dist, needs_intersection, medium, si,
                 transmittance);
        sampler->loop_put(loop);
        loop.init();
      
        while (loop(dr::detach(active))) {
            Float remaining_dist = max_dist - total_dist;
            ray.maxt = remaining_dist;
            active &= remaining_dist > 0.f;
     
            if (dr::none_or<false>(active))
                break;

            Mask escaped_medium = false;
            Mask active_medium  = active && dr::neq(medium, nullptr);
            Mask active_surface = active && !active_medium;      

            if (dr::any_or<true>(active_medium)) {
                UInt32 v0(0), v1(0);
                if constexpr (dr::is_array_v<Float>) {
                    UInt32 seed = UInt32(sampler->next_1d(active_medium) * UINT32_MAX);
                    UInt32 idx = dr::arange<UInt32>(sampler->wavefront_size());
                    std::tie(v0, v1) = sample_tea_32(seed, idx);
                } else {
                    v0 = UInt32(sampler->next_1d(active_medium) * UINT32_MAX);
                }
                auto [mei, integral_tr] = medium->integrate_tr(ray, v0, v1, m_est, sampler->wavefront_size(), active_medium); // transmission integral \w supervoxel majorant and control grid

                Mask intersect = needs_intersection && active_medium;
                if (dr::any_or<true>(intersect))
                    dr::masked(si, intersect) = scene->ray_intersect(ray, intersect);

                needs_intersection &= !active_medium;
                escaped_medium = active_medium && !mei.is_valid();
                transmittance[active_medium] *= integral_tr;
                active_medium &= mei.is_valid();
            }

            // Handle interactions with surfaces
            Mask intersect = active_surface && needs_intersection;
            if (dr::any_or<true>(intersect))
                dr::masked(si, intersect)    = scene->ray_intersect(ray, intersect);
            needs_intersection &= !intersect;
            active_surface |= escaped_medium;
            dr::masked(total_dist, active_surface) += si.t;

            active_surface &= si.is_valid() && active && !active_medium;
            if (dr::any_or<true>(active_surface)) {
                auto bsdf         = si.bsdf(ray);
                Spectrum bsdf_val = bsdf->eval_null_transmission(si, active_surface);
                bsdf_val = si.to_world_mueller(bsdf_val, si.wi, si.wi);
                dr::masked(transmittance, active_surface) *= bsdf_val;
            }

            // Update the ray with new origin & t parameter
            dr::masked(ray, active_surface) = si.spawn_ray(ray.d);
            ray.maxt = remaining_dist;
            needs_intersection |= active_surface;

            // Continue tracing through scene if non-zero weights exist
            active &= (active_medium || active_surface) && dr::any(dr::neq(unpolarized_spectrum(transmittance), 0.f));

            // If a medium transition is taking place: Update the medium pointer
            Mask has_medium_trans = active_surface && si.is_medium_transition();
            if (dr::any_or<true>(has_medium_trans)) {
                dr::masked(medium, has_medium_trans) = si.target_medium(ray.d);
            }
            // Log(Info, "sampling emitter exit");       
        }
        return { transmittance * emitter_val, ds };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("VolumetricSimplePathIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::select(dr::isfinite(w), w, 0.f);
    };

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(ResidualRatioTrackingIntegrator, MonteCarloIntegrator);
MI_EXPORT_PLUGIN(ResidualRatioTrackingIntegrator, "Residual Ratio Tracking integrator");
NAMESPACE_END(mitsuba)
