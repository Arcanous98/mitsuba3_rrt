#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/scene.h>
// #include <mitsuba/render/texture.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Medium<Float, Spectrum>::Medium() : m_is_homogeneous(false), m_has_spectral_extinction(true), m_is_absorptive(false) {}

MI_VARIANT Medium<Float, Spectrum>::Medium(const Properties &props)
    : m_majorant_grid(nullptr), m_control_grid(nullptr), m_id(props.id()) {

    for (auto &[name, obj] : props.objects(false)) {
        auto *phase = dynamic_cast<PhaseFunction *>(obj.get());
        if (phase) {
            if (m_phase_function)
                Throw("Only a single phase function can be specified per medium");
            m_phase_function = phase;
            props.mark_queried(name);
        }
    }
    if (!m_phase_function) {
        // Create a default isotropic phase function
        m_phase_function =
            PluginManager::instance()->create_object<PhaseFunction>(Properties("isotropic"));
    }

    m_majorant_factor = props.get<ScalarFloat>("majorant_factor", 1.01);
    m_majorant_resolution_factor = props.get<size_t>("majorant_resolution_factor", 0);
    m_control_density = dr::NaN<ScalarFloat>;

    m_sample_emitters = props.get<bool>("sample_emitters", true);
    dr::set_attr(this, "use_emitter_sampling", m_sample_emitters);
    dr::set_attr(this, "phase_function", m_phase_function.get());
    // Properties props_sampler("stratified");
    // ref<Sampler> sp_strat = static_cast<Sampler *>(PluginManager::instance()->create_object<Sampler>(props_sampler));
    // sp_strat->seed(sp.template next_uint32(active_tracking), idx);
}

MI_VARIANT Medium<Float, Spectrum>::~Medium() {}

MI_VARIANT void Medium<Float, Spectrum>::traverse(TraversalCallback *callback) {
    callback->put_object("phase_function", m_phase_function.get(), +ParamFlags::Differentiable);
}

MI_VARIANT
typename Medium<Float, Spectrum>::MediumInteraction3f
Medium<Float, Spectrum>::sample_interaction(const Ray3f &ray, Float sample,
                                            UInt32 channel, Mask _active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, _active);

    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);

    const Float desired_tau = -dr::log(1 - sample);
    Float sampled_t(0.f);
    if (m_majorant_grid) {
        // --- Spatially-variying majorant (supergrid).
        // 1. Prepare for DDA traversal
        // Adapted from: https://github.com/francisengelmann/fast_voxel_traversal/blob/9664f0bde1943e69dbd1942f95efc31901fbbd42/main.cpp
        // TODO: allow precomputing all this (but be careful when ray origin is updated)
        auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);

        // 2. Traverse the medium with DDA until we reach the desired
        // optical depth.
        Mask active_tracking(active);
        Mask reached(!active);
        Float tau_acc(0.f);
        dr::Loop<Mask> dda_loop("Medium::sample_interaction_dda");
        dda_loop.put(active_tracking, reached, dda_t, dda_tmax, tau_acc, mei);
        dda_loop.init();
        while (dda_loop(dr::detach(active_tracking))) {
            // Figure out which axis we hit first.
            // `t_next` is the ray's `t` parameter when hitting that axis.
            Float t_next = dr::min(dda_tmax);
            Vector3f tmax_update;
            Mask got_assigned = false;
            for (size_t k = 0; k < 3; ++k) {
                Mask active_k = dr::eq(dda_tmax[k], t_next);
                tmax_update[k] = dr::select(!got_assigned && active_k, dda_tdelta[k], 0);
                got_assigned |= active_k;
            }

            // Lookup and accumulate majorant in current cell.
            dr::masked(mei.t, active_tracking) = 0.5f * (dda_t + t_next);
            dr::masked(mei.p, active_tracking) = ray(mei.t);
            // TODO: avoid this vcall, could lookup directly from the array
            // of floats (but we still need to account for the bbox, etc).
            Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
            Float tau_next = tau_acc + local_majorant * (t_next - dda_t);

            // For rays that will stop within this cell, figure out
            // the precise `t` parameter where `desired_tau` is reached.
            Float t_precise = dda_t + (desired_tau - tau_acc) / local_majorant;
            reached |= active_tracking && (tau_next >= desired_tau) && (t_precise < maxt) && (local_majorant > 0);
            dr::masked(dda_t, active_tracking) = dr::select(reached, t_precise, t_next);

            // Prepare for next iteration
            active_tracking &= !reached && (t_next < maxt);
            dr::masked(dda_tmax, active_tracking) = dda_tmax + tmax_update;
            dr::masked(tau_acc, active_tracking) = tau_next;
        }
        // Adopt the stopping location, making sure to convert to the main
        // ray's parametrization.
        sampled_t = dr::select(reached, dda_t, dr::Infinity<Float>);
    } else {
        // --- A single majorant for the whole volume.
        mei.combined_extinction = dr::detach(get_majorant(mei, active));
        Float m                = extract_channel(mei.combined_extinction, channel);
        sampled_t = mint + (desired_tau / m);
    }

    Mask valid_mei = active && (sampled_t <= maxt);
    dr::masked(mei.t, active) = dr::select(valid_mei, sampled_t, dr::Infinity<Float>);
    dr::masked(mei.p, active) = ray(sampled_t);

    if (m_majorant_grid) {
        // Otherwise it was already looked up above
        mei.combined_extinction = dr::detach(m_majorant_grid->eval_1(mei, valid_mei));
    }
    std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) =
        get_scattering_coefficients(mei, active);
    return mei;
}

MI_VARIANT
typename Medium<Float, Spectrum>::MediumInteraction3f
Medium<Float, Spectrum>::sample_interaction_real(const Ray3f &ray,
                                                 UInt32 seed_v0,
                                                 UInt32 seed_v1,
                                                 Mask _active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, _active);

    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);
    UInt32 channel(0);
    MediumInteraction3f mei_next = mei;
    Mask escaped = !active;
    // Spectrum weight = dr::full<Spectrum>(1.f, dr::width(ray));
    dr::Loop<Mask> loop("Medium::sample_interaction_real");

    // Get global majorant once and for all
    auto combined_extinction = get_majorant(mei, active);
    mei.combined_extinction  = combined_extinction;
    Float global_majorant = extract_channel(combined_extinction, channel);

    using PCG32 = mitsuba::PCG32<UInt32>;
    PCG32 sp;
    if constexpr (dr::is_array_v<Float>) {
        sp.seed(1, seed_v0, seed_v1);
    } else {
        sp.seed(1, seed_v0, PCG32_DEFAULT_STREAM);
    }
    loop.put(active, mei, mei_next, escaped, sp.state, sp.inc);
    // sampler->loop_put(loop);
    loop.init();

    while (loop(active)) {
        // Repeatedly sample from homogenized medium
        Float desired_tau = -dr::log(1 - sp.template next_float<Float>(active));
        Float sampled_t = mei_next.mint + desired_tau / global_majorant;

        Mask valid_mei = active && (sampled_t < maxt);
        mei_next.t     = sampled_t;
        mei_next.p     = ray(sampled_t);
        std::tie(mei_next.sigma_s, mei_next.sigma_n, mei_next.sigma_t) =
            get_scattering_coefficients(mei_next, valid_mei);

        // Determine whether it was a real or null interaction
        Float r = extract_channel(mei_next.sigma_t, channel) / global_majorant;
        Mask did_scatter = valid_mei && (sp.template next_float<Float>(active) < r);
        mei[did_scatter] = mei_next;

        mei_next.mint = sampled_t;
        escaped |= active && (mei_next.mint >= maxt);
        active &= !did_scatter && !escaped;
        mei.weight = dr::select(did_scatter, mei.sigma_s[0] / mei.sigma_t[0], 1.f);
    }

    dr::masked(mei.t, escaped) = dr::Infinity<Float>;
    mei.p                      = ray(mei.t);

    return mei;
}

MI_VARIANT
typename Medium<Float, Spectrum>::MediumInteraction3f
Medium<Float, Spectrum>::sample_interaction_real_super(const Ray3f &ray,
                                                 UInt32 seed_v0,
                                                 UInt32 seed_v1,
                                                 Mask _active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, _active);

    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);
    auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);

    Float sampled_t(0.f);
    // Ratio Tracking
    Float tau_acc(0.f);
    using PCG32 = mitsuba::PCG32<UInt32>;
    PCG32 sp;
    if constexpr (dr::is_array_v<Float>) {
        sp.seed(1, seed_v0, seed_v1);
    } else {
        sp.seed(1, seed_v0, PCG32_DEFAULT_STREAM);
    }

    Float desired_tau = 0.f;

    Mask active_tracking(active);
    Mask needs_new_target(active);
    dr::Loop<Mask> loop("Ratio Tracking Distance Sampler with DDA SuperGrid");
    Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
    
    loop.put(dda_t, dda_tmax, dda_tdelta, mei, local_majorant, tau_acc, needs_new_target, active_tracking, desired_tau, sp.state, sp.inc);
    // sampler->loop_put(loop);
    loop.init();
    while (loop(dr::detach(active_tracking))){
        dr::masked(desired_tau, active_tracking && needs_new_target) = -dr::log(1 - sp.template next_float<Float>(needs_new_target));
        dr::masked(tau_acc, active_tracking && needs_new_target) = 0.f;

        Float t_next = dr::min(dda_tmax);
        Vector3f tmax_update;
        Mask got_assigned = false;
        for (size_t k = 0; k < 3; ++k) {
            Mask active_k = dr::eq(dda_tmax[k], t_next);
            tmax_update[k] = dr::select(!got_assigned && active_k, dda_tdelta[k], 0);
            got_assigned |= active_k;
        }

        dr::masked(mei.t, active_tracking) = 0.5f * (dda_t + t_next);
        dr::masked(mei.p, active_tracking) = ray(mei.t);
        
        local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
        Float tau_next = tau_acc + local_majorant * (t_next - dda_t);

        // For rays that will stop within this cell, figure out
        // the precise `t` parameter where `desired_tau` is reached.
        Float t_precise = dda_t + (desired_tau - tau_acc) / local_majorant;
        Mask reached = active_tracking && (tau_next >= desired_tau) && (t_precise < maxt);
        Mask escaped = active_tracking && (t_next >= maxt);

        dr::masked(dda_t, active_tracking) = dr::select(reached, t_precise, dr::select(escaped, dr::Infinity<Float>, t_next));
        dr::masked(dda_tmax, active_tracking && !reached) = dda_tmax + tmax_update;
        dr::masked(tau_acc, active_tracking && !reached) = tau_next;

        needs_new_target = active_tracking && (reached || escaped);

        dr::masked(mei.t, needs_new_target) = dda_t;
        dr::masked(mei.p, needs_new_target) = ray(mei.t);

        std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, needs_new_target);
        Float r = dr::select(dr::neq(local_majorant, 0), mei.sigma_t[0] / local_majorant, 0);
        Mask scattering = reached && sp.template next_float<Float>(reached) < r;

        // Prepare for next iteration
        active_tracking &= !scattering;
        active_tracking &= !escaped; 

        mei.t = dr::select(scattering, dda_t, dr::Infinity<Float>);
        mei.weight = dr::select(scattering, mei.sigma_s[0] / mei.sigma_t[0], 1.f);
    }
    mei.p = ray(mei.t);
    return mei;
}

MI_VARIANT
typename Medium<Float, Spectrum>::MediumInteraction3f
Medium<Float, Spectrum>::sample_interaction_super_weighted_dt(const Ray3f &ray,
                                                            UInt32 seed_v0,
                                                            UInt32 seed_v1,
                                                            Mask _active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, _active);

    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);
    auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);

    Float sampled_t(0.f);
    // Ratio Tracking
    Float tau_acc(0.f);
    using PCG32 = mitsuba::PCG32<UInt32>;
    PCG32 sp;
    if constexpr (dr::is_array_v<Float>) {
        sp.seed(1, seed_v0, seed_v1);
    } else {
        sp.seed(1, seed_v0, PCG32_DEFAULT_STREAM);
    }

    Float desired_tau = 0.f;
    Float weight(1.f);
    Mask active_tracking(active);
    Mask needs_new_target(active);
    dr::Loop<Mask> loop("Ratio Tracking Distance Sampler with DDA SuperGrid");
    Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
    
    loop.put(weight, dda_t, dda_tmax, dda_tdelta, mei, local_majorant, tau_acc, needs_new_target, active_tracking, desired_tau, sp.state, sp.inc);
    // sampler->loop_put(loop);
    loop.init();
    while (loop(dr::detach(active_tracking))){
        dr::masked(desired_tau, active_tracking && needs_new_target) = -dr::log(1 - sp.template next_float<Float>(needs_new_target));
        dr::masked(tau_acc, active_tracking && needs_new_target) = 0.f;

        Float t_next = dr::min(dda_tmax);
        Vector3f tmax_update;
        Mask got_assigned = false;
        for (size_t k = 0; k < 3; ++k) {
            Mask active_k = dr::eq(dda_tmax[k], t_next);
            tmax_update[k] = dr::select(!got_assigned && active_k, dda_tdelta[k], 0);
            got_assigned |= active_k;
        }

        dr::masked(mei.t, active_tracking) = 0.5f * (dda_t + t_next);
        dr::masked(mei.p, active_tracking) = ray(mei.t);
        
        local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
        Float tau_next = tau_acc + local_majorant * (t_next - dda_t);

        // For rays that will stop within this cell, figure out
        // the precise `t` parameter where `desired_tau` is reached.
        Float t_precise = dda_t + (desired_tau - tau_acc) / local_majorant;
        Mask reached = active_tracking && (tau_next >= desired_tau) && (t_precise < maxt);
        Mask escaped = active_tracking && (t_next >= maxt);

        dr::masked(dda_t, active_tracking) = dr::select(reached, t_precise, dr::select(escaped, dr::Infinity<Float>, t_next));
        dr::masked(dda_tmax, active_tracking && !reached) = dda_tmax + tmax_update;
        dr::masked(tau_acc, active_tracking && !reached) = tau_next;

        needs_new_target = active_tracking && (reached || escaped);

        dr::masked(mei.t, needs_new_target) = dda_t;
        dr::masked(mei.p, needs_new_target) = ray(mei.t);

        std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, needs_new_target);
        // we assume that the local majorant is ALWAYS bigger than the local density eval
        Float residual = dr::select(dr::neq(local_majorant, 0), local_majorant - mei.sigma_t[0], 1.f);
        Float w_null = dr::select(dr::neq(local_majorant, 0), residual / local_majorant, 1.f);
        Float w_real = dr::select(dr::neq(local_majorant, 0), mei.sigma_t[0] / local_majorant, 1.f);
        Float p_null = dr::select(dr::neq(mei.sigma_t[0], 0), residual / (residual + mei.sigma_t[0]), 1.f);
        Float p_real = dr::select(dr::neq(mei.sigma_t[0], 0), mei.sigma_t[0] / (residual + mei.sigma_t[0]), 0.f);
        Float w_scattering = dr::select(dr::neq(mei.sigma_t[0], 0), mei.sigma_s[0] / mei.sigma_t[0], 1.f);

        Mask scattering = reached && sp.template next_float<Float>(reached) < p_real;

        // Prepare for next iteration
        active_tracking &= !scattering;
        active_tracking &= !escaped; 

        mei.t = dr::select(scattering, dda_t, dr::Infinity<Float>);
        mei.weight = dr::select(scattering, weight * w_real * w_scattering / p_real, 1.f);

        dr::masked(weight, reached) *= w_null / p_null;
    }
    mei.p = ray(mei.t);
    return mei;
}



MI_VARIANT
MI_INLINE
std::tuple<typename Medium<Float, Spectrum>::MediumInteraction3f, Float, Float,
           typename Medium<Float, Spectrum>::Mask>
Medium<Float, Spectrum>::prepare_interaction_sampling(const Ray3f &ray,
                                                      Mask active) const {
    // Initialize basic medium interaction fields
    MediumInteraction3f mei = dr::zeros<MediumInteraction3f>();
    mei.wi                  = -ray.d;
    mei.sh_frame            = Frame3f(mei.wi);
    mei.time                = ray.time;
    mei.wavelengths         = ray.wavelengths;
    mei.medium              = this;

    auto [aabb_its, mint, maxt] = intersect_aabb(ray);
    aabb_its &= (dr::isfinite(mint) || dr::isfinite(maxt));
    active &= aabb_its;
    dr::masked(mint, !active) = 0.f;
    dr::masked(maxt, !active) = dr::Infinity<Float>;

    mint = dr::maximum(0.f, mint);
    maxt = dr::minimum(ray.maxt, maxt);
    mei.mint = mint;

    return std::make_tuple(mei, mint, maxt, active);
}

MI_VARIANT
MI_INLINE
dr::Array<Float, 16> Medium<Float, Spectrum>::piecewise_maj(const Ray3f &ray, Mask _active) const {
    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);
    auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);

    using Vector16f = dr::Array<Float, 16>;
    Mask active_traversal(active);
    Vector16f pdf(0.f);
    Float segment_length = (maxt - mint) / 16.f;
    dr::Loop<Mask> loop("Construct Piece-wise PDF");
    loop.put(pdf, dda_t, dda_tdelta, dda_tmax, active_traversal);
    loop.init();
    while (loop(dr::detach(active_traversal))){

        Float t_next = dr::min(dda_tmax);
        Vector3f tmax_update;
        Mask got_assigned = false;
        for (size_t k = 0; k < 3; ++k) {
            Mask active_k = dr::eq(dda_tmax[k], t_next);
            tmax_update[k] = dr::select(!got_assigned && active_k, dda_tdelta[k], 0);
            got_assigned |= active_k;
        }

        dr::masked(mei.t, active_traversal) = 0.5f * (dda_t + t_next);
        dr::masked(mei.p, active_traversal) = ray(mei.t);

        Float local_majorant = m_majorant_grid->eval_1(mei, active_traversal);
        
        UInt32 majorant_idx = dr::floor((mei.t - mint) / segment_length);
        for (size_t k = 0; k < 16; ++k) {
            Mask assign_this = dr::eq(majorant_idx, k);
            pdf[k] = dr::select(assign_this, dr::maximum(local_majorant, pdf[k]), pdf[k]);
        }
        
        Mask escaped = active_traversal && (t_next >= maxt);

        dr::masked(dda_t, active_traversal) = dr::select(escaped, dr::Infinity<Float>, t_next);
        dr::masked(dda_tmax, active_traversal) = dda_tmax + tmax_update;

        // Prepare for next iteration
        active_traversal &= !escaped; 
    }
    
    return pdf;
}
MI_VARIANT
MI_INLINE
std::tuple<Float, typename Medium<Float, Spectrum>::Vector3f,
           typename Medium<Float, Spectrum>::Vector3f>
Medium<Float, Spectrum>::prepare_dda_traversal(const Volume *majorant_grid,
                                               const Ray3f &ray, Float mint,
                                               Float maxt, Mask /*active*/) const {
    const auto &bbox   = majorant_grid->bbox();
    const auto extents = bbox.extents();
    Ray3f local_ray(
        /* o */ (ray.o - bbox.min) / extents,
        /* d */ ray.d / extents, ray.time, ray.wavelengths);
    const ScalarVector3i res  = majorant_grid->resolution();
    Vector3f local_voxel_size = 1.f / res;

    // The id of the first and last voxels hit by the ray
    Vector3i current_voxel(dr::floor(local_ray(mint) / local_voxel_size));
    Vector3i last_voxel(dr::floor(local_ray(maxt) / local_voxel_size));
    // By definition, current and last voxels should be valid voxel indices.
    current_voxel = dr::clamp(current_voxel, 0, res - 1);
    last_voxel    = dr::clamp(last_voxel, 0, res - 1);

    // Increment (in number of voxels) to take at each step
    Vector3i step = dr::select(local_ray.d >= 0, 1, -1);

    // Distance along the ray to the next voxel border from the current position
    Vector3f next_voxel_boundary = (current_voxel + step) * local_voxel_size;
    next_voxel_boundary += dr::select(
        dr::neq(current_voxel, last_voxel) && (local_ray.d < 0), local_voxel_size, 0);

    // Value of ray parameter until next intersection with voxel-border along each axis
    auto ray_nonzero = dr::neq(local_ray.d, 0);
    Vector3f dda_tmax =
        dr::select(ray_nonzero, (next_voxel_boundary - local_ray.o) / local_ray.d,
                   dr::Infinity<Float>);

    // How far along each component of the ray we must move to move by one voxel
    Vector3f dda_tdelta = dr::select(
        ray_nonzero, step * local_voxel_size / local_ray.d, dr::Infinity<Float>);

    // Current ray parameter throughout DDA traversal
    Float dda_t = mint;

    // Note: `t` parameters on the reparametrized ray yield locations on the
    // normalized majorant supergrid in [0, 1]^3. But they are also directly
    // valid parameters on the original ray, yielding positions in the
    // bbox-aligned supergrid.
    return { dda_t, dda_tmax, dda_tdelta };
}

MI_VARIANT
MI_INLINE Float
Medium<Float, Spectrum>::extract_channel(Spectrum value, UInt32 channel) {
    Float result = value[0];
    if constexpr (is_rgb_v<Spectrum>) { // Handle RGB rendering
        dr::masked(result, dr::eq(channel, 1u)) = value[1];
        dr::masked(result, dr::eq(channel, 2u)) = value[2];
    } else {
        DRJIT_MARK_USED(channel);
    }
    return result;
}
MI_VARIANT
std::pair<typename Medium<Float, Spectrum>::UnpolarizedSpectrum,
          typename Medium<Float, Spectrum>::UnpolarizedSpectrum>
Medium<Float, Spectrum>::transmittance_eval_pdf(const MediumInteraction3f &mi,
                                                const SurfaceInteraction3f &si,
                                                Mask active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
    
    Float t      = dr::minimum(mi.t, si.t) - mi.mint;
    UnpolarizedSpectrum tr  = dr::exp(-t * mi.combined_extinction);
    UnpolarizedSpectrum pdf = dr::select(si.t < mi.t, tr, tr * mi.combined_extinction);
    return { tr, pdf };
}
MI_VARIANT
std::pair<typename Medium<Float, Spectrum>::MediumInteraction3f,
          typename Medium<Float, Spectrum>::UnpolarizedSpectrum>
Medium<Float, Spectrum>::integrate_tr(const Ray3f &ray,
                                    UInt32 seed_v0,
                                    UInt32 seed_v1,
                                    uint32_t estimator_selector,
                                    uint32_t idx,
                                    Mask _active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, _active);
    using Vector16f = dr::Array<Float, 16>;
    
    // TODO: add early exits when transmittance reaches a lower bound
    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);
    
    Float t(mint);
    Spectrum TotalTr(1.f);
    Float max_delta_t = maxt-mint;
    Mask active_tracking(active);

    using PCG32 = mitsuba::PCG32<UInt32>;
    PCG32 sp;
    
    if constexpr (dr::is_array_v<Float>) {
        sp.seed(1, seed_v0, seed_v1);
    } else {
        sp.seed(1, seed_v0, PCG32_DEFAULT_STREAM);
    }
    if (estimator_selector == 0) {
        // Ratio Tracking
        Spectrum T(1.f);
        Spectrum combined_extinction = get_majorant(mei, active);
        Float m = combined_extinction[0]; // default control mu is the average of volume densities
        dr::Loop<Mask> loop("Ratio Tracking Estimator");
        loop.put(T, t, mei, TotalTr, active_tracking, m, sp.state, sp.inc);
        // sampler->loop_put(loop);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            // Float sample = sampler->next_1d(active_tracking);
            Float sample = sp.template next_float<Float>(active_tracking);
            t -= (dr::log(1 - sample) / m);

            Mask exceeded_limits = t >= maxt && active_tracking;
            active_tracking &= !exceeded_limits;

            dr::masked(mei.t, exceeded_limits) =  dr::Infinity<Float>;
            dr::masked(mei.p, exceeded_limits) =  ray(mei.t);
            dr::masked(TotalTr, exceeded_limits) = Spectrum(T);

            dr::masked(mei.p, active_tracking) = ray(t);
            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, active_tracking);
            Float r = dr::select(active_tracking, mei.sigma_t[0] / m, 0.f);
            dr::masked(T, active_tracking) *= (1.f - r);
        }
    } else if (estimator_selector == 1) {
        // Residual Ratio Trackings
        Spectrum T(1.f);
        Spectrum combined_extinction = get_majorant(mei, active);
        Float m = combined_extinction[0]; // default control mu is the average of volume densities
        Spectrum combined_control_extinction = get_control_sigma_t(mei, active_tracking);
        Spectrum combined_residual = combined_extinction - combined_control_extinction;
        Float m_c = combined_control_extinction[0]; // default control mu is the average of volume densities
        Float m_r = combined_residual[0]; //Novak14: select mu_r conservatively by selecting the max difference between mu and mu_c
        Spectrum Tc = dr::exp(- m_c * (max_delta_t));
        Spectrum Tr = 1.f;
        dr::Loop<Mask> loop("Residual Ratio Tracking Estimator");
        loop.put(Tr, t, mei, TotalTr, active_tracking, sp.state, sp.inc);
        // sampler->loop_put(loop);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            // Float sample = sampler->next_1d(active_tracking);
            Float sample = sp.template next_float<Float>(active_tracking);
            t -= (dr::log(1 - sample) / m_r);

            Mask exceeded_limits = t >= maxt && active_tracking;
            active_tracking &= !exceeded_limits;  

            dr::masked(mei.t, exceeded_limits) =  dr::Infinity<Float>;
            dr::masked(mei.p, exceeded_limits) =  ray(mei.t);
            dr::masked(TotalTr, exceeded_limits) =  Tr*Tc;
            
            dr::masked(mei.p, active_tracking) = ray(t);
            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, active_tracking);
            dr::masked(Tr, active_tracking) *= (Spectrum(1.f) - (mei.sigma_t - combined_control_extinction) / combined_residual); 
        }
    } else if (estimator_selector == 2) {
        // Ratio Tracking \w local majorants
        auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);

        Mask needs_new_target(active);
        Spectrum T(1.f);
        Float tau_acc(0.f);
        Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
        Float desired_tau = 0.f;

        dr::Loop<Mask> loop("Ratio Tracking Integrator with Local Majorants");
        loop.put(T, dda_t, mei, local_majorant, TotalTr, dda_tdelta, dda_tmax, active_tracking, needs_new_target, desired_tau, tau_acc, sp.state, sp.inc);
        // sampler->loop_put(loop);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            dr::masked(desired_tau, active_tracking && needs_new_target) = -dr::log(1 - sp.template next_float<Float>(needs_new_target));
            dr::masked(tau_acc, active_tracking && needs_new_target) = 0.f;

            Float t_next = dr::min(dda_tmax);
            Vector3f tmax_update;
            Mask got_assigned = false;
            for (size_t k = 0; k < 3; ++k) {
                Mask active_k = dr::eq(dda_tmax[k], t_next);
                tmax_update[k] = dr::select(!got_assigned && active_k, dda_tdelta[k], 0);
                got_assigned |= active_k;
            }

            dr::masked(mei.t, active_tracking) = 0.5f * (dda_t + t_next);
            dr::masked(mei.p, active_tracking) = ray(mei.t);

            local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
            Float tau_next = tau_acc + local_majorant * (t_next - dda_t);

            Float t_precise = dda_t + (desired_tau - tau_acc) / local_majorant;
            Mask reached = active_tracking && (tau_next >= desired_tau) && (t_precise < maxt);
            Mask escaped = active_tracking && (t_next >= maxt);

            dr::masked(dda_t, active_tracking) = dr::select(reached, t_precise, dr::select(escaped, dr::Infinity<Float>, t_next));
            dr::masked(dda_tmax, active_tracking && !reached) = dda_tmax + tmax_update;
            dr::masked(tau_acc, active_tracking && !reached) = tau_next;

            needs_new_target = active_tracking && (reached || escaped);

            dr::masked(mei.t, needs_new_target) = dda_t;
            dr::masked(mei.p, needs_new_target) = ray(mei.t);

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, needs_new_target);
            Float r = dr::select(reached, mei.sigma_t[0] / local_majorant, 0);

            dr::masked(T, active_tracking) *= (Spectrum(1.f) - r);

            // Prepare for next iteration
            active_tracking &= !escaped; 

            dr::masked(mei.t, escaped) =  dr::Infinity<Float>;
            dr::masked(mei.p, escaped) =  ray(mei.t);
            dr::masked(TotalTr, escaped) = T; 
        }
    } else if (estimator_selector == 3) {
        // Residual Ratio Tracking \w local majorants and local control coefficients
        auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);
        Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
        Float local_control = m_control_grid->eval_1(mei, active_tracking);
        Float local_residual = local_majorant - local_control;
        Mask needs_new_target(active);

        Spectrum Tc(1.f);// = dr::exp(- local_control * (dr::min(dda_tmax) - mint));
        Spectrum Tr(1.f);

        Float tau_acc(0.f);
        Float desired_tau = -dr::log(1 - sp.template next_float<Float>(active_tracking));
        dr::Loop<Mask> loop("Residual Ratio Tracking Integrator with Local Majorants and Control Coeffs");
        loop.put(Tr, Tc, TotalTr, dda_t, dda_tdelta, dda_tmax, mei, local_majorant, local_control, local_residual, active_tracking, needs_new_target, desired_tau, tau_acc, sp.state, sp.inc);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            dr::masked(desired_tau, active_tracking && needs_new_target) = -dr::log(1 - sp.template next_float<Float>(needs_new_target));
            dr::masked(tau_acc, active_tracking && needs_new_target) = 0.f;

            Float t_next = dr::min(dda_tmax);
            Vector3f tmax_update;
            Mask got_assigned = false;
            for (size_t k = 0; k < 3; ++k) {
                Mask active_k = dr::eq(dda_tmax[k], t_next);
                tmax_update[k] = dr::select(!got_assigned && active_k, dda_tdelta[k], 0);
                got_assigned |= active_k;
            }

            dr::masked(mei.t, active_tracking) = 0.5f * (dda_t + t_next);
            dr::masked(mei.p, active_tracking) = ray(mei.t);

            local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
            local_control = m_control_grid->eval_1(mei, active_tracking);
            local_residual = local_majorant - local_control;//dr::clamp(local_majorant - local_control,0.00001,dr::Infinity<Float>);

            Float tau_next = tau_acc + local_residual * (t_next - dda_t);

            Float t_precise = dda_t + (desired_tau - tau_acc) / local_majorant;
            Mask reached = active_tracking && (tau_next >= desired_tau) && (t_precise < maxt);
            Mask escaped = active_tracking && (t_next >= maxt);

            Float distance_travelled = dr::select(reached, t_precise - dda_t, dr::select(escaped, maxt - dda_t, t_next - dda_t));

            dr::masked(dda_t, active_tracking) = dr::select(reached, t_precise, dr::select(escaped, dr::Infinity<Float>, t_next));
            dr::masked(dda_tmax, active_tracking && !reached) = dda_tmax + tmax_update;
            dr::masked(tau_acc, active_tracking && !reached) = tau_next;

            needs_new_target = active_tracking && (reached || escaped);

            dr::masked(mei.t, needs_new_target) = dda_t;
            dr::masked(mei.p, needs_new_target) = ray(mei.t);

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, reached);
            Float r = dr::select(reached, (mei.sigma_t[0] - local_control) / local_residual, 0.f);

            dr::masked(Tr, active_tracking) *= (Spectrum(1.f) - r);
            dr::masked(Tc, active_tracking) *= dr::exp(- local_control * distance_travelled);

            active_tracking &= !escaped; 
            dr::masked(mei.t, escaped) =  dr::Infinity<Float>;
            dr::masked(mei.p, escaped) =  ray(mei.t);
            dr::masked(TotalTr, escaped) = Tr*Tc; 
        }
    } else if (estimator_selector == 4) {
        // Next Flight Estimator
        Spectrum tau_accum_f(1.f);
        // Vector16f piecewise_majorant = piecewise_maj(ray, active_tracking);
        // Float segment_length(max_delta_t/16.f);
        // Float local_majorant = dr::clamp(piecewise_majorant[0],1e-2,dr::Infinity<Float>);

        Spectrum combined_extinction = get_majorant(mei, active_tracking);
        // Spectrum T(1.f);
        // for (size_t i = 0; i<16; i++){
        //     T *= dr::exp(- segment_length * piecewise_majorant[i]);
        // }
        Spectrum T = dr::exp(- max_delta_t * combined_extinction);
        Float m = combined_extinction[0]; // default control mu is the average of volume densities
        dr::Loop<Mask> loop("Next Flight Estimator");
        loop.put(T, t, tau_accum_f, mei, TotalTr, active_tracking, sp.state, sp.inc);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            Float sample = sp.template next_float<Float>(active_tracking);
            t -= (dr::log(1 - sample) / m);

            Mask exceeded_limits = t >= maxt && active_tracking;
            active_tracking &= !exceeded_limits;

            dr::masked(mei.t, exceeded_limits) =  dr::Infinity<Float>;
            dr::masked(mei.p, exceeded_limits) =  ray(mei.t);
            dr::masked(TotalTr, exceeded_limits) = T;

            dr::masked(mei.p, active_tracking) = ray(t);
            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, active_tracking);
            Float r = dr::select(active_tracking, mei.sigma_t[0] / m, 0.f);
            dr::masked(tau_accum_f, active_tracking) *= (Spectrum(1.f) - r);
            
            dr::masked(T, active_tracking) += tau_accum_f * dr::exp(- combined_extinction * (maxt - t));
        }

    } else if (estimator_selector == 5){
        // Ray Marching (Biased)
        Spectrum combined_extinction = get_majorant(mei, active_tracking);
        Float m = combined_extinction[0]; // default control mu is the average of volume densities
        Spectrum T(0.f);
        Float step = 1.f / m;
        dr::Loop<Mask> loop("Ray Marching Estimator");
        loop.put(t, T, step, mei, TotalTr, active_tracking, m, sp.state, sp.inc);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            // Float sample = sampler->next_1d(active_tracking);
            dr::masked(step, active_tracking) = dr::minimum(step, maxt - t);

            Float jump = t + sp.template next_float<Float>(active_tracking) * step;

            Mask exceeded_limits = t >= maxt && active_tracking;
            active_tracking &= !exceeded_limits;

            dr::masked(mei.t, exceeded_limits) =  dr::Infinity<Float>;
            dr::masked(mei.p, exceeded_limits) =  ray(mei.t);
            dr::masked(TotalTr, exceeded_limits) = dr::exp(-T);

            dr::masked(mei.p, active_tracking) = ray(jump);
            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, active_tracking);
            dr::masked(T, active_tracking) += mei.sigma_t * step;
            dr::masked(t, active_tracking) += step;
        }
    } else if (estimator_selector == 6){
        // Power Series Cumulative
        // commented code tried to leverage a piecewise constant majorant along the line but it has artifacts currently. Do not use
        Spectrum combined_extinction = get_majorant(mei, active_tracking);//keeps getting stuck with large majorants
        // Vector16f piecewise_majorant = piecewise_maj(ray, active_tracking);
        // Float segment_length(max_delta_t/16.f);
        Spectrum T(0.f);
        Float i(1.f);
        Float max_samples(1000.f); //this will introduce some bias but better than getting stuck
        Spectrum accum_weight(1.f);
        // Stratified sampler
        Float sample(0.f);
        UInt32 dim_index(0);
        if (false) {
            UInt32 sample_indices = dim_index * idx;
            dr::masked(dim_index, active_tracking) +=1;
            UInt32 perm_seed = seed_v0 + dim_index;
            UInt32 p = permute_kensler(sample_indices, uint32_t(16), perm_seed, active_tracking);
            Float j = sp.template next_float<Float>(active_tracking);
            sample = (p + j) / uint32_t(16);
        } else {
            sample = sp.template next_float<Float>(active_tracking);
        }
        //
        Float rr = dr::clamp(sample, 0.0000001f, 1.f);
        // Spectrum Tc(1.f);
        // for (size_t i = 0; i<16; i++){
        //     Tc *= dr::exp(- segment_length * piecewise_majorant[i]);
        // }
        Spectrum Tc = dr::exp(- max_delta_t * combined_extinction);
        Float weight_i(1.f);
        Float max_abs_w(1.f);
        Float random_t(mint);
        dr::Loop<Mask> loop("Power Series Cumulative Estimator");
        loop.put(random_t, weight_i, dim_index, i, T, rr, accum_weight, mei, TotalTr, active_tracking, sp.state, sp.inc);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            // Stratified sampler
            Float sample(0.f);
            if (false) {
                UInt32 sample_indices = dim_index * idx;
                dr::masked(dim_index, active_tracking) +=1;
                UInt32 perm_seed = seed_v0 + dim_index;
                UInt32 p = permute_kensler(sample_indices, uint32_t(16), perm_seed, active_tracking);
                Float j = sp.template next_float<Float>(active_tracking);
                sample = (p + j) / uint32_t(16);
            } else {
                sample = sp.template next_float<Float>(active_tracking);
            }
            dr::masked(random_t, active_tracking) = dr::clamp((sample * max_delta_t + mint), mint, maxt);
            dr::masked(mei.p, active_tracking) = ray(random_t);

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, active_tracking);
            
            // UInt32 majorant_idx = dr::floor((random_t - mint) / segment_length);
            // // TODO: make this more efficient...
            // Float local_majorant(0.f);
            // for (size_t k = 0; k < 16; ++k) {
            //     Mask assign_this = dr::eq(majorant_idx, k);
            //     local_majorant = dr::select(assign_this, piecewise_majorant[k], local_majorant);
            // }
            // dr::masked(weight_i, active_tracking) = (1.f / i) * (local_majorant - mei.sigma_t[0]) * segment_length;
            dr::masked(weight_i, active_tracking) = (1.f / i) * (combined_extinction[0] - mei.sigma_t[0]) * max_delta_t;

            // dr::masked(weight_i, active_tracking) = dr::select(dr::isnan(weight_i), 1.f, weight_i);
            
            dr::masked(T, active_tracking) += accum_weight;//dr::select(dr::isnan(accum_weight[0]), Spectrum(0.f), accum_weight);

            dr::masked(max_abs_w, active_tracking) = dr::max(dr::abs(accum_weight * weight_i));
            // dr::masked(max_abs_w, active_tracking) = dr::select(dr::isnan(max_abs_w), 1.f, max_abs_w);

            Mask accept = active_tracking && max_abs_w > 1.f;
            dr::masked(max_abs_w, accept) = 1.f;

            Mask terminate_recoursion = active_tracking && (max_abs_w <= rr) || (i > max_samples);
            active_tracking &= !terminate_recoursion;
            
            dr::masked(rr, active_tracking) /= max_abs_w;// dr::select(dr::isnan(rr / max_abs_w), rr, rr / max_abs_w);
            dr::masked(i, active_tracking) += 1.f;
            dr::masked(accum_weight, accept) *= weight_i;//dr::select(dr::isnan((accum_weight * weight_i)[0]), accum_weight, accum_weight * weight_i);

            dr::masked(mei.t, terminate_recoursion) =  dr::Infinity<Float>;
            dr::masked(mei.p, terminate_recoursion) =  ray(mei.t);
            dr::masked(TotalTr, terminate_recoursion) = dr::select(dr::isnan(T), 1.f, dr::clamp(T * Tc, 0.f, 1.f));
        }
    } else if (estimator_selector == 7){
        // Power Series CMF -- same as Power Series Cumulative but does not start RR until a certain amount of density has been reached
        Float sample(0.f);
        UInt32 dim_index(0);
        if (false) {
            UInt32 sample_indices = dim_index * idx;
            dr::masked(dim_index, active_tracking) +=1;
            UInt32 perm_seed = seed_v0 + dim_index;
            UInt32 p = permute_kensler(sample_indices, uint32_t(16), perm_seed, active_tracking);
            Float j = sp.template next_float<Float>(active_tracking);
            sample = (p + j) / uint32_t(16);
        } else {
            sample = sp.template next_float<Float>(active_tracking);
        }

        Spectrum combined_extinction = get_majorant(mei, active_tracking); //keeps getting stuck on large majorants
        Spectrum accum_cdf(0.f);
        Spectrum T(0.f);
        Float i(1.f);
        Float max_samples(1000.f); //this will introduce bias but ensure we don't get trapped in the recoursion
        Spectrum accum_weight(1.f);
        Float rr = dr::clamp(sample, 0.0000001f, 1.f); // dr::clamp(sp.template next_float<Float>(active_tracking), 0.00001f, 1.f);

        Spectrum tau_c =  max_delta_t * combined_extinction;
        Spectrum Tc = dr::exp(- tau_c);
        Float cutoff(0.99f);
        Float rr_cut(1.f);
        Spectrum prev_pdf = Tc;
        Mask start_rr(!active_tracking);

        dr::Loop<Mask> loop_cdf("Power Series CMF Estimator");
        loop_cdf.put(start_rr, dim_index, i, T, TotalTr, rr, accum_cdf, prev_pdf, accum_weight, mei, active_tracking, sp.state, sp.inc);
        loop_cdf.init();
        while (loop_cdf(dr::detach(active_tracking))){
            // stratified sampler
            Float sample(0.f);
            if (false) {
                UInt32 sample_indices = dim_index * idx;
                dr::masked(dim_index, active_tracking) +=1;
                UInt32 perm_seed = seed_v0 + dim_index;
                UInt32 p = permute_kensler(sample_indices, uint32_t(16), perm_seed, active_tracking);
                Float j = sp.template next_float<Float>(active_tracking);
                sample = (p + j) / uint32_t(16);
            } else {
                sample = sp.template next_float<Float>(active_tracking);
            }
            //
            Float random_t = sample * max_delta_t + mint; //sp.template next_float<Float>(active_tracking) * max_delta_t + mint;
            dr::masked(mei.p, active_tracking) = ray(random_t);
            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, active_tracking);
                        
            Float weight_i = (1.f / i) * (combined_extinction[0] - mei.sigma_t[0]) * max_delta_t;
            weight_i = dr::select(dr::isnan(weight_i), 1.f, weight_i);
            dr::masked(accum_cdf, active_tracking) += prev_pdf;
            dr::masked(prev_pdf, active_tracking) *= (tau_c / i);
            dr::masked(T, active_tracking) += accum_weight;
            dr::masked(accum_weight, active_tracking) *= dr::select(start_rr, weight_i / rr_cut, weight_i);
            dr::masked(i, active_tracking) += 1.f;

            rr_cut = tau_c[0] / i;
            start_rr = active_tracking && (accum_cdf[0] >= cutoff);
            Mask terminate_rr = start_rr && (rr_cut <= rr) || (i > max_samples);
            active_tracking &= !terminate_rr;
            dr::masked(rr, start_rr) /= rr_cut;

            dr::masked(mei.t, terminate_rr) =  dr::Infinity<Float>;
            dr::masked(mei.p, terminate_rr) =  ray(mei.t);
            dr::masked(TotalTr, terminate_rr) = dr::select(dr::isnan(T), 1.f, dr::clamp(T * Tc, 0.f, 1.f));
        }
    } else if (estimator_selector == 8) {
        // Next Flight Estimator \w local majorants
        Spectrum tau_accum_f(1.f);
        auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);
        Vector16f piecewise_majorant = piecewise_maj(ray, active_tracking);
        Float segment_length(max_delta_t/16.f);

        Spectrum combined_extinction = get_majorant(mei, active_tracking); //keeps getting stuck on large majorants
        Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
        Float desired_tau(0.f);
        Float tau_acc(0.f);
        Mask needs_new_target(active_tracking);

        Spectrum T(0.f);
        dr::Loop<Mask> loop("Next Flight Estimator with Local Majorants");
        loop.put(local_majorant, needs_new_target, desired_tau, tau_acc, dda_t, dda_tmax, dda_tdelta, T, tau_accum_f, mei, TotalTr, active_tracking, sp.state, sp.inc);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            dr::masked(desired_tau, active_tracking && needs_new_target) = -dr::log(1 - sp.template next_float<Float>(needs_new_target));
            dr::masked(tau_acc, active_tracking && needs_new_target) = 0.f;

            Float t_next = dr::min(dda_tmax);
            Vector3f tmax_update;
            Mask got_assigned = false;
            for (size_t k = 0; k < 3; ++k) {
                Mask active_k = dr::eq(dda_tmax[k], t_next);
                tmax_update[k] = dr::select(!got_assigned && active_k, dda_tdelta[k], 0);
                got_assigned |= active_k;
            }

            dr::masked(T, active_tracking) += dr::exp(- local_majorant * (t_next - dda_t));

            dr::masked(mei.t, active_tracking) = 0.5f * (dda_t + t_next);
            dr::masked(mei.p, active_tracking) = ray(mei.t);

            local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
            Float tau_next = tau_acc + local_majorant * (t_next - dda_t);

            Float t_precise = dda_t + (desired_tau - tau_acc) / local_majorant;
            Mask reached = active_tracking && (tau_next >= desired_tau) && (t_precise < maxt);
            Mask escaped = active_tracking && (t_next >= maxt);

            dr::masked(dda_t, active_tracking) = dr::select(reached, t_precise, dr::select(escaped, dr::Infinity<Float>, t_next));
            dr::masked(dda_tmax, active_tracking && !reached) = dda_tmax + tmax_update;
            dr::masked(tau_acc, active_tracking && !reached) = tau_next;

            needs_new_target = active_tracking && (reached || escaped);

            dr::masked(mei.t, needs_new_target) = dda_t;
            dr::masked(mei.p, needs_new_target) = ray(mei.t);

            // Prepare for next iteration
            active_tracking &= !escaped; 

            dr::masked(mei.t, escaped) =  dr::Infinity<Float>;
            dr::masked(mei.p, escaped) =  ray(mei.t);
            dr::masked(TotalTr, escaped) = T; 

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, needs_new_target);
            Float r = dr::select(dr::neq(local_majorant, 0) && reached, mei.sigma_t[0] / local_majorant, 0.f);
            Float next_tau = 1.f - r;

            Spectrum piecewise_extinction(1.f);
            UInt32 majorant_idx = dr::floor((t - mint) / segment_length);
            for (size_t k = 0; k < 16; ++k) {
                Mask assign_this = k >= majorant_idx;
                Mask this_segment = dr::eq(majorant_idx, k);
                piecewise_extinction *= dr::select(assign_this, dr::exp(-piecewise_majorant[k] * dr::select(this_segment, segment_length * (k + 1) - t, segment_length)), 1.f);
            }
            dr::masked(T, active_tracking) += tau_accum_f * piecewise_extinction;

            dr::masked(tau_accum_f, reached) *= next_tau;
            dr::masked(T, reached) += tau_accum_f * piecewise_extinction;//dr::exp(- combined_extinction * (maxt - dda_t)); //this is wrong. I would need a piecewise pdf precomputed
        }
    } else {
        std::cout<<"Estimator: "<<estimator_selector<<std::endl;
        NotImplementedError("select one of the following estimators: rt, rrt, rt_local, rrt_local, nf, rm, ps_cum, ps_cmf");
    }
    
    return {mei, TotalTr};
}


MI_IMPLEMENT_CLASS_VARIANT(Medium, Object, "medium")
MI_INSTANTIATE_CLASS(Medium)
NAMESPACE_END(mitsuba)
