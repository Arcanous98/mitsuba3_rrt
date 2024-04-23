#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/scene.h>
// #include <mitsuba/render/texture.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Medium<Float, Spectrum>::Medium() : m_is_homogeneous(false), m_has_spectral_extinction(true) {}

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
    Float sampled_t;
    if (m_majorant_grid) {
        // --- Spatially-variying majorant (supergrid).
        // 1. Prepare for DDA traversal
        // Adapted from: https://github.com/francisengelmann/fast_voxel_traversal/blob/9664f0bde1943e69dbd1942f95efc31901fbbd42/main.cpp
        // TODO: allow precomputing all this (but be careful when ray origin is updated)
        auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);

        // 2. Traverse the medium with DDA until we reach the desired
        // optical depth.
        Mask active_dda = active;
        Mask reached = false;
        Float tau_acc = 0.f;
        dr::Loop<Mask> dda_loop("Medium::sample_interaction_dda");
        dda_loop.put(active_dda, reached, dda_t, dda_tmax, tau_acc, mei);
        dda_loop.init();
        while (dda_loop(dr::detach(active_dda))) {
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
            dr::masked(mei.t, active_dda) = 0.5f * (dda_t + t_next);
            dr::masked(mei.p, active_dda) = ray(mei.t);
            // TODO: avoid this vcall, could lookup directly from the array
            // of floats (but we still need to account for the bbox, etc).
            Float majorant = m_majorant_grid->eval_1(mei, active_dda);
            Float tau_next = tau_acc + majorant * (t_next - dda_t);

            // For rays that will stop within this cell, figure out
            // the precise `t` parameter where `desired_tau` is reached.
            Float t_precise = dda_t + (desired_tau - tau_acc) / majorant;
            reached |= active_dda && (majorant > 0) && (t_precise < maxt) && (tau_next >= desired_tau);
            dr::masked(dda_t, active_dda) = dr::select(reached, t_precise, t_next);

            // Prepare for next iteration
            active_dda &= !reached && (t_next < maxt);
            dr::masked(dda_tmax, active_dda) = dda_tmax + tmax_update;
            dr::masked(tau_acc, active_dda) = tau_next;
        }

        // Adopt the stopping location, making sure to convert to the main
        // ray's parametrization.
        sampled_t = dr::select(reached, dda_t, dr::Infinity<Float>);
    } else {
        // --- A single majorant for the whole volume.
        mei.combined_extinction = dr::detach(get_majorant(mei, active));
        Float m                = extract_channel(mei.combined_extinction, channel);
        sampled_t              = mint + (desired_tau / m);
    }

    Mask valid_mei = active && (sampled_t <= maxt);
    mei.t          = dr::select(valid_mei, sampled_t, dr::Infinity<Float>);
    mei.p          = ray(sampled_t);

    if (m_majorant_grid) {
        // Otherwise it was already looked up above
        mei.combined_extinction = dr::detach(m_majorant_grid->eval_1(mei, valid_mei));
    }
    std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) =
        get_scattering_coefficients(mei, valid_mei);
    return mei;
}

MI_VARIANT
typename Medium<Float, Spectrum>::MediumInteraction3f
Medium<Float, Spectrum>::sample_interaction_real(const Ray3f &ray,
                                                 Sampler *sampler,
                                                 Mask _active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumSample, _active);

    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);
    UInt32 channel(0);
    MediumInteraction3f mei_next = mei;
    Mask escaped = !active;
    Spectrum weight = dr::full<Spectrum>(1.f, dr::width(ray));
    dr::Loop<Mask> loop("Medium::sample_interaction_real");

    // Get global majorant once and for all
    auto combined_extinction = get_majorant(mei, active);
    mei.combined_extinction  = combined_extinction;
    Float global_majorant = extract_channel(combined_extinction, channel);

    loop.put(active, mei, mei_next, escaped, weight);
    sampler->loop_put(loop);
    loop.init();

    while (loop(active)) {
        // Repeatedly sample from homogenized medium
        Float desired_tau = -dr::log(1 - sampler->next_1d(active));
        Float sampled_t = mei_next.mint + desired_tau / global_majorant;

        Mask valid_mei = active && (sampled_t < maxt);
        mei_next.t     = sampled_t;
        mei_next.p     = ray(sampled_t);
        std::tie(mei_next.sigma_s, mei_next.sigma_n, mei_next.sigma_t) =
            get_scattering_coefficients(mei_next, valid_mei);

        // Determine whether it was a real or null interaction
        Float r = extract_channel(mei_next.sigma_t, channel) / global_majorant;
        Mask did_scatter = valid_mei && (sampler->next_1d(valid_mei) < r);
        mei[did_scatter] = mei_next;

        Spectrum event_pdf = mei_next.sigma_t / combined_extinction;
        event_pdf = dr::select(did_scatter, event_pdf, 1.f - event_pdf);
        weight[active] *= event_pdf / dr::detach(event_pdf);

        mei_next.mint = sampled_t;
        escaped |= active && (mei_next.mint >= maxt);
        active &= !did_scatter && !escaped;
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

    Float desired_tau = -dr::log(1 - sp.template next_float<Float>(active));

    Mask active_tracking(active);
    Mask escaped(!active);
    dr::Loop<Mask> loop("Ratio Tracking Distance Sampler with DDA SuperGrid");
    Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
    
    loop.put(dda_t, dda_tmax, dda_tdelta, mei, local_majorant, tau_acc, escaped, active_tracking, desired_tau, sp.state, sp.inc);
    // sampler->loop_put(loop);
    loop.init();
    while (loop(dr::detach(active_tracking))){
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
        // TODO: avoid this vcall, could lookup directly from the array
        // of floats (but we still need to account for the bbox, etc).
        local_majorant = m_majorant_grid->eval_1(mei, active_tracking);
        Float tau_next = tau_acc + local_majorant * (t_next - dda_t);

        // For rays that will stop within this cell, figure out
        // the precise `t` parameter where `desired_tau` is reached.
        Float t_precise = dda_t + (desired_tau - tau_acc) / local_majorant;

        Mask reached = active_tracking && (tau_next >= desired_tau) && (local_majorant > 0)  && (t_precise < maxt);

        dr::masked(dda_t, active_tracking) = dr::select(reached, t_precise, t_next);
        dr::masked(dda_tmax, active_tracking) = dda_tmax + tmax_update;
        desired_tau = dr::select(reached, -dr::log(1 - sp.template next_float<Float>(reached)), desired_tau);
        tau_acc = dr::select(reached, 0.f, tau_next);

        dr::masked(mei.t, reached) = dda_t;
        dr::masked(mei.p, reached) = ray(mei.t);

        Spectrum sigma_s(0.f), sigma_n(0.f), sigma_t(0.f);
        std::tie(sigma_s, sigma_n, sigma_t) = get_scattering_coefficients(mei, reached);
        Float r = dr::select(dr::neq(local_majorant, 0), sigma_t[0] / local_majorant, 0);
        Mask scattering = reached && sp.template next_float<Float>(reached) < r;

        // Prepare for next iteration
        escaped |= active && (dda_t >= maxt);
        active_tracking &= !scattering && !escaped && (t_next < maxt); 

        mei.t = dr::select(scattering, dda_t, dr::Infinity<Float>);
        mei.sigma_s     = dr::select(scattering, sigma_s, 1.f);
        mei.sigma_t     = dr::select(scattering, sigma_t, 1.f);
        mei.sigma_n     = dr::select(scattering, sigma_n, 1.f);        
    }
    dr::masked(mei.t, escaped) = dr::Infinity<Float>;
    mei.p                      = ray(mei.t);
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
                                    Mask _active) const {
    MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, _active);

    // TODO: add early exits when transmittance reaches a lower bound
    auto [mei, mint, maxt, active] = prepare_interaction_sampling(ray, _active);
    auto [dda_t, dda_tmax, dda_tdelta] = prepare_dda_traversal(
            m_majorant_grid.get(), ray, mint, maxt, active);
    
    Float t(mint);
    Spectrum TotalTr(1.f);
    Float max_delta_t = maxt-mint;
    Float T(1.f);
    Mask active_tracking(active);
    Float local_majorant = m_majorant_grid->eval_1(mei, active_tracking);

    using PCG32 = mitsuba::PCG32<UInt32>;
    PCG32 sp;
    
    if constexpr (dr::is_array_v<Float>) {
        sp.seed(1, seed_v0, seed_v1);
    } else {
        sp.seed(1, seed_v0, PCG32_DEFAULT_STREAM);
    }
    if (estimator_selector == 0) {
        // Ratio Tracking
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
        Mask exceeded_limits(!active_tracking);
        Spectrum T(1.f);
        Float tau_acc(0.f);
        Float desired_tau = -dr::log(1 - sp.template next_float<Float>(active_tracking));
        dr::Loop<Mask> loop("Ratio Tracking Integrator with Local Majorants");
        loop.put(T, dda_t, mei, local_majorant, TotalTr, dda_tdelta, dda_tmax, active_tracking, exceeded_limits, desired_tau, tau_acc, sp.state, sp.inc);
        // sampler->loop_put(loop);
        loop.init();
        while (loop(dr::detach(active_tracking))){
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

            Mask reached = active_tracking && (tau_next >= desired_tau) && (local_majorant > 0)  && (t_precise < maxt);

            dr::masked(dda_t, active_tracking) = t_next;
            dr::masked(dda_tmax, active_tracking) = dda_tmax + tmax_update;
            dr::masked(desired_tau, reached) = -dr::log(1 - sp.template next_float<Float>(reached));
            tau_acc = dr::select(reached, local_majorant * (t_next - t_precise), tau_next);

            dr::masked(mei.t, reached) = dda_t;
            dr::masked(mei.p, reached) = ray(mei.t);

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, reached);
            Float r = dr::select(reached, mei.sigma_t[0] / local_majorant, 0);
            Mask real = reached && sp.template next_float<Float>(reached) < r;

            dr::masked(T, active_tracking) *= (Spectrum(1.f) - r);

            exceeded_limits = active_tracking && (dda_t >= maxt);
            active_tracking &= !exceeded_limits;

            dr::masked(mei.t, exceeded_limits) =  dr::Infinity<Float>;
            dr::masked(mei.p, exceeded_limits) =  ray(mei.t);
            dr::masked(TotalTr, exceeded_limits) = T; 
        }
    } else if (estimator_selector == 3) {
        // Residual Ratio Tracking \w local majorants and local control coefficients
        Float max_delta_t = maxt-mint;

        Float local_control = m_control_grid->eval_1(mei, active_tracking);
        Float local_residual = local_majorant - local_control;

        Spectrum Tc = dr::exp(- local_control * (dr::min(dda_tmax) - mint));
        Spectrum Tr(1.f);

        Float tau_acc(0.f);
        Float desired_tau = -dr::log(1 - sp.template next_float<Float>(active_tracking));
        dr::Loop<Mask> loop("Ratio Tracking Integrator with Local Majorants");
        loop.put(Tr, Tc, TotalTr, dda_t, dda_tdelta, dda_tmax, mei, local_majorant, local_control, local_residual, active_tracking, desired_tau, tau_acc, sp.state, sp.inc);
        // sampler->loop_put(loop);
        loop.init();
        while (loop(dr::detach(active_tracking))){
            Mask exceeded_limits = active_tracking && (dda_t >= maxt);
            active_tracking &= !exceeded_limits;

            dr::masked(mei.t, exceeded_limits) =  dr::Infinity<Float>;
            dr::masked(mei.p, exceeded_limits) =  ray(mei.t);
            dr::masked(TotalTr, exceeded_limits) = Tc*Tr; 

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
            local_residual = local_majorant - local_control;

            Float tau_next = tau_acc + local_residual * (t_next - dda_t);

            Float t_precise = dda_t + (desired_tau - tau_acc) / local_residual;

            Mask reached = active_tracking && (tau_next >= desired_tau) && (local_majorant > 0)  && (t_precise < maxt);

            Float distance_travelled = t_next - dda_t;

            dr::masked(dda_t, active_tracking) = t_next;
            dr::masked(dda_tmax, active_tracking) = dda_tmax + tmax_update;
            dr::masked(desired_tau, reached) = -dr::log(1 - sp.template next_float<Float>(reached));
            tau_acc = dr::select(reached, local_residual * (t_next - t_precise), tau_next);

            dr::masked(mei.t, reached) = dda_t;
            dr::masked(mei.p, reached) = ray(mei.t);

            std::tie(mei.sigma_s, mei.sigma_n, mei.sigma_t) = get_scattering_coefficients(mei, reached);
            Float r = dr::select(reached, (mei.sigma_t[0] - local_control) / local_residual, 0);

            Mask real = reached && sp.template next_float<Float>(reached) < r;

            dr::masked(Tr, real) *= (Spectrum(1.f) - r);
            dr::masked(Tc, active_tracking) *= dr::exp(- local_control * distance_travelled);
        }
    } else {
        std::cout<<"Estimator: "<<estimator_selector<<std::endl;
        NotImplementedError("select one of the following estimators: rt, rrt, rt_local, rrt_local");
    }
    return {mei, TotalTr};
}


MI_IMPLEMENT_CLASS_VARIANT(Medium, Object, "medium")
MI_INSTANTIATE_CLASS(Medium)
NAMESPACE_END(mitsuba)
