#include <mitsuba/core/frame.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/volume.h>

NAMESPACE_BEGIN(mitsuba)


/**!

.. _medium-heterogeneous:

Heterogeneous medium (:monosp:`heterogeneous`)
-----------------------------------------------

.. pluginparameters::

 * - albedo
   - |float|, |spectrum| or |volume|
   - Single-scattering albedo of the medium (Default: 0.75).
   - |exposed|, |differentiable|

 * - sigma_t
   - |float|, |spectrum| or |volume|
   - Extinction coefficient in inverse scene units (Default: 1).
   - |exposed|, |differentiable|

 * - scale
   - |float|
   - Optional scale factor that will be applied to the extinction parameter.
     It is provided for convenience when accommodating data based on different
     units, or to simply tweak the density of the medium. (Default: 1)
   - |exposed|

 * - sample_emitters
   - |bool|
   - Flag to specify whether shadow rays should be cast from inside the volume (Default: |true|)
     If the medium is enclosed in a :ref:`dielectric <bsdf-dielectric>` boundary,
     shadow rays are ineffective and turning them off will significantly reduce
     render time. This can reduce render time up to 50% when rendering objects
     with subsurface scattering.

 * - (Nested plugin)
   - |phase|
   - A nested phase function that describes the directional scattering properties of
     the medium. When none is specified, the renderer will automatically use an instance of
     isotropic.
   - |exposed|, |differentiable|


This plugin provides a flexible heterogeneous medium implementation, which acquires its data
from nested volume instances. These can be constant, use a procedural function, or fetch data from
disk, e.g. using a 3D grid.

The medium is parametrized by the single scattering albedo and the extinction coefficient
:math:`\sigma_t`. The extinction coefficient should be provided in inverse scene units.
For instance, when a world-space distance of 1 unit corresponds to a meter, the
extinction coefficient should have units of inverse meters. For convenience,
the scale parameter can be used to correct the units. For instance, when the scene is in
meters and the coefficients are in inverse millimeters, set scale to 1000.

Both the albedo and the extinction coefficient can either be constant or textured,
and both parameters are allowed to be spectrally varying.

.. tabs::
    .. code-tab:: xml
        :name: lst-heterogeneous

        <!-- Declare a heterogeneous participating medium named 'smoke' -->
        <medium type="heterogeneous" id="smoke">
            <!-- Acquire extinction values from an external data file -->
            <volume name="sigma_t" type="gridvolume">
                <string name="filename" value="frame_0150.vol"/>
            </volume>

            <!-- The albedo is constant and set to 0.9 -->
            <float name="albedo" value="0.9"/>

            <!-- Use an isotropic phase function -->
            <phase type="isotropic"/>

            <!-- Scale the density values as desired -->
            <float name="scale" value="200"/>
        </medium>

        <!-- Attach the index-matched medium to a shape in the scene -->
        <shape type="obj">
            <!-- Load an OBJ file, which contains a mesh version
                 of the axis-aligned box of the volume data file -->
            <string name="filename" value="bounds.obj"/>

            <!-- Reference the medium by ID -->
            <ref name="interior" id="smoke"/>
            <!-- If desired, this shape could also declare
                a BSDF to create an index-mismatched
                transition, e.g.
                <bsdf type="dielectric"/>
            -->
        </shape>

    .. code-tab:: python

        # Declare a heterogeneous participating medium named 'smoke'
        'smoke': {
            'type': 'heterogeneous',

            # Acquire extinction values from an external data file
            'sigma_t': {
                'type': 'gridvolume',
                'filename': 'frame_0150.vol'
            },

            # The albedo is constant and set to 0.9
            'albedo': 0.9,

            # Use an isotropic phase function
            'phase': {
                'type': 'isotropic'
            },

            # Scale the density values as desired
            'scale': 200
        },

        # Attach the index-matched medium to a shape in the scene
        'shape': {
            'type': 'obj',
            # Load an OBJ file, which contains a mesh version
            # of the axis-aligned box of the volume data file
            'filename': 'bounds.obj',

            # Reference the medium by ID
            'interior': 'smoke',
            # If desired, this shape could also declare
            # a BSDF to create an index-mismatched
            # transition, e.g.
            # 'bsdf': {
            #     'type': 'isotropic'
            # },
        }
*/
template <typename Float, typename Spectrum>
class HeterogeneousMedium final : public Medium<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Medium, m_is_homogeneous, m_has_spectral_extinction, m_is_absorptive,
                    m_phase_function, m_majorant_grid,m_control_grid,
                   m_majorant_resolution_factor, m_majorant_factor)
    MI_IMPORT_TYPES(Scene, Sampler, Texture, Volume)

    HeterogeneousMedium(const Properties &props) : Base(props) {
        m_albedo         = props.volume<Volume>("albedo", 0.75f);
        m_sigmat         = props.volume<Volume>("sigma_t", 1.f);
        m_is_absorptive  = props.get<bool>("absorptive_medium", false);
        // m_emission       = props.volume<Volume>("emission", 0.f);

        ScalarFloat scale = props.get<ScalarFloat>("scale", 1.0f);
        m_has_spectral_extinction = props.get<bool>("has_spectral_extinction", false);
        m_scale = scale;
    
        update_majorant_supergrid();
        if (m_majorant_resolution_factor > 0) {
            Log(Info, "Using majorant supergrid with resolution %s", m_majorant_grid->resolution());
            const ScalarFloat vmax = m_majorant_factor * scale * m_sigmat->max();
            const ScalarFloat vmean = m_majorant_factor * scale * m_sigmat->avg();

            m_control_sigma_t = dr::opaque<Float>(dr::maximum(1e-6f, vmean));
            m_max_density = dr::opaque<Float>(dr::maximum(1e-6f, vmax));
        } else {
            const ScalarFloat vmax = m_majorant_factor * scale * m_sigmat->max();
            const ScalarFloat vmean = m_majorant_factor * scale * m_sigmat->avg();

            m_control_sigma_t = dr::opaque<Float>(dr::maximum(1e-6f, vmean));
            m_max_density = dr::opaque<Float>(dr::maximum(1e-6f, vmax));
            Log(Info, "Heterogeneous medium will use majorant: %s (majorant factor: %s)",
                m_max_density, m_majorant_factor);
        }

        dr::set_attr(this, "is_homogeneous", m_is_homogeneous);
        dr::set_attr(this, "has_spectral_extinction", m_has_spectral_extinction);
        dr::set_attr(this, "is_absorptive", m_is_absorptive);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("scale", m_scale, +ParamFlags::NonDifferentiable);
        callback->put_object("albedo",   m_albedo.get(), +ParamFlags::Differentiable);
        callback->put_object("sigma_t",  m_sigmat.get(), +ParamFlags::Differentiable);
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {
        // unsupported majorant grid optimization
        m_max_density = dr::opaque<Float>(m_scale.scalar() * m_sigmat->max());
    }

    UnpolarizedSpectrum
    get_majorant(const MediumInteraction3f & /* mi */,
                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        return m_max_density;
    }

    UnpolarizedSpectrum
    get_control_sigma_t(const MediumInteraction3f & /* mi */,
                 Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);
        return m_control_sigma_t;
    }

    std::tuple<UnpolarizedSpectrum, UnpolarizedSpectrum, UnpolarizedSpectrum>
    get_scattering_coefficients(const MediumInteraction3f &mi,
                                Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::MediumEvaluate, active);

        auto sigmat = m_scale.scalar() * m_sigmat->eval(mi, active);
        if (has_flag(m_phase_function->flags(), PhaseFunctionFlags::Microflake))
            sigmat *= m_phase_function->projected_area(mi, active);

        auto sigmas = sigmat * m_albedo->eval(mi, active);
        // auto sigman = m_max_density - sigmat;
        Float maj = (m_majorant_resolution_factor > 0) ? m_majorant_grid->eval_1(mi, active) : m_max_density;
        auto sigman = maj - sigmat;

        return { sigmas, sigman, sigmat };
    }

    void update_majorant_supergrid() {
        if (m_majorant_resolution_factor <= 0)
            return;

        // Build a majorant grid, with the scale factor baked-in for convenience
        auto [majorants, control_mu] = m_sigmat->local_majorants(m_majorant_resolution_factor, m_majorant_factor * m_scale.scalar());
        dr::eval(majorants);
        dr::eval(control_mu);

        Properties props("gridvolume");
        props.set_string("filter_type", "nearest");
        props.set_transform("to_world", m_sigmat->world_transform());
        using TensorHandle = typename Properties::TensorHandle;
        props.set_tensor_handle("data", TensorHandle(std::make_shared<TensorXf>(majorants)));
        m_majorant_grid = (Volume *) PluginManager::instance()->create_object<Volume>(props).get();
        Log(Info, "Majorant supergrid updated (resolution: %s)", m_majorant_grid->resolution());

        Properties props_c("gridvolume");
        props_c.set_string("filter_type", "nearest");
        props_c.set_transform("to_world", m_sigmat->world_transform());
        using TensorHandle = typename Properties::TensorHandle;
        props_c.set_tensor_handle("data", TensorHandle(std::make_shared<TensorXf>(control_mu)));
        m_control_grid = (Volume *) PluginManager::instance()->create_object<Volume>(props_c).get();
        Log(Info, "Control supergrid updated (resolution: %s)", m_control_grid->resolution());
    }

    std::tuple<Mask, Float, Float>
    intersect_aabb(const Ray3f &ray) const override {
        return m_sigmat->bbox().ray_intersect(ray);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "HeterogeneousMedium[" << std::endl
            << "  albedo  = " << string::indent(m_albedo) << std::endl
            << "  sigma_t = " << string::indent(m_sigmat) << std::endl
            << "  scale   = " << string::indent(m_scale.scalar()) << std::endl
            << "  max_density     = " << string::indent(m_max_density) << std::endl
            << "  majorant_factor = " << string::indent(m_majorant_factor) << std::endl
            << "  majorant_resolution_factor   = " << string::indent(m_majorant_resolution_factor) << std::endl
            // << "  majorant_grid                = " << string::indent(m_majorant_grid) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Volume> m_sigmat, m_albedo;
    field<Float> m_scale;

    Float m_max_density;
    Float m_control_sigma_t;

};

MI_IMPLEMENT_CLASS_VARIANT(HeterogeneousMedium, Medium)
MI_EXPORT_PLUGIN(HeterogeneousMedium, "Heterogeneous Medium")
NAMESPACE_END(mitsuba)
