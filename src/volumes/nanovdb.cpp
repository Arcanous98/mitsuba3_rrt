#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/srgb.h>
#include <mitsuba/render/volume.h>
#include <mitsuba/render/volumegrid.h>

#include <drjit/dynamic.h>
#include <drjit/texture.h>
#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/SampleFromVoxels.h>
// #include <nanovdb/util/CudaDeviceBuffer.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class NanoVDBVolume final : public Volume<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Volume, update_bbox, m_to_local, m_bbox, m_channel_count)
    MI_IMPORT_TYPES(VolumeGrid)

    using Allocator = pstd::pmr::polymorphic_allocator<std::byte>;

    NanoVDBVolume(const Properties &props) : Base(props) {
        m_raw = props.get<bool>("raw", false);
        m_accel = props.get<bool>("accel", true);
        
        // TODO allow RGB or Spectral grids
        ScalarUInt32 channel_count = 1;
        FileResolver *fs = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        if (!fs::exists(file_path))
            Log(Error, "\"%s\": file does not exist!", file_path);

        // TODO load nanovdb volume
        nanovdb::GridHandle<NanoVDBBuffer> m_vdb_grid;
        Allocator alloc;
        m_density_volume = read_nanovdb_grid<NanoVDBBuffer>(file_path, alloc);
        nanovdb::BBox<nanovdb::Vec3R> bbox = m_density_volume->worldBBox();
        BoundingBox3f bounds = BoundingBox3f(Point3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]), Point3f(bbox.max()[0], bbox.max()[1], bbox.max()[2]));

        if (true){
            // Use a single global majorant
        } else {
            // Use a Majorant grid for local majorants
            size_t channel_count = 2;
            Float* majorant_grid_array = nullptr;
            // define grid size in voxel count
            // TODO load grid res from options in html file
            Vector3u majorant_grid_res(64);
            ScalarUInt32 grid_size = dr::prod(majorant_grid_res);
            // create majorant volume grid with single channel, desired grid res
            size_t shape[4] = {
                    (size_t) majorant_grid_res.z,
                    (size_t) majorant_grid_res.y,
                    (size_t) majorant_grid_res.x,
                    channel_count
            };
            majorant_grid_array = dr::zeros<Float>(grid_size);
            
            // Following code has been adapted from pbrtv4. It fills up the majorant grid.
            for (size_t index=0; index<grid_size; index++){
                // Indices into majorantGrid
                int x = index % majorant_grid_res.x;
                int y = (index / majorant_grid_res.x) % majorant_grid_res.y;
                int z = index / (majorant_grid_res.x * majorant_grid_res.y);

                // World (aka medium) space bounds of this max grid cell
                BoundingBox3f wb(bounds.lerp(Point3f(Float(x) / majorant_grid_res.x,
                                                Float(y) / majorant_grid_res.y,
                                                Float(z) / majorant_grid_res.z)),
                                bounds.lerp(Point3f(Float(x + 1) / majorant_grid_res.x,
                                                Float(y + 1) / majorant_grid_res.y,
                                                Float(z + 1) / majorant_grid_res.z)));

                // Compute corresponding NanoVDB index-space bounds in floating-point.
                nanovdb::Vec3R i0 = m_density_volume->worldToIndexF(
                    nanovdb::Vec3R(wb.min.x, wb.min.y, wb.min.z));
                nanovdb::Vec3R i1 = m_density_volume->worldToIndexF(
                    nanovdb::Vec3R(wb.max.x, wb.max.y, wb.max.z));

                // Now find integer index-space bounds, accounting for both
                // filtering and the overall index bounding box.
                auto bbox = m_density_volume->indexBBox();
                Float delta = 1.f;  // Filter slop
                int nx0 = dr::maximum(int(i0[0] - delta), bbox.min()[0]);
                int nx1 = dr::minimum(int(i1[0] + delta), bbox.max()[0]);
                int ny0 = dr::maximum(int(i0[1] - delta), bbox.min()[1]);
                int ny1 = dr::minimum(int(i1[1] + delta), bbox.max()[1]);
                int nz0 = dr::maximum(int(i0[2] - delta), bbox.min()[2]);
                int nz1 = dr::minimum(int(i1[2] + delta), bbox.max()[2]);

                float max_value = 0;
                float mean_value = 0;
                auto accessor = m_density_volume->getAccessor();
                // Apparently nanovdb integer bounding boxes are inclusive on
                // the upper end...
                for (int nz = nz0; nz <= nz1; ++nz)
                    for (int ny = ny0; ny <= ny1; ++ny)
                        for (int nx = nx0; nx <= nx1; ++nx)
                            float val = accessor.getValue({nx, ny, nz});
                            max_value = dr::maximum(max_value, val);
                            mean_value += val;
                // write into the value buffer
                mean_value /= (nz1 * ny1 * nx1);
                dr::scatter(majorant_grid_array, max_value, index * 2);
                dr::scatter(majorant_grid_array, mean_value, index * 2 + 1);
            }
            m_majorant_grid = Texture3f(TensorXf(majorant_grid_array, 4, shape),
                                        m_accel, m_accel, "trilinear", "clamp");

        }
        
        m_density_volume->tree().extrema(m_min, m_max);
        m_channel_count = channel_count;

    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("data", m_texture.tensor(), +ParamFlags::Differentiable);
        Base::traverse(callback);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "data")) {
            const size_t channels = nchannels();
            if (channels != 1 && channels != 3 && channels != 6)
                Throw("parameters_changed(): The volume data %s was changed "
                      "to have %d channels, only volumes with 1, 3 or 6 "
                      "channels are supported!", to_string(), channels);

            m_texture.set_tensor(m_texture.tensor());

            if (!m_fixed_max)
                m_max = (float) dr::max_nested(dr::detach(m_texture.value()));
        }
    }

    UnpolarizedSpectrum eval(const Interaction3f &it,
                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        nanovdb::Vec3<float> pIndex =
            m_density_volume->worldToIndexF(nanovdb::Vec3<float>(it.p.x, it.p.y, it.p.z));
        using NanoVDBSampler = nanovdb::SampleFromVoxels<nanovdb::FloatGrid::TreeType, 1, false>;
        Float d = NanoVDBSampler(m_density_volume->tree())(pIndex);

        return Spectrum(d);
    }

    Float eval_1(const Interaction3f &it, Mask active = true) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        // only density nanovdb volumes supported
        return Float(eval(it, active));
    }

    void eval_n(const Interaction3f &it, Float *out, Mask active = true) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        // unsupported
    }

    Vector3f eval_3(const Interaction3f &it,
                    Mask active = true) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        //unsupported RGB volumes 
        return dr::zeros<Vector3f>();
    }

    dr::Array<Float, 6> eval_6(const Interaction3f &it,
                               Mask active = true) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        //unsupported Spectral rendering with nanovdb volumes
        return dr::zeros<dr::Array<Float, 6>>();
    }

    ScalarFloat max() const override { return m_max; }
    ScalarFloat min() const override { return m_min; }
    ScalarFloat avg() const override { return m_avg; }

    void max_per_channel(ScalarFloat *out) const override {
        for (size_t i=0; i<m_max_per_channel.size(); ++i)
            out[i] = m_max_per_channel[i];
    }
    void min_per_channel(ScalarFloat *out) const override {
        for (size_t i=0; i<m_min_per_channel.size(); ++i)
            out[i] = m_min_per_channel[i];
    }
    void avg_per_channel(ScalarFloat *out) const override {
        for (size_t i=0; i<m_avg_per_channel.size(); ++i)
            out[i] = m_avg_per_channel[i];
    }

    ScalarVector3i resolution() const override {
        const size_t *shape = m_texture.shape();
        return { (int) shape[2], (int) shape[1], (int) shape[0] };
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "GridVolume[" << std::endl
            << "  to_local = " << string::indent(m_to_local, 13) << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "  dimensions = " << resolution() << "," << std::endl
            << "  max = " << m_max << "," << std::endl
            << "  channels = " << m_texture.shape()[3] << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

protected:
    class NanoVDBBuffer {
        public:
            static inline void ptrAssert(void *ptr, const char *msg, const char *file, int line,
                                        bool abort = true) {
                    if (abort)
                        LOG_FATAL("%p: %s (%s:%d)", ptr, msg, file, line);
                    else
                        LOG_ERROR("%p: %s (%s:%d)", ptr, msg, file, line);
                }

                NanoVDBBuffer() = default;
                NanoVDBBuffer(Allocator alloc) : alloc(alloc) {}
                NanoVDBBuffer(size_t size, Allocator alloc = {}) : alloc(alloc) { init(size); }
                NanoVDBBuffer(const NanoVDBBuffer &) = delete;
                NanoVDBBuffer(NanoVDBBuffer &&other) noexcept
                    : alloc(std::move(other.alloc)),
                    bytesAllocated(other.bytesAllocated),
                    ptr(other.ptr) {
                    other.bytesAllocated = 0;
                    other.ptr = nullptr;
                }
                NanoVDBBuffer &operator=(const NanoVDBBuffer &) = delete;
                NanoVDBBuffer &operator=(NanoVDBBuffer &&other) noexcept {
                    // Note, this isn't how std containers work, but it's expedient for
                    // our purposes here...
                    clear();
                    // operator= was deleted? Fine.
                    new (&alloc) Allocator(other.alloc.resource());
                    bytesAllocated = other.bytesAllocated;
                    ptr = other.ptr;
                    other.bytesAllocated = 0;
                    other.ptr = nullptr;
                    return *this;
                }
                ~NanoVDBBuffer() { clear(); }

                void init(uint64_t size) {
                    if (size == bytesAllocated)
                        return;
                    if (bytesAllocated > 0)
                        clear();
                    if (size == 0)
                        return;
                    bytesAllocated = size;
                    ptr = (uint8_t *)alloc.allocate_bytes(bytesAllocated, 128);
                }

                const uint8_t *data() const { return ptr; }
                uint8_t *data() { return ptr; }
                uint64_t size() const { return bytesAllocated; }
                bool empty() const { return size() == 0; }

                void clear() {
                    alloc.deallocate_bytes(ptr, bytesAllocated, 128);
                    bytesAllocated = 0;
                    ptr = nullptr;
                }

                static NanoVDBBuffer create(uint64_t size, const NanoVDBBuffer *context = nullptr) {
                    return NanoVDBBuffer(size, context ? context->GetAllocator() : Allocator());
                }

                Allocator GetAllocator() const { return alloc; }

        private:
            Allocator alloc;
            size_t bytesAllocated = 0;
            uint8_t *ptr = nullptr;
        };

    template <typename Buffer>
    static nanovdb::GridHandle<Buffer> read_nanovdb_grid(const std::string &filename, Allocator alloc) {
        NanoVDBBuffer buf(alloc);
        nanovdb::GridHandle<Buffer> grid;
        grid = nanovdb::io::readGrid<Buffer>(filename, 0 /* not verbose */, buf);
        return grid;
    }
    // DDA Majorant methods
    // TODO include control coefficients (and suppose mu_r can be approximated by majorant - mu_c)
    // alternatively implement Kutz17 for "local" control media (decomposition tracking)
    Float dda_majorant_eval_point(Point3f &p, Mask active) const {
        Float maj_sigma_t;
        m_majorant_grid.eval(p, maj_sigma_t, active);
        return maj_sigma_t;
    }

    DDAInteraction3f dda_majorant_iterator(Ray3f &ray, Mask active){
        Vector3f diag = m_bbox.extents();
        Ray3f ray_grid(Point3f(m_bbox.offset(ray.o)), Vector3f(ray.d.x / diag.x, ray.d.y / diag.y, ray.d.z / diag.z));
        Point3f ray_grid_p = ray_grid(ray.mint);
        Point3u voxel_limit, voxel, step;
        Point3f delta_t, next_crossing_t;
        for (int axis = 0; axis < 3; ++axis){
            // Initialize ray stepping parameters for _axis_
            // Compute current voxel for axis and handle negative zero direction
            voxel[axis] = dr::clamp(ray_grid_p[axis] * m_majorant_grid.shape()[axis], 0, m_majorant_grid.shape()[axis] - 1);
            delta_t[axis] = 1 / (dr::abs(ray_grid.d[axis]) * m_majorant_grid.shape()[axis]);
            if (ray_grid.d[axis] == -0.f) 
                ray_grid.d[axis] = 0.f;

            if (ray_grid.d[axis] >= 0) {
                // Handle ray with positive direction for voxel stepping
                Float next_voxel_p = Float(voxel[axis] + 1) / m_majorant_grid.shape()[axis];
                next_crossing_t[axis] =
                    ray.mint + (next_voxel_p - ray_grid_p[axis]) / ray_grid.d[axis];
                step[axis] = 1;
                voxel_limit[axis] = m_majorant_grid.shape()[axis];

            } else {
                // Handle ray with negative direction for voxel stepping
                Float next_voxel_p = Float(voxel[axis]) / m_majorant_grid.shape()[axis];
                next_crossing_t[axis] =
                    ray.mint + (next_voxel_p - ray_grid_p[axis]) / ray_grid.d[axis];
                step[axis] = -1;
                voxel_limit[axis] = -1;
            }
        }
        DDAInteraction3f dda = dr::zeros<DDAInteraction3f>();
        dda.next_crossing_t = next_crossing_t;
        dda.ray = ray_grid;
        dda.sigma_t, dda.sigma_c = m_majorant_grid.eval(ray_grid_p, active);
        dda.voxel_limit = voxel_limit;
        dda.voxel = voxel;
        dda.delta_t = delta_t;
    }

    void dda_next_majorant(DDAInteraction3f *dda_it, Mask active){
        if (dda_it.ray.mint >= dda_it.ray.maxt)
            return;
        // Find _stepAxis_ for stepping to next voxel and exit point _tVoxelExit_
        int bits = ((dda_it.next_crossing_t[0] < dda_it.next_crossing_t[1]) << 2) +
                   ((dda_it.next_crossing_t[0] < dda_it.next_crossing_t[2]) << 1) +
                   ((dda_it.next_crossing_t[1] < dda_it.next_crossing_t[2]));
        const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        int stepAxis = cmpToAxis[bits];
        Float t_voxel_exit = dr::mininimum(dda_it.ray.maxt, dda_it.next_crossing_t[stepAxis]);
        // Get _maxDensity_ for current voxel and initialize _RayMajorantSegment_, _seg_
        dda_it.sigma_t = m_majorant_grid.eval(dda_it.ray(t_voxel_exit), active);

        // Advance to next voxel in maximum density grid
        dda_it.ray.mint = t_voxel_exit;
        if (dda_it.next_crossing_t[stepAxis] > dda_it.ray.maxt)
            dda_it.ray.mint = dda_it.ray.maxt;
        voxel[stepAxis] += step[stepAxis];
        if (voxel[stepAxis] == dda_it.voxel_limits[stepAxis])
            dda_it.ray.mint = dda_it.ray.maxt;
        dda_it.next_crossing_t[stepAxis] += dda_it.delta_t[stepAxis];

        return;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "NanoVDB Volume[" << std::endl
            << "  to_local = " << string::indent(m_to_local, 13) << "," << std::endl
            << "  bbox = " << string::indent(m_bbox) << "," << std::endl
            << "  dimensions = " << resolution() << "," << std::endl
            << "  max = " << m_max << "," << std::endl
            << "  channels = " << m_texture.shape()[3] << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()

protected:
    const nanovdb::FloatGrid *m_density_volume = nullptr;
    nanovdb::GridHandle<NanoVDBBuffer> m_vdb_grid;
    Texture3f m_texture;
    Texture3f m_majorant_grid;
    bool m_accel;
    bool m_raw;
    bool m_fixed_max = false;
    ScalarFloat m_max;
    ScalarFloat m_min;
    ScalarFloat m_avg;
    std::vector<ScalarFloat> m_max_per_channel;
    std::vector<ScalarFloat> m_min_per_channel;
    std::vector<ScalarFloat> m_avg_per_channel;
    BoundingBox3f m_bbox;
};

MI_IMPLEMENT_CLASS_VARIANT(NanoVDBVolume, Volume)
MI_EXPORT_PLUGIN(NanoVDBVolume, "NanoVDB Volume texture")

NAMESPACE_END(mitsuba)