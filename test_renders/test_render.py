import os
from os.path import realpath, join, dirname
import sys

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import numpy as np
import torch
torch.manual_seed(0)

mi.set_log_level(mi.LogLevel.Debug)

def cube_test_scene(resx=512, resy=512, spp=16, density_scale=1.0):
    T = mi.ScalarTransform4f

    grid = np.array(torch.rand((128, 128, 128), dtype=torch.float32, device="cpu"))

    grid = mi.cuda_ad_rgb.TensorXf(grid)
    to_world = T.translate([-0.5, -0.5, -0.5]).scale([2, 2, 2])

    return {
        'type': 'scene',
        # -------------------- Sensor --------------------
        'sensor': {
            'type': 'perspective',
            'fov': 30,
            'to_world': T.look_at(
                origin=[4.0, 4.0, 4.0],
                target=[0, -0.15, 0],
                up=[0, 1, 0]
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': spp,
            },
            'film': {
                'type': 'hdrfilm',
                'width' : resx,
                'height': resy,
                'rfilter': {
                    'type': 'box',
                },
                'pixel_format': "rgb",
            }
        },
        # Mostly just to avoid the warning
        'integrator': {
            'type': 'path',
        },
        # -------------------- Light --------------------
        # 'light': {
        #     'type': 'constant',
        #     'radiance': {'type': 'rgb', 'value': [1.0, 0.8, 0.2]},
        # },
        'light': {
            'type': 'envmap',
            'filename': "./scenes/teapot-full/textures/venice_sunset_4k.exr",
        },
        # -------------------- Media --------------------
        'medium1': {
            'type': 'heterogeneous',
            'scale': density_scale,
            'majorant_resolution_factor': 8,
            # 'albedo': {
            #     'type': 'constvolume',
            #     'value': {'type': 'rgb', 'value': [0.8, 0.9, 0.7]},
            # },
            'sigma_t': {
                'type': 'gridvolume',
                'data': grid,
                'to_world': to_world,
            },
            'albedo': 0.4,
        },
        # -------------------- Shapes --------------------
        'cube': {
            # Cube covers [0, 0, 0] to [1, 1, 1] by default
            'type': 'cube',
            'bsdf': { 'type': 'null', },
            'interior': {
                'type': 'ref',
                'id':  'medium1'
            },
            'to_world': to_world,
        },
    }

def test_novak14_supervoxels():
    output_dir = os.path.join("./test_renders/")
    os.makedirs(output_dir, exist_ok=True)

    scene_dict = cube_test_scene(density_scale=2.0)
    scene = mi.load_dict(scene_dict)
    integrator =  mi.load_dict({
                    'type': 'volpath_novak14',
                    'max_depth': 100,
                    'rr_depth': 1000,
                    'transmittance_estimator': "rrt_local", #rt, rrt, rt_local, rrt_local
                    'distance_sampler': "ff_local", #ff, ff_local
                   })
    img = mi.render(scene, integrator=integrator, spp=8)
    import matplotlib.pyplot as plt

    plt.axis("off")
    plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping
    plt.show()
    mi.util.write_bitmap(output_dir+"volpath_novak14_supervoxels.exr", img)
if __name__ == '__main__':
    # dr.set_flag(dr.JitFlag.VCallRecord, False)
    # dr.set_flag(dr.JitFlag.LoopRecord, False)
    test_novak14_supervoxels()

