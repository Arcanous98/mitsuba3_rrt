import os
from os.path import realpath, join, dirname
import sys

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import numpy as np
import torch
torch.manual_seed(0)
import matplotlib.pyplot as plt
import time

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

def load_volume_scene(gaussian_count="10k",
                      grid_res=256,
                      path_env = "./scenes/teapot-full/textures/syferfontein_1d_clear_puresky_4k.exr", 
                    #   path_env = "./scenes/teapot-full/textures/venice_sunset_4k.exr", syferfontein_1d_clear_puresky_4k rustig_koppie_puresky_4k
                      density_scale=1.0, 
                      albedo=0.99,
                      res=1080):
    T = mi.ScalarTransform4f

    t_env = T.scale([0.1, 0.1, 0.1])

    path_vol = join(f'./scenes/volumes/volume_grid_sigmat_{gaussian_count}_{grid_res}.npy')
    grid = np.load(path_vol)
    grid = mi.cuda_ad_rgb.TensorXf(grid)
    to_world_grid = mi.ScalarTransform4f(np.load(join(f'./scenes/volumes/volume_grid_to_world_{gaussian_count}_{grid_res}.npy')))

    bsphere = mi.ScalarBoundingSphere3f([-50.9844, -14.5577, 7.01233], 960.642)

    if True: # Teaser config
        origin = bsphere.center - dr.normalize(mi.ScalarVector3f(0.0, 0.5, 1)) * bsphere.radius * 8.0
        target = bsphere.center + mi.ScalarVector3f(80.0, 110.0, 0.0)
        up     = mi.ScalarVector3f(0, 1, 0)
        fov    = 7.5
        width  = 1360
        height = 720

    sensor_dict = {
        'type': 'perspective',
        'fov': fov,
        'near_clip': 0.0001,
        'far_clip': 1000000.0,
        'to_world': mi.ScalarTransform4f.look_at(origin=origin, target=target, up=up).rotate([0,0,1],90).rotate([1,0,0],-0.7).rotate([0,1,0],-0.5),
        'film': {
            'type': 'hdrfilm',
            'width': width,
            'height': height,
            'filter': { 'type': 'gaussian' },
        }
    }
    return {
        'type': 'scene',
        # -------------------- Sensor --------------------
        'sensor': sensor_dict,
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
            'filename': path_env,
            'to_world': t_env,
            'scale': 1.0,
        },
        # -------------------- Media --------------------
        'medium1': {
            'type': 'heterogeneous',
            'absorptive_medium': False,
            'scale': density_scale,
            'majorant_resolution_factor': 8,
			"majorant_factor": 1.01,
            # 'albedo': {
            #     'type': 'constvolume',
            #     'value': {'type': 'rgb', 'value': [0.8, 0.9, 0.7]},
            # },
            'sigma_t': {
                'type': 'gridvolume',
                'data': grid,
                'to_world': to_world_grid,
            },
            'albedo': albedo,
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
            'to_world': mi.ScalarTransform4f.scale(0.95 * 3000.0)
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
    img = mi.render(scene, integrator=integrator, spp=128)

    plt.axis("off")
    plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping
    plt.show()
    mi.util.write_bitmap(output_dir+"volpath_novak14_supervoxels.exr", img)

def render_cloud(integrator_name="volpath_novak14",
                 estimator="rrt",
                 distance_sampler="ff",
                 spp=1,
                 init_time=0):

    output_dir = os.path.join("./test_renders/")
    os.makedirs(output_dir, exist_ok=True)

    scene_dict = load_volume_scene()
    scene = mi.load_dict(scene_dict)
    if integrator_name == "volpath":
         output_name = output_dir+"disney_cloud_"+integrator_name+".exr"
         integrator =  mi.load_dict({
                    'type': integrator_name,
                    'max_depth': 100,
                    'rr_depth': 1000,
                   })
    else:
        output_name = output_dir+"disney_cloud_"+integrator_name+"_"+estimator+"_"+distance_sampler+".exr"
        integrator =  mi.load_dict({
                        'type': integrator_name,
                        'max_depth': 100,
                        'rr_depth': 1000,
                        'transmittance_estimator': estimator, #rt, rrt, rt_local, rrt_local
                        'distance_sampler': distance_sampler, #ff, ff_local
                    })
        
    loading_time = time.time() - init_time
    img = mi.render(scene, integrator=integrator, spp=spp)
    # plt.axis("off")
    # plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping
    # plt.show()
    mi.util.write_bitmap(output_name, img)
    return loading_time

if __name__ == '__main__':
    # dr.set_flag(dr.JitFlag.VCallRecord, False)
    # dr.set_flag(dr.JitFlag.LoopRecord, False)
    # test_novak14_supervoxels()
    init_time = time.time()
    loading_time = render_cloud(integrator_name="volpath_novak14",
                                estimator="ps_cum",
                                distance_sampler="ff_local",
                                spp=8,
                                init_time=init_time)
    rendering_time = time.time() - init_time
    print("Rendering time (including jitting): "+str(rendering_time - loading_time))
    

