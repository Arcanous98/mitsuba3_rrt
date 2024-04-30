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
import json

# mi.set_log_level(mi.LogLevel.Debug)

def compute_metrics(reference, target):
    target = torch.tensor(target)
    reference = torch.tensor(reference)
    mse = torch.mean((target - reference) ** 2)
    rmse = torch.sqrt(mse)
    # lpips
    return [mse.item(), rmse.item()]

def load_volume_scene(gaussian_count="10k",
                      grid_res=256,
                      path_env = "./scenes/teapot-full/textures/syferfontein_1d_clear_puresky_4k.exr", 
                    #   path_env = "./scenes/teapot-full/textures/venice_sunset_4k.exr", syferfontein_1d_clear_puresky_4k rustig_koppie_puresky_4k
                      density_scale=5.0, 
                      albedo=0.99,
                      render_wdas_cloud=False,
                      absorptive_only_test=False,
                      majorant_res_factor = 1):
    
    T = mi.ScalarTransform4f
    t_env = T.scale([0.1, 0.1, 0.1])

    if absorptive_only_test:
        albedo = 0.0

    if (not render_wdas_cloud):
        majorant_res_factor = 1
        density_scale = 5.0
        path_vol = join(f'./scenes/volumes/smoke.vol')
        sensor_dict = {
            'type': 'perspective',
            'fov': 35,
            'near_clip': 0.0001,
            'far_clip': 1000000.0,
            'to_world': T.look_at(origin=np.array([0,0,4]), target=np.array([0,0,0]), up=np.array([0,1,0])),
            'film': {
                'type': 'hdrfilm',
                'width': 800,
                'height': 800,
                'filter': { 'type': 'gaussian' },
            }
        }
        medium_T = T.scale(2.0).translate(np.array([-0.5,-0.5,-0.5]))#T.scale(2.0).translate(np.array([-1.0,-1.0,-1.0]))
        medium_dict = {
            'type': 'heterogeneous',
            'absorptive_medium': absorptive_only_test,
            'scale': density_scale,
            'majorant_resolution_factor': majorant_res_factor,
			"majorant_factor": 1.01,
            # 'albedo': {
            #     'type': 'constvolume',
            #     'value': {'type': 'rgb', 'value': [0.8, 0.9, 0.7]},
            # },
            'sigma_t': {
                'type': 'gridvolume',
                'filename': path_vol,
                'to_world': medium_T,
            },
            'albedo': albedo,
        }
        
        T_cube = T.scale(1.0)
    else:
        majorant_res_factor = 8
        if absorptive_only_test:
            density_scale = 0.2
        else:
            density_scale = 1.0
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
            'to_world': T.look_at(origin=origin, target=target, up=up).rotate([0,0,1],90).rotate([1,0,0],-0.7).rotate([0,1,0],-0.5),
            'film': {
                'type': 'hdrfilm',
                'width': width,
                'height': height,
                'filter': { 'type': 'gaussian' },
            }
        }
        medium_dict = {
            'type': 'heterogeneous',
            'absorptive_medium': absorptive_only_test,
            'scale': density_scale,
            'majorant_resolution_factor': majorant_res_factor,
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
        }
        
        T_cube = T.scale(0.95 * 3000.0)
    
    if absorptive_only_test:
        light_source_dict = {
            'type': 'constant',
            'radiance': 1.0
        }
    else:
        light_source_dict = {
            'type': 'envmap',
            'filename': path_env,
            'to_world': t_env,
            'scale': 1.0,
        }
    return {
        'type': 'scene',
        # -------------------- Sensor --------------------
        'sensor': sensor_dict,
        # -------------------- Medium --------------------
        'medium1': medium_dict,
        # Mostly just to avoid the warning
        'integrator': {
            'type': 'path',
        },
        # -------------------- Light --------------------
        # 'light': {
        #     'type': 'constant',
        #     'radiance': {'type': 'rgb', 'value': [1.0, 0.8, 0.2]},
        # },
        'light': light_source_dict,
        # -------------------- Media --------------------
        # -------------------- Shapes --------------------
        'cube': {
            # Cube covers [0, 0, 0] to [1, 1, 1] by default
            'type': 'cube',
            'bsdf': { 'type': 'null', },
            'interior': {
                'type': 'ref',
                'id':  'medium1'
            },
            'to_world': T_cube, 
        },
    }

def render_volume(integrator_name="volpath_novak14",
                 estimator="rrt",
                 distance_sampler="ff",
                 render_wdas_cloud=False,
                 absorptive_only_test=False,
                 display_figures=False,
                 save_results=True,
                 run_count=1,
                 spp=1,
                 init_time=0,
                 output_dir = "./test_renders/"):

    output_dir = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    scene_dict = load_volume_scene(render_wdas_cloud=render_wdas_cloud,
                                   absorptive_only_test=absorptive_only_test)
    scene = mi.load_dict(scene_dict)
    
    if render_wdas_cloud:
        if integrator_name == "volpath":
            output_name = output_dir+"disney_cloud_"+integrator_name+"_"+str(spp)+".exr"
            integrator =  mi.load_dict({
                        'type': integrator_name,
                        'max_depth': 100,
                        'rr_depth': 1000,
                    })
        else:
            output_name = output_dir+"disney_cloud_"+integrator_name+"_"+estimator+"_"+distance_sampler+"_"+str(spp)+".exr"
            integrator =  mi.load_dict({
                            'type': integrator_name,
                            'max_depth': 100,
                            'rr_depth': 1000,
                            'transmittance_estimator': estimator, #rt, rrt, rt_local, rrt_local
                            'distance_sampler': distance_sampler, #ff, ff_local
                        })
    else:
        if integrator_name == "volpath":
                output_name = output_dir+"smoke_"+integrator_name+"_"+str(spp)+".exr"
                integrator =  mi.load_dict({
                            'type': integrator_name,
                            'max_depth': 100,
                            'rr_depth': 1000,
                        })
        else:
            output_name = output_dir+"smoke_"+integrator_name+"_"+estimator+"_"+distance_sampler+"_"+str(spp)+".exr"
            integrator =  mi.load_dict({
                            'type': integrator_name,
                            'max_depth': 100,
                            'rr_depth': 1000,
                            'transmittance_estimator': estimator, #rt, rrt, rt_local, rrt_local
                            'distance_sampler': distance_sampler, #ff, ff_local
                        })
        
    loading_time = time.time() - init_time
    for i in range(run_count):
        img = mi.render(scene, integrator=integrator, spp=spp)

    if display_figures:
        plt.axis("off")
        plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping
        plt.show()

    if save_results:
        mi.util.write_bitmap(output_name, img)
    return [loading_time, torch.tensor(np.array(img))]

def run_experiment(asset = "cloud",
                    experiment = "absorption"):
    if experiment == "absorption":
        output_dir = os.path.join("./absorption_exp_figures/")
        os.makedirs(output_dir, exist_ok=True)
        absorptive_only_test = True

    elif experiment == "high_albedo":
        output_dir = os.path.join("./high_albedo_exp_figures/")
        os.makedirs(output_dir, exist_ok=True)
        absorptive_only_test = False
    
    estimators = ["rt", "rrt", "rt_local", "rrt_local", "nf", "rm", "ps_cum", "ps_cmf"]
    samplers   = ["ff_weighted_local"]
    spp_counts = [1, 2, 4, 8]
    run_count  = 10
    results_txt = {}

    if asset == "cloud":
        render_wdas_cloud = True
    else:
        render_wdas_cloud = False

    __, reference = render_volume(integrator_name="volpath",
                                estimator="",
                                distance_sampler="",
                                init_time=0,
                                absorptive_only_test = absorptive_only_test,
                                render_wdas_cloud = render_wdas_cloud, 
                                run_count = 1,
                                spp = 1024,
                                output_dir = output_dir)

    for estimator in estimators:
        for sampler in samplers:
            for spp in spp_counts:
                exp_name = estimator + "_" + sampler + "_" + str(spp)
                init_time = time.time()
                loading_time, img = render_volume(integrator_name="volpath_novak14",
                                                estimator=estimator,
                                                distance_sampler=sampler,
                                                init_time=init_time,
                                                absorptive_only_test = absorptive_only_test,
                                                render_wdas_cloud = render_wdas_cloud, 
                                                run_count = run_count,
                                                spp = spp,
                                                output_dir = output_dir)
                rendering_time = (time.time() - init_time) 
                rendering_time_per_run  = (rendering_time - loading_time) / run_count
                mse, rmse = compute_metrics(reference, img)
                results_txt[exp_name] = {'render_time': rendering_time_per_run,
                                         'mse': mse,
                                         'rmse': rmse}
                                         
    for spp in spp_counts:
        exp_name = "delta_tracking_track_length_local_majorants" + "_" + str(spp)
        init_time = time.time()
        loading_time, img = render_volume(integrator_name="volpath",
                                    estimator=estimator,
                                    distance_sampler=sampler,
                                    init_time=init_time,
                                    absorptive_only_test = absorptive_only_test,
                                    render_wdas_cloud = render_wdas_cloud, 
                                    run_count = run_count,
                                    spp = spp,
                                    output_dir = output_dir)
        rendering_time = (time.time() - init_time) 
        rendering_time_per_run  = (rendering_time - loading_time) / run_count
        mse, rmse = compute_metrics(reference, img)
        results_txt[exp_name] = {'render_time': rendering_time_per_run,
                                 'mse': mse,
                                 'rmse': rmse}
    
    if render_wdas_cloud:
        output_dir_txt = output_dir+"/results_cloud.txt"
    else:
        output_dir_txt = output_dir+"/results_smoke.txt"
    with open(output_dir_txt,"w") as file:
        keys = list(results_txt)
        for exp in range(len(keys)):
            print(keys[exp]+": ", file=file)
            print(results_txt[keys[exp]], file=file)
            print('\n', file=file)

if __name__ == '__main__':
    run_experiment(asset = "smoke",
                   experiment = "absorption")