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
import math

mi.set_log_level(mi.LogLevel.Debug)

def compute_metrics(reference, target):
    target = torch.tensor(target)
    reference = torch.tensor(reference)
    mse = torch.mean((target - reference) ** 2)
    rmse = torch.sqrt(mse)
    # lpips
    return [mse.item(), rmse.item()]

def halton(dim: int, nbpts: int):
    h = np.full(nbpts * dim, np.nan)
    p = np.full(nbpts, np.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = math.log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(math.ceil(lognbpts / math.log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1))

        for j in range(nbpts):
            d = j + 1
            sum_ = math.fmod(d, b) * p[0]
            for t in range(1, n):
                d = math.floor(d / b)
                sum_ += math.fmod(d, b) * p[t]

            h[j*dim + i] = sum_
    return h.reshape(nbpts, dim)

def warpHalton(sequence):
    hSeq = np.zeros((len(sequence[:,0]),3))
    for i,pair in enumerate(sequence):
        new_point = warp_squareToUniformSphere(pair)
        hSeq[i,:] = new_point
    return hSeq

def warp_squareToUniformSphere(sample):
    z = 1.0 - 2.0 * sample[0]
    y = math.sin(2.0 * math.pi * sample[1]) * 2.0 * math.sqrt(sample[0] * (1.0 - sample[0]))
    x = math.cos(2.0 * math.pi * sample[1]) * 2.0 * math.sqrt(sample[0] * (1.0 - sample[0]))
    return np.array((x, y, z))

def load_sensor(position, scale):
    T = mi.ScalarTransform4f
    # Apply two rotations to convert from spherical coordinates to world 3D coordinates.
    origin = T.translate(position) @ mi.ScalarPoint3f([0, 0, scale])

    return mi.load_dict({
        'type': 'orthographic',
        'to_world': T.look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 1
        },
        'film': {
            'type': 'hdrfilm',
            'width': 1000,
            'height': 1000,
            'rfilter': {'type': 'gaussian'},
        },
    })

def load_volume_scene(density_scale=1.0, 
                      albedo=0.0,
                      majorant_res_factor = 8,
                      disable_supervoxels = False,
                      majorant_factor = 1.01):
    
    T = mi.ScalarTransform4f

    if disable_supervoxels:
        majorant_res_factor = 0
    else:
        majorant_res_factor = 8

    path_vol = join(f'./scenes/volumes/dust_explosion.vol')
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
    medium_T = T.scale(1.75).translate(np.array([-0.5,-0.5,-0.5]))#T.scale(2.0).translate(np.array([-1.0,-1.0,-1.0]))
    medium_dict = {
        'type': 'heterogeneous',
        'absorptive_medium': True,
        'scale': density_scale,
        'majorant_resolution_factor': majorant_res_factor,
        "majorant_factor": majorant_factor,
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
    light_source_dict = {
        'type': 'constant',
        'radiance': 1.0
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
                 display_figures=False,
                 save_results=True,
                 spp=1,
                 init_time=0,
                 output_dir = "./dust_explosion_abs/",
                 max_depth = 1000,
                 rr_depth  = 1000,
                 disable_supervoxels = False,
                 num_views = 10):

    output_dir = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if estimator =="ps_cum" or estimator=="ps_cmf":
        majorant_factor = 1.01
    else:
        majorant_factor = 1.01

    scene_dict = load_volume_scene(majorant_factor = majorant_factor,
                                   disable_supervoxels = disable_supervoxels)
    # load sensors
    sensors = []
    samples = np.random.random_sample((num_views * 2,))
    scale = 5.0
    poses = []
    for i in range(num_views):
        pose = warp_squareToUniformSphere([samples[i * 2],samples[i * 2 + 1]])
        poses.append(pose)
        sensors.append(load_sensor(pose, scale))

    scene = mi.load_dict(scene_dict)

    if integrator_name == "volpath":
        integrator =  mi.load_dict({
                    'type': integrator_name,
                    'max_depth': max_depth,
                    'rr_depth': rr_depth,
                })
    else:
        integrator =  mi.load_dict({
                        'type': integrator_name,
                        'max_depth': max_depth,
                        'rr_depth': rr_depth,
                        'transmittance_estimator': estimator, #rt, rrt, rt_local, rrt_local
                        'distance_sampler': distance_sampler, #ff, ff_local
                    })
    
    loading_time = time.time() - init_time

    for i in range(num_views):
        img = mi.render(scene, integrator=integrator, spp=spp, sensor = sensors[i])
        output_name = join(output_dir, str(i).zfill(4)+'.exr')
        mi.util.write_bitmap(output_name, img)

    import pickle
    with open(join(output_dir,'sensors.npy'), 'wb') as fp:
        pickle.dump(poses, fp)
        fp.close()
    return loading_time

def render_views():
    estimator = "rrt"
    sampler   = "ff"
    reference_spp = 32
    num_views  = 100

    init_time = time.time()
    _ = render_volume(  integrator_name="volpath_novak14",
                        estimator=estimator,
                        distance_sampler=sampler,
                        init_time=init_time,
                        disable_supervoxels = True,
                        spp = reference_spp,
                        num_views = num_views)

if __name__ == '__main__':
    render_views()
