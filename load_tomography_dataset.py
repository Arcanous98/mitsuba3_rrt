import os
from os.path import realpath, join, dirname
import sys

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import numpy as np
import time
import json
import math

mi.set_log_level(mi.LogLevel.Debug)

def load_volume_scene(density_scale=1.0, 
                      albedo=0.0,
                      path_vol = join(f'./scenes/volumes/dust_explosion.vol')):
    
    T = mi.ScalarTransform4f

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
    medium_T = T.scale(1.75).translate(np.array([-0.5,-0.5,-0.5]))
    medium_dict = {
        'type': 'heterogeneous',
        'scale': density_scale,
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

if __name__ == '__main__':
    path_vol = join(f'./scenes/volumes/dust_explosion.vol')
    path_cams = join(f'./dust_explosion_abs/poses.npy')
    # load scene (compatible with standard Mitsuba3, no supervoxels)
    scene_dict = load_volume_scene(path_vol = path_vol)
    # load camera poses
    cam_positions = np.load(path_cams)
    sensors = []
    for cmp in cam_positions:
        sensors.append(load_sensor(cmp, 5.0))
    