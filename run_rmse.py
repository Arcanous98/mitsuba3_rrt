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
mi.set_log_level(mi.LogLevel.Debug)

def compute_metrics(reference, target):
    target = torch.tensor(target)
    reference = torch.tensor(reference)
    mse = torch.mean((target - reference) ** 2)
    rmse = torch.sqrt(mse)
    # lpips
    return [mse.item(), rmse.item()]


# target = mi.cuda_ad_rgb.TensorXf(mi.Bitmap('gaussians.exr'))
# reference = mi.cuda_ad_rgb.TensorXf(mi.Bitmap('gaussians_ref.exr'))

target = mi.cuda_ad_rgb.TensorXf(mi.Bitmap('volume_grids.exr'))
reference = mi.cuda_ad_rgb.TensorXf(mi.Bitmap('volume_grids_ref.exr'))

print(compute_metrics(reference, target))
