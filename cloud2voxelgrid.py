import os
from os.path import realpath, join, dirname
import sys

# import drjit as dr
# import mitsuba as mi
# mi.set_variant('cuda_ad_rgb')
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d, Axes3D
import time
import json
from tqdm import tqdm

PRINT = False
# Load the dataset generated by the EM pipeline
DEVICE = "cuda"
filename = ("dust_explosion_4_g10000_2.npy")
dataset = np.load(filename)

num_gaussians = dataset.shape[0] # if args.max_gaussians is None else args.max_gaussians

first = 0
last  = first + num_gaussians

means     = np.ravel(dataset[first:last, 0:3]).reshape(num_gaussians, 3)
scales    = np.ravel(dataset[first:last, 3:6]).reshape(num_gaussians, 3)
rotations = np.ravel(dataset[first:last, 6:15]).reshape(num_gaussians, 9)
sigmats   = np.ravel(dataset[first:last, 15]).reshape(num_gaussians, 1)

if PRINT:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    print(means.shape)
    ax.scatter(means[:,0], means[:,1], means[:,2], s=np.max(scales,axis=1),c='b')
    # ax.scatter(means[:,0], means[:,1], means[:,2], s=np.min(scales,axis=1)*100.0,c='b',alpha=0.2)

    plt.show()

# Create tensor of covariance matrices
R = torch.tensor(rotations).reshape(num_gaussians,3,3).to(DEVICE)
S_vect = torch.tensor(scales).to(DEVICE)
S = torch.diag_embed(S_vect)
E = R * S * torch.transpose_copy(S,1,2) * torch.transpose_copy(R,1,2)
E_inv = torch.inverse(E)
M = torch.tensor(means).to(DEVICE)
alpha = torch.tensor(sigmats).to(DEVICE)

S_norm = 1 / (((2 * torch.pi) ** 1.5) * torch.sqrt(torch.linalg.det(E)))

# def sample_point(p: torch.tensor):
#     P = p.repeat(num_gaussians,1)
#     P_m = torch.unsqueeze(P - M,-1)
#     v = torch.sum(S_norm * torch.exp( - 0.5 * (torch.transpose(P_m,1,2) @ E_inv @ P_m).reshape(num_gaussians)) * alpha) 
#     return v

def sample_point(p: torch.tensor):
    P = p.repeat(num_gaussians,1)
    P_m = torch.unsqueeze(P - M,-1)
    v = torch.sum(S_norm * torch.exp( - 0.5 * (torch.transpose(P_m,1,2) @ E_inv @ P_m).flatten()) * alpha) 
    return v

# def sample_point_batch(p: torch.tensor, num_points):
#     P_m = p.repeat(num_gaussians,1) - M.repeat(num_points,1)
#     v = torch.sum((S_norm.repeat(num_points,1).flatten() * torch.exp( - 0.5 * torch.sum(P_m * torch.diagonal(E_inv,dim1=1,dim2=2).repeat(num_points,1) * P_m,dim=1)) * alpha.repeat(num_points,1).flatten()).reshape(num_gaussians,num_points),dim=0) 
#     return v

RES = 1024
SAMPLES_PER_VOXEL_DIM = 2
TOTAL_SAMPLES = RES * (SAMPLES_PER_VOXEL_DIM ** 3)
S_DIM = RES * SAMPLES_PER_VOXEL_DIM
grid = torch.zeros((RES, RES, RES),dtype=torch.double).to(DEVICE).flatten()
grid_x, grid_y, grid_z = torch.meshgrid(torch.arange(0,RES,1),torch.arange(0,RES,1),torch.arange(0,RES,1))
# Uniform voxel grid sampling
sample_x, sample_y, sample_z = torch.meshgrid(torch.linspace(-1 + 0.5/S_DIM, 1 - 0.5/S_DIM,S_DIM),torch.linspace(-1 + 0.5/S_DIM, 1 - 0.5/S_DIM,S_DIM),torch.linspace(-1 + 0.5/S_DIM, 1 - 0.5/S_DIM,S_DIM))

def sample_point_batch_per_g(p: torch.tensor):
    P_m = torch.unsqueeze(p - M, -1)
    v = S_norm * torch.exp( - 0.5 * (torch.transpose(P_m,1,2) @ E_inv @ P_m)).flatten() * alpha
    return v.flatten()

batched_alpha = alpha.repeat(S_DIM,1).flatten()
batched_e_inv = E_inv.repeat(S_DIM,1,1)
batched_means = M.repeat(S_DIM,1)
batched_s_norm = S_norm.repeat(S_DIM,1).flatten()

def sample_point_batch(p: torch.tensor, num_points):
    P_m = torch.unsqueeze(p.repeat(num_gaussians,1) - batched_means, -1)
    v = torch.sum((batched_s_norm * torch.exp( - 0.5 * (torch.transpose(P_m,1,2) @ batched_e_inv @ P_m)).flatten() * batched_alpha).reshape(num_gaussians,num_points),dim=0) 
    return v

# # one sample at a time
# for i in tqdm(range(S_DIM)):
#     for j in range(S_DIM):
#         for k in range(S_DIM):
#             sample = torch.tensor([sample_x[i,j,k],sample_y[i,j,k],sample_z[i,j,k]]).to(DEVICE)
#             grid_pos = torch.floor(sample * (RES//2) + RES//2).type(torch.int64)
#             flat_grid_pos = grid_pos[0] * RES * RES + grid_pos[1] * RES + grid_pos[2] 
#             grid.scatter_add_(0, flat_grid_pos, sample_point(sample))

# batch along 1 dim
for i in tqdm(range(S_DIM)):
    for j in range(S_DIM):
        sample = torch.stack((sample_x[i,j,:],sample_y[i,j,:],sample_z[i,j,:]),-1).to(DEVICE)
        grid_pos = torch.floor(sample * (RES//2) + RES//2).type(torch.int64)
        flat_grid_pos = grid_pos[:,0] * RES * RES + grid_pos[:,1] * RES + grid_pos[:,2] 
        grid.scatter_add_(0, flat_grid_pos, sample_point_batch(sample, S_DIM))

#batch along 2 dim
# for i in tqdm(range(S_DIM)):
#     sample = torch.stack((sample_x[i,:,:],sample_y[i,:,:],sample_z[i,:,:]),-1).to(DEVICE).reshape(S_DIM * S_DIM, 3)
#     grid_pos = torch.floor(sample * (RES//2) + RES//2).type(torch.int64)
#     flat_grid_pos = grid_pos[:,0] * RES * RES + grid_pos[:,1] * RES + grid_pos[:,2] 
#     grid.scatter_add_(0, flat_grid_pos, sample_point_batch(sample, S_DIM * S_DIM))

# # Sample rounds
# def inv_sigmoid(values):
#     return torch.log(values * torch.reciprocal(1-values))
# # Importance sampling from the GMM mixture
# L, G = torch.linalg.eig(E)
# SAMPLE_ROUNDS = 2_000
# grid_calls = torch.zeros((RES, RES, RES),dtype=torch.int32).to(DEVICE).flatten()
# for i in tqdm(range(SAMPLE_ROUNDS)):
#     y_s = torch.rand(size=(num_gaussians, 3), device=DEVICE)
#     x_normal = inv_sigmoid(y_s)
#     sample = torch.real(torch.transpose(torch.unsqueeze(x_normal * L, -1),1,2) @ G).reshape(num_gaussians,3) + M
#     grid_pos = torch.floor(sample * (RES//2) + RES//2).type(torch.int64)
#     flat_grid_pos = grid_pos[:,0] * RES * RES + grid_pos[:,1] * RES + grid_pos[:,2] 
#     grid.scatter_add_(0, flat_grid_pos, sample_point_batch_per_g(sample))
#     grid_calls.scatter_add_(0, flat_grid_pos, torch.ones_like(flat_grid_pos,dtype=torch.int32))

# grid = grid * torch.reciprocal(grid_calls)
# grid[torch.isnan(grid)] = 0.0

grid /= (SAMPLES_PER_VOXEL_DIM ** 3)

np.save("dust_1024_2.npy",grid.to("cpu").reshape(RES,RES,RES))
print(torch.max(grid))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.array(grid_x.to("cpu")), np.array(grid_y.to("cpu")), np.array(grid_z.to("cpu")), s= 1e-3 * np.array(grid.to("cpu")),c='r')
ax.scatter(means[:,0]*RES//2+RES//2,means[:,1]*RES//2+RES//2,means[:,2]*RES//2+RES//2, s=np.max(scales,axis=1),c='b')
plt.show()
