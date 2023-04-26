import functools
import glob
import logging
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from Preprocessing import  voxel2patch

# from torch_geometric.data import Data


def load(raw_file: str):
    f = open(raw_file, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()

    raw_data = np.uint32(raw_data)
    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7  # bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
    all_ts = all_ts / 1e6  # µs -> s
    all_p = all_p.astype(np.float64)
    all_p[all_p == 0] = -1
    #events = np.array([all_x, all_y, all_ts, all_p])
    events = np.stack([all_x, all_y, all_ts, all_p], axis=1)
    print(events.shape)
    print(events[0])
    print(events[1])
    # print(" y ", all_y)
    # print(" x ", all_x)
    # print(" p ", all_p)
    # print(" t ", all_ts)

    return events

    # x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
    # return Data(x=x, pos=pos)


# def voxelize(events, resolution, duration):
#     """
#     Voxelize event data given a list of events with coordinates and time stamps.
#
#     Args:
#         events (list): List of events, where each event is a tuple (x, y, t, p),
#             where x and y are the spatial coordinates, t is the time stamp,
#             and p is the polarity (+1 or -1).
#         resolution (float): The resolution of the voxel grid.
#         duration (float): The duration of the voxel grid.
#
#     Returns:
#         A 3D numpy array representing the voxel grid, where each voxel contains the
#         accumulated polarity of events that occurred within its bounds.
#     """
#     # Calculate the dimensions of the voxel grid
#     width = int(np.ceil(1.0 / resolution))
#     height = int(np.ceil(1.0 / resolution))
#     depth = int(np.ceil(duration))
#
#     # Create an empty voxel grid
#     voxel_grid = torch.zeros((width, height, depth), dtype=torch.float32)
#
#     # Iterate over each event and accumulate its polarity in the corresponding voxel
#     for event in events:
#         x, y, t, p = event
#         ix = int(np.floor(x / resolution))
#         iy = int(np.floor(y / resolution))
#         it = int(np.floor(t))
#         print((ix, iy, it))
#         if ix >= 0 and ix < width and iy >= 0 and iy < height and it >= 0 and it < depth:
#             voxel_grid[ix, iy, it] += p
#
#     return voxel_grid




# def subdivide_array(events, nx, ny, nt):
#     """
#     Subdivides a 4D numpy array into a grid of nx by ny by nt subarrays.
#     """
#     # get the shape of the input array
#     shape = events.shape
#     print('shape: ', shape)
#
#     # compute the size of each subarray
#     nx_size = shape[0] // nx
#     ny_size = shape[1] // ny
#     nt_size = shape[2] // nt
#
#     # reshape the array into a grid of subarrays
#     subarrays = events.reshape((nx, nx_size, ny, ny_size, nt, nt_size, shape[3]))
#
#     return subarrays


# nx = 10
# ny = 10
# nt = 10
#
#
# x_min, x_max = points[:, 0].min(), points[:, 0].max()
# y_min, y_max = points[:, 1].min(), points[:, 1].max()
# t_min, t_max = points[:, 2].min(), points[:, 2].max()
# x_bins = np.linspace(x_min, x_max, nx + 1)
# y_bins = np.linspace(y_min, y_max, ny + 1)
# t_bins = np.linspace(t_min, t_max, nt + 1)
#
# # digitize the points into the nx by ny by nt grid nodes
# x_idx = np.digitize(points[:, 0], x_bins)
# y_idx = np.digitize(points[:, 1], y_bins)
# t_idx = np.digitize(points[:, 2], t_bins)
#
# # create an empty nested list to hold the grid data
# grid_data = [[[[] for _ in range(nt)] for _ in range(ny)] for _ in range(nx)]

# loop over the points and assign them to the appropriate grid node
# for i, (x, y, t, p) in enumerate(points):
#     x_idx_i = x_idx[i] - 1 if x_idx[i] > 0 else 0
#     y_idx_i = y_idx[i] - 1 if y_idx[i] > 0 else 0
#     t_idx_i = t_idx[i] - 1 if t_idx[i] > 0 else 0
#     grid_data[x_idx_i][y_idx_i][t_idx_i].append((x, y, t, p))
#
#
# # print the number of points in each grid node
# for i in range(nx):
#     for j in range(ny):
#         for k in range(nt):
#             print(f"Grid node ({i}, {j}, {k}): {len(grid_data[i, j, k])} points")
#


# data = load("image_0005.bin")
# voxels = voxelize(data, 10, 10, 10)
# patches = voxel2patch(voxels)


# x, y, z = data[0], data[1], data[2]
# c = data[3]
# # extract the fourth dimension
# fig, ax = plt.subplots()
# scatter = ax.scatter(x, y, c=c, cmap='viridis')
# plt.colorbar(scatter)
#
# # set the labels for the axes
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
#
# # show the plot
# plt.show()