import functools
import glob
import logging
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

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
    events = np.array([all_x, all_y, all_ts, all_p])
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

def voxelize(events, nx, ny, nt):
    # Range of data
    print('Number of events:', len(events))
    print('x range: ', min(events[0]), ' - ', max(events[0]))
    print('y range: ', min(events[1]), ' - ', max(events[1]))
    print('t range: ', min(events[2]), ' - ', max(events[2]))
    print('Shape of events: ', events.shape)

    dx = (max(events[0]) - min(events[0])) // nx
    dy = (max(events[1]) - min(events[1])) // ny
    dt = (max(events[2]) - min(events[2])) / nt

    voxel_list = []

    for x in range(nx):
        print('voxel ', x)
        for y in range(ny):
            for t in range(nt):

                x_min = x * dx + min(events[0])
                x_max = (x + 1) * dx + min(events[0])

                y_min = y * dy + min(events[1])
                y_max = (y + 1) * dy + min(events[1])

                t_min = t * dt + min(events[2])
                t_max = (t + 1) * dt + min(events[2])

                print('x_min: ', x_min, 'x_max: ', x_max)
                print('y_min: ', y_min, 'y_max: ', y_max)
                print('t_min: ', t_min, 't_max: ', t_max)

                selection = np.where(x_min <= events[0][:])
                selection = np.where(x_max > selection[0][:])

                selection = np.where(y_min <= selection[1][:])
                selection = np.where(y_max > selection[1][:])

                selection = np.where(t_min <= selection[2][:])
                selection = np.where(t_max > selection[2][:])


                event_list = []
                # change coordinates of events to local coordinates
                for event in selection:
                    event[0] -= x * dx
                    event[1] -= y * dy
                    event[2] -= t * dt

                    # make sure polarity is of format 1/-1
                    if event[3] == 1:
                        pass
                    else:
                        event[3] = -1

                    event_list.append(event)

                voxel_list.append([x, y, t, selection])

    return voxel_list


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

data = load("image_0005.bin")
print(voxelize(data, 10, 10, 10))
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