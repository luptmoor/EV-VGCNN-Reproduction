import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def delta(x, y):
    if x == 0 and y == 0:
        return 1
    else:
        return 0

def voxel2patch(voxel_width, voxel_height, voxel_list):
    """
    Integrates the events contained in a voxel to a 2D patch
    :param voxel_width: width of voxels (int), should be constant for all voxels
    :param voxel_height: height of voxels (int), should be constant for all voxels
    :param voxel_list: list of Np voxels to be integrated, voxels should be resembled as lists with event locations and data
    :return: patch_list: tensor of flattened patches resulting from voxel list (Np x voxel_width*voxel_height)
    """

    print(len(voxel_list), 'voxels are being converted to patches.')
    flatten = nn.Flatten()

    patch_list = torch.zeros(len(voxel_list), voxel_width, voxel_height)

    i = 0
    for voxel in voxel_list:
        event_list = voxel # Voxels are just lists of the events inside them
        # Create patch with voxel width and height
        patch = torch.zeros(voxel_width, voxel_height)

        # Determine patch values according to formula from paper
        for x in range(voxel_width):
            for y in range(voxel_height):
                patch[x, y] = np.sum([delta(x - event[0], y - event[1]) * event[2] * event[3] for event in event_list])

        patch_list[i] = patch
        i += 1


    # Return list of flattened patches
    return flatten(patch_list)



def voxelize(events, nx, ny, nt):
    # Range of data
    print('Number of events:', len(events))
    print('x range: ', min(events[:, 0]), ' - ', max(events[:, 0]))
    print('y range: ', min(events[:, 1]), ' - ', max(events[:, 1]))
    print('t range: ', min(events[:, 2]), ' - ', max(events[:, 2]))
    print('Shape of events: ', events.shape)

    dx = (max(events[0]) - min(events[0])) // nx
    dy = (max(events[1]) - min(events[1])) // ny
    dt = (max(events[2]) - min(events[2])) / nt

    voxel_list = []

    for x in range(nx):
        print('row ', x, ' / ', nx)
        for y in range(ny):
            for t in range(nt):
                selection = []
                x_min = x * dx
                x_max = (x + 1) * dx

                y_min = y * dy
                y_max = (y + 1) * dy

                t_min = t * dt
                t_max = (t + 1) * dt
                #
                # print('x_min: ', x_min, 'x_max: ', x_max)
                # print('y_min: ', y_min, 'y_max: ', y_max)
                # print('t_min: ', t_min, 't_max: ', t_max)

                for event in events:
                    if x_min <= event[0] <= x_max and y_min <= event[1] <= y_max and t_min <= event[2] <= t_max:
                        selection.append(event)


                # change coordinates of events to local coordinates
                for event in selection:
                    event[0] -= x * dx
                    event[1] -= y * dy
                    event[2] -= t * dt

                voxel_list.append([x, y, t, selection])

    print(len(voxel_list), ' voxels have been created.')
    print('Verification: input grid was', nx, 'by', ny, 'by', nt, '=', nx * ny * nt)
    print('Voxel size: ', dx, dy, dt)
    return dx, dy, dt, voxel_list

def plot_events(points, dx, dy, dz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([p[0] for p in points], [p[1] for p in points], [p[2] for p in points])

    # Add grid lines
    x_range = max([p[0] for p in points]) - min([p[0] for p in points])
    y_range = max([p[1] for p in points]) - min([p[1] for p in points])
    z_range = max([p[2] for p in points]) - min([p[2] for p in points])
    x_ticks = [i for i in range(int(min([p[0] for p in points])), int(max([p[0] for p in points])) + 1, dx)]
    y_ticks = [i for i in range(int(min([p[1] for p in points])), int(max([p[1] for p in points])) + 1, dy)]
    z_ticks = [i for i in range(int(min([p[2] for p in points])), int(max([p[2] for p in points])) + 1, dz)]
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_zticks(z_ticks)
    ax.xaxis.set_tick_params(pad=10)
    ax.yaxis.set_tick_params(pad=10)
    ax.zaxis.set_tick_params(pad=10)
    ax.grid(True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# "Unit Test"
# voxel_list = [[[0.2, 0.3, 0.4, -1], [0.4, 0.1, 0.1, 1], [1, 1, 2, 1]]]
# pl = voxel2patch(2, 3, voxel_list)
# print(pl)

