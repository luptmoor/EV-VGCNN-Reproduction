import numpy as np
import torch
import torch.nn as nn

def delta(x, y):
    if x == 0 and y == 0:
        return 1
    else:
        return 0

def voxel2patch(voxel_list):
    """
    Integrates the events contained in a voxel to a 2D patch
    :param voxel_list: list of voxels to be integrated with Np objects
    :return: patch_list: tensor of flattened patches resulting from voxel list (Np x D)
    """

    print(len(voxel_list), 'voxels are being converted to patches.')
    flatten = nn.Flatten()
    patch_list = torch.zeros([len(voxel_list)])
    print(patch_list)
    i = 0
    for voxel in voxel_list:
        voxel_height = voxel[0]
        voxel_width = voxel[1]
        event_list = voxel[3]
        # Create patch with voxel width and height
        patch = torch.zeros([voxel_width, voxel_height])

        # Determine patch values according to formula from paper
        for x in range(voxel_width):
            for y in range(voxel_height):
                patch[x, y] = np.sum([delta(x - event[0], y - event[1]) * event[2] * event[3] for event in event_list])
        print(patch)
        patch_list[i] = patch
        i += 1

    print(len(patch_list), 'patches have been converted from the voxels.')

    # Return list of flattened patches
    return flatten(patch_list)


voxel_list = [[1, 2, 3, [[0.2, 0.3, 0.4, -1], [0.4, 0.1, 0.1, 1]]]]
pl = voxel2patch(voxel_list)
print(pl)

