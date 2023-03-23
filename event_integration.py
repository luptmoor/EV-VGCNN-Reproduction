import numpy as np

def delta(x, y):
    if x == 0 and y == 0:
        return 1
    else:
        return 0

def voxel2patch(voxel_list):
    """
    Integrates the events contained in a voxel to a 2D patch
    :param voxel_list: list of voxels to be integrated
    :return: patch_list: list of patches resulting from voxel list
    """

    print(len(voxel_list), 'voxels are being converted to patches.')
    patch_list = []
    for voxel in voxel_list:
        voxel_height = voxel[0]
        voxel_width = voxel[1]
        event_list = voxel[3]
        patch = np.array(voxel_width, voxel_height)

        for x in voxel_width:
            for y in voxel_height:
                patch[x, y] = np.sum([delta(x - event[0], y - event[1]) * event[2] * event[3] for event in event_list])

        patch_list.append(patch)

    print(len(patch_list), 'patches have been converted from the voxels.')
    return patch_list
