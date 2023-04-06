import functools
import glob
import logging
import numpy as np
import os
import torch

from torch_geometric.data import Data


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
        events = np.column_stack((all_x, all_y, all_ts, all_p))
        print(" y ", all_y)
        print(" x ", all_x)
        print(" p ", all_p)
        print(" t ", all_ts)


        return events

        # x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
        # return Data(x=x, pos=pos)

import numpy as np

def voxelize(events, resolution, duration):
    """
    Voxelize event data given a list of events with coordinates and time stamps.

    Args:
        events (list): List of events, where each event is a tuple (x, y, t, p),
            where x and y are the spatial coordinates, t is the time stamp,
            and p is the polarity (+1 or -1).
        resolution (float): The resolution of the voxel grid.
        duration (float): The duration of the voxel grid.

    Returns:
        A 3D numpy array representing the voxel grid, where each voxel contains the
        accumulated polarity of events that occurred within its bounds.
    """
    # Calculate the dimensions of the voxel grid
    width = int(np.ceil(1.0 / resolution))
    height = int(np.ceil(1.0 / resolution))
    depth = int(np.ceil(duration))

    # Create an empty voxel grid
    voxel_grid = torch.zeros((width, height, depth), dtype=torch.float32)

    # Iterate over each event and accumulate its polarity in the corresponding voxel
    for event in events:
        x, y, t, p = event
        ix = int(np.floor(x / resolution))
        iy = int(np.floor(y / resolution))
        it = int(np.floor(t))
        print((ix, iy, it))
        if ix >= 0 and ix < width and iy >= 0 and iy < height and it >= 0 and it < depth:
            voxel_grid[ix, iy, it] += p

    return voxel_grid

data = load("image_0005.bin")

output = voxelize(data, 0.1, 1)

print(output)
