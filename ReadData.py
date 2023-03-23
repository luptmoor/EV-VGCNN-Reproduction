import functools
import glob
import logging
import numpy as np
import os
import torch

from torch_geometric.data import Data


def load(raw_file: str) -> Data:
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
        events = torch.from_numpy(events).float().cuda()

        x, pos = events[:, -1:], events[:, :3]   # x = polarity, pos = spatio-temporal position
        return Data(x=x, pos=pos)

print(load("image_0005.bin"))