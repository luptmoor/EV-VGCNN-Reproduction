import functools
import glob
import logging
import numpy as np
import os
import torch

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence


class BinaryFileDataset(Dataset):
    def __init__(self, folder_dir, max_length=10000):
        self.folder_dir = folder_dir
        self.filenames = os.listdir(folder_dir)
        self.max_length = max_length  # Set the maximum length to a fixed value
        self.data = []  # Initialize an empty list to store the data
        
        for filename in self.filenames:
            full_path = os.path.join(self.folder_dir, filename)
            f = open(full_path, 'rb')
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
            events_padded = torch.tensor(events[:self.max_length])  # Limit the maximum length
            self.data.append(events_padded)
        
        self.labels = torch.tensor([0] * len(self.filenames))  # Set the label for each file
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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


folder_dir = r"C:\Users\SID TUDelft\Documents\TU DELFT\Msc1\Deep Learning\Deep Learning Project\EV-VGCNN-Reproduction-main\buddha"

# # Create an instance of the dataset
dataset = BinaryFileDataset(folder_dir)

# # Create a data loader to load the dataset in batches
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

MAX_LENGTH = dataset.max_length  # Get the maximum length from the dataset

# # Iterate over the data loader
# Iterate over the data loader
for batch_idx, (data, label) in enumerate(dataloader):
    # Pad all tensors in the batch to the same length
    padded_data = []
    for d in data:
        pad_size = MAX_LENGTH - d.shape[0]
        padded_d = pad_sequence([d], batch_first=True, padding_value=0)[0] # updated argument name
        padded_data.append(padded_d)
    padded_data = torch.stack(padded_data, dim=0)

    # Stack the padded tensors along the 0th dimension
    print(f"Batch {batch_idx}: data shape = {padded_data.shape}, label shape = {label.shape}")

print(label[0], padded_data[0])