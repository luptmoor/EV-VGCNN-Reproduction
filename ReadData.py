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
    def __init__(self, folder_dir, max_length=5000):
        self.folder_dir = folder_dir
        self.filenames = glob.glob(os.path.join(folder_dir, "*.bin"))
        self.max_length = max_length
        self.data = []
        self.labels = []
        label_dict = {'accordion': 0,
                      'airplanes': 1,
                      'anchor': 2,
                      'ant': 3,
                      'BACKGROUND_Google' : 4,
                      'barrel' : 5,
                      'bass' : 6,
                      'beaver' : 7,
                      'binocular' : 8,
                      'bonsai' : 9,
                      'brain' : 10,
                      'brontosaurus' : 11,
                      'buddha' : 12,
                      'butterfly' : 13,
                      'camera' : 14,
                      'cannon' : 15,
                      'car_side' : 16,
                      'ceiling_fan' : 17,
                      'cellphone' : 18,
                      'chair' : 19,
                      'chandelier': 20,
                      'cougar_body': 21,
                      'cougar_face': 22,
                      'crab': 23,
                      'crayfish': 24,
                      'crocodile': 25,
                      'crocodile_head': 26,
                      'cup': 27,
                      'dalmatian': 28,
                      'dollar_bill': 29,
                      'dolphin': 30,
                      'dragonfly': 31,
                      'electric_guitar': 32,
                      'elephant': 33,
                      'emu': 34,
                      'euphonium': 35,
                      'ewer': 36,
                      'Faces_easy': 37,
                      'ferry': 38,
                      'flamingo': 39,
                      'flamingo_head': 40,
                      'garfield': 41,
                      'gerenuk': 42,
                      'gramophone': 43,
                      'grand_piano': 44,
                      'hawksbill': 45,
                      'headphone': 46,
                      'hedgehog': 47,
                      'helicopter': 48,
                      'ibis': 49,
                      'inline_skate': 50,
                      'joshua_tree': 51,
                      'kangaroo': 52,
                      'ketch': 53,
                      'lamp': 54,
                      'laptop': 55,
                      'Leopards': 56,
                      'llama': 57,
                      'lobster': 58,
                      'lotus': 59,
                      'mandolin': 60,
                      'mayfly': 61,
                      'menorah' : 62,
                      'metronome' : 63,
                      'minaret' : 64,
                      'Motorbikes' : 65,
                      'nautilus' : 66,
                      'octopus' : 67,
                      'okapi' : 68,
                      'pagoda' : 69,
                      'panda' : 70,
                      'pigeon' : 71,
                      'pizza' : 72,
                      'platypus' : 73,
                      'pyramid' : 74,
                      'revolver' : 75,
                      'rhino' : 76,
                      'rooster' : 77,
                      'saxophone' : 78,
                      'schooner' : 79,
                      'scissors' : 80,
                      'scorpion' : 81,
                      'sea_horse' : 82,
                      'snoopy' : 83,
                      'soccer_ball' : 84,
                      'stapler' : 85,
                      'starfish' : 86,
                      'stegosaurus' : 87,
                      'stop_sign' : 88,
                      'strawberry' : 89,
                      'sunflower' : 90,
                      'tick' : 91,
                      'trilobite' : 92,
                      'umbrella' : 93,
                      'watch' : 94,
                      'water_lilly' : 95,
                      'wheelchair' : 96,
                      'wild_cat' : 97,
                      'windsor_chair' : 98,
                      'wrench' : 99,
                      'yin_yang' : 100}
            
        for filename in self.filenames:
            full_path = filename
            f = open(full_path, 'rb')
            raw_data = np.fromfile(f, dtype=np.uint8)
            f.close()
            raw_data = np.uint32(raw_data)
            all_y = raw_data[1::5]
            all_x = raw_data[0::5]
            all_p = (raw_data[2::5] & 128) >> 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
            all_ts = all_ts / 1e6
            all_p = all_p.astype(np.float64)
            all_p[all_p == 0] = -1
            events = np.column_stack((all_x, all_y, all_ts, all_p))
            events_padded = torch.tensor(events[:self.max_length])

            # every image is voxelized and then added to dataloader
            voxel_list = voxelize(events_padded, 10, 10, 10)

            #voxel selection:
            selected_voxels = []
            for voxel in voxel_list:
                if not len(voxel[3]) == 0:
                    selected_voxels.append(voxel)
            print(len(selected_voxels), 'have been selected.')

            # coordinate vectors:
            coordinate_list = [voxel[0:3] for voxel in selected_voxels]
            print('Coordinate vector: ')
            print(coordinate_list)

            # voxel2patch:
            feature_vector = voxel2patch(23, 10, voxel[3])
            print(feature_vector)

            self.data.append(events_padded)
            
            # Get the parent directory name and one-hot encode it as the label
            folder_name = os.path.basename(os.path.dirname(filename))
            label_idx = label_dict[folder_name]
            label = np.zeros(101)
            label[label_idx] = 1
            self.labels.append(label)
        
        self.labels = torch.tensor(self.labels)

    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# folder_dirs = [
#    r"./Caltech101/accordion",
#    # r"./Caltech101/airplanes",
#    # r"./Caltech101/anchor",
#    # r"./Caltech101/ant",
#    # r"./Caltech101/BACKGROUND_Google",
#    # r"./Caltech101/barrel",
#    # r"./Caltech101/bass",
#    # r"./Caltech101/beaver",
#    # r"./Caltech101/binocular",
#    # r"./Caltech101/bonsai",
#    # r"./Caltech101/brain",
#    # r"./Caltech101/brontosaurus",
#    # r"./Caltech101/buddha",
#    # r"./Caltech101/butterfly",
#    # r"./Caltech101/camera",
#    # r"./Caltech101/cannon",
#    # r"./Caltech101/car_side",
#    # r"./Caltech101/ceiling_fan",
#    # r"./Caltech101/cellphone",
#    # r"./Caltech101/chair",
#    # r"./Caltech101/chandelier",
#    # r"./Caltech101/cougar_body",
#    # r"./Caltech101/cougar_face",
#    # r"./Caltech101/crab",
#    # r"./Caltech101/crayfish",
#    # r"./Caltech101/crocodile",
#    # r"./Caltech101/crocodile_head",
#    # r"./Caltech101/cup",
#    # r"./Caltech101/dalmatian",
#    # r"./Caltech101/dollar_bill",
#    # r"./Caltech101/dolphin",
#    # r"./Caltech101/dragonfly",
#    # r"./Caltech101/electric_guitar",
#    # r"./Caltech101/elephant",
#    # r"./Caltech101/emu",
#    # r"./Caltech101/euphonium",
#    # r"./Caltech101/ewer",
#    # r"./Caltech101/Faces_easy",
#    # r"./Caltech101/ferry",
#    # r"./Caltech101/flamingo",
#    # r"./Caltech101/flamingo_head",
#    # r"./Caltech101/garfield",
#    # r"./Caltech101/gerenuk",
#    # r"./Caltech101/gramophone",
#    # r"./Caltech101/grand_piano",
#    # r"./Caltech101/hawksbill",
#    # r"./Caltech101/headphone",
#    # r"./Caltech101/hedgehog",
#    # r"./Caltech101/helicopter",
#    # r"./Caltech101/ibis",
#    # r"./Caltech101/inline_skate",
#    # r"./Caltech101/joshua_tree",
#    # r"./Caltech101/kangaroo",
#    # r"./Caltech101/ketch",
#    # r"./Caltech101/lamp",
#    # r"./Caltech101/laptop",
#    # r"./Caltech101/Leopards",
#    # r"./Caltech101/llama",
#    # r"./Caltech101/lobster",
#    # r"./Caltech101/lotus",
#    # r"./Caltech101/mandolin",
#    # r"./Caltech101/mayfly",
#    # r"./Caltech101/menorah",
#    # r"./Caltech101/metronome",
#    # r"./Caltech101/minaret",
#    # r"./Caltech101/Motorbikes",
#    # r"./Caltech101/nautilus",
#    # r"./Caltech101/octopus",
#    # r"./Caltech101/okapi",
#    # r"./Caltech101/pagoda",
#    # r"./Caltech101/panda",
#    # r"./Caltech101/pigeon",
#    # r"./Caltech101/pizza",
#    # r"./Caltech101/platypus",
#    # r"./Caltech101/pyramid",
#    # r"./Caltech101/revolver",
#    # r"./Caltech101/rhino",
#    # r"./Caltech101/rooster",
#    # r"./Caltech101/saxophone",
#    # r"./Caltech101/schooner",
#    # r"./Caltech101/scissors",
#    # r"./Caltech101/scorpion",
#    # r"./Caltech101/sea_horse",
#    # r"./Caltech101/snoopy",
#    # r"./Caltech101/soccer_ball",
#    # r"./Caltech101/stapler",
#    # r"./Caltech101/starfish",
#    # r"./Caltech101/stegosaurus",
#    # r"./Caltech101/stop_sign",
#    # r"./Caltech101/strawberry",
#    # r"./Caltech101/sunflower",
#    # r"./Caltech101/tick",
#    # r"./Caltech101/trilobite",
#    # r"./Caltech101/umbrella",
#    # r"./Caltech101/watch",
#    # r"./Caltech101/water_lilly",
#    # r"./Caltech101/wheelchair",
#    # r"./Caltech101/wild_cat",
#    # r"./Caltech101/windsor_chair",
#    # r"./Caltech101/wrench",
#    # r"./Caltech101/yin_yang"
# ]
#
# # Create an instance of the dataset
# all_labels = []
# nr = 0
# for folder_dir in folder_dirs:
#     # Create an instance of the dataset for the current folder
#     dataset = BinaryFileDataset(folder_dir)
#     #nr = nr+1
#
#     # Create a data loader to load the dataset in batches
#     batch_size = 64
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     MAX_LENGTH = dataset.max_length  # Get the maximum length from the dataset
#
#     # Iterate over the data loader
#     for batch_idx, (data, label) in enumerate(dataloader):
#         # Pad all tensors in the batch to the same length
#         padded_data = []
#         if(batch_idx == 0):
#             all_labels.append(label[0])
#         for d in data:
#             pad_size = MAX_LENGTH - d.shape[0]
#             padded_d = pad_sequence([d], batch_first=True, padding_value=0)[0] # updated argument name
#             padded_data.append(padded_d)
#         padded_data = torch.stack(padded_data, dim=0)
#         #print('######## DATA: ', padded_data)
#         #print('######LABEL: ', all_labels)
#
#         # Stack the padded tensors along the 0th dimension
#         print(f"Folder: {folder_dir}, Batch {batch_idx}: data shape = {padded_data.shape}, label shape = {label.shape}")
#         #print(nr)
# # Concatenate all the labels into a single tensor
# labels = torch.stack(all_labels, dim=0)
#
