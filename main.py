import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Our functions
from network import *
from ReadData import BinaryFileDataset
from Preprocessing import voxel2patch, voxelize




# 1. Load Data
training_data_loader = load_dataloader(0)

model_VGCNN = VGCNN_MFRL()

optimizer = optim.Adam(model_VGCNN.parameters(), lr=1e-3) # Stochastic Gradient Descent (idk if this is ideal)
loss_fn = nn.MSELoss()  # Mean-squared error (idk if this ideal, but probably)

for i, data in enumerate(training_data_loader):
    inputs, labels = data
    print(inputs[:, :, 2])
    # outputs = model_VGCNN.forward(input)
    # loss = loss_fn(outputs, labels)
    # loss.backward()  # backprop
    # optimizer.step()  # SGD




dummy = input('Everything fine till here.')










# 2. Voxelize (Serban)
#    example: xc  yc tc   x0 y0 t0 p0   x1  y1 t1 p1 (events)           xc means x-coordinate of the voxel
#voxel_list = [[1, 2, 3, [[1, 2, 3, -1], [2, 3, 1, 1]]]]
vec = torch.Tensor([0, 2, 3])
data = load("image_0005.bin")
voxel_list = voxelize(data, 10, 10, 10)

voxel_width = 23.3
voxel_height = 10

# 3. Graph construction:
# 3A Coordinate vector (takes first 3 entries of voxel list)
coordinate_vector = torch.Tensor(voxel_list[:][:2])

# 3B Feature vector (takes last entry of voxel_list (events) and integrates them)
feature_vector = voxel2patch(voxel_width, voxel_height, voxel_list[:][3])

# (4) Feed forward into VGCNN (not actually needed here but cool to check if it works)
model_VGCNN = VGCNN_MFRL()
prediction = model_VGCNN.forward(coordinate_vector, feature_vector)
print(prediction)


dummy = input('Everything fine till here.')
# 5 Data Loader (pair input vectors and ground truth vectors, also specify batch size)


# 5 Training loop (Sergio ( ͡° ͜ʖ ͡°) )

num_epochs = 20  # (idk if ideal)
optimizer = optim.Adam(model_VGCNN.parameters(), lr=1e-3) # Stochastic Gradient Descent (idk if this is ideal)
loss_fn = nn.MSELoss()  # Mean-squared error (idk if this ideal, but probably)

for epoch in range(num_epochs):
    print('Epoch ', epoch+1)
    for i, data in enumerate(training_data_loader):
        inputs, labels = data
        outputs = model_VGCNN.forward(input)
        loss = loss_fn(outputs, labels)
        loss.backward()  # backprop
        optimizer.step()  # SGD







