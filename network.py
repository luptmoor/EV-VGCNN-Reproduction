import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from event_integration import voxel2patch
#from dataloader import load_dataloader
import numpy as np


from ReadData import load, voxelize
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def cosine_similarity(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    x1x2 = torch.matmul(src, dst.transpose(1, 2))
    x1_2 = torch.sum(src ** 2, -1).view(B, N, 1)
    x2_2 = torch.sum(dst ** 2, -1).view(B, M, 1).permute(0, 2, 1)
    x1_1_x2_2 = torch.matmul(x1_2, x2_2) + 1e-8
    return torch.div(x1x2, x1_1_x2_2)

"""index points """
def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


"""knn according to square distance"""
def knn(nsample, xyz, new_xyz, cosine=False):
    if cosine:
        dist = cosine_similarity(xyz, new_xyz)
    else:
        dist = square_distance(xyz, new_xyz)
    idx = torch.topk(dist, k=nsample, dim=-1, largest=False)[1]  # [B, N, nsample]
    return idx


"""graph pooling layer"""
class random_pool(nn.Module):
    def __init__(self, output_coord_num: int, neighbor_num: int, if_aggregate: bool):
        super().__init__()
        self.output_coord_num = output_coord_num
        self.neighbor_num = neighbor_num
        self.if_aggregate = if_aggregate

    def forward(self, critic_voxel_feature, critic_voxel_coordinate, coord_orig=None):
        B, N = critic_voxel_feature.size(0), critic_voxel_feature.size(2)

        pool_num = self.output_coord_num
        sample_idx = torch.randperm(N)[:pool_num].type_as(critic_voxel_feature).long()
        new_coord = critic_voxel_coordinate[:, :, sample_idx]
        new_feature = critic_voxel_feature[:, :, sample_idx]
        if coord_orig is not None:
            return new_feature, new_coord, coord_orig[:, :, sample_idx]
        return new_feature, new_coord

"""basenet"""
class relation_net_multi(nn.Module):
    def __init__(self, in_channel, num_neighbors, feature_in_dim, feature_out_dim, ifcosine=False):
        super().__init__()
        self.ifcosine = ifcosine
        in_channel = 2 * in_channel
        self.num_neighbors = num_neighbors
        self.num_scale = len(self.num_neighbors)

        self.coord_relation_net = nn.ModuleList()
        self.feat_relation_net = nn.ModuleList()

        self.feat_self_net = nn.Sequential(
            nn.Conv1d(feature_in_dim, feature_out_dim, kernel_size=1),
            nn.BatchNorm1d(feature_out_dim),
            nn.LeakyReLU(0.0, inplace=False)
        )

        for i in range(self.num_scale):
            if i == 0:
                self.coord_relation_net.append(nn.Sequential(
                    nn.Conv2d(in_channel, self.num_neighbors[i], kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.num_neighbors[i]),
                    nn.Tanh(),
                ))
            else:
                self.coord_relation_net.append(nn.Sequential(
                    nn.Conv2d(in_channel, self.num_neighbors[i] - self.num_neighbors[i - 1], kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.num_neighbors[i] - self.num_neighbors[i - 1]),
                    nn.Tanh(),
                ))

            self.feat_relation_net.append(nn.Sequential(
                nn.Conv2d(feature_in_dim, feature_out_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(feature_out_dim),
                nn.ReLU(inplace=True),
            ))

    def forward(self, voxel_feature, voxel_coordinate):
        features = self.feat_self_net(voxel_feature)

        voxel_feature, voxel_coordinate = \
            voxel_feature.permute(0, 2, 1).contiguous(), \
                voxel_coordinate.permute(0, 2, 1).contiguous()  # B x dim x N -> B x N x dim

        neighbor_idx_ls = self.get_neighbor_idx(voxel_coordinate, )

        for i in range(self.num_scale):
            features += self.single_scale_aggre(voxel_feature, voxel_coordinate, neighbor_idx_ls[i], i)

        return features

    def single_scale_aggre(self, voxel_feature, voxel_coordinate, neighbor_idx_ls, idx):
        neighbor_index = neighbor_idx_ls
        num_neighbors = neighbor_index.shape[2]

        feature_neighbour = index_points(voxel_feature, neighbor_index)
        feature_neighbour = self.feat_relation_net[idx](feature_neighbour.permute(0, 3, 1, 2).contiguous()).permute(0,2,3,1).contiguous()

        coordinate_neighbour = index_points(voxel_coordinate, neighbor_index)

        self_coordinate = coordinate_neighbour[:, :, 0, :].unsqueeze(2).repeat(1, 1, num_neighbors,
                                                                               1)
        coordinate_relation = torch.cat((self_coordinate,
                                         self_coordinate - coordinate_neighbour,), -1)

        coordinate_relation = coordinate_relation.permute(0, 3, 1, 2).contiguous()

        relation = self.coord_relation_net[idx](coordinate_relation).transpose(1, 2).contiguous()

        feature_with_relation = torch.einsum('...i j, ... j d->... i d', relation, feature_neighbour)
        feature_with_relation = rearrange(feature_with_relation, "b n k d -> b d n k")
        feature_aggregation = torch.sum(feature_with_relation, dim=-1)

        return feature_aggregation

    def get_neighbor_idx(self, voxel_coordinate):
        neighbor_sum = np.sum(np.array(self.num_neighbors))
        neighbor_index = knn(neighbor_sum, voxel_coordinate, voxel_coordinate, cosine=self.ifcosine)
        self_idx = neighbor_index[:, :, 0]
        neighbor_idx_list = []
        for i in range(self.num_scale):
            if i == 0:
                neighbor_index_new = neighbor_index[:, :, :self.num_neighbors[i]]
            else:
                neighbor_index_new = neighbor_index[:, :, self.num_neighbors[i - 1]:self.num_neighbors[i]]
                neighbor_index_new[:, :, 0] = self_idx
            neighbor_idx_list.append(neighbor_index_new)
        return neighbor_idx_list

# Network class
class VGCNN_MFRL(nn.Module):
    def __init__(self, feat_dim=25, num_classes = 101, num_neighbors = [10, 25]):
        super().__init__()

        num_neighbors = num_neighbors
        num_points_input = 1024
        pool_out = num_points_input - 128
        """1st stage"""

        """process 2D semantic feauture for each voxel"""
        self.aggregation_1 = relation_net_multi(in_channel=3, num_neighbors=num_neighbors, feature_in_dim=feat_dim, feature_out_dim=64)
        self.pool_layer_1 = random_pool(output_coord_num=pool_out, neighbor_num=6, if_aggregate=False)
        pool_out -= 128

        """2nd stage"""
        self.aggregation_2 = relation_net_multi(in_channel=3, num_neighbors=num_neighbors, feature_in_dim=64, feature_out_dim=64,
                                             ifcosine=False)
        self.pool_layer_2 = random_pool(output_coord_num=pool_out, neighbor_num=6, if_aggregate=False)
        pool_out -= 128

        """3rd stage"""
        self.aggregation_3 = relation_net_multi(in_channel=3, num_neighbors=num_neighbors, feature_in_dim=64, feature_out_dim=128,
                                             ifcosine=False)
        self.pool_layer_3 = random_pool(output_coord_num=pool_out, neighbor_num=6, if_aggregate=False)

        """4th stage"""
        self.aggregation_4 = relation_net_multi(in_channel=3, num_neighbors=num_neighbors, feature_in_dim=128, feature_out_dim=128,
                                             ifcosine=False)

        """6th stage"""
        self.aggregation_6 = nn.Sequential(nn.Conv1d(128, 512, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(512),
                                   nn.LeakyReLU(negative_slope=0.2))

        """Classifier"""
        self.fc_1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_3 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, voxel_feature, voxel_coordinate):
        """initial stage"""
        B, N = voxel_coordinate.shape[0], voxel_coordinate.shape[2]

        """1st stage"""
        voxel_feature_aggr_1 = self.aggregation_1(voxel_feature, voxel_coordinate)  # (B, 64, N)
        pool_feat_1, pool_coord_1 = voxel_feature_aggr_1, voxel_coordinate


        """2nd stage"""
        voxel_feature_aggr_2 = self.aggregation_2(pool_feat_1, pool_coord_1)
        pool_feat_2, pool_coord_2 = voxel_feature_aggr_2, pool_coord_1

        """3rd stage"""
        voxel_feature_aggr_3 = self.aggregation_3(pool_feat_2, pool_coord_2)
        pool_feat_3, pool_coord_3 = voxel_feature_aggr_3, pool_coord_2

        """4th stage"""
        voxel_feature_aggr_4 = self.aggregation_4(pool_feat_3, pool_coord_3)

        voxel_feature_aggr_6 = self.aggregation_6(voxel_feature_aggr_4)

        """pooling opperation along the feature axis"""
        feature_pool_max = F.adaptive_max_pool1d(voxel_feature_aggr_6, 1).view(voxel_feature.shape[0], -1)
        feature_pool_avg = F.adaptive_avg_pool1d(voxel_feature_aggr_6, 1).view(voxel_feature.shape[0], -1)
        feature_pool = torch.cat([feature_pool_max, feature_pool_avg], dim=1)

        """fc layer with dropout operation to produce final predictions"""
        feature_pool_s = self.fc_1(feature_pool)
        feature_pool_s = self.fc_2(feature_pool_s)
        output_s = self.fc_3(feature_pool_s)

        return output_s

########################## END OF DEFINITION ################################

# 1. Load Data (Serban)
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

# 5 Data Loader (pair input vectors and ground truth vectors, also specify batch size)
training_data_loader, validation_data_loader = load_dataloader(batch_size=32)  # to be implemented, comment from here onwards if annoying

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

print('Done')





