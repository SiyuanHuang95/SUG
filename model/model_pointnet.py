from model.Model import Pointnet_c
from model.model_utils_old import *
from model.pointnet2_utils import PointNetSetAbstraction

class Pointnet_cls(nn.Module):
    def __init__(self, num_class=10):
        super(Pointnet_cls, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        self.mlp1 = fc_layer(1024, 512)
        self.dropout1 = nn.Dropout2d(p=0.7)
        self.mlp2 = fc_layer(512, 256)
        self.dropout2 = nn.Dropout2d(p=0.7)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x, adapt=False):
        batch_size = x.size(0)
        point_num = x.size(2)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.squeeze()  # batchsize*1024
        if adapt == True:
            mid_feature = x
        x = self.mlp1(x)  # batchsize*512
        x = self.dropout1(x)
        x = self.mlp2(x)  # batchsize*256
        x = self.dropout2(x)
        x = self.mlp3(x)  # batchsize*10
        if adapt == False:
            return x
        else:
            return x, mid_feature


class Pointnet2_cls(nn.Module):
    def __init__(self, num_class=10, normal_channel=False):
        super(Pointnet2_cls, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_class)

    def forward(self, xyz):
        xyz = xyz.squeeze(-1) # 64 * 3 * 1024
        B = xyz.shape[0]
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        return x

K = 20
class DGCNN(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()
        self.k = K

        self.input_transform_net = transform_net(6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(num_f_prev, 512, kernel_size=1, bias=False)
        # self.node_fea_adapt = adapt_layer_off()
        # self.conv1d = nn.Conv1d(128, 64, 1)

        # self.dim_redu = nn.MaxPool1d(3, stride=16) # B * 64 *1024 -> B * 64 * 64
        self.classifier = Pointnet_c(dgcnn_flag=True)

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)
        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        # x0 = get_graph_feature(x, self.args, k=self.k)  # x0: [b, 6, 1024, 20]
        # align to a canonical space (e.g., apply rotation such that all inputs will have the same rotation)
        # transformd_x0 = self.input_transform_net(x0)  # transformd_x0: [3, 3]
        # x = torch.matmul(transformd_x0, x)

        # returns a tensor of (batch_size, 6, #points, #neighboors)
        # interpretation: each point is represented by 20 NN, each of size 6
        x = get_graph_feature(x, k=self.k)  # x: [b, 6, 1024, 20]
        # process point and inflate it from 6 to e.g., 64
        x = self.conv1(x)  # x: [b, 64, 1024, 20]
        # per each feature (from e.g., 64) take the max value from the representative vectors
        # Conceptually this means taking the neighbor that gives the highest feature value.
        # returns a tensor of size e.g., (batch_size, 64, #points)
        x1 = x.max(dim=-1, keepdim=False)[0] # 64 * 64 * 1024

        x = get_graph_feature(x1, k=self.k)  # 64 * 64 * 1024 *20
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0] # 64 * 64 * 1024

        # x_, node_fea, node_off = self.node_fea_adapt(x2.view(batch_size, 64, 1024, 1), x_loc)
        # x2 = self.conv1d(x_.squeeze(-1))

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x5 = self.conv5(x_cat)  # [b, 1024, 1024]
        x5 = F.leaky_relu(self.bn5(x5), negative_slope=0.2)
        x1 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x5, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = self.classifier(x)

        return x


class Pointnet_cls_old(nn.Module):
    def __init__(self, num_class=40):
        super(Pointnet_cls_old, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        self.mlp1 = fc_layer(1024, 512)
        self.dropout1 = nn.Dropout2d(p=0.7)
        self.mlp2 = fc_layer(512, 256)
        self.dropout2 = nn.Dropout2d(p=0.7)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x, adapt=False):
        batch_size = x.size(0)
        point_num = x.size(2)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1)
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.squeeze()  # batchsize*1024
        if adapt == True:
            mid_feature = x
        x = self.mlp1(x)  # batchsize*512
        x = self.dropout1(x)
        x = self.mlp2(x)  # batchsize*256
        x = self.dropout2(x)
        x = self.mlp3(x)  # batchsize*10
        if adapt == False:
            return x
        else:
            return x, mid_feature