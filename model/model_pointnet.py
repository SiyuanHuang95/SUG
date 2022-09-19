from model.model_utils import *
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
