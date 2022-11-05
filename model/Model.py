from sqlite3 import adapt
from model.model_utils import *
from model.pointnet2_utils import PointNetSetAbstraction
from model.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG

import model.pointnet2.pytorch_utils as pt_utils
from model.Ptran_transformer import TransformerBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

# Channel Attention
class  CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm1d(4096)

    def forward(self, x):
        y = self.conv_du(x)
        y = x * y + x
        y = y.view(y.shape[0], -1)
        y = self.bn(y)

        return y

# Grad Reversal
class GradReverse(torch.autograd.Function):
    def __init__(self, lambd):
        self.lambd = lambd

    @staticmethod
    def forward(x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd).forward(x)

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
        self.node_fea_adapt = adapt_layer_off()
        self.conv1d = nn.Conv1d(128, 64, 1)

        self.dim_redu = nn.MaxPool1d(3, stride=16) # B * 64 *1024 -> B * 64 * 64
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

        x_, node_fea, node_off = self.node_fea_adapt(x2.view(batch_size, 64, 1024, 1), x_loc)
        x2 = self.conv1d(x_.squeeze(-1))

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

        if node:
            return x, node_fea, None
        else:
            return x, node_fea

class Pointnet2_g(nn.Module):
    def __init__(self, normal_channel=False):
        super(Pointnet2_g, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.num_class = 10
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        # self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64], group_all=False)
        # self.adapt_layer_off = adapt_layer_off() # 64 -> 128 / input: [64, 3, 64, 64]
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)

        self.channel_redu = nn.Conv2d(512, 64, 1)
        self.dim_redu = nn.MaxPool1d(3, stride=8) # B * 64 * 512 -> B * 64 * 64
        
    def forward(self, xyz, node=False):
        xyz = xyz.squeeze(-1) # 64 * 3 * 1024
        B = xyz.shape[0]
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        # x_loc = xyz.squeeze(-1)
        l1_xyz, l1_points, node_fea = self.sa1(xyz, norm, adapt=True)
        # l1_xyz: 64 * 3 * 512  l1_points: 64 * 128 * 512
        # node_fea: 64 * 64 * 512
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # 64 * 3 * 128 // 64 * 256 * 128
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # 64 * 3 * 1 // 64 * 1024 * 1
        x = l3_points.view(B, 1024)
        
        node_fea = self.dim_redu(node_fea).view(B, 64, 64, 1)

        if node:
            return x, node_fea, None
        else:
            return x, node_fea

NPOINTS = [512, 128, None]
RADIUS = [[0.2],[ 0.4], [None]]
NSAMPLE = [[32], [64], [None]]
MLPS = [[[64, 64, 128]], [[128, 128, 256]],
        [[256, 512, 1024]]]
FP_MLPS = [[256, 256], [1024, 1024], [1024, 1024]]

DP_RATIO = 0.4

class Ponintnet2MSG_g(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        self.SA_modules = nn.ModuleList()
        channel_in = input_channels

        skip_channel_list = [input_channels]
        for k in range(NPOINTS.__len__()):
            mlps = MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=NPOINTS[k],
                    radii=RADIUS[k],
                    nsamples=NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=True
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(FP_MLPS.__len__()):
            pre_channel = FP_MLPS[k + 1][-1] if k + 1 < len(FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + FP_MLPS[k])
            )


    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, node=False):
        xyz, features = self._break_up_pc(pointcloud.squeeze(-1))

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz) # [[63 * 3 * 3]]  
            l_features.append(li_features)  # 64 * 1021 * 3

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        # pred_cls = self.cls_layer(l_features[0]).transpose(1, 2).contiguous()  # (B, N, 1)
        return l_features[0]

# Generator
class Pointnet_g(nn.Module):
    def __init__(self):
        super(Pointnet_g, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        # SA Node Module
        self.conv3 = adapt_layer_off()  # (64->128)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)

        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x) # 64 * 3 * 3
        x = x.transpose(2, 1) # 64 * 1024 * 64 * 1

        x = x.squeeze(-1) # 64 * 1024 * 64
        x = torch.bmm(x, transform) 
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        # x: 64*64*1024 *1  x_loc: 64*3*1024
        x, node_fea, node_off = self.conv3(x, x_loc)
        # node_off: 64 * 3 * 64 node_fea: 64 * 64* 64 *1
        # x = [B, dim, num_node, 1]/[64, 128, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
        x = self.conv4(x)
        x = self.conv5(x)

        x, _ = torch.max(x, dim=2, keepdim=False)

        x = x.squeeze(-1)

        x = self.bn1(x)

        if node:
            return x, node_fea, node_off
        else:
            return x, node_fea


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class PTran_g(nn.Module):
    def __init__(self):
        super(PTran_g, self).__init__()
        npoints, nblocks, nneighbor, n_c, d_points = 1024, 4, 16, 10, 3
        transformer_dim = 512  # TODO to see whether can be 1024
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):

        pass

# Classifier
class Pointnet_c(nn.Module):
    def __init__(self, num_class=10, dgcnn_flag=False):
        super(Pointnet_c, self).__init__()
        # classifier in PointDAN
        # self.fc = nn.Linear(1024, num_class)
        
        if dgcnn_flag:
            activate = 'leakyrelu' 
            bias = True
        else:
            activate = 'relu'
            bias = False

        # classifier in PointNet
        # bn On or Off?
        self.mlp1 = fc_layer(1024, 512, bn=True, activation=activate, bias=bias)
        self.dropout1 = nn.Dropout2d(p=0.4)
        # should check 0.7 -> 0.4?
        self.mlp2 = fc_layer(512, 256, bn=True, activation=activate, bias=True)
        self.dropout2 = nn.Dropout2d(p=0.4)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x, adapt=False):
        x = self.mlp1(x)  # batchsize*512
        x = self.dropout1(x)
        x = self.mlp2(x)  # batchsize*256
        if adapt == True:
            mid_feature = x 
        # mid_feature: bs * 256
        x = self.dropout2(x)
        x = self.mlp3(x)  # batchsize*10
        if adapt == False:
            return x
        else:
            return x, mid_feature


class Net_MDA(nn.Module):
    def __init__(self, model_name='Pointnet'):
        super(Net_MDA, self).__init__()
        self.dgcnn_flag = False
        if model_name == 'Pointnet':
            self.g = Pointnet_g()
        elif model_name == "Pointnet2":
            self.g = Pointnet2_g()
        elif model_name == "DGCNN":
            self.g = DGCNN()
            self.dgcnn_flag = True
        else:
            raise NotImplementedError("Unsupported model name")

        self.attention_s = CALayer(64 * 64)
        self.attention_t = CALayer(64 * 64)
        self.c1 = Pointnet_c(dgcnn_flag=self.dgcnn_flag)
        self.c2 = Pointnet_c(dgcnn_flag=self.dgcnn_flag)

    def forward(self, x, constant=1, adaptation=False, node_vis=False, mid_feat=False, node_adaptation_s=False,
                node_adaptation_t=False, semantic_adaption=False):
        x, feat_ori, node_idx = self.g(x, node=True)
        batch_size = feat_ori.size(0)

        # sa node visualization
        if node_vis:
            return node_idx

        # collect mid-level feat
        if mid_feat:
            return x, feat_ori
            # x: 64 * 1024  feat_ori: 64 * 64 * 64

        if node_adaptation_s:
            # source domain sa node feat
            feat_node = feat_ori.contiguous().view(batch_size, -1)
            feat_node_s = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_s
        elif node_adaptation_t:
            # target domain sa node feat
            feat_node = feat_ori.contiguous().view(batch_size, -1)
            feat_node_t = self.attention_t(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_t

        if adaptation:
            x = grad_reverse(x, constant)

        if not semantic_adaption:
            y1 = self.c1(x, adapt=semantic_adaption)
            y2 = self.c2(x, adapt=semantic_adaption)
            return y1, y2
        else:
            y1, sem_feature1 = self.c1(x, adapt=semantic_adaption)
            y2, sem_feature2 = self.c2(x, adapt=semantic_adaption)
            return y1, y2, sem_feature1, sem_feature2
 