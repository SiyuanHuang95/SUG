from sqlite3 import adapt
from model.model_utils import *
import pdb
import os
import torch.nn.functional as F


# Channel Attention
class  CALayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
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


# Classifier
class Pointnet_c(nn.Module):
    def __init__(self, num_class=10):
        super(Pointnet_c, self).__init__()
        # classifier in PointDAN
        # self.fc = nn.Linear(1024, num_class)
        
        # classifier in PointNet
        self.mlp1 = fc_layer(1024, 512, bn=False)
        self.dropout1 = nn.Dropout2d(p=0.7)
        self.mlp2 = fc_layer(512, 256, bn=False)
        self.dropout2 = nn.Dropout2d(p=0.7)
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
        if model_name == 'Pointnet':
            self.g = Pointnet_g()
            self.attention_s = CALayer(64 * 64)
            self.attention_t = CALayer(64 * 64)
            self.c1 = Pointnet_c()
            self.c2 = Pointnet_c()

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
            feat_node = feat_ori.view(batch_size, -1)
            feat_node_s = self.attention_s(feat_node.unsqueeze(2).unsqueeze(3))
            return feat_node_s
        elif node_adaptation_t:
            # target domain sa node feat
            feat_node = feat_ori.view(batch_size, -1)
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
 