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
        # x_loc: 64 * 3 * 1024
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        # x: 64 * 1024 *3 *1
        x = self.conv1(x) # 64*3*1024*1 -> 64*64*1024*1
        x = self.conv2(x) # 64*64*1024*1 -> 64*64*1024*1

        transform = self.trans_net2(x) # 64 * 3 * 3
        x = x.transpose(2, 1) # 64 * 1024 * 64 * 1
        x = x.squeeze(-1) # 64 * 1024 * 64
        x = torch.bmm(x, transform) 
        x = x.unsqueeze(3)
        x = x.transpose(2, 1) # 64 *64 *1024 *1

        x, node_fea, node_off = self.conv3(x, x_loc)
        # node_off: 64 * 3 * 64 node_fea: 64 * 64* 64 *1
        # x = [B, dim, num_node, 1]/[64, 128, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
        x = self.conv4(x) # 64 *128 *1024 *1
        x = self.conv5(x) # 64 *128 *1024 *1

        x, _ = torch.max(x, dim=2, keepdim=False) # 64 * 1024 * 1

        x = x.squeeze(-1) # 64 * 1024

        x = self.bn1(x)

        if node:
            return x, node_fea, node_off
        else:
            return x, node_fea


# Generator
class Pointnet_g_layer1(nn.Module):
    def __init__(self):
        super(Pointnet_g_layer1, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(128, 128)
        self.conv1 = conv_2d(3, 64, 1)
        # SA Node Module
        self.node_fea = nn.Conv1d(64,64,1,stride=16)
        self.node_conv = conv_2d(64, 128, 1)

        self.conv3 = conv_2d(128, 128, 1)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)
        # x_loc: 64 * 3 * 1024
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)

        # x: 64 * 1024 *3 *1
        x = self.conv1(x) # 64*3*1024*1 -> 64*64*1024*1
        node_fea = self.node_fea(x.squeeze(-1))
        x = self.node_conv(x)
        transform = self.trans_net2(x) # 64 * 3 * 3
        x = x.transpose(2, 1) # 64 * 1024 * 64 * 1

        x = x.squeeze(-1) # 64 * 1024 * 64
        x = torch.bmm(x, transform) 
        x = x.unsqueeze(3)
        x = x.transpose(2, 1) # 64 *64 *1024 *1
       
        x =self.conv3(x)
        x = self.conv4(x) # 64 *128 *1024 *1
        x = self.conv5(x) # 64 *128 *1024 *1

        x, _ = torch.max(x, dim=2, keepdim=False) # 64 * 1024 * 1
        x = x.squeeze(-1) # 64 * 1024
        x = self.bn1(x)

        if node:
            return x, node_fea, None
        else:
            return x, node_fea



# Generator
class Pointnet_g_layer2(nn.Module):
    def __init__(self):
        super(Pointnet_g_layer2, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        # SA Node Module
        self.conv2 = adapt_layer_off()  # (64->128)

        self.conv3 = conv_2d(128, 128, 1)
        self.conv4 = conv_2d(128, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)
        # x_loc: 64 * 3 * 1024
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        # x: 64 * 1024 *3 *1
        x = self.conv1(x) # 64*3*1024*1 -> 64*64*1024*1
        transform = self.trans_net2(x) # 64 * 3 * 3
        x = x.transpose(2, 1) # 64 * 1024 * 64 * 1

        x = x.squeeze(-1) # 64 * 1024 * 64
        x = torch.bmm(x, transform) 
        x = x.unsqueeze(3)
        x = x.transpose(2, 1) # 64 *64 *1024 *1

        x, node_fea, node_off = self.conv2(x, x_loc)
        # node_off: 64 * 3 * 64 node_fea: 64 * 64* 64 *1
        # x = [B, dim, num_node, 1]/[64, 128, 1024, 1]; x_loc = [B, xyz, num_node] / [64, 3, 1024]
       
       
        x =self.conv3(x)
        x = self.conv4(x) # 64 *128 *1024 *1
        x = self.conv5(x) # 64 *128 *1024 *1

        x, _ = torch.max(x, dim=2, keepdim=False) # 64 * 1024 * 1
        x = x.squeeze(-1) # 64 * 1024
        x = self.bn1(x)

        if node:
            return x, node_fea, node_off
        else:
            return x, node_fea

class Pointnet_g_layer4(nn.Module):
    def __init__(self):
        super(Pointnet_g_layer4, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = adapt_layer_off()  # (64->128)
        self.conv5 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)
        # x_loc: 64 * 3 * 1024
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        # x: 64 * 1024 *3 *1
        x = self.conv1(x) # 64*3*1024*1 -> 64*64*1024*1
        x = self.conv2(x) # 64*64*1024*1 -> 64*64*1024*1
        transform = self.trans_net2(x) # 64 * 3 * 3
        x = x.transpose(2, 1) # 64 * 1024 * 64 * 1

        x = x.squeeze(-1) # 64 * 1024 * 64
        x = torch.bmm(x, transform) 
        x = x.unsqueeze(3)
        x = x.transpose(2, 1) # 64 *64 *1024 *1

        x = self.conv3(x) # 64 *128 *1024 *1
        x, node_fea, node_off = self.conv4(x, x_loc)
        x = self.conv5(x) # 64 *128 *1024 *1

        x, _ = torch.max(x, dim=2, keepdim=False) # 64 * 1024 * 1

        x = x.squeeze(-1) # 64 * 1024

        x = self.bn1(x)

        if node:
            return x, node_fea, node_off
        else:
            return x, node_fea

class Pointnet_g_layer5(nn.Module):
    def __init__(self):
        super(Pointnet_g_layer5, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 64, 1)
        self.conv5 = adapt_layer_off()  # (64->128)
        self.conv6 = conv_2d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(1024)

    def forward(self, x, node=False):
        x_loc = x.squeeze(-1)
        # x_loc: 64 * 3 * 1024
        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(-1)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        # x: 64 * 1024 *3 *1
        x = self.conv1(x) # 64*3*1024*1 -> 64*64*1024*1
        x = self.conv2(x) # 64*64*1024*1 -> 64*64*1024*1
        transform = self.trans_net2(x) # 64 * 3 * 3
        x = x.transpose(2, 1) # 64 * 1024 * 64 * 1

        x = x.squeeze(-1) # 64 * 1024 * 64
        x = torch.bmm(x, transform) 
        x = x.unsqueeze(3)
        x = x.transpose(2, 1) # 64 *64 *1024 *1

        x = self.conv3(x) # 64 *128 *1024 *1
        x = self.conv4(x) # 64 *128 *1024 *1
        x, node_fea, node_off = self.conv5(x, x_loc)
        x = self.conv6(x)
        x, _ = torch.max(x, dim=2, keepdim=False) # 64 * 1024 * 1

        x = x.squeeze(-1) # 64 * 1024

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

    def forward(self, x, adapt=False, layer_index=2):
        x = self.mlp1(x)  # batchsize*512
        if adapt and layer_index==1:
            mid_feature = x
        x = self.dropout1(x)
        x = self.mlp2(x)  # batchsize*256
        if adapt and layer_index == 2:
            mid_feature = x 
        # mid_feature: bs * 256
        x = self.dropout2(x)
        x = self.mlp3(x)  # batchsize*10
        if adapt and layer_index == 3:
            mid_feature = x
        if adapt == False:
            return x
        else:
            return x, mid_feature



PN_G_Dict = {
    "1" : Pointnet_g_layer1(),
    "2" : Pointnet_g_layer2(),
    "3" : Pointnet_g(),
    "4" : Pointnet_g_layer4(),
    "5" : Pointnet_g_layer5()
}


class Net_MDA(nn.Module):
    def __init__(self, model_name='Pointnet', layer="3"):
        super(Net_MDA, self).__init__()
        if model_name == 'Pointnet':
            self.g = PN_G_Dict[layer]
            self.attention_s = CALayer(64 * 64)
            self.attention_t = CALayer(64 * 64)
            self.c1 = Pointnet_c()
            self.c2 = Pointnet_c()

    def forward(self, x, constant=1, adaptation=False, node_vis=False, mid_feat=False, node_adaptation_s=False,
                node_adaptation_t=False, semantic_adaption=False, sem_ada_layer=2):
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
            y1 = self.c1(x, adapt=semantic_adaption, layer_index=sem_ada_layer)
            y2 = self.c2(x, adapt=semantic_adaption, layer_index=sem_ada_layer)
            return y1, y2
        else:
            y1, sem_feature1 = self.c1(x, adapt=semantic_adaption, layer_index=sem_ada_layer)
            y2, sem_feature2 = self.c2(x, adapt=semantic_adaption, layer_index=sem_ada_layer)
            return y1, y2, sem_feature1, sem_feature2
 