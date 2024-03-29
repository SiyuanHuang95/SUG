import torch
import torch.nn as nn
import model.point_utils as point_utils

import torch.nn.functional as F
from torch.autograd import Variable,Function

class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False)
            )
        elif activation == 'tanh':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.Tanh()
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu', bias=False):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=False)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                # nn.BatchNorm1d(out_ch),
                nn.LayerNorm(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    def __init__(self, in_ch, K=3):
        super(transform_net, self).__init__()
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1)
        self.conv2d2 = conv_2d(64, 128, 1)
        self.conv2d3 = conv_2d(128, 1024, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = nn.Linear(256, K * K)

    def forward(self, x, DGCNN_Flag=False):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        if DGCNN_Flag:
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device='cuda')
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


class adapt_layer_off(nn.Module):
    def __init__(self, num_node=64, offset_dim=3, trans_dim_in=64, trans_dim_out=64, fc_dim=64):
        super(adapt_layer_off, self).__init__()
        self.num_node = num_node
        self.offset_dim = offset_dim
        self.trans = conv_2d(trans_dim_in, trans_dim_out, 1)
        self.pred_offset = nn.Sequential(
            nn.Conv2d(trans_dim_out, offset_dim, kernel_size=1, bias=False),
            nn.Tanh())
        self.residual = conv_2d(trans_dim_in, fc_dim, 1)

    def forward(self, input_fea, input_loc):
        # Initialize node
        fpoint_idx = point_utils.farthest_point_sample(input_loc, self.num_node)  # (B, num_node)
        fpoint_loc = point_utils.index_points(input_loc, fpoint_idx)  # (B, 3, num_node)
        fpoint_fea = point_utils.index_points(input_fea, fpoint_idx)  # (B, C, num_node)
        group_idx = point_utils.query_ball_point(0.3, 64, input_loc, fpoint_loc)  # (B, num_node, 64)
        group_fea = point_utils.index_points(input_fea, group_idx)  # (B, C, num_node, 64)
        group_fea = group_fea - fpoint_fea.unsqueeze(3).expand(-1, -1, -1, self.num_node)

        # Learn node offset
        seman_trans = self.pred_offset(group_fea)  # (B, 3, num_node, 64)
        group_loc = point_utils.index_points(input_loc, group_idx)  # (B, 3, num_node, 64)
        group_loc = group_loc - fpoint_loc.unsqueeze(3).expand(-1, -1, -1, self.num_node)
        node_offset = (seman_trans * group_loc).mean(dim=-1)

        # Update node and get node feature
        node_loc = fpoint_loc + node_offset.squeeze(-1)  # (B,3,num_node)
        group_idx = point_utils.query_ball_point(None, 64, input_loc, node_loc)
        residual_fea = self.residual(input_fea)
        group_fea = point_utils.index_points(residual_fea, group_idx)
        node_fea, _ = torch.max(group_fea, dim=-1, keepdim=True)

        # Interpolated back to original point
        output_fea = point_utils.upsample_inter(input_loc, node_loc, input_fea, node_fea, k=3).unsqueeze(3)

        return output_fea, node_fea, node_offset


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss, -alpha * (1-yi)** gamma * ce_loss(xi,yi)
        :param alpha:   
        :param gamma:   
        :param num_classes:    
        :param size_average:   
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            # use same weight for all-cls
            alpha = [ 1 / num_classes] * num_classes
            self.alpha = torch.Tensor(alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss
        :param preds:   size:[B,N,C] or [B,C]
        :param labels:  size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  
        # torch.pow((1-preds_softmax), self.gamma) -> (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx
    

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # Run on cpu or gpu
    idx_base = torch.arange(0, batch_size, device='cuda:0').view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature