import torch
import torch.nn as nn
from model.PTran_utils import PointNetFeaturePropagation, PointNetSetAbstraction
from model.Ptran_transformer import TransformerBlock

import easydict


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        

class Backbone(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        # fc1: bs * 1024 * 3 -> bs * 1024 * 32
        self.transformer1 = TransformerBlock(32, cfg.model.transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, cfg.model.transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        # x : batch_size * 3 * 1024 * 1
        x_ = x.squeeze(-1)
        x_ = x_.permute(0, 2, 1) # bs * 1024 * 3
        xyz = x_[..., :3]
        x1 = self.fc1(x_) # x1:bs * 1024 * 32
        points = self.transformer1(xyz, x1)[0] # bs * 1024 * 32

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats
        # points: bs * 4 * 512
        # xyz_fea: [
        # 0: bs * 1024 * 3     bs * 1024 * 32
        # 1: bs * 256 * 3      bs * 256 * 64
        # 2: bs * 64 * 3       bs * 64 * 128
        # 3: bs * 16 * 3       bs * 16 * 256
        # 4: bs * 4 * 3        bs * 4 * 512
        # ]


class PointTransformerCls(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg  = easydict.EasyDict()
            cfg["model"] = {"nneighbor": 16, "nblocks": 4, "transformer_dim": 512}
            cfg["num_point"] = 1024
            cfg["num_class"] = 10
            cfg["input_dim"] = 3
        self.backbone = Backbone(cfg)
        npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        return res

######
# https://github.com/qq456cvb/Point-Transformers/issues/13
# batch_size: 32
# epoch: 500
# learning_rate: 0.0005
# gpu: 0
# num_point: 1024
# optimizer: Adam
# weight_decay: 0.0001
# normal: true
# model:
#   nneighbor: 16
#   nblocks: 4
#   transformer_dim: 512
#   name: Hengshuang
######