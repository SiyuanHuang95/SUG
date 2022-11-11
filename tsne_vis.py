from sqlite3 import adapt
from model.model_pointnet import Pointnet_cls_old
from model.Model import Net_MDA

import os
import numpy as np
from sklearn.manifold import TSNE


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataloader import include_dataset_full_information, UnifiedPointDG
from utils.visual_utils import visualize_feature_scatter

device = "cuda"
num_class = 10

def load_model(ckpt_pth, source_only=True):
    if source_only:
        model = Pointnet_cls_old(num_class=40)  
    else:
        model = Net_MDA(model_name="Pointnet")
    model = model.to(device=device)

    state_dict = torch.load(ckpt_pth)
    model.load_state_dict(state_dict["model_state"], strict=True)

    return model

def extract_dataset_feature_map(model, dataset_type="scannet", save_path=None, source_only=False):
    mid_features_numpy, label_numpy = None, None
    if save_path is not None:
        feat_full_path = os.path.join(save_path, dataset_type + "_fea.npy")
        label_full_path = os.path.join(save_path, dataset_type + "_lbl.npy")

        if os.path.exists(feat_full_path):
            mid_features_numpy = np.load(feat_full_path)
            label_numpy = np.load(label_full_path)
            print(f"Direct Load: {feat_full_path}")
            return mid_features, label_numpy

    cls_pts, cls_lables = include_dataset_full_information(dataset_type, status="test")

    cls_dataset = UnifiedPointDG(dataset_type=dataset_type, pts=cls_pts, labels=cls_lables, aug=False)
    cls_dataloader = DataLoader(cls_dataset, batch_size=64, shuffle=False, num_workers=2)

    mid_features = []
    labels = []

    for batch_cls in cls_dataloader:
        data, label = batch_cls
        data = data.to(device=device)
        if source_only:
            _, mid_feature = model(data, adapt=True)
        else:
            mid_feature, _ = model(data, mid_feat=True)  # batch_size * num_cls + batch_size * 1024
        # bugs in original model:logits is not from the softmax, but from the mlp
        # also, the dim of logist is 40
        mid_features.extend(mid_feature.cpu().detach().numpy().tolist())
        labels.extend(label.cpu().detach().numpy().tolist())

    mid_features_numpy = np.array(mid_features).reshape([-1, 1024])
    label_numpy = np.array(labels).reshape([-1, 1])

    if save_path is not None:
        feat_full_path = os.path.join(save_path, dataset_type + "_fea.npy")
        label_full_path = os.path.join(save_path, dataset_type + "_lbl.npy")
        
        print(f"Save {dataset_type} mid_features to {save_path}")
        np.save(feat_full_path, mid_features_numpy)
        np.save(label_full_path, label_numpy)

    return mid_features_numpy, label_numpy


def tsne(features, save_path):
    if os.path.exists(save_path):
        tsne_features = np.load(save_path)
        return tsne_features
    tsne = TSNE(n_components=2, init='pca', random_state=0, method='exact', verbose=False)
    tsne_features = tsne.fit_transform(features)

    np.save(save_path, tsne_features)
    return tsne_features

def run(source_dataset, source_only, cpkt_path=None):
    if source_only:
        pre_trained =  "/point_dg/data/output/Source_Baseline/ckpt/Source_exp/Source_Baseline"
        ckpt_folder = os.path.join(pre_trained, source_dataset)
        cpkt_pth = os.path.join(ckpt_folder, "checkpoint_epoch_150.pth")
    else:
        cpkt_pth = cpkt_path

    dataset_list = ["scannet", "shapenet", "modelnet"]
    test_datasets = list(set(dataset_list) - {source_dataset})

    if source_only:
        save_path = os.path.join("/point_dg/workspace/tsne/source/", source_dataset)
    else:
        save_path = os.path.join("/point_dg/workspace/tsne/subdataset/", source_dataset)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = load_model(ckpt_pth=cpkt_pth, source_only=source_only)
    print(f"Load model from {cpkt_pth}")
    for dataset in test_datasets:
        feature_map, labels = extract_dataset_feature_map(model=model, dataset_type=dataset, save_path=save_path, source_only=source_only)
        save_path_tsne = os.path.join(save_path, dataset + "_tsne.npy")
        tsne_featus = tsne(features=feature_map, save_path=save_path_tsne)

        # visualize_feature_scatter(tsne_featus, cls=labels, file_path=os.path.join(save_path, dataset +"tsne.png"))


if __name__ == "__main__":
    root_dir = "/point_dg/workspace"
    modelnet_ckpt = os.path.join(root_dir, "ckpt/0_ra_modelnet_checkpoint_epoch_40.pth")
    # shapenet_cpkt = os.path.join(root_dir, "shapenet_checkpoint_epoch_120.pth")
    run("modelnet", False, cpkt_path=modelnet_ckpt)
    # run("shapenet", False, cpkt_path=shapenet_cpkt)