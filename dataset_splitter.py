import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data.dataloader import UnifiedPointDG
from model.model_pointnet import Pointnet_cls
from utils.train_files_spliter import include_dataset_full_information, include_dataset_one_class, data_root, num_class, dataset_list
from utils.visual_utils import visualize_feature_scatter

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.special import kl_div
from scipy.cluster.hierarchy import fclusterdata

from multiprocessing import Pool, pool


def split_dataset_clusters(config):
    pre_trained_ = config["pre_trained_"]
    dataset_type = config["dataset_type"]
    cluster_num = config["cluster_num"]
    model = config["model"]
    mid_features_numpy, logits_numpy = extract_feature_map_class(pre_trained_, model=model)
    # cluster feature maps within class
    # cluster prediction uncertainity cross class
    raw_pts, raw_labels = init_dataloader(dataset_type=dataset_type, get_raw_data=True)
    for i in range(num_class):
        index_cls = raw_labels == i
        cluster_cls = kmeans_clustering(mid_features_numpy[index_cls], cluster_num=cluster_num, cls=i)
        spliter_cls_data(pts_all=raw_pts, cluster_labels=cluster_cls, cls=i, method="kmeans")
    
    cluster_labels = entropy_clustering(logits_numpy, cluster_num=cluster_num)
    spliter_cls_data(pts_all=raw_pts, cluster_labels=cluster_labels, cls=-1, method="entropy")



def extract_feature_map_class(pre_trained_, save_path=None, dataset_type="modelnet", cls=-1, model=None):
    if save_path is not None and os.path.exists(save_path) and False:
        mid_features_numpy = np.load(save_path)
        print(f"Direct load mid-features from {save_path}")
        cls_pts, cls_lables = include_dataset_one_class(dataset_type, status="train", cls=cls)
    else:
        device = 'cuda'
        cls_dataloader = init_dataloader(dataset_type=dataset_type, cls=cls)
        model = init_model(pre_trained_=pre_trained_, model=model)

        mid_features = []
        logits_list = []
        for batch_cls in cls_dataloader:
            data, label = batch_cls
            data = data.to(device=device)
            logits, mid_feature = model(data, adapt=True)  # batch_size * num_cls + batch_size * 1024

            mid_features.extend(mid_feature.cpu().detach().numpy().tolist())
            logits_list.extend(logits.cpu().detach().numpy().tolist())

        mid_features_numpy = np.array(mid_features).reshape([-1, 1024])
        logits_numpy = np.array(logits_list).reshape([-1, num_class])
    
        if save_path is not None:
            print(f"Save {dataset_type} mid_features to {save_path}")
            np.save(save_path, mid_features_numpy)
    return mid_features_numpy, logits_numpy
    

def kmeans_clustering(feature_maps, cluster_num=4, cls=-1):
    feature_maps_ = reduction_tsne(feature_maps, num_comps=2)
    kmeans_model = KMeans(n_clusters=4, verbose=False).fit(feature_maps_)

    spliter_save_path = os.path.join(data_root, dataset_type, "spliter")
    if not os.path.exists(spliter_save_path):
        os.makedirs(spliter_save_path)
    fig_path = os.path.join(spliter_save_path, str(cls)+"_clsuter.png")
    visualize_feature_scatter(feature_maps_, labels_=kmeans_model.labels_,
                              cluster_centers=kmeans_model.cluster_centers_, cls=cls, file_path=fig_path)
    return kmeans_model.labels_


def reduction_tsne(features, num_comps=3, visualize=False):
    tsne = TSNE(n_components=num_comps, init='pca', random_state=0, method='exact', verbose=False)

    tsne_features = tsne.fit_transform(features)
    if visualize:
        visualize_feature_scatter(tsne_features)
    return tsne_features


def entropy_clustering(probs, cluster_num=4):
    """
        Ref: https://github.com/ej0cl6/deep-active-learning/blob/master/query_strategies/entropy_sampling.py
    """
    EPS = 1e-30
    log_probs = torch.log(probs + EPS)
    uncertainties = (probs*log_probs).sum(1)  # data_size * 1
    indices = uncertainties.sort()[1]

    dataset_size = probs.shape[0]
    cluster_labels = np.ones(dataset_size)
    cluster_size = int(dataset_size // cluster_num)
    for i in range(cluster_num):
        cluster_labels[indices < (cluster_size * i+1)] = i

    return cluster_labels


def kl_divergence_distance(x, y):
    return kl_div(x,y) * 0.5 + kl_div(y,x) * 0.5
    

def kl_clustering(preds, cluster_num=4):
    return fclusterdata(preds, metric=kl_divergence_distance, criterion='maxclust', t=cluster_num)


def spliter_cls_data(pts_all, cluster_labels, cls, method:str, save_path=None):
    assert pts_all.shape[0] == cluster_labels.shape[0], "The cluster labels and Pts shape mismatch"
    for k in range(len(set(cluster_labels))):
        cluster_index = cluster_labels == k
        cluster_pts = pts_all[cluster_index, :]

        if save_path is None:
            save_path = os.path.join(data_root, dataset_type, "spliter")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        npy_file = method + str(cls) + "_" + str(k) + "_" + str(cluster_pts.shape[0]) + ".npy"
        # file name: method + class_idx - cluster_idx - num_pts
        npy_save_path = os.path.join(save_path, npy_file)

        np.save(npy_save_path, cluster_pts)
        print(f"Save Class {cls} Cluster {k} with number {cluster_pts.shape[0]} to {npy_save_path}")


def init_model(pre_trained_, model=None):
    device = 'cuda'
    if model is None:
        model = Pointnet_cls()
        model = model.to(device=device)
    state_dict = torch.load(pre_trained_)
    model.load_state_dict(state_dict["model_state"], strict=True)
    return model


def init_dataloader(dataset_type="modelnet", cls=-1, get_raw_data=False):
    if cls != -1:
        cls_pts, cls_lables = include_dataset_one_class(dataset_type, status="train", cls=cls)
        assert len(set(cls_lables.tolist())) == 1, "The class in labels is more than 1!"
    else:
        cls_pts, cls_lables = include_dataset_full_information(dataset_type=dataset_type, status='train')
    
    if get_raw_data:
        return cls_pts, cls_lables

    cls_dataset = UnifiedPointDG(dataset_type=dataset_type, pts=cls_pts, labels=cls_lables, aug=False)
    print(f"For cls {cls}, the sample num is {len(cls_dataset)}")
    cls_dataloader = DataLoader(cls_dataset, batch_size=64, shuffle=False, num_workers=2)
    return cls_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arg parser')
    parser.add_argument('--pre_trained', type=str, default=None, help='pretrained_model')
    parser.add_argument('--dataset', type=str, default="modelnet")
    parser.add_argument('--process_all', action="store_true", default=False, help="Whether to process all")
    args = parser.parse_args()
    # pre_trained = "/home/siyuan/4-data/PointDA_data/output/ckpt/source_train/modelnet/checkpoint_epoch_150.pth"
    if args.process_all:
        pool = Pool(processes=len(dataset_list))
        process_list = []
        for dataset_type in dataset_list:
            # when --process_all set, the --pred_trained is the folder contains all ckpt
            ckpt_folder = os.path.join(args.pre_trained, dataset_type)
            cpkt_pth = os.path.join(ckpt_folder, "checkpoint_epoch_150.pth")
            if not os.path.join(cpkt_pth):
                raise FileNotFoundError("The Pre-Trained Ckpt not found")
            process_list.append({"pre_trained_": cpkt_pth, "dataset_type":dataset_type, "cluster_num":4, "model":None})
        pool.map(split_dataset_clusters, process_list)

    else:
        config_ = {"pre_trained_": args.pre_trained, "dataset_type":args.dataset, "cluster_num":4, "model":None}
        split_dataset_clusters(config_)
