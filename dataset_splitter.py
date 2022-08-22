import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data.dataloader import UnifiedPointDG
from model.model_pointnet import Pointnet_cls
from utils.train_files_spliter import include_dataset_full_information, include_dataset_one_class, data_root, num_class, dataset_list
from utils.visual_utils import visualize_feature_scatter
from utils.common_utils import check_numpy_to_torch

import os
import shutil
import torch
import torch.nn.functional as F

import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.special import kl_div
from scipy.cluster.hierarchy import fclusterdata


def split_dataset_clusters(config):
    pre_trained_ = config["pre_trained_"]
    dataset_type = config["dataset_type"]
    cluster_num = config["cluster_num"]
    model = config["model"]

    spliter_save_path = os.path.join(data_root, dataset_type, "spliter")
    if os.path.exists(spliter_save_path):
        shutil.rmtree(spliter_save_path, ignore_errors=True)
        print(f"Remove the old folder")

    mid_features_numpy, logits_numpy = extract_feature_map_class(pre_trained_, model=model, dataset_type=dataset_type)
    # cluster feature maps within class
    # cluster prediction uncertainity cross class
    raw_pts, raw_labels = init_dataloader(dataset_type=dataset_type, get_raw_data=True)
    probs_numpy = F.softmax(torch.from_numpy(logits_numpy), dim=1).numpy()
    cluster_labels_entropy, entropys = entropy_clustering(probs_numpy, cluster_num=cluster_num)
    for i in range(num_class):
        index_cls = raw_labels == i
        entropys_cls = entropys[index_cls]
        cluster_cls = kmeans_clustering(mid_features_numpy[index_cls], dataset_type=dataset_type,cluster_num=cluster_num, cls=i)
        if cluster_cls is None:
        # adjust the cluster number
        # ref:https://www.jianshu.com/p/0e74342b9b0b
        # https://zhuanlan.zhihu.com/p/98918878
            continue
        spliter_cls_data(pts_all=raw_pts[index_cls], cluster_labels=cluster_cls, cls=i, method="kmeans", dataset_type=dataset_type, cls_entropy=entropys_cls)
    
    spliter_cls_data(pts_all=raw_pts, cluster_labels=cluster_labels_entropy, cls=-1, method="entropy", dataset_type=dataset_type, raw_labels=raw_labels, cls_entropy=entropys)


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
            # bugs in original model:logits is not from the softmax, but from the mlp
            # also, the dim of logist is 40
            mid_features.extend(mid_feature.cpu().detach().numpy().tolist())
            logits_list.extend(logits[:, :num_class].cpu().detach().numpy().tolist())

        mid_features_numpy = np.array(mid_features).reshape([-1, 1024])
        logits_numpy = np.array(logits_list).reshape([-1, num_class])
    
        if save_path is not None:
            print(f"Save {dataset_type} mid_features to {save_path}")
            np.save(save_path, mid_features_numpy)
    return mid_features_numpy, logits_numpy
    

def kmeans_clustering(feature_maps, dataset_type, cluster_num=4, cls=-1):
    spliter_save_path = os.path.join(data_root, dataset_type, "spliter")
    if not os.path.exists(spliter_save_path):
        os.makedirs(spliter_save_path)
    
    fig_path = os.path.join(spliter_save_path, "kmeans_" + str(cls)+"_clsuter.png")
    if os.path.exists(fig_path) and False:
        return None

    feature_maps_ = reduction_tsne(feature_maps, num_comps=2)
    kmeans_model = KMeans(n_clusters=4, verbose=False).fit(feature_maps_)

    # should reorder the cluster-idx based on the cluster_centers_ distance
    labels_, centers_ = kmeans_cluster_idx_update(kmeans_model.labels_, kmeans_model.cluster_centers_)
    visualize_feature_scatter(feature_maps_, labels_=labels_, cluster_centers=centers_, cls=cls, file_path=fig_path)
    return labels_


def kmeans_cluster_idx_update(labels_, cluster_centers_):
    new_labels = np.zeros_like(labels_)
    new_cluster_centers = np.zeros_like(cluster_centers_)
    anchor_center = cluster_centers_[0]
    distances = [np.linalg.norm(anchor_center - cluster_center) for cluster_center in cluster_centers_]
    indices = np.argsort(np.array(distances)).squeeze()
    for i in range(len(cluster_centers_)):
        cluster_idx = labels_ == i
        new_labels[cluster_idx] = indices.tolist().index(i)
        new_cluster_centers[i] = cluster_centers_[indices[i]]
    return new_labels, new_cluster_centers
        

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
    uncertainties = cal_probs2entropy(probs)
    uncertainties = uncertainties.cpu().numpy()
    indices = np.argsort(uncertainties)
    dataset_size = probs.shape[0]
    cluster_labels = np.ones(dataset_size)

    cluster_with_hist = True
    # cluster the entropy with fixed number, could be overlapped
    # should cluster with the hist
    if not cluster_with_hist:
        cluster_size = int(dataset_size // cluster_num)
        for i in range(cluster_num):
            pos=np.where( (indices>=cluster_size * i ) & (indices<cluster_size * (i+1))) 
            cluster_labels[pos] = i
    else:
        value_edges = np.histogram(uncertainties, bins=cluster_num)[1]
        for i in range(cluster_num):
            pos=np.where( (uncertainties>= value_edges[i] ) & (uncertainties<  value_edges[i+1]))
            cluster_labels[pos] = i
    return cluster_labels, uncertainties


def cal_probs2entropy(probs):

    EPS = 1e-30
    probs = check_numpy_to_torch(probs)[0] 
    log_probs = torch.log(probs+ EPS)
    uncertainties = -(probs*log_probs).sum(1)  # data_size * 1

    return uncertainties


def kl_divergence_distance(x, y):
    return kl_div(x,y) * 0.5 + kl_div(y,x) * 0.5
    

def kl_clustering(preds, cluster_num=4):
    return fclusterdata(preds, metric=kl_divergence_distance, criterion='maxclust', t=cluster_num)


def spliter_cls_data(pts_all, cluster_labels, cls, method:str, dataset_type:str, save_path=None, raw_labels=None, cls_entropy=None):
    assert pts_all.shape[0] == cluster_labels.shape[0], "The cluster labels and Pts shape mismatch"
    if cls == -1 and raw_labels is None:
        raise RuntimeError("When process all cls, label infos need to be added")
    for k in range(len(set(cluster_labels))):
        cluster_index = cluster_labels == k
        cluster_pts = pts_all[cluster_index, :]

        cluster_entropy = None
        if cls_entropy is not None:
            cluster_entropy = np.median(cls_entropy[cluster_index]).tolist()

        if cls == -1:
            cluster_lbl = raw_labels[cluster_index]

        if save_path is None:
            save_path = os.path.join(data_root, dataset_type, "spliter")

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if cls_entropy is None:
            npy_file = method + "_" + str(cls) + "_" + str(k) + "_" + str(cluster_pts.shape[0]) + ".npy"
        else:
            npy_file = method + "_" + str(cls) + "_" + str(k) + "_" + str(cluster_pts.shape[0]) + "_entropy_" + str(cluster_entropy) + ".npy"
        # file name: method + class_idx - cluster_idx - num_pts
        npy_save_path = os.path.join(save_path, npy_file)
        np.save(npy_save_path, cluster_pts)
        print(f"Save Class {cls} Cluster {k} with number {cluster_pts.shape[0]} to {npy_save_path}")

        if cls == -1:
            npy_file = method + "_" + str(cls) + "_" + str(k) + "_" + str(cluster_pts.shape[0]) + "_labels.npy"
            npy_save_path = os.path.join(save_path, npy_file)
            np.save(npy_save_path, cluster_lbl)

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
    args.pre_trained =  "/point_dg/data/output/Source_Baseline/ckpt/Source_exp/Source_Baseline"
    args.process_all = True
    
    if args.process_all:
        process_list = []
        for dataset_type in dataset_list:
            # when --process_all set, the --pred_trained is the folder contains all ckpt
            ckpt_folder = os.path.join(args.pre_trained, dataset_type)
            cpkt_pth = os.path.join(ckpt_folder, "checkpoint_epoch_150.pth")
            # cpkt only loads the 150-th epoch,not the best one
            if not os.path.join(cpkt_pth):
                raise FileNotFoundError("The Pre-Trained Ckpt not found")
            process_list.append({"pre_trained_": cpkt_pth, "dataset_type":dataset_type, "cluster_num":4, "model":None})
        for procss_config in process_list:
            split_dataset_clusters(procss_config)
            # planned to use multi-process pool here, cuda not allowd...not fixed yet

    else:
        config_ = {"pre_trained_": args.pre_trained, "dataset_type":args.dataset, "cluster_num":4, "model":None}
        split_dataset_clusters(config_)
