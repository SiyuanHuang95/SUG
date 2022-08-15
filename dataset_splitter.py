import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data.dataloader import UnifiedPointDG
from model.model_pointnet import Pointnet_cls
from utils.train_files_spliter import include_dataset_one_class, data_root, num_class, dataset_list
from utils.visual_utils import visualize_feature_scatter
from utils.common_utils import Timer, time_str

import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def extract_feature_map_class(pre_trained_, save_path=None, dataset_type="modelnet", cls=0, model=None):
    if save_path is not None and os.path.exists(save_path) and False:
        mid_features_numpy = np.load(save_path)
        print(f"Direct load mid-features from {save_path}")
        cls_pts, cls_lables = include_dataset_one_class(dataset_type, status="train", cls=cls)
    else:
        cls_pts, cls_lables = include_dataset_one_class(dataset_type, status="train", cls=cls)
        assert len(set(cls_lables.tolist())) == 1, "The class in labels is more than 1!"
        cls_dataset = UnifiedPointDG(dataset_type=dataset_type, pts=cls_pts, labels=cls_lables, aug=False)
        print(f"For cls {cls}, the sample num is {len(cls_dataset)}")
        cls_dataloader = DataLoader(cls_dataset, batch_size=64, shuffle=False, num_workers=2)
        device = 'cuda'
        if model is None:
            model = Pointnet_cls()
            model = model.to(device=device)
        state_dict = torch.load(pre_trained_)
        model.load_state_dict(state_dict["model_state"], strict=True)

        mid_features = []
        for batch_cls in cls_dataloader:
            data, label = batch_cls
            data = data.to(device=device)
            logits, mid_feature = model(data, adapt=True)  # batch_size * 1024

            mid_features.extend(mid_feature.cpu().detach().numpy().tolist())

        mid_features_numpy = np.array(mid_features).reshape([-1, 1024])
        if save_path is not None:
            print(f"Save {dataset_type} mid_features to {save_path}")
            np.save(save_path, mid_features_numpy)

    mid_features_numpy = reduction_tsne(mid_features_numpy, num_comps=2)
    kmeans_model = KMeans(n_clusters=4, verbose=False).fit(mid_features_numpy)

    spliter_save_path = os.path.join(data_root, dataset_type, "spliter")
    if not os.path.exists(spliter_save_path):
        os.makedirs(spliter_save_path)
    fig_path = os.path.join(spliter_save_path, str(cls)+"_clsuter.png")
    visualize_feature_scatter(mid_features_numpy, labels_=kmeans_model.labels_,
                              cluster_centers=kmeans_model.cluster_centers_, cls=cls, file_path=fig_path)
    spliter_cls_data(cls_pts, kmeans_model.labels_, cls, spliter_save_path)


def reduction_tsne(features, num_comps=3, visualize=False):
    tsne = TSNE(n_components=num_comps, init='pca', random_state=0, method='exact', verbose=False)

    tsne_features = tsne.fit_transform(features)
    if visualize:
        visualize_feature_scatter(tsne_features)
    return tsne_features


def spliter_cls_data(pts_all, cluster_labels, cls, save_path):
    assert pts_all.shape[0] == cluster_labels.shape[0], "The cluster labels and Pts shape mismatch"
    for k in range(len(set(cluster_labels))):
        cluster_index = cluster_labels == k
        cluster_pts = pts_all[cluster_index, :]

        npy_file = str(cls) + "_" + str(k) + "_" + str(cluster_pts.shape[0]) + ".npy"
        # file name: class_idx - cluster_idx - num_pts
        npy_save_path = os.path.join(save_path, npy_file)

        np.save(npy_save_path, cluster_pts)
        print(f"Save Class {cls} Cluster {k} with number {cluster_pts.shape[0]} to {npy_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arg parser')
    parser.add_argument('--pre_trained', type=str, default=None, help='pretrained_model')
    parser.add_argument('--dataset', type=str, default="modelnet")
    parser.add_argument('--process_all', action="store_true", default=False, help="Whether to process all")
    args = parser.parse_args()
    # pre_trained = "/home/siyuan/4-data/PointDA_data/output/ckpt/source_train/modelnet/checkpoint_epoch_150.pth"
    time_used = Timer()
    if args.process_all:
        for dataset_type in dataset_list:
            time_used.s()
            # when --process_all set, the --pred_trained is the folder contains all ckpt
            ckpt_folder = os.path.join(args.pre_trained, dataset_type)
            cpkt_pth = os.path.join(ckpt_folder, "checkpoint_epoch_150.pth")
            if not os.path.join(cpkt_pth):
                raise FileNotFoundError("The Pre-Trained Ckpt not found")
            for i in range(num_class):
                extract_feature_map_class(pre_trained_=cpkt_pth, cls=i, dataset_type=dataset_type)
            t_used = time_str(time_used.t())
            print(f"The {dataset_type} used time: {t_used}")
    else:
        time_used.s()
        for i in range(num_class):
            extract_feature_map_class(pre_trained_=args.pre_trained, cls=i, dataset_type=args.dataset)
        t_used = time_str(time_used.t())
        print(f"The {args.dataset} used time: {t_used}")