import os
from typing import List
import h5py
import glob
import numpy as np
import pickle
import random
import datetime

data_root = "/point_dg/data"
# data_root = "/data/point_cloud_classification/PointDA_data"
# data_root = "/home/siyuan/4-data/PointDA_data"
num_class = 10
dataset_list = ["scannet", "shapenet", "modelnet"]


def split_dataset(dataset_type, split_config, logger, status='train'):
    dataset_path = os.path.join(data_root, dataset_type)
    full_pts, full_label = include_dataset_full_information(
        dataset_type, status=status)
    assert full_pts.shape[0] == full_label.shape[0], "The label size should be identical with pts"

    dataset_spliter = {}
    index_subset_1, index_subset_2 = None, None
    subset_2_size = 1 if split_config["SUBSET_FULLSIZE"] else 0.5
    size_usage = split_config["SAMPLE_RATE"] + subset_2_size
    # index_config_naming = str(datetime.datetime.now()) + split_config["split_method"] + "_" + str(
    #     split_config["sample_rate"]) + ".pkl"
    index_config_naming = "size_" + str(size_usage) + split_config["METHOD"] + "_" + str(
        split_config["SAMPLE_RATE"]) + ".pkl"
    index_file_storage = os.path.join(dataset_path, index_config_naming)
    if os.path.exists(index_file_storage):
        logger.info(f"Direct load the indexing history from {index_file_storage}")
        with open(index_file_storage, "rb") as f:
            indexs = pickle.load(f)
            index_subset_1 = indexs['index1']
            index_subset_2 = indexs['index2']

    if index_subset_1 is None:
        if split_config["METHOD"] == "Random":
            dataset_size = full_pts.shape[0]
            index_array = np.arange(dataset_size)

            subset_size = int(dataset_size * split_config["SAMPLE_RATE"])
            index_subset_1 = np.random.choice(
                index_array, replace=False, size=subset_size)

            if not split_config["SUBSET_FULLSIZE"]:
                index_subset_2 = np.setdiff1d(index_array, index_subset_1)
            else:
                index_subset_2 = index_array

            indexs = {'index2': index_subset_2, "index1": index_subset_1}

            with open(index_file_storage, "wb") as f:
                pickle.dump(indexs, f)
            logger.info(f"Save indexing history to {index_file_storage}")

            dataset_spliter = {
                "subset_1": {
                    "pts": full_pts[index_subset_1, :],
                    "label": full_label[index_subset_1]
                },

                "subset_2": {
                    "pts": full_pts[index_subset_2, :],
                    "label": full_label[index_subset_2]
                }
            }
            return dataset_spliter

        elif split_config["METHOD"] == "Cluster":
            return include_dataset_from_splitter(dataset_type, split_config, method="kmeans")

        elif split_config["METHOD"] == "Entropy":
            return include_dataset_from_splitter(dataset_type, split_config, method="entropy")

        else:
            raise NotImplementedError("Not Implemented Error")


def include_dataset_full_information(dataset_type, status='train'):
    """
        Load full train information for one dataset.
    """
    pts_path = os.path.join(data_root, dataset_type, status + "_pts.npy")
    full_pts = np.load(pts_path)

    label_path = os.path.join(data_root, dataset_type, status + "_label.npy")
    full_label = np.load(label_path)
    return full_pts, full_label


def include_dataset_one_class(dataset_type, status='train', cls=0):
    """
    Args:
        dataset_type: scannet/shapenet/modelnet
        status: train of test
        cls: 0-9, number
    Returns: pts and labels of cls in dataset
    """
    full_pts, full_labels = include_dataset_full_information(dataset_type, status)
    index_cls = full_labels == cls
    return full_pts[index_cls], full_labels[index_cls]


def include_dataset_from_splitter(dataset_type, spliter_config, subset_num=2, method="kmeans"):
    spliter_path = os.path.join(data_root, dataset_type, "spliter")
    if not os.path.exists(spliter_path):
        raise RuntimeError("No Spliter Folder Found, Need to Generate Dataset Cluster First!")

    subset_1_pts, subset_1_labels = [], []
    subset_2_pts, subset_2_labels = [], []  # could be full-size
    if "kmeans" in method:
        cluster_num = len(glob.glob(str(spliter_path) + "/" + method + "_1_*.npy"))
        subset_1_cluster = int(cluster_num * spliter_config["SAMPLE_RATE"])
        sample_method = "random"
        if spliter_config.get("MERGE_CLUSTER_METHOD", None):
            sample_method = spliter_config["MERGE_CLUSTER_METHOD"]
        for i in range(num_class):
            subset_cls_1, subset_cls_2 = load_splitter_npy_list(spliter_path, spliter_config, method, i, sample_method, subset_1_cluster)

            cls_1_pts, cls_1_labels = load_npy_pts_and_labels(subset_cls_1, cls=i)
            cls_2_pts, cls_2_labels = load_npy_pts_and_labels(subset_cls_2, cls=i)
            subset_1_pts.extend(cls_1_pts)
            subset_1_labels.extend(cls_1_labels)
            subset_2_pts.extend(cls_2_pts)
            subset_2_labels.extend(cls_2_labels)
    elif method == "entropy":
        cluster_num = len(glob.glob(str(spliter_path) + "/" + method + "_-1_*.npy"))
        subset_1_cluster = int(cluster_num * spliter_config["SAMPLE_RATE"])
        choice_list = [[0, 1], [2, 3]]
        subset_cls_1, subset_cls_2 = load_splitter_npy_list(spliter_path, spliter_config, method, cls=-1, choice_list=choice_list, choice_method=choice_list)
        subset_1_pts, subset_1_labels = load_npy_pts_and_labels(subset_cls_1, cls=-1)
        subset_2_pts, subset_2_labels = load_npy_pts_and_labels(subset_cls_2, cls=-1)

    dataset_spliter = {
        "subset_1": {
            "pts": np.array(subset_1_pts),
            "label": np.array(subset_1_labels)
        },

        "subset_2": {
            "pts": np.array(subset_2_pts),
            "label": np.array(subset_2_labels)
        }
    }
    return dataset_spliter


def load_splitter_npy_list(path, spliter_config, method="kmeans", cls=-1, \
                            choice_method="random", subset_1_cluster=2, choice_list=None):
    cls_npy_list = glob.glob(str(path) + "/" + method + "_" + str(cls) + "_*.npy")
    cls_npy_list = [npy for npy in cls_npy_list if "_label" not in npy]
    if choice_method == "random":
        random.shuffle(cls_npy_list)
        subset_cls_1 = cls_npy_list[0:subset_1_cluster]
        if spliter_config["SUBSET_FULLSIZE"]:
            # 50% + 100%
            subset_cls_2 = cls_npy_list
        else:
            subset_cls_2 = cls_npy_list[subset_1_cluster:]

    elif choice_method == "Entropy":
        sort_with_entropy(cls_list=cls_npy_list)
        subset_cls_1 = cls_npy_list[0:subset_1_cluster]
        if spliter_config["SUBSET_FULLSIZE"]:
            # 50% + 100%
            subset_cls_2 = cls_npy_list
        else:
            subset_cls_2 = cls_npy_list[subset_1_cluster:]
    else:
        if choice_list is None:
            raise RuntimeError("When not random, should set the choice list")
        subset_cls_1 = [cls_npy_list[i] for i in choice_list[0]]
        subset_cls_2 = [cls_npy_list[i] for i in choice_list[1]]
    return subset_cls_1, subset_cls_2
    

def sort_with_entropy(cls_list:List):
    def get_entropy(file_name):
        return float(file_name.split("_entropy_")[-1].split(".npy")[0]) 
    return cls_list.sort(key=get_entropy)


def load_npy_pts_and_labels(npy_list, cls):
    """
    Args:
        npy_list: list store the NPY splitter files
        cls: current cls index

    Returns:
        pts and label in list
    """
    pts, labels = [], []
    pts = np.array(load_npy_list(npy_list))
    if cls != -1:
        labels = (np.ones(pts.shape[0]) * cls).tolist()
    else:
        label_file = [file.split("_entropy")[0] + "_labels.npy" for file in npy_list]
        labels = load_npy_list(label_file)
    return pts.tolist(), labels


def load_npy_list(npy_list):
    infos = []
    for npy_file in npy_list:
        infos.extend(np.load(npy_file))
    return infos

def extract_scannet_to_npy(scannet_path):
    for split_set in ["train", "test"]:
        pts_save_npy = os.path.join(scannet_path, split_set + "_pts.npy")
        label_save_npy = os.path.join(scannet_path, split_set + "_label.npy")

        data_path = os.path.join(scannet_path, split_set + "_files.txt")
        with open(data_path, 'r') as f:
            lines = f.readlines()
        file_list = [os.path.join(
            scannet_path, line.rstrip().split('/')[-1]) for line in lines]

        point_list = []
        label_list = []
        for pth in file_list:
            data_file = h5py.File(pth, 'r')
            point = data_file['data'][:]
            label = data_file['label'][:]

            point_list.append(point)
            label_list.append(label)

        data = np.concatenate(point_list, axis=0)
        label = np.concatenate(label_list, axis=0)
        assert data.shape[0] == label.shape[0], "The label size should be identical with pts"

        print(f" Extracted pts are saved to {pts_save_npy}")
        np.save(pts_save_npy, data)
        np.save(label_save_npy, label)


def extract_shapenet_to_npy(shapenet_path, dataset="shapenet"):
    for split_set in ["train", "test"]:
        pts_save_npy = os.path.join(shapenet_path, split_set + "_pts.npy")
        label_save_npy = os.path.join(shapenet_path, split_set + "_label.npy")

        categorys = glob.glob(os.path.join(shapenet_path, '*'))
        categorys = sorted([c.split(os.path.sep)[-1] for c in categorys])

        pts_list = glob.glob(os.path.join(
            shapenet_path, "*", split_set, "*.npy"))
        point_list = []
        label_list = []
        for pts in pts_list:
            point_list.append(np.load(pts))
            label_list.append(categorys.index(
                pts.split(dataset)[-1].split(str("/") + split_set)[0].split("/")[-1]))

        data = np.array(point_list)
        label = np.array(label_list)

        assert data.shape[0] == label.shape[0], "The label size should be identical with pts"
        print(f" Extracted pts are saved to {pts_save_npy}")
        np.save(pts_save_npy, data)
        np.save(label_save_npy, label)


def extract_modelnet_to_npy(modelnet_path):
    extract_shapenet_to_npy(modelnet_path, dataset="modelnet")


def rename_npy_files(data_path):
    """
        This function is mainly used for renaming npy files in ShapeNet/plant
    """
    counter = 500000  # to avoid being same idx with other files
    for spliter_set in ["train", "test"]:
        data_path_full = os.path.join(data_path, spliter_set)
        npy_pts = os.listdir(data_path_full)
        for npy_file in npy_pts:
            if ".npy" not in npy_file:
                continue
            old_name = os.path.join(data_path_full, npy_file)
            print(old_name)
            pts = np.load(old_name)
            new_name = os.path.join(data_path_full, str(counter) + ".npy")
            # os.rename(old_name, new_name) -> after rename, file is broken to open
            np.save(new_name, pts)
            os.remove(old_name)
            counter += 1


if __name__ == "__main__":
    init_dataset = False
    if init_dataset:
        # rename_npy_files(os.path.join(scannet_path, "plant"))
        funcs = {
            "scannet": extract_scannet_to_npy,
            "shapenet": extract_shapenet_to_npy,
            "modelnet": extract_modelnet_to_npy
        }
        for dataset in ["scannet", "shapenet", "modelnet"]:
            dataset_path = os.path.join(data_root, dataset)
            funcs[dataset](dataset_path)