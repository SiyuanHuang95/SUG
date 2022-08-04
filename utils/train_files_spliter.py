import os
from tkinter import N
import h5py
import glob
import numpy as np

data_root = "/point_dg/data"


def split_dataset(dataset_type, split_config=None, status='train'):
    if split_config is None:
        split_config = {
            "split_method" : "random",
            "subset_2_fullsize": True
        }

    dataset_path = os.path.join(data_root, dataset_type)
    full_pts, full_label = include_dataset_full_information(
        dataset_path, status=status)
    assert full_pts.shape[0] == full_label.shape[0], "The label size should be identical with pts"

    dataset_spliter = {}
    if split_config["split_method"] == "random":
        dataset_size = full_pts.shape[0]
        index_array = np.arange(dataset_size)

        subset_size = dataset_size // 2
        index_subset_1 = np.random.choice(
            index_array, replace=False, size=subset_size)
        
        if not split_config["subset_2_fullsize"]:
            index_subset_2 = np.setdiff1d(index_array, index_subset_1)
        else:
            index_subset_2 = index_array

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


def include_dataset_full_information(dataset_path, status='train'):
    """
        Load full train information for one dataset.
    """
    pts_path = os.path.join(dataset_path, status+"_pts.npy")
    full_pts = np.load(pts_path)

    label_path = os.path.join(dataset_path, status+"_label.npy")
    full_label = np.load(label_path)

    return full_pts, full_label


def extract_scannet_to_npy(scannet_path):
    for split_set in ["train", "test"]:
        pts_save_npy = os.path.join(scannet_path, split_set+"_pts.npy")
        label_save_npy = os.path.join(scannet_path, split_set + "_label.npy")

        data_path = os.path.join(scannet_path, split_set+"_files.txt")
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
        pts_save_npy = os.path.join(shapenet_path, split_set+"_pts.npy")
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
                pts.split(dataset)[-1].split(str("/")+split_set)[0].split("/")[-1]))

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
    # rename_npy_files(os.path.join(scannet_path, "plant"))
    funcs = {
        "scannet": extract_scannet_to_npy,
        "shapenet": extract_shapenet_to_npy,
        "modelnet": extract_modelnet_to_npy
    }
    for dataset in ["scannet", "shapenet", "modelnet"]:
        dataset_path = os.path.join(data_root, dataset)
        funcs[dataset](dataset_path)
    # extract_scannet_to_npy(scannet_path)
