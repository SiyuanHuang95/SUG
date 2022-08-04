import os
from tkinter import N
import h5py
import glob
import numpy as np

def create_full_file_txt(data_root):
    pass

def extract_scannet_to_npy(scannet_path):
    for split_set in ["train", "test"]:
        pts_save_npy = os.path.join(scannet_path, split_set+"_pts.npy")
        label_save_npy = os.path.join(scannet_path, split_set + "_label.npy")
        
        data_path = os.path.join(scannet_path, split_set+"_files.txt")
        with open(data_path,'r') as f:
            lines = f.readlines()
        file_list =  [os.path.join(scannet_path, line.rstrip().split('/')[-1]) for line in lines]

        point_list = []
        label_list = []
        for pth in file_list:
            data_file = h5py.File(pth, 'r')
            point = data_file['data'][:]
            label = data_file['label'][:]
            
            point_list.append(point)
            label_list.append(label)

        data = np.concatenate(point_list, axis=0)
        label =  np.concatenate(label_list, axis=0)

        print(f" Extracted pts are saved to {pts_save_npy}")
        np.save(pts_save_npy, data)
        np.save(label_save_npy, label)


def extract_shapenet_to_npy(shapenet_path, dataset="shapenet"):
    for split_set in ["train", "test"]:
        pts_save_npy = os.path.join(scannet_path, split_set+"_pts.npy")
        label_save_npy = os.path.join(scannet_path, split_set + "_label.npy")

        categorys = glob.glob(os.path.join(shapenet_path, '*'))
        categorys = sorted([c.split(os.path.sep)[-1] for c in categorys])

        pts_list = glob.glob(os.path.join(shapenet_path, "*", split_set, "*.npy"))
        point_list = []
        label_list = []
        for pts in pts_list:
            point_list.append(np.load(pts))
            label_list.append(categorys.index(pts.split(dataset)[-1].split(str("/")+split_set)[0].split("/")[-1]))

        data = np.concatenate(point_list, axis=0)
        label =  np.array(label_list)

        print(f" Extracted pts are saved to {pts_save_npy}")
        np.save(pts_save_npy, data)
        np.save(label_save_npy, label)


def extract_modelnet_to_npy(modelnet_path):
    extract_shapenet_to_npy(modelnet_path, dataset="modelnet")

def rename_npy_files(data_path):
    """
        This function is mainly used for renaming npy files in ShapeNet/plant
    """
    counter = 500000 # to avoid being same idx with other files
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
    scannet_path = "/point_dg/data/scannet"
    # rename_npy_files(os.path.join(scannet_path, "plant"))
    # extract_modelnet_to_npy(scannet_path)
    extract_scannet_to_npy(scannet_path)