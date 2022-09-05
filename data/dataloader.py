import torch
import torch.utils.data as data
import os
import sys
import h5py
import numpy as np
import glob
import random
from data.data_utils import *
from utils.train_files_spliter import split_dataset, include_dataset_full_information


def load_dir(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir, name), 'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]


def get_info(shapes_dir, isView=False):
    names_dict = {}
    if isView:
        for shape_dir in shapes_dir:
            name = '_'.join(os.path.split(shape_dir)[
                                1].split('.')[0].split('_')[:-1])
            if name in names_dict:
                names_dict[name].append(shape_dir)
            else:
                names_dict[name] = [shape_dir]
    else:
        for shape_dir in shapes_dir:
            name = os.path.split(shape_dir)[1].split('.')[0]
            names_dict[name] = shape_dir

    return names_dict


class Modelnet40_data(data.Dataset):
    def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True):
        super(Modelnet40_data, self).__init__()

        self.status = status
        self.pc_list = []
        self.lbl_list = []
        self.pc_input_num = pc_input_num
        self.aug = aug

        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        # sorted(categorys)
        categorys = sorted(categorys)

        if status == 'train':
            npy_list = glob.glob(os.path.join(pc_root, '*', 'train', '*.npy'))
        else:
            npy_list = glob.glob(os.path.join(pc_root, '*', 'test', '*.npy'))
        # names_dict = get_info(npy_list, isView=False)

        for _dir in npy_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

        print(f'{status} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc = normal_pc(pc)
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        # print(pc.shape)
        pc = np.expand_dims(pc.transpose(), axis=2)
        return torch.from_numpy(pc).type(torch.FloatTensor), lbl

    def __len__(self):
        return len(self.pc_list)


class Shapenet_data(data.Dataset):
    def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True, data_type='*.npy', pts_list=None):
        super(Shapenet_data, self).__init__()

        self.status = status
        self.pc_list = []
        self.lbl_list = []
        self.pc_input_num = pc_input_num
        self.aug = aug
        self.data_type = data_type

        categorys = glob.glob(os.path.join(pc_root, '*'))
        categorys = [c.split(os.path.sep)[-1] for c in categorys]
        # sorted(categorys)
        categorys = sorted(categorys)

        if status == 'train':
            pts_list = glob.glob(os.path.join(
                pc_root, '*', 'train', self.data_type))
        elif status == 'test':
            pts_list = glob.glob(os.path.join(
                pc_root, '*', 'test', self.data_type))
        else:
            pts_list = glob.glob(os.path.join(
                pc_root, '*', 'validation', self.data_type))
        # names_dict = get_info(pts_list, isView=False)

        for _dir in pts_list:
            self.pc_list.append(_dir)
            self.lbl_list.append(categorys.index(_dir.split('/')[-3]))

        print(f'{status} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        if self.data_type == '*.pts':
            pc = np.array([[float(value) for value in xyz.split(' ')]
                           for xyz in open(self.pc_list[idx], 'r') if len(xyz.split(' ')) == 3])[:self.pc_input_num, :]
        elif self.data_type == '*.npy':
            pc = np.load(self.pc_list[idx])[
                 :self.pc_input_num].astype(np.float32)
        pc = normal_pc(pc)
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        pad_pc = np.zeros(
            shape=(self.pc_input_num - pc.shape[0], 3), dtype=float)
        pc = np.concatenate((pc, pad_pc), axis=0)
        pc = np.expand_dims(pc.transpose(), axis=2)
        return torch.from_numpy(pc).type(torch.FloatTensor), lbl

    def __len__(self):
        return len(self.pc_list)


class Scannet_data_h5(data.Dataset):

    def __init__(self, pc_root, status='train', pc_input_num=1024, aug=True):
        super(Scannet_data_h5, self).__init__()
        self.num_points = pc_input_num
        self.status = status
        self.aug = aug
        # self.label_map = [2, 3, 4, 5, 6, 7, 9, 10, 14, 16]

        if self.status == 'train':
            data_pth = load_dir(pc_root, name='train_files.txt')
        else:
            data_pth = load_dir(pc_root, name='test_files.txt')

        point_list = []
        label_list = []
        for pth in data_pth:
            data_file = h5py.File(pth, 'r')
            point = data_file['data'][:]
            label = data_file['label'][:]

            # idx = [index for index, value in enumerate(list(label)) if value in self.label_map]
            # point_new = point[idx]
            # label_new = np.array([self.label_map.index(value) for value in label[idx]])

            point_list.append(point)
            label_list.append(label)
        self.data = np.concatenate(point_list, axis=0)
        self.label = np.concatenate(label_list, axis=0)

    def __getitem__(self, idx):
        point_idx = np.arange(0, self.num_points)
        np.random.shuffle(point_idx)
        point = self.data[idx][point_idx][:, :3]
        label = self.label[idx]

        pc = normal_pc(point)
        if self.aug:
            pc = rotation_point_cloud(pc)
            pc = jitter_point_cloud(pc)
        # print(pc.shape)
        pc = np.expand_dims(pc.transpose(), axis=2)
        return torch.from_numpy(pc).type(torch.FloatTensor), label

    def __len__(self):
        return self.data.shape[0]


class UnifiedPointDG(data.Dataset):
    def __init__(self, dataset_type, pts, labels, status='train', pc_input_num=1024, aug=True):
        super(UnifiedPointDG, self).__init__()

        self.num_points = pc_input_num
        self.status = status
        self.aug = aug
        self.dataset_type = dataset_type

        self.pts = pts
        self.labels = labels

        self.class_num = 10
        self.dataset_size = pts.shape[0]
        self.indices = [[] for _ in range(self.class_num)]
         
        for i, label in enumerate(labels):
            self.indices[int(label)].append(i)
        self.cls_num_counter = [ len(cls_index) for cls_index in self.indices]

        print(f"Create {status} Dataset {dataset_type} with pts {self.dataset_size}")
        print(f"Cls number {self.cls_num_counter}")

    def classes(self):
        return self.indices

    def cls_wights(self, weighting="number_inverse"):
        if weighting == "number_inverse":
            num_inv = [ 1/num_cls for num_cls in self.cls_num_counter]
            weights = [num_inv_ / sum(num_inv) for num_inv_ in num_inv]
            return weights
        elif weighting == "exp_inverse":
            exp_cls = [np.exp(-cls_num / self.dataset_size) for cls_num in self.cls_num_counter]
            weights = [exp_cls_ / sum(exp_cls) for exp_cls_ in exp_cls]
            return weights
        elif weighting == "DLSA":
            # Constructing Balance from Imbalance: Cewu Lu
            q = 2.0
            sample_num_neg_power = [np.power(cls_num, -q) for cls_num in self.cls_num_counter]
            weights = [cls_weight / sum(sample_num_neg_power) for cls_weight in sample_num_neg_power]
            return weights
        else:
            return [ 1 / self.class_num] * self.class_num

    def __getitem__(self, index):
        raw_pts = self.pts[index][:, :3]  # for ScanNet, only x-y-z features are used
        label = self.labels[index]
        # TODO should do normal once, to speed-up the whole process
        pts = normal_pc(raw_pts)

        if self.aug:
            pts = rotation_point_cloud(pts)
            pts = jitter_point_cloud(pts)

        if pts.shape[0] < self.num_points:
            pad_pc = np.zeros(
                shape=(self.num_points - pts.shape[0], 3), dtype=float)
            pts = np.concatenate((pts, pad_pc), axis=0)
        elif pts.shape[0] > self.num_points:
            point_idx = np.arange(0, pts.shape[0])
            np.random.shuffle(point_idx)
            pts = pts[point_idx[:self.num_points]]
        pts = np.expand_dims(pts.transpose(), axis=2)
        return torch.from_numpy(pts).type(torch.FloatTensor), label

    def __len__(self):
        return self.pts.shape[0]


def create_splitted_dataset(dataset_type, status="train", config=None, logger=None):
    dataset_list = ["scannet", "shapenet", "modelnet"]
    assert dataset_type in dataset_list, f"Not supported dataset {dataset_type}!"

    dataset_spliter = split_dataset(
        dataset_type, status=status, logger=logger, split_config=config)

    dataset_subsets = []
    for subset in dataset_spliter.keys():
        pts = dataset_spliter[subset]["pts"]
        label = dataset_spliter[subset]["label"]
        dataset_subsets.append(UnifiedPointDG(
            dataset_type=dataset_type, pts=pts, labels=label, status=status))

    return dataset_subsets


def create_single_dataset(dataset_type, status="train", aug=False):
    dataset_list = ["scannet", "shapenet", "modelnet"]
    assert dataset_type in dataset_list, f"Not supported dataset {dataset_type}!"

    pts, labels = include_dataset_full_information(dataset_type, status)
    assert len(set(labels.tolist())) == 10, "The class in labels is less than 10!"
    return UnifiedPointDG(dataset_type=dataset_type, pts=pts, labels=labels, status=status, aug=aug)


if __name__ == "__main__":
    pass
