import logging
import os
import torch
import numpy as np
import subprocess

import torch.distributed as dist
import torch.multiprocessing as mp
import random

from datetime import datetime


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size
    

def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    # os.environ['MASTER_PORT'] = str(tcp_port)
    # os.environ['MASTER_ADDR'] = 'localhost'
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Set the Random Seed!")


def worker_init_fn(worker_id, seed=666):
    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s %(filename)s %(funcName)s %(lineno)d %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def exp_log_folder_creator(cfg, extra_tag=None):
    """
        return output_dir, cpkt_dir string
    """
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d %H:%M:%S')

    if 'data' not in cfg["DATA_ROOT"]:
        dir_root = os.path.join(cfg["DATA_ROOT"], 'PointDA_data/')
    else:
        dir_root = cfg["DATA_ROOT"]

    output_dir = os.path.join(dir_root, 'output', cfg["EXTRA_TAG"])
    ckpt_dir = os.path.join(output_dir, 'ckpt', cfg["EXPERIMENT"], cfg["EXTRA_TAG"])
    if extra_tag is not None:
        output_dir = os.path.join(output_dir, extra_tag)
        ckpt_dir = os.path.join(ckpt_dir, extra_tag)
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    else:
        output_dir = os.path.join(output_dir, today_str)
        os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(ckpt_dir): 
        os.makedirs(ckpt_dir)
    else:
        ckpt_dir = os.path.join(ckpt_dir, today_str)
        os.makedirs(ckpt_dir, exist_ok=True)
    return output_dir, ckpt_dir


def create_one_hot_labels(original_labels, num_class=10):
    one_hot_labels = torch.zeros(original_labels.shape[0], num_class)
    one_hot_labels[range(original_labels.shape[0]), original_labels] = 1
    return one_hot_labels


def get_most_overlapped_element(vec_a, vec_b, num_class=10):
    """
        Get the Overlapped Elements of two class sets
        vec_a, vec_b: Batch_size * 1, containing the batch labels
    """
    sorted_a, indices_a = torch.sort(vec_a)
    sorted_b, indices_b = torch.sort(vec_b)
    assert torch.max(sorted_a) < num_class, "The input class is larger than pre-defined"
    a_pointer = 0
    b_pointer = 0
    selecter_a = []
    selecter_b = []
    for i in range(num_class):
        a_i = (sorted_a == i).sum()
        b_i = (sorted_b == i).sum()
            
        selecter_a_i_num = min(a_i, b_i)
        selecter_a_i = [a_pointer+i for i in range(selecter_a_i_num)]
        selecter_b_i = [b_pointer+i for i in range(selecter_a_i_num)]
        a_pointer += a_i
        b_pointer += b_i
        selecter_a.extend(selecter_a_i)
        selecter_b.extend(selecter_b_i)
    
    ind_aa = [int(indices_a[i]) for i in selecter_a]
    ind_bb = [int(indices_b[i]) for i in selecter_b]

    return ind_aa, ind_bb