import logging
import os
import torch

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
    if 'data' not in cfg["DATA_ROOT"]:
        dir_root = os.path.join(cfg["DATA_ROOT"], 'PointDA_data/')
    else:
        dir_root = cfg["DATA_ROOT"]

    output_dir = os.path.join(dir_root, 'output', cfg["EXTRA_TAG"])
    ckpt_dir = os.path.join(output_dir, 'ckpt', cfg["EXPERIMENT"], cfg["EXTRA_TAG"])
    if extra_tag is not None:
        output_dir = os.path.join(output_dir, extra_tag)
        ckpt_dir = os.path.join(ckpt_dir, extra_tag)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    return output_dir, ckpt_dir


def create_one_hot_labels(original_labels, num_class=10):
    one_hot_labels = torch.zeros(original_labels.shape[0], num_class)
    one_hot_labels[range(original_labels.shape[0]), original_labels] = 1
    return one_hot_labels
