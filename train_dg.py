from copy import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model_pointnet import Pointnet_cls as Pointnet_cls
import model.Model as mM
from data.dataloader import create_splitted_dataset, create_single_dataset
import time
import numpy as np
import os
import glob
import argparse
import pdb
import model.mmd as mmd
# from utils import *
import math
import warnings
from multiprocessing import Pool
import copy
from utils.eval_utils import eval_worker
from utils.train_utils import save_checkpoint, checkpoint_state, adjust_learning_rate, discrepancy

from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='scannet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=200)
parser.add_argument('-models', '-m', type=str, help='alignment model', default='MDA')
parser.add_argument('-lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('-scaler', type=float, help='scaler of learning rate', default=1.)
parser.add_argument('-weight', type=float, help='weight of src loss', default=1.)
parser.add_argument('-datadir', type=str, help='directory of data', default='./dataset/')
parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./logs')
parser.add_argument('-target_cls_loss', type=float, help="the wights for cls loss from target split", default=1.0)
parser.add_argument('-class_mmd', action='store_true', help="Use MMD loss only within the same cls", default=False)
parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
parser.add_argument('--max_ckpt_save_num', type=int, default=50, help='max number of saved checkpoint')
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
parser.add_argument('--spliter_fullsize', action='store_true', default=False)
parser.add_argument('--train_base',type=int, default=1, help="Use which part as main training, default 1 to be full label" )
args = parser.parse_args()

if not os.path.exists(os.path.join(os.getcwd(), args.tb_log_dir)):
    os.makedirs(os.path.join(os.getcwd(), args.tb_log_dir))
writer = SummaryWriter(log_dir=args.tb_log_dir)

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = args.batchsize * len(args.gpu.split(','))
LR = args.lr
weight_decay = 5e-4
momentum = 0.9
max_epoch = args.epochs
num_class = 10
if 'data' not in args.datadir:
    dir_root = os.path.join(args.datadir, 'PointDA_data/')
else:
    dir_root = args.datadir

output_dir = os.path.join(dir_root, 'output')
ckpt_dir = os.path.join(output_dir, 'ckpt', 'DG_exp')
if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)


def main():
    print('Start Training\nInitiliazing\n')
    print('The source domain is set to:', args.source)

    dataset_list = ["scannet", "shapenet", "modelnet"]
    test_datasets = list(set(dataset_list) - {args.source})
    print('The datasets used for testing:', test_datasets)

    # Data loading
    split_config = {
        "split_method": "random",
        "subset_fullsize": args.spliter_fullsize,
        "sample_rate": 0.5,
        "train_base": args.train_base
    }
    source_train_subsets = create_splitted_dataset(dataset_type=args.source, status="train", config=split_config)
    source_train_dataset = source_train_subsets[split_config["train_base"]]
    target_train_dataset1 = source_train_subsets[1-split_config["train_base"]]
    # split 2 is fullsize

    source_test_dataset = create_single_dataset(args.source, status="test", aug=False)
    target_test_dataset1 = create_single_dataset(test_datasets[0], status="test", aug=False)
    target_test_dataset2 = create_single_dataset(test_datasets[-1], status="test", aug=False)

    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_train1 = len(target_train_dataset1)
    num_target_test1 = len(target_test_dataset1)
    num_target_test2 = len(target_test_dataset2)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                         drop_last=True)
    target_train_dataloader1 = DataLoader(target_train_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                          drop_last=True)

    source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                        drop_last=True)
    target_test_dataloader1 = DataLoader(target_test_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                         drop_last=True)
    target_test_dataloader2 = DataLoader(target_test_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                         drop_last=True)
    performance_test_sets = {"source": source_test_dataloader, "test1": target_test_dataloader1,
                             "test2": target_test_dataloader2}

    print(f"Num of source train: {num_source_train}, Num of target train: {num_target_train1}")
    print(
        f"Num of source test: {num_source_test}, Num of test on {test_datasets[0]} {num_target_test1}, on {test_datasets[-1]} {num_target_test2}")
    print('batch_size:', BATCH_SIZE)

    best_test_acc = {"source": 0, "test1": 0, "test2": 0}
    # pool_eval = Pool(processes=len(dataset_list))
    # AssertionError: daemonic processes are not allowed to have children

    # Model
    model = mM.Net_MDA()
    model = model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    remain_epoch = 50

    # Optimizer

    params = [{'params': v} for k, v in model.g.named_parameters() if 'pred_offset' not in k]

    optimizer_g = optim.Adam(params, lr=LR, weight_decay=weight_decay)
    lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs + remain_epoch)

    optimizer_c = optim.Adam([{'params': model.c1.parameters()}, {'params': model.c2.parameters()}], lr=LR * 2,
                             weight_decay=weight_decay)
    lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=args.epochs + remain_epoch)

    optimizer_dis = optim.Adam([{'params': model.g.parameters()}, {'params': model.attention_s.parameters()},
                                {'params': model.attention_t.parameters()}],
                               lr=LR * args.scaler, weight_decay=weight_decay)
    lr_schedule_dis = optim.lr_scheduler.CosineAnnealingLR(optimizer_dis, T_max=args.epochs + remain_epoch)


    for epoch in range(max_epoch):
        since_e = time.time()

        lr_schedule_g.step(epoch=epoch)
        lr_schedule_c.step(epoch=epoch)
        adjust_learning_rate(optimizer_dis, epoch, args.lr, args.scaler, writer)

        writer.add_scalar('lr_g', lr_schedule_g.get_lr()[0], epoch)
        writer.add_scalar('lr_c', lr_schedule_c.get_lr()[0], epoch)

        model.train()

        loss_total = 0
        loss_adv_total = 0
        loss_node_total = 0
        correct_total = 0
        data_total = 0
        data_t_total = 0
        cons = math.sin((epoch + 1) / max_epoch * math.pi / 2)

        # Training
        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_train_dataloader, target_train_dataloader1)):
            data, label = batch_s
            data_t, label_t = batch_t

            data = data.to(device=device)
            label = label.to(device=device).long()
            data_t = data_t.to(device=device)
            label_t = label_t.to(device=device).long()

            pred_s1, pred_s2 = model(data)
            pred_t1, pred_t2 = model(data_t, constant=cons, adaptation=True)

            # Classification loss
            loss_s1 = criterion(pred_s1, label)
            loss_s2 = criterion(pred_s2, label)

            # Adversarial loss -> let two heads of the model output similiar
            loss_adv = - 1 * discrepancy(pred_t1, pred_t2)

            loss_s = loss_s1 + loss_s2
            if args.target_cls_loss > 0:
                loss_t1 = criterion(pred_t1, label)
                loss_t2 = criterion(pred_t2, label)
                loss_t = loss_t1 + loss_t2
                loss = args.weight * loss_s + loss_adv + args.target_cls_loss * loss_t
            else:
                loss = args.weight * loss_s + loss_adv

            loss.backward()
            optimizer_g.step()
            optimizer_c.step()
            optimizer_g.zero_grad()
            optimizer_c.zero_grad()

            # Local Alignment
            feat_node_s = model(data, node_adaptation_s=True)  # shape: batch_size * 4096
            feat_node_t = model(data_t, node_adaptation_t=True)
            sigma_list = [0.01, 0.1, 1, 10, 100]
            if args.class_mmd:
                # Only enfore the feature align when the source and target have the same label
                same_class_index = torch.eq(label, label_t)
                selected_feat_node_s = feat_node_s[same_class_index]
                selected_feat_node_t = feat_node_t[same_class_index]

                loss_node_adv = 1 * mmd.mix_rbf_mmd2(selected_feat_node_s, selected_feat_node_t, sigma_list)
            else:
                loss_node_adv = 1 * mmd.mix_rbf_mmd2(feat_node_s, feat_node_t, sigma_list)
            loss = loss_node_adv
            loss.backward()
            optimizer_dis.step()
            optimizer_dis.zero_grad()

            loss_total += loss_s.item() * data.size(0)
            loss_adv_total += loss_adv.item() * data.size(0)
            loss_node_total += loss_node_adv.item() * data.size(0)
            data_total += data.size(0)
            data_t_total += data_t.size(0)

            if (batch_idx + 1) % 10 == 0:
                print(
                    'Train:{} [{} {}/{}  loss_s: {:.4f} \t loss_adv: {:.4f} \t loss_node_adv: {:.4f} \t cons: {:.4f}]'.format(
                        epoch, data_total, data_t_total, num_source_train, loss_total / data_total,
                                                                           loss_adv_total / data_total,
                                                                           loss_node_total / data_total, cons
                    ))

        # Testing
        with torch.no_grad():
            model.eval()
            # Could be accelerated with multi-process?
            for eval_dataset in performance_test_sets.keys():
                eval_dict = {
                    "model": copy.deepcopy(model),
                    "dataloader": performance_test_sets[eval_dataset],
                    "dataset": eval_dataset,
                    "best_target_acc": best_test_acc[eval_dataset],
                    "device": device,
                    "criterion": criterion,
                    "epoch": epoch
                }
                eval_result = eval_worker(eval_dict)
                best_test_acc[eval_dataset] = eval_result["best_target_acc"]
                writer_item = 'acc/' + eval_result["dataset"] + "_test_acc"
                writer.add_scalar(writer_item, eval_result["best_target_acc"], epoch)

        trained_epoch = epoch + 1
        if trained_epoch % args.ckpt_save_interval == 0:
            ckpt_list = [os.path.join(ckpt_dir, cpkt) for cpkt in os.listdir(ckpt_dir) if ".pth" in cpkt]
            ckpt_list.sort(key=os.path.getmtime)
            if ckpt_list.__len__() >= args.max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - args.max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name = os.path.join(ckpt_dir, args.source + ('_checkpoint_epoch_%d' % trained_epoch))
            print(f"Save current ckpt to {ckpt_name}")
            save_checkpoint(checkpoint_state(model, epoch=trained_epoch), filename=ckpt_name)

        time_pass_e = time.time() - since_e
        print('The {} epoch takes {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 60, time_pass_e % 60))
        print(args)
        print(' ')


if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))
