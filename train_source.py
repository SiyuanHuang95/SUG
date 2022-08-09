import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model_pointnet import Pointnet_cls as Pointnet_cls
from data.dataloader import Modelnet40_data, Shapenet_data, Scannet_data_h5
from data.dataloader import create_single_dataset

import time
import os
import argparse
from copy import copy
from tensorboardX import SummaryWriter
from utils.train_utils import save_checkpoint, checkpoint_state
from utils.eval_utils import eval_worker


# Command setting
parser = argparse.ArgumentParser(description='Main')
parser.add_argument('-source', '-s', type=str, help='source dataset', default='scannet')
parser.add_argument('-target1', '-t1', type=str, help='target dataset', default='modelnet')
parser.add_argument('-target2', '-t2', type=str, help='target dataset', default='shapenet')
parser.add_argument('-batchsize', '-b', type=int, help='batch size', default=64)
parser.add_argument('-gpu', '-g', type=str, help='cuda id', default='0')
parser.add_argument('-epochs', '-e', type=int, help='training epoch', default=200)
parser.add_argument('-lr', type=float, help='learning rate', default=0.001)
parser.add_argument('-datadir', type=str, help='directory of data', default='/repository/yhx/')
parser.add_argument('-tb_log_dir', type=str, help='directory of tb', default='./logs/src_m_s_ss')
parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
parser.add_argument('--max_ckpt_save_num', type=int, default=50, help='max number of saved checkpoint')
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

output_dir = os.path.join(dir_root , 'output')
ckpt_dir = os.path.join(output_dir , 'ckpt', 'source_train', args.source)
if not os.path.exists(output_dir): os.makedirs(output_dir) 
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir) 

def main():
    print ('Start Training\nInitiliazing\n')
    print('The source domain is set to:', args.source)

    dataset_list = ["scannet", "shapenet", "modelnet"]
    test_datasets = list(set(dataset_list) - {args.source})
    print('The datasets used for testing:', test_datasets)

    data_func={'modelnet': Modelnet40_data, 'scannet': Scannet_data_h5, 'shapenet': Shapenet_data}

    source_train_dataset = create_single_dataset(dataset_type=args.source,status="train", aug=True)
    source_test_dataset = create_single_dataset(dataset_type=args.source,status="test", aug=False)
    target_test_dataset1 = create_single_dataset(dataset_type=test_datasets[0], status="test", aug=False)
    target_test_dataset2 = create_single_dataset(test_datasets[-1], status="test", aug=False)

    num_source_train = len(source_train_dataset)
    num_source_test = len(source_test_dataset)
    num_target_test1 = len(target_test_dataset1)
    num_target_test2 = len(target_test_dataset2)

    source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    target_test_dataloader1 = DataLoader(target_test_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    target_test_dataloader2 = DataLoader(target_test_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)
    performance_test_sets = {"source": source_test_dataloader, "test1": target_test_dataloader1,
                             "test2": target_test_dataloader2}

    print('num_source_train: {:d}, num_source_test: {:d}, num_target_test1: {:d}, num_target_test2: {:d}'.format(
        num_source_train, num_source_test, num_target_test1, num_target_test2))
    print('batch_size:', BATCH_SIZE)

    # Model

    model = Pointnet_cls()
    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    # Optimizer
    remain_epoch=50

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs+remain_epoch)
    # lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_test_acc = {"source": 0, "test1": 0, "test2": 0}

    for epoch in range(max_epoch):
        lr_schedule.step(epoch=epoch)
        print(lr_schedule.get_lr())
        writer.add_scalar('lr', lr_schedule.get_lr(), epoch)

        model.train()
        loss_total = 0
        data_total = 0

        for batch_idx, batch_s in enumerate(source_train_dataloader):

            data, label = batch_s
            data = data.to(device=device)
            label = label.to(device=device).long()

            output_s = model(data)
            loss_s = criterion(output_s, label)
            loss_s.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_total += loss_s.item() * data.size(0)
            data_total += data.size(0)

            if (batch_idx + 1) % 10 == 0:
                print('Train:{} [{} /{}  loss: {:.4f} \t]'.format(
                epoch, data_total, num_source_train, loss_total/data_total))
                trained_epoch = epoch + 1
        
        if trained_epoch % args.ckpt_save_interval == 0:
            ckpt_list = [os.path.join(ckpt_dir, cpkt) for cpkt in os.listdir(ckpt_dir) if ".pth" in cpkt]
            ckpt_list.sort(key=os.path.getmtime)

            if ckpt_list.__len__() >= args.max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - args.max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name = os.path.join(ckpt_dir , ('checkpoint_epoch_%d' % trained_epoch) )
            print(f"Save current ckpt to {ckpt_name}")
            save_checkpoint(checkpoint_state(model, epoch=trained_epoch), filename=ckpt_name)

        with torch.no_grad():
            model.eval()

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
            

if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))

