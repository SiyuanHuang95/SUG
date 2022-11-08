import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.model_pointnet import Pointnet_cls, Pointnet2_cls, DGCNN
from model.Ptran_model import PointTransformerCls
from model.KPConv_model import KPFCls, p2p_fitting_regularizer
from data.dataloader import Modelnet40_data, Shapenet_data, Scannet_data_h5
from data.dataloader import create_single_dataset

import time
import os
import copy
import datetime

from tensorboardX import SummaryWriter
from utils.eval_utils import eval_worker
from utils.train_utils import save_checkpoint, checkpoint_state, adjust_learning_rate, discrepancy
from utils.common_utils import create_logger, exp_log_folder_creator
from utils.config import parser_config, log_config_to_file
from utils.train_files_spliter import dataset_list

def main():
    args, cfg = parser_config()

    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    BATCH_SIZE = args.batch_size * len(args.gpu.split(','))

    output_dir, ckpt_dir = exp_log_folder_creator(cfg, extra_tag=args.source)
    log_name = 'log_train_source%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(output_dir, log_name)
    logger = create_logger(log_file=log_file)

    logger.info('**********************Start logging**********************')
    if not os.path.exists(os.path.join(output_dir, 'tensorboard')):
        os.makedirs(os.path.join(output_dir,'tensorboard'))
    writer = SummaryWriter(log_dir=str(os.path.join(output_dir,'tensorboard')))

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    logger.info('Start Training\nInitiliazing\n')
    logger.info(f'The source domain is set to: {args.source}')

    test_datasets = list(set(dataset_list) - {args.source})
    logger.info(f'The datasets used for testing: {test_datasets}')

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

    logger.info('num_source_train: {:d}, num_source_test: {:d}, num_target_test1: {:d}, num_target_test2: {:d}'.format(
        num_source_train, num_source_test, num_target_test1, num_target_test2))
    logger.info(f'batch_size: {BATCH_SIZE}')

    num_cls = cfg["DATASET"]["NUM_CLASS"]
    # Model
    if cfg.get("Model", "PointNet") == "PointNet2":
        model = Pointnet2_cls(num_class=num_cls)
    elif cfg.get("Model", "PointNet") == "DGCNN":
        model = DGCNN()
    elif cfg.get("Model", "PointNet") == "PTran":
        model = PointTransformerCls()
    elif cfg.get("Model", "PointNet") == "KPConv":
        model = KPFCls()
    else:
        model = Pointnet_cls(num_class=num_cls)
    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=device)

    # Optimizer
    remain_epoch=50
    max_epoch_num = cfg["OPTIMIZATION"]["NUM_EPOCHES"]
    LR = cfg["OPTIMIZATION"]["LR"]
    weight_decay = cfg["OPTIMIZATION"]["WEIGHT_DECAY"]

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch_num+remain_epoch)
    # lr_schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_test_acc = {"source": [0, 0], "test1":[0, 0], "test2":[0, 0]}
    dataset_remapping = {"source":args.source, "test1": test_datasets[0],
                             "test2": test_datasets[1]}
    # best_target_acc_epoch + best_target_acc

    for epoch in range(max_epoch_num):
        since_e = time.time()
        lr_schedule.step(epoch=epoch)
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

            if cfg.get("Model", "PointNet") == "KPConv":
                reg_loss = p2p_fitting_regularizer(model.encoder.encoder_blocks, deform_fitting_power=model.deform_fitting_power)
                loss_s += reg_loss
                
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
            logger.info(f"Save current ckpt to {ckpt_name}")
            save_checkpoint(checkpoint_state(model, epoch=trained_epoch), filename=ckpt_name)

        with torch.no_grad():
            model.eval()
            for eval_dataset in performance_test_sets.keys():
                eval_dict = {
                    "model": copy.deepcopy(model),
                    "dataloader": performance_test_sets[eval_dataset],
                    "dataset": eval_dataset,
                    "best_target_acc": best_test_acc[eval_dataset][1],
                    "device": device,
                    "criterion": criterion,
                    "epoch": epoch,
                    "best_target_acc_epoch": best_test_acc[eval_dataset][0],
                    "dataset_name": dataset_remapping[eval_dataset],
                    "num_class": cfg["DATASET"]["NUM_CLASS"],
                    "source_flag": True
                }
                eval_result = eval_worker(eval_dict, logger)
                best_test_acc[eval_dataset][1] = eval_result["best_target_acc"]
                best_test_acc[eval_dataset][0] = eval_result["best_target_acc_epoch"]
                writer_item = 'acc/' + eval_result["dataset"] + "_test_acc"
                writer.add_scalar(writer_item, eval_result["best_target_acc"], epoch)
        
        time_pass_e = time.time() - since_e
        logger.info('The {} epoch takes {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 60, time_pass_e % 60))
        logger.info('****************Finished One Epoch****************')
            

if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))

