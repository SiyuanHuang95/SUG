import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from easydict import EasyDict

from model.model_pointnet import Pointnet_cls as Pointnet_cls
import model.Model as mM
import time
import os
import glob
import pdb
import model.mmd as mmd
# from utils import *
import math
import warnings
import datetime
import copy
from utils.eval_utils import eval_worker
from utils.train_utils import save_checkpoint, checkpoint_state, adjust_learning_rate, discrepancy, Sampler
from model.model_utils import focal_loss
from utils.common_utils import create_logger, exp_log_folder_creator, set_random_seed
from utils.config import parser_config, log_config_to_file
from data.dataloader import create_splitted_dataset, create_single_dataset


from tensorboardX import SummaryWriter

warnings.filterwarnings("ignore")

def main():
    args, cfg = parser_config()

    if args.fix_random_seed:
        set_random_seed(666 + cfg.LOCAL_RANK)

    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    BATCH_SIZE = args.batch_size * len(args.gpu.split(','))

    output_dir, ckpt_dir = exp_log_folder_creator(cfg, extra_tag=args.source)
    log_name = 'log_train_dg%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
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

    dataset_list = ["scannet", "shapenet", "modelnet"]
    test_datasets = list(set(dataset_list) - {args.source})
    logger.info(f'The datasets used for testing: {test_datasets}')

    # Data loading
    multi_spliter = False
    # when split_config is a dict which means only one split method is used
    split_config = cfg["DATASET_SPLITTER"]
    if type(split_config) is EasyDict:
        source_train_subsets = create_splitted_dataset(dataset_type=args.source, status="train", logger=logger, config=split_config)
        source_train_dataset = source_train_subsets[split_config["TRAIN_BASE"]]
        target_train_dataset1 = source_train_subsets[1-split_config["TRAIN_BASE"]]
    elif type(split_config) is list:
        logger.info(f"{len(split_config)} types split methods are used.")
        multi_spliter = True
        source_train_datasets = []
        target_train_datasets = []

        for config_ in split_config:
            source_train_subsets = create_splitted_dataset(dataset_type=args.source, status="train", logger=logger, config=config_)
            source_train_datasets.append(source_train_subsets[config_["TRAIN_BASE"]])
            target_train_datasets.append(source_train_subsets[1-config_["TRAIN_BASE"]])
    else:
        raise RuntimeError(f"Unsupported Splitter Config {type(split_config)}")
    # split 2 is fullsize

    source_test_dataset = create_single_dataset(args.source, status="test", aug=False)
    target_test_dataset1 = create_single_dataset(test_datasets[0], status="test", aug=False)
    target_test_dataset2 = create_single_dataset(test_datasets[-1], status="test", aug=False)

    if not multi_spliter:
        num_source_train = len(source_train_dataset)
        num_target_train1 = len(target_train_dataset1)
        logger.info(f"Num of source train: {num_source_train}, Num of target train: {num_target_train1}")

        if cfg.get("CLASS_BALANCE", False):
            sampler_1 = Sampler(source_train_dataset.classes(), class_per_batch=10, batch_size=BATCH_SIZE)
            sampler_2 = Sampler(target_train_dataset1.classes(), class_per_batch=10, batch_size=BATCH_SIZE)
            source_train_dataloader = DataLoader(source_train_dataset, batch_sampler=sampler_1, num_workers=2)
            target_train_dataloader = DataLoader(target_train_dataset1, batch_sampler=sampler_2, num_workers=2)
        else:
            source_train_dataloader = DataLoader(source_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                                drop_last=True)
            target_train_dataloader = DataLoader(target_train_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                                drop_last=True)
    else:
        source_train_dataloaders = []
        target_train_dataloaders = []
        for st,tt in zip(source_train_datasets, target_train_datasets):
            num_source_train = len(st)
            num_target_train1 = len(tt)
            logger.info(f"For Current Split Method: Num of source train: {num_source_train}, Num of target train: {num_target_train1}")

            if cfg.get("CLASS_BALANCE", False):
                sampler_1 = Sampler(st.classes(), class_per_batch=10, batch_size=BATCH_SIZE)
                sampler_2 = Sampler(tt.classes(), class_per_batch=10, batch_size=BATCH_SIZE)
                source_train_dataloader = DataLoader(st, batch_sampler=sampler_1, num_workers=2)
                target_train_dataloader = DataLoader(tt, batch_sampler=sampler_2, num_workers=2)
            else:
                source_train_dataloader = DataLoader(st, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                                    drop_last=True)
                target_train_dataloader = DataLoader(tt, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                                    drop_last=True)

            source_train_dataloaders.append(source_train_dataloader)
            target_train_dataloaders.append(target_train_dataloader)

    num_source_test = len(source_test_dataset)
    num_target_test1 = len(target_test_dataset1)
    num_target_test2 = len(target_test_dataset2)
    logger.info(f"Num of source test: {num_source_test}, Num of test on {test_datasets[0]} {num_target_test1}, on {test_datasets[-1]} {num_target_test2}")

    

    source_test_dataloader = DataLoader(source_test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                        drop_last=True)
    target_test_dataloader1 = DataLoader(target_test_dataset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                         drop_last=True)
    target_test_dataloader2 = DataLoader(target_test_dataset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                                         drop_last=True)
    performance_test_sets = {"source": source_test_dataloader, "test1": target_test_dataloader1,
                             "test2": target_test_dataloader2}

    logger.info(f'batch_size: {BATCH_SIZE}')

    best_test_acc = {"source": [0, 0], "test1":[0, 0], "test2":[0, 0]}
    # best_target_acc_epoch + best_target_acc
    dataset_remapping = {"source":args.source, "test1": test_datasets[0],
                             "test2": test_datasets[1]}
    # pool_eval = Pool(processes=len(dataset_list))
    # AssertionError: daemonic processes are not allowed to have children

    # Model
    model = mM.Net_MDA(model_name=cfg.get("Model", "Pointnet"))
    logger.info(model)
    model = model.to(device=device)
    KPC_Flag = cfg["Model"] == "KPConv"

    opt_cfg = cfg["OPTIMIZATION"]

    if opt_cfg.get("CLS_LOSS", "CrossEntropyLoss") == "FocalLoss":
        cls_weights=None
        if opt_cfg.get("CLS_WEIGHT", None):
            cls_weights = source_train_dataset.cls_wights(weighting=opt_cfg["CLS_WEIGHT"])
        criterion = focal_loss(num_classes=cfg["DATASET"]["NUM_CLASS"], gamma=opt_cfg["FOCAL_GAMMA"], alpha=cls_weights)
        logger.info(f"FocalLoss: alpha {cls_weights}")
        logger.info(f"FocalLoss: gamma {opt_cfg['FOCAL_GAMMA']}")
    elif opt_cfg.get("CLS_LOSS", "CrossEntropyLoss") == "ClassWeighting":
        gamma = 0.0
        # when gamma is zero, FL degrades to class re-weighting
        if not opt_cfg.get("CLS_WEIGHT", None):
            raise RuntimeError("When setting ClassWeighting, CLS_WEIGHT should be provided")
        cls_weights = source_train_dataset.cls_wights(weighting=opt_cfg["CLS_WEIGHT"], q_=opt_cfg.get("DLSA_Q", None))
        criterion = focal_loss(num_classes=cfg["DATASET"]["NUM_CLASS"], gamma=gamma, alpha=cls_weights)
        logger.info(f"ClassWeighting: Weights: {cls_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device=device)

    # Optimizer Setting
    remain_epoch = 50
    max_epoch_num = opt_cfg["NUM_EPOCHES"]
    LR = opt_cfg["LR"]
    weight_decay = opt_cfg["WEIGHT_DECAY"]
    scaler = opt_cfg["LR_SCALER"]
    pure_cls_epoch = cfg["METHODS"]["PURE_CLS_EPOCH"]

    params = [{'params': v} for k, v in model.g.named_parameters() if 'pred_offset' not in k]

    optimizer_g = optim.Adam(params, lr=LR, weight_decay=weight_decay)
    lr_schedule_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=max_epoch_num + remain_epoch)

    optimizer_c = optim.Adam([{'params': model.c1.parameters()}, {'params': model.c2.parameters()}], lr=LR * 2,
                             weight_decay=weight_decay)
    lr_schedule_c = optim.lr_scheduler.CosineAnnealingLR(optimizer_c, T_max=max_epoch_num + remain_epoch)

    optimizer_dis = optim.Adam([{'params': model.g.parameters()}, {'params': model.attention_s.parameters()},
                                {'params': model.attention_t.parameters()}],
                               lr=LR * scaler, weight_decay=weight_decay)
    lr_schedule_dis = optim.lr_scheduler.CosineAnnealingLR(optimizer_dis, T_max=max_epoch_num + remain_epoch)

    cls_eval = opt_cfg.get("CLS_EVAL", True)

    for epoch in range(max_epoch_num):
        since_e = time.time()

        lr_schedule_g.step(epoch=epoch)
        lr_schedule_c.step(epoch=epoch)
        adjust_learning_rate(optimizer_dis, epoch, LR, scaler, writer)

        writer.add_scalar('lr_g', lr_schedule_g.get_lr()[0], epoch)
        writer.add_scalar('lr_c', lr_schedule_c.get_lr()[0], epoch)

        model.train()

        loss_cls_total = 0
        loss_adv_total = 0
        loss_geo_total = 0
        loss_sem_total = 0
        correct_total = 0
        data_total = 0
        data_t_total = 0
        cons = math.sin((epoch + 1) / max_epoch_num * math.pi / 2)

        # choose for current epoch, which split method will be used
        if multi_spliter:
            idx = epoch % len(source_train_dataloaders)
            source_train_dataloader = source_train_dataloaders[idx]
            target_train_dataloader = target_train_dataloaders[idx]
        
        # Training
        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_train_dataloader, target_train_dataloader)):
            data, label = batch_s
            data_t, label_t = batch_t

            # 64 * 3 * 1024
            data = data.to(device=device)
            # label： 64
            label = label.to(device=device).long()
            data_t = data_t.to(device=device)
            label_t = label_t.to(device=device).long()
            

            # Senmantic MMD loss
            # data: 64 * 3 * 1024 * 1
            pred_s1, pred_s2, sem_fea_s1, sem_fea_s2 = model(data, semantic_adaption=True)
            if cfg["METHODS"].get("GRL", None):
                pred_t1, pred_t2, sem_fea_t1, sem_fea_t2 = model(data_t, semantic_adaption=True, constant=cons, adaptation=True)
            else:
                pred_t1, pred_t2, sem_fea_t1, sem_fea_t2 = model(data_t, semantic_adaption=True)
            # no need for GRL now
            # sem_fea_s1:64 * 256

            # Classification loss
            loss_s1 = criterion(pred_s1, label)
            loss_s2 = criterion(pred_s2, label)

            # Adversarial loss -> let two heads of the model output similar
            loss_adv = - cfg["METHODS"]["ADV_WEIGHT"] * discrepancy(pred_t1, pred_t2)
            # TODO Ablation to check wether need add loss_adv

            loss_s = loss_s1 + loss_s2
            if cfg["METHODS"]["TARGET_LOSS"] > 0:
                loss_t1 = criterion(pred_t1, label)
                loss_t2 = criterion(pred_t2, label)
                loss_t = loss_t1 + loss_t2
                loss = cfg["METHODS"]["SRC_LOSS_WEIGHT"] * loss_s + loss_adv + cfg["METHODS"]["TARGET_LOSS"] * loss_t
            else:
                loss = cfg["METHODS"]["SRC_LOSS_WEIGHT"] * loss_s + loss_adv

            loss_cls = cfg["METHODS"]["CLS_WEIGHT"] * loss
            if epoch < pure_cls_epoch or cfg["METHODS"]["MMD_WEIGHT"] <= 0:
                loss = loss_cls
                loss.backward()
                optimizer_dis.step()
                optimizer_g.step()
                optimizer_c.step()
                optimizer_g.zero_grad()
                optimizer_c.zero_grad()
                optimizer_dis.zero_grad()
                loss_cls_total += loss_cls.item() * data.size(0)
            else:

                # should only activate the MMD loss after some epoch, to let the classifier gain the basic ability for classifying            
                # ******************************** MMD alignment part *****************************************
                # geometric info
                # Local Alignment -> self-adaptive node: contains geometry info
                feat_node_s = model(data, node_adaptation_s=True)  # shape: batch_size * 4096 -> 64 * 64
                feat_node_t = model(data_t, node_adaptation_t=True)
                # Add geometric weights
                geo_mmd_cfg = cfg["METHODS"]["GEO_MMD"][0]
                loss_geo_mmd =  cfg["METHODS"]["MMD_WEIGHT"] * geo_mmd_cfg["GEO_SCALE"] * mmd.mmd_cal(label, feat_node_s, label_t, feat_node_t, geo_mmd_cfg, data_s=data, data_t=data_t, KPC=KPC_Flag)           
                
                sem_mmd_cfg = cfg["METHODS"]["SEM_MMD"][0]
                loss_sem_mmd = None
                if  sem_mmd_cfg["SEM_SCALE"] > 0:
                    loss_sem_mmd_1 = sem_mmd_cfg["SEM_SCALE"] * mmd.mmd_cal(label, sem_fea_s1, label_t, sem_fea_t1, sem_mmd_cfg, data_s=pred_s1, data_t=pred_t1, KPC=KPC_Flag)
                    loss_sem_mmd_2 = sem_mmd_cfg["SEM_SCALE"] * mmd.mmd_cal(label, sem_fea_s2, label_t, sem_fea_t2, sem_mmd_cfg, data_s=pred_s2, data_t=pred_t2, KPC=KPC_Flag)
                    loss_sem_mmd = cfg["METHODS"]["MMD_WEIGHT"] * (0.5 * loss_sem_mmd_1 + 0.5 * loss_sem_mmd_2)

                if loss_sem_mmd is not None:
                    loss = loss_cls + loss_geo_mmd + loss_sem_mmd
                else:
                    loss = loss_cls + loss_geo_mmd

                loss.backward()
                optimizer_dis.step()
                optimizer_g.step()
                optimizer_c.step()
                optimizer_g.zero_grad()
                optimizer_c.zero_grad()
                optimizer_dis.zero_grad()

                
                loss_geo_total += loss_geo_mmd.item() * data.size(0)
                if loss_sem_mmd is not None:
                    loss_sem_total += loss_sem_mmd.item() * data.size(0)

            data_total += data.size(0)
            data_t_total += data_t.size(0)

            loss_cls_total += loss_cls.item() * data.size(0)
            loss_adv_total += loss_adv.item() * data.size(0)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Train Epoch {epoch} [{data_total} {data_t_total}/{num_source_train}:] loss_cls {loss_cls_total / data_total} ")
                if epoch >= pure_cls_epoch:
                    logger.info(f"loss_adv: {loss_adv_total / data_total} loss_geo_mmd {loss_geo_total / data_total} loss_sem_mmd {loss_sem_total / data_total}")

        # Testing
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
                    "cls_eval": cls_eval
                }
                eval_result = eval_worker(eval_dict, logger)
                best_test_acc[eval_dataset][1] = eval_result["best_target_acc"]
                best_test_acc[eval_dataset][0] = eval_result["best_target_acc_epoch"]
                writer_item = 'acc/' + eval_result["dataset"] + "_" + dataset_remapping[eval_result["dataset"]] + "_best_acc"
                writer.add_scalar(writer_item, eval_result["best_target_acc"], epoch)

                writer_item = 'acc/' + eval_result["dataset"] + "_" + dataset_remapping[eval_result["dataset"]] + "_cur_acc"
                writer.add_scalar(writer_item, eval_result["cur_target_acc"], epoch)

        trained_epoch = epoch + 1
        if trained_epoch % args.ckpt_save_interval == 0:
            ckpt_list = [os.path.join(ckpt_dir, cpkt) for cpkt in os.listdir(ckpt_dir) if ".pth" in cpkt]
            ckpt_list.sort(key=os.path.getmtime)
            if ckpt_list.__len__() >= args.max_ckpt_save_num:
                for cur_file_idx in range(0, len(ckpt_list) - args.max_ckpt_save_num + 1):
                    os.remove(ckpt_list[cur_file_idx])

            ckpt_name = os.path.join(ckpt_dir, args.source + ('_checkpoint_epoch_%d' % trained_epoch))
            logger.info(f"Save current ckpt to {ckpt_name}")
            save_checkpoint(checkpoint_state(model, epoch=trained_epoch), filename=ckpt_name)

        time_pass_e = time.time() - since_e
        logger.info('The {} epoch takes {:.0f}m {:.0f}s'.format(epoch, time_pass_e // 60, time_pass_e % 60))
        logger.info('****************Finished One Epoch****************')


if __name__ == '__main__':
    since = time.time()
    main()
    time_pass = since - time.time()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_pass // 60, time_pass % 60))