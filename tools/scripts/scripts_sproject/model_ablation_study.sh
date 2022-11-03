#!/bin/bash

#SBATCH --partition=shlab_adg_s2
#SBATCH -N 1

#SBATCH --output=/mnt/petrelfs/huangsiyuan/PointDG/logs_sproject/model_pn_%j.out
#SBATCH --error=/mnt/petrelfs/huangsiyuan/PointDG/logs_sproject/model_pn_%j.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=model_pn
#SBATCH  --quotatype=auto
#SBATCH --cpus-per-task=4

echo "SLURM_JOBID: " $SLURM_JOBID
echo "Start Training"

# python ../../../train_dg.py --source modelnet  --cfg ../../cfgs/cfgs_sproject/DG_unified_loss_onedataset_modelnet.yaml --ckpt_save_interval 40 --set METHODS.CLS_WEIGHT 10.0
python ../../../train_dg_single_gpu.py --source modelnet  --cfg ../../cfgs/cfgs_sproject/DG_unified_loss_onedataset_modelnet_pointnet_ra_sub.yaml --ckpt_save_interval 40 --batch_size 64 --set Model Pointnet 
# python ../../../train_dg.py --source modelnet  --cfg ../../cfgs/cfgs_sproject/DG_unified_loss_onedataset_modelnet_tune.yaml --ckpt_save_interval 40


