#!/bin/bash

#SBATCH --partition=shlab_adg
#SBATCH -N 1
#SBATCH --mail-type=end
#SBATCH --mail-user=huangsiyuan@pjlab.org.cn
#SBATCH --output=/mnt/lustre/huangsiyuan/pointdg/logs_sproject/model_tune%j.out
#SBATCH --error=/mnt/lustre/huangsiyuan/pointdg/logs_sproject/model_tune%j.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=model_tune
#SBATCH  --quotatype=auto


echo "SLURM_JOBID: " $SLURM_JOBID
echo "Start Training"

# python ../../../train_dg.py --source modelnet  --cfg ../../cfgs/cfgs_sproject/DG_unified_loss_onedataset_modelnet.yaml --ckpt_save_interval 40 --set METHODS.CLS_WEIGHT 10.0
python ../../../train_dg.py --source modelnet  --cfg ../../cfgs/cfgs_sproject/DG_unified_loss_onedataset_modelnet_tune.yaml --ckpt_save_interval 40 --set METHODS.ADV_WEIGHT 0.5
# python ../../../train_dg.py --source modelnet  --cfg ../../cfgs/cfgs_sproject/DG_unified_loss_onedataset_modelnet_tune.yaml --ckpt_save_interval 40


