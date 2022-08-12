# Soft-MMD DG
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_soft_mmd.yaml --set METHODS.TARGET_LOSS 0.5 METHODS.SRC_LOSS_WEIGHT 0.5
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_soft_mmd.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_soft_mmd.yaml
