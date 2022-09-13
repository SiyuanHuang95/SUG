# Baseline DG
python ../../../train_dg.py --source scannet --cfg ../cfgs/cfgs_local/DG_soft_mmd_cluster.yaml
python ../../../train_dg.py --source modelnet  --cfg ../cfgs/cfgs_local/DG_soft_mmd_cluster.yaml
python ../../../train_dg.py --source shapenet  --cfg ../cfgs/cfgs_local/DG_soft_mmd_cluster.yaml
