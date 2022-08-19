# Baseline DG
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_max_mmd_cluster_entropy.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_max_mmd_cluster_entropy.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_max_mmd_cluster_entropy.yaml
