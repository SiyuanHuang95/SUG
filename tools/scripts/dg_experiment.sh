# Baseline DG
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_scannet_low_weight.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_baseline.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_baseline.yaml
