# Baseline DG
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_baseline.yaml --set METHODS.TARGET_LOSS 0.5 METHODS.SRC_LOSS_WEIGHT 0.5  EXTRA_TAG LOW_WEIGHT_SCANNET
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_baseline.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_baseline.yaml
