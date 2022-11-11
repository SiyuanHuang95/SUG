# max_mmd geo
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_mix_spliter_soft.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_mix_spliter_soft.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_mix_spliter_soft.yaml
