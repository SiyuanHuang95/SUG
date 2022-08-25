# max_mmd geo
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_max_mmd_geometric.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_max_mmd_geometric.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_max_mmd_geometric.yaml

# max_mmd geo_cls
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_max_mmd_geometric_cls.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_max_mmd_geometric_cls.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_max_mmd_geometric_cls.yaml

# soft geo
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_soft_mmd_geometric.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_soft_mmd_geometric.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_soft_mmd_geometric.yaml

# soft geo_cls
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_soft_mmd_geometric_cls.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_soft_mmd_geometric_cls.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_soft_mmd_geometric_cls.yaml


# max_mmd geo hist
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_max_mmd_geometric_hist.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_max_mmd_geometric_hist.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_max_mmd_geometric_hist.yaml

# max_mmd geo_cls
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_max_mmd_geometric_hist_cls.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_max_mmd_geometric_hist_cls.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_max_mmd_geometric_hist_cls.yaml

# soft geo
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_soft_mmd_geometric_hist.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_soft_mmd_geometric_hist.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_soft_mmd_geometric_hist.yaml

# soft geo_cls
python ../../train_dg.py --source scannet --cfg ../cfgs/DG_soft_mmd_geometric_hist_cls.yaml
python ../../train_dg.py --source modelnet  --cfg ../cfgs/DG_soft_mmd_geometric_hist_cls.yaml
python ../../train_dg.py --source shapenet  --cfg ../cfgs/DG_soft_mmd_geometric_hist_cls.yaml