python ../train.py -source scannet -scaler 1 -weight 1 -tb_log_dir ./logs/dg_sc2m -datadir /point_dg/data/
python ../train.py -source modelnet -scaler 1 -weight 0.5 -tb_log_dir ./logs/dg_m2sc -datadir /point_dg/data/
python ../train.py -source shapenet -scaler 1 -weight 1 -tb_log_dir ./logs/dg_sh2m -datadir /point_dg/data/
