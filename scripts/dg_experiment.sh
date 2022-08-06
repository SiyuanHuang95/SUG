# Baseline DG: Spliter Random (50%+ 50%(label)) + class-mmd
python ../train.py -source scannet -scaler 1 -weight 1 -tb_log_dir ./logs/dg_sc2m -datadir /point_dg/data/  -target_cls_loss 1 -class_mmd  2>&1 | tee /point_dg/workspace/logs/20220806_dg_scannet.txt
python ../train.py -source modelnet -scaler 1 -weight 0.5 -tb_log_dir ./logs/dg_m2sc -datadir /point_dg/data/  -target_cls_loss 1  -class_mmd    2>&1 | tee /point_dg/workspace/logs/20220806_dg_modelnet.txt
python ../train.py -source shapenet -scaler 1 -weight 1 -tb_log_dir ./logs/dg_sh2m -datadir /point_dg/data/   -target_cls_loss 1   -class_mmd   2>&1 | tee /point_dg/workspace/logs/20220806_dg_shapenet.txt
