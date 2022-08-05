python ../train_source.py -source scannet -target1 modelnet -target2 shapenet -datadir /point_dg/data/  -tb_log_dir ./logs/source_scannet
python ../train_source.py -source shapenet -target1 scannet -target2 modelnet -datadir /point_dg/data/  -tb_log_dir ./logs/source_shapenet
python ../train_source.py -source modelnet -target1 scannet -target2 shapenet -datadir /point_dg/data/  -tb_log_dir ./logs/source_modelnet