echo "Waiting"
# sleep 4h 
# Baseline DG gamma 2.0
echo "Start"

# python ../../train_dg.py --source shapenet  --cfg ../cfgs/shapenet2_shapenet.yaml --ckpt_save_interval 40
python ../../train_dg.py --source scannet  --cfg ../cfgs/shapenet2_scannet.yaml --ckpt_save_interval 40
# python ../../train_dg.py --source modelnet  --cfg ../cfgs/shapenet2_modelnet.yaml --ckpt_save_interval 40 

