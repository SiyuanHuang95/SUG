# 3D Transfer Learning - PointDAN

This repo contains the source code and dataset for our NeurIPS 2019 paper:

[**PointDAN: A Multi-Scale 3D Domain Adaption Network for Point Cloud Representation**](http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation)
<br>
2019 Conference on Neural Information Processing Systems (NeurIPS 2019)
<br>
[paper](http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation),
[arXiv](https://arxiv.org/abs/1911.02744),
[bibtex](http://papers.nips.cc/paper/8940-pointdan-a-multi-scale-3d-domain-adaption-network-for-point-cloud-representation/bibtex)

![PointDAN](/Figs/PointDAN.png)


## Dataset
![PointDA-10](/Figs/PointDA-10.png)
The [PointDA-10](https://drive.google.com/file/d/1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J/view?usp=sharing) dataset is extracted from three popular 3D object/scene datasets (i.e., [ModelNet](https://modelnet.cs.princeton.edu/), [ShapeNet](https://shapenet.cs.stanford.edu/iccv17/) and [ScanNet](http://www.scan-net.org/)) for cross-domain 3D objects classification.

The new version of PointDA dataset will come soon!

## Requirements
- Python 3.6
- PyTorch 1.0

### Docker Environment
1. Ensure that you have installed Docker, NVIDIA-Container toolkit
2. Export data env variable, which will let the docker have the volume attach to the data folder.

```
    export DATA=/hdd1/huangsiyuan/PointDA_data/
```
3. To let the docker have the GPU access:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        } 
    },
    "default-runtime": "nvidia" 
}

```
4. build the docker:

```
make build
```

5. Finally, run the container with
```
make run
```

6. If you meet the problem:
```
ERROR: Couldn’t connect to Docker daemon at http+docker://localunixsocket - is it running?
```
- sudo gpasswd -a ${USER} docker
- sudo su
- su huangsiyuan

## File Structure
```
.
├── README.md
├── logs                            
├── dataset
│   └──PointDA_data                              
│      ├── modelnet                      
│      ├── scannet
│      └── shapenet             
├── dataloader.py
├── data_utils.py
├── main.sh
├── mmd.py
├── Model.py
├── model_utils.py
├── train.py            
└── train_source.py                                   
```

## Data Download
Download the [PointDA-10](https://drive.google.com/file/d/1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J/view?usp=sharing) and extract it as the dataset fold.

## Train & Test
If you run the experiment on one adaptation scanerio, like scannet to modelnet,
```
python train.py -source scannet -target modelnet
```
, or run experiments on all adaptation scenarios.
```
bash main.sh
```
