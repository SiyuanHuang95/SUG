# Domain Generalization for 3D Point Cloud 

## TODO List

[X] Give the acc report on the another dataset

[ ] Set the data path with config

[x] Add config parse

[x] should bind the volume of logs/ with local host 

## Requirements

### Setup from scratch
- Python 3.8
- PyTorch 1.8.0
- others from requirements.txt

### Docker Environment
1. Ensure that you have installed Docker, NVIDIA-Container toolkit
2. Export data and log (optional) env variable, which will let the docker have the volume attach to the data and log folder.

```
    export DATA=/hdd1/huangsiyuan/PointDA_data/
    export LOG=/hdd1/huangsiyuan/logs/
```
3. To let the docker have the GPU access, make nvidia the default runtime in /etc/docker/daemon.json:
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
Save the file and run sudo systemctl restart docker to restart docker.
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


## Data Download
Download the [PointDA-10](https://drive.google.com/file/d/1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J/view?usp=sharing) and extract it as the dataset fold. 

## Train & Test
If you run the experiment on one generalization scanerio, like scannet to others,
```
python train.py -source scannet
```
, or run experiments on all adaptation scenarios.
```
bash main.sh
```
