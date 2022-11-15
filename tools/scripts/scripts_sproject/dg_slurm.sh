#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
NNum=$3
SP_TYPE=$4
PY_ARGS=${@:5}

# GPUS_PER_NODE=${GPUS_PER_NODE:-4}
# CPUS_PER_TASK=${CPUS_PER_TASK:-8}

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun -p ${PARTITION} \
    -N ${NNum} \
    --quotatype=${SP_TYPE} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ../../../train_dg.py --launcher slurm --tcp_port $PORT ${PY_ARGS}

# sh dg_slurm.sh shlab_adg_s2 dg 1 auto --cfg ../../cfgs/cfgs_sproject/DG_unified_loss_onedataset_modelnet.yaml --ckpt_save_interval 50 
