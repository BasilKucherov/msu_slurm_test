#!/bin/bash
#SBATCH --job-name=pytorch_dataparallel
#SBATCH --output=pytorch_dataparallel_%j.out
#SBATCH --error=pytorch_dataparallel_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=batch

WORKDIR="/scratch/${USER}/msu_slurm_test/pytorch_single_node"
CHECKPOINT_DIR="${WORKDIR}/checkpoints_dataparallel"
DATA_DIR="/scratch/${USER}/datasets/mnist"
ENROOT_IMAGE="/scratch/${USER}/nvcr.io+nvidia+pytorch+24.04-py3.sqsh"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATA_DIR}

CONTAINER_DATA_DIR="/workspace/datasets/mnist"

echo "Starting DataParallel training with $(nvidia-smi --list-gpus | wc -l) GPUs"
srun --container-image=${ENROOT_IMAGE} \
     --container-mounts=/scratch/${USER}:/workspace \
     --container-workdir=/workspace/msu_slurm_test/pytorch_single_node \
     python train_mnist_dataparallel.py \
     --epochs=10 \
     --batch-size=512 \
     --checkpoint-dir=/workspace/msu_slurm_test/pytorch_single_node/checkpoints_dataparallel \
     --data-dir=${CONTAINER_DATA_DIR}

echo "DataParallel training completed!" 
