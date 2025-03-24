#!/bin/bash
#SBATCH --job-name=pytorch_distributed
#SBATCH --output=pytorch_distributed_%j.out
#SBATCH --error=pytorch_distributed_%j.err
#SBATCH --nodes=2                  # Number of nodes
#SBATCH --ntasks-per-node=4        # Number of tasks per node (one task per GPU)
#SBATCH --cpus-per-task=4          # CPUs per task
#SBATCH --gpus-per-node=4          # GPUs per node
#SBATCH --time=02:00:00
#SBATCH --partition=batch

export MASTER_PORT=29500
export MASTER_ADDR=$(hostname -s)

WORKDIR="/scratch/${USER}/msu_slurm_test/pytorch_single_node"
CHECKPOINT_DIR="${WORKDIR}/checkpoints_distributed"
DATA_DIR="/scratch/${USER}/datasets/mnist"
ENROOT_IMAGE="/scratch/${USER}/nvcr.io+nvidia+pytorch+24.04-py3.sqsh"

NNODES=${SLURM_JOB_NUM_NODES}
NTASKS=${SLURM_NTASKS}

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${DATA_DIR}

CONTAINER_DATA_DIR="/workspace/datasets/mnist"

echo "Starting distributed training on ${NNODES} nodes with ${NTASKS} total tasks"
echo "Master node: ${MASTER_ADDR}"

srun --container-image=${ENROOT_IMAGE} \
     --container-mounts=/scratch/${USER}:/workspace \
     --container-workdir=/workspace/msu_slurm_test/pytorch_single_node \
     python train_mnist_distributed.py \
     --epochs=10 \
     --batch-size=64 \
     --checkpoint-dir=/workspace/msu_slurm_test/pytorch_single_node/checkpoints_distributed \
     --data-dir=${CONTAINER_DATA_DIR} \
     --backend=nccl

echo "Distributed training completed!" 
