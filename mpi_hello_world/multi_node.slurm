#!/bin/bash
#SBATCH --job-name=mpi_multi_node
#SBATCH --output=/scratch/$USER/msu_slurm_test/mpi_hello_world/multi_node.out
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=8
#SBATCH --container-image=/scratch/$USER/nvcr.io+nvidia+pytorch+24.04-py3.sqsh

cd /scratch/$USER/msu_slurm_test/mpi_hello_world
srun --mpi=pmix ./hello_mpi
