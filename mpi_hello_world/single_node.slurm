#!/bin/bash
#SBATCH --job-name=mpi_single_node
#SBATCH --output=/scratch/$USER/msu_slurm_test/mpi_hello_world/single_node.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --container-image=/scratch/$USER/nvcr.io+nvidia+pytorch+24.04-py3.sqsh

cd /scratch/$USER/msu_slurm_test/mpi_hello_world
srun --mpi=pmix ./hello_mpi
