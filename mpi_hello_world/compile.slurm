#!/bin/bash
#SBATCH --job-name=mpi_compile
#SBATCH --output=/scratch/$USER/msu_slurm_test/mpi_hello_world/compile.out
#SBATCH --container-image=/scratch/$USER/nvcr.io+nvidia+pytorch+24.04-py3.sqsh

cd /scratch/$USER/msu_slurm_test/mpi_hello_world
make
