#!/bin/bash
#SBATCH --job-name=mpi_multi_node
#SBATCH --output=multi_node_%j.out
#SBATCH --error=multi_node_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=20
#SBATCH --ntasks-per-node=10 # на каждой ноде (узле) будет по 10 MPI-процессов
#SBATCH --cpus-per-task=1
#SBATCH --distribution=block

srun --mpi=pmix \
     --cpu-bind=cores \
     ./hello_mpi
