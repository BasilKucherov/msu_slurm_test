#!/bin/bash
#SBATCH --job-name=mpi_single_node
#SBATCH --output=single_node_%j.out
#SBATCH --error=single_node_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --distribution=block


srun --mpi=pmix \
     --cpu-bind=cores \
     ./hello_mpi
