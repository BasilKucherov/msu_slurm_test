#!/bin/bash
#SBATCH --job-name=mpi_compile
#SBATCH --output=compile_%A_%a.out
#SBATCH --error=compile_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1

srun --container-image=/scratch/$USER/nvcr.io+nvidia+pytorch+24.04-py3.sqsh \
     --container-mounts=/scratch/$USER/msu_slurm_test/mpi_hello_world:/workdir \
     bash -c "
  cd /workdir
  make
"
