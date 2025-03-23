#!/bin/bash
#SBATCH --job-name=mpi_multi_node
#SBATCH --output=multi_node_%A_%a.out
#SBATCH --error=multi_node_%A_%a.err
#SBATCH --nodes=3
#SBATCH --ntasks=24

srun --container-image=/scratch/$USER/nvcr.io+nvidia+pytorch+24.04-py3.sqsh \
     --container-mounts=/scratch/$USER/msu_slurm_test/mpi_hello_world:/workdir \
     --mpi=pmix \
     bash -c "
  cd /workdir
  mpirun ./hello_world
"
