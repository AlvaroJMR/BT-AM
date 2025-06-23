#!/bin/bash
#SBATCH --job-name=run-BT-AM # Nombre del trabajo
#SBATCH --output=run-BT-AM-%j.out # Archivo de salida
#SBATCH --error=run-BT-AM-%j.err # Archivo de errores
#SBATCH --partition=standard # Partición donde ejecutar el trabajo
#SBATCH --nodes=2 # Número de nodos
#SBATCH --ntasks=18 # Número total de procesos MPI del trabajo
#SBATCH --cpus-per-task=10 # Número de hilos OpenMP por proceso MPI
#SBATCH --time=02:00:00 # Tiempo máximo de ejecución (horas:minutos:segundos)
#SBATCH --mem=18G # Memoria total por nodo

## Configure environments
module purge
module load GCC/12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load OpenMPI/4.1.5-GCC-12.3.0
module load Eigen/3.4.0-GCCcore-12.3.0
module load PETSc/3.23.3-foss-2023a-kokkos

MPI_RUN=mpiexec
MPI_P=${SLURM_NTASKS}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --mpi=pmix ./exe-BT-AM