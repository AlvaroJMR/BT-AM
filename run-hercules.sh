#!/bin/bash
#SBATCH --job-name=run-BT-AM # Nombre del trabajo
#SBATCH --output=run-BT-AM-%j.out # Archivo de salida
#SBATCH --error=run-BT-AM-%j.err # Archivo de errores
#SBATCH --partition=standard # Partición donde ejecutar el trabajo
#SBATCH --nodes=2 # Número de nodos
#SBATCH --ntasks=18 # Número total de procesos MPI del trabajo
#SBATCH --cpus-per-task=10 # Número de hilos OpenMP por proceso MPI
#SBATCH --mem-per-cpu=2G # Memoria por núcleo solicitado
#SBATCH --time=24:00:00 # Tiempo máximo de ejecución (horas:minutos:segundos)

## Configure environments
clear
module purge
module load GCC/12.3.0
module load CMake/3.26.3-GCCcore-12.3.0
module load PETSc/3.23.3-foss-2023a-kokkos

MPI_RUN=mpiexec
MPI_P=${SLURM_NTASKS}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

${MPI_RUN} -np ${MPI_P} ./exe-BT-AM