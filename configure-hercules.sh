#!/bin/bash

clear
module purge

## Generates standard UNIX makefiles.
PLATFORM="Unix Makefiles"

## Common modules
module load make/4.4.1-GCCcore-12.3.0

module load CMake/3.26.3-GCCcore-12.3.0

module load OpenMPI/4.1.5-GCC-12.3.0

module load GCC/12.3.0

module load Eigen/3.4.0-GCCcore-12.3.0

module load PETSc/3.23.2-kokkos-2025

## Export variables
export PKG_CONFIG_PATH=$PETSC_DIR/lib/pkgconfig

C_COMPILER=mpicc
CXX_COMPILER=mpicxx
MAKE=make

## If build does not exists, create it
BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
  mkdir ${BUILD_DIR}
fi

## Navigate inside of build
cd ${BUILD_DIR}

if [ -f "$FILE" ]; then
    ${MAKE} -k
else 
    cmake .. \
     -DCMAKE_BUILD_TYPE="Release" \
     -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
     -DUSE_KOKKOS=ON \
     -G "${PLATFORM}"
    
fi

if [[ "$PLATFORM" == "Unix Makefiles" ]]
then
${MAKE} -j8
elif [[ "$PLATFORM" == "Ninja" ]]
then
ninja
fi

cd ..

 