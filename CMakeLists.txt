cmake_minimum_required(VERSION 3.19)

project("BT: Álvaro Montaño Rosa" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Add MPI (Message Passing Interface)
option(USE_MPI "Use MPI library" ON)

# Add VTK (Visualization Toolkit)
option(USE_VTK "Use VTK library" OFF)

# Add OpenMP (Open Multi-Processing) 
option(USE_OPENMP "Use OpenMP library" ON)

# Add SLEPc (Scalable Library for Eigenvalue Problem Computations)
option(USE_SLEPc "Use SLEPc library" OFF)

# Add test environment
option(src_BUILD_TESTING "Use Check library" ON)

option(USE_KOKKOS "Use Kokkos" ON)


# Include coverage
option(ENABLE_COVERAGE "Use Check library" OFF)
if(ENABLE_COVERAGE)
  set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/CMake-codecov/cmake" ${CMAKE_MODULE_PATH})
endif()

SET(CMAKE_COLOR_MAKEFILE ON)
SET(CMAKE_VERBOSE_MAKEFILE OFF)
set(EXECUTABLE_OUTPUT_PATH "../")

# GNU specific settings
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
endif()

# Intel specific settings
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
endif()

# Clang specific settings
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")
endif()

# Check if Ninja
set(USED_CMAKE_GENERATOR "${CMAKE_GENERATOR}" CACHE STRING "Expose CMAKE_GENERATOR" FORCE)
if (USED_CMAKE_GENERATOR MATCHES "Ninja")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

# Check OS
IF (APPLE)
    message("MACOSX system")
    add_definitions("-DSYS_MACOSX")
    SET(BLA_VENDOR Apple)
ELSEIF (UNIX)
    message("UNIX system")
    add_definitions("-DSYS_LINUX")
    SET(BLA_VENDOR OpenBLAS)
ELSE ()
    message(FATAL_ERROR "Your system has not been recongnised." )
ENDIF ()

# Compile settings
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

if (CMAKE_BUILD_TYPE MATCHES "Debug")
message(STATUS "Debug mode")
add_definitions("-DDEBUG_MODE")
set(CMAKE_CXX_FLAGS_DEBUG "--coverage -O0 -g -Wall -Wpedantic -Wextra -Wunused-variable")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
message(STATUS "Release mode")
#set(CMAKE_CXX_FLAGS_RELEASE "-openmp --coverage -Ofast")
#set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
endif()

set(Kokkos_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Kokkos)

###############################################################################
## dependencies ###############################################################
###############################################################################

# Add math library
link_libraries(m)

# Add PkgConfig
find_package(PkgConfig REQUIRED)

# Add PETSc
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
if (PETSC_FOUND)
  link_libraries(PkgConfig::PETSC)
  message ("PETSC version linked: ${PETSC_VERSION}")
  if(${PETSC_VERSION} VERSION_GREATER_EQUAL "3.21.1")
    add_definitions("-DPETSC_NEW")
  else()
    add_definitions("-DPETSC_OLD")
  endif()
else (PETSC_FOUND)
    message (FATAL_ERROR "${PROJECT_NAME}: Unable to find the PETSC folder.")
endif(PETSC_FOUND)

# Add SLEPc
if (USE_SLEPc)
  pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET SLEPc)
  if (SLEPC_FOUND)
    link_libraries(PkgConfig::SLEPC)
    message ("SLEPc version linked: ${SLEPC_VERSION}")
    add_definitions("-DUSE_SLEPC")
  else (SLEPC_FOUND)
    message (FATAL_ERROR "${PROJECT_NAME}: Unable to find the SLEPc folder.")
  endif(SLEPC_FOUND)  
endif()

# Add MPI
if(USE_MPI)
  find_package(MPI REQUIRED)
  if (MPI_FOUND)
    add_definitions("-DUSE_MPI")
    include_directories(${MPI_CXX_INCLUDE_DIRS})
    link_libraries(${MPI_CXX_LIBRARIES})
  endif()
endif()

# Add OpenMP
if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
  if (OPENMP_FOUND)
    add_definitions("-DUSE_OPENMP")
    link_libraries(OpenMP::OpenMP_CXX)
  endif()
endif()

# Add VTK
if (USE_VTK)
  find_package(VTK REQUIRED)
  if (VTK_FOUND)
    add_definitions("-DUSE_VTK")
    include_directories(${VTK_INCLUDE_DIRS})
    link_libraries(${VTK_LIBRARIES})
  else (VTK_FOUND)
    message (FATAL_ERROR "${PROJECT_NAME}: Unable to find the VTK folder.")
  endif(VTK_FOUND)
endif(USE_VTK)

if(DEFINED PETSC_DIR AND DEFINED PETSC_ARCH)
  set(PETSC_INCLUDE_DIR "${PETSC_DIR}/${PETSC_ARCH}/include")
  message(STATUS "Usando directorio de includes de PETSc: ${PETSC_INCLUDE_DIR}")
  include_directories(${PETSC_INCLUDE_DIR})
else()
  message(WARNING "Las variables PETSC_DIR y/o PETSC_ARCH no están definidas.")
endif()

if(USE_KOKKOS)
  find_package(Kokkos CONFIG REQUIRED)
  #find_package(Kokkos CONFIG REQUIRED COMPONENTS cuda serial)
  add_definitions(-DUSE_KOKKOS=1)
else()
  add_definitions(-DUSE_KOKKOS=0)
endif()

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-relaxed-constexpr")
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -lineinfo")

###############################################################################
## Create executable ##########################################################
###############################################################################

# Include directories
include_directories(BEFORE
  src/
  external/eigen3
  /usr/lib/x86_64-linux-gnu/hdf5/serial/include
)

# Source code directories
file(GLOB_RECURSE src_src 
  src/ADP/MgHx-ADP.cpp 
  src/ADP/density.cpp
  src/Atoms/Atom.cpp    
  src/Atoms/Ghosts.cpp
  src/Atoms/Neighbors.cpp  
  src/Atoms/Topology.cpp 
  src/IO/dump-input.cpp
  src/IO/hdf5-io.cpp
  src/IO/dump-output.cpp
  src/Numerical/cubic-spline.cpp
  src/Numerical/Quadrature-Measure.cpp
  src/Numerical/Quadrature-Hermitian-3th.cpp
  src/Numerical/Quadrature-Hermitian-5th.cpp
  src/Numerical/Quadrature-Multipole.cpp      
  src/Periodic-Boundary/boundary_conditions.cpp
)

add_executable("exe-BT-AM" ${src_src} main-BT-AM.cpp)
target_link_libraries(exe-BT-AM PRIVATE Kokkos::kokkos)
target_compile_definitions("exe-BT-AM" PUBLIC -DGH3TH_INTEGRAL)
target_compile_definitions("exe-BT-AM" PUBLIC -DNUMERICAL_DERIVATIVES=0)
target_compile_definitions("exe-BT-AM" PUBLIC -DEIGEN_USE_GPU)
#target_compile_options(exe-BT-AM 
#  PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
#)
#target_compile_options(exe-BT-AM
#  PRIVATE 
#    --expt-relaxed-constexpr
#)