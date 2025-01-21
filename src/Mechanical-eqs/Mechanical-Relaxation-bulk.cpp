/**
 * @file Mechanical-eqs/min-V/Mechanical-Relaxation-bulk.cpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief This function performs the mechanical relaxation of a periodic sistem
 * of atomic positions under zero temperature.
 * @version 0.1
 * @date 2023-07-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cstdlib>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "ADP/MgHx-mf-V-bulk.hpp"
#include "Atoms/Atom.hpp"
#include "Atoms/Ghosts.hpp"
#include "Atoms/Neighbors.hpp"
#include "Atoms/Topology.hpp"
#include "Macros.hpp"
#include "petscdm.h"
#include "petscdmda.h"
#include "petscdmlabel.h"
#include <Eigen/Dense>
#include <ctime>
#include <fstream>
#include <iomanip>  // to print more decimals
#include <iostream>
#include <math.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

typedef std::numeric_limits<double> dbl;

extern double petsc_abstol;
extern double petsc_rtol;
extern double petsc_stol;
extern double petsc_maxit;
extern double petsc_maxf;

extern char petsc_ngmres_m[];

extern char petsc_linesearch_minlambda[];
extern char petsc_linesearch_damping[];
extern char petsc_linesearch_max_it[];

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern Eigen::Vector3d box_origin_0;

extern Eigen::Vector3d lattice_x_B0;
extern Eigen::Vector3d lattice_y_B0;
extern Eigen::Vector3d lattice_z_B0;

extern Eigen::Vector3d lattice_x_Bn;
extern Eigen::Vector3d lattice_y_Bn;
extern Eigen::Vector3d lattice_z_Bn;

extern Eigen::Vector3d local_domain_ll;
extern Eigen::Vector3d local_domain_ur;

/*! User-context for PETSc */
struct Equilibrium_dV_dF_ctx {

  /*! @param n_sites_local: Number of local sites */
  PetscInt n_sites_local;

  /*! @param n_sites_local_ghosted: Number of local sites including ghosts */
  PetscInt n_sites_local_ghosted;

  /*! @param n_sites_ghost: Number of ghost sites */
  PetscInt n_sites_ghost;

  /*! @param n_mechanical_sites_local: Get local number of mechanical atoms
   * (excluding ghosts) */
  PetscInt n_mechanical_sites_local;

  /*! @param idx_q_ptr: */
  PetscInt* idx_q_ptr;

  /*! @param mechanical_neighs_idx: mechanical neighs of each site */
  IS* mechanical_neighs_idx;

  /*! @param active_mech_sites: List of active mechanical sites */
  IS active_mech_sites;

  /*! @param mean_q_ptr: Pointer to the mean position of the atoms */
  PetscScalar* mean_q_ptr;

  /*! @param mf_rho_ptr: Meanfiel value of the energy density */
  PetscScalar* mf_rho_ptr;

  /*! @param xi_ptr: Pointer to the mean occupancy of the atoms */
  PetscScalar* xi_ptr;

  /*! @param specie_ptr: Pointer to the speci indicator of the atoms */
  AtomicSpecie* specie_ptr;

  /*! @param system_equations Definition of the equation */
  dmd_equations system_equations;
};

static PetscErrorCode evaluate_RHS(SNES snes, Vec X, Vec Y, void* ctx);

static PetscErrorCode monitor_equilibrium(SNES snes, PetscInt its,
                                          PetscReal fnorm, void* ctx);

/************************************************************************/

PetscErrorCode mechanical_relaxation_bulk(DMD* Simulation,
                                          dmd_equations system_equations) {

  PetscFunctionBeginUser;

  unsigned int dim = NumberDimensions;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get atomistic topology
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  //! Get local number of sites in the simulation (without ghost)
  PetscInt n_sites_local = Simulation->n_sites_local;

  //! Get local number of sites in the simulation (with ghost)
  PetscInt n_sites_local_ghosted;
  PetscCall(
      DMSwarmGetLocalSize(Simulation->atomistic_data, &n_sites_local_ghosted));

  //! Get number of ghost particles
  PetscInt n_sites_ghost = n_sites_local_ghosted - n_sites_local;

  //! Get number of mechanical sites in the local domain (excluding ghosts)
  PetscInt n_mechanical_sites_local = Simulation->n_mechanical_sites_local;

  //! Get list of mechanical sites in the local domain (including ghosts)
  IS active_mech_sites = Simulation->active_mech_sites;

  //! Get list of mechanical neighbors
  IS* mechanical_neighs_idx = Simulation->mechanical_neighs_idx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get copies of the mean position vector all over the ranks
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar* mean_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, DMSwarmPICField_coor,
                            NULL, NULL, (void**)&mean_q_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get local pointer of the meanfield energy density
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar* mf_rho_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "mf-rho", NULL, NULL,
                            (void**)&mf_rho_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the molar fraction vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscScalar* xi_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "molar-fraction", NULL,
                            NULL, (void**)&xi_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the atomic specie vector all over the ranks
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  AtomicSpecie* specie_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "specie", NULL, NULL,
                            (void**)&specie_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Get index of the particles
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt* idx_q_ptr;
  PetscCall(DMSwarmGetField(Simulation->atomistic_data, "idx", NULL, NULL,
                            (void**)&idx_q_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Define PETSc variables for the mechanical equilibrium
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  SNES snes;  //! Nonlinear solver context
  KSP ksp;    //! linear solver context
  PC pc;      //! preconditioner context
  Vec X;      //! Solution vector X := {F}
  Vec Y;      //! Residual vector Y := {dV-dF}
  Mat J;      //! Jacobian matrix J := {d2V-dF}
  SNESLineSearch linesearch;
  PetscInt SNES_iterations;

  // absolute convergence tolerance
  PetscReal abstol = 1e-12;
  // relative convergence tolerance
  PetscReal rtol = 1e-10;
  // convergence tolerance in terms of the norm of the change in
  // the solution between steps, || delta x || < stol*|| x ||
  PetscReal stol = petsc_abstol;
  // maximum number of iterations
  PetscInt maxit = 20;
  // maximum number of function evaluations
  PetscInt maxf = petsc_maxf;
  // Set a reason for convergence/divergence of solver (SNES)
  SNESConvergedReason SNES_reason;
  const char* SNES_strreason;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    User-defined context
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  struct Equilibrium_dV_dF_ctx ctx;
  ctx.n_sites_local = n_sites_local;
  ctx.n_sites_local_ghosted = n_sites_local_ghosted;
  ctx.n_sites_ghost = n_sites_ghost;
  ctx.n_mechanical_sites_local = n_mechanical_sites_local;
  ctx.idx_q_ptr = idx_q_ptr;
  ctx.active_mech_sites = active_mech_sites;
  ctx.mechanical_neighs_idx = mechanical_neighs_idx;
  ctx.mean_q_ptr = mean_q_ptr;
  ctx.mf_rho_ptr = mf_rho_ptr;
  ctx.xi_ptr = xi_ptr;
  ctx.specie_ptr = specie_ptr;
  ctx.system_equations = system_equations;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Initialice solution vector X := {F}
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &X));
  PetscCall(VecSetFromOptions(X));
  PetscScalar X_value = 1.0;
  for (PetscInt X_idx = 0; X_idx < dim; X_idx++) {
    PetscCall(VecSetValues(X, 1, &X_idx, &X_value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Initialize residual vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, dim, &Y));
  PetscCall(VecSetFromOptions(Y));
  PetscScalar Y_value = 100.0;
  for (PetscInt Y_idx = 0; Y_idx < dim; Y_idx++) {
    PetscCall(VecSetValues(Y, 1, &Y_idx, &Y_value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Initialize the nonzero structure of the Jacobian. This is artificial
  because clearly if we had a routine to compute the Jacobian we wouldn't
  need to use finite differences.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_SELF, &J));
  PetscCall(MatSetType(J, MATAIJ));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, dim, dim));
  for (PetscInt i = 0; i < dim; i++) {
    for (PetscInt j = 0; j < dim; j++) {
      PetscCall(MatSetValue(J, i, j, 1.0, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(J, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));

#ifdef DEBUG_MODE
  MatView(J, PETSC_VIEWER_DRAW_WORLD);
  MatViewFromOptions(J, NULL, "-mat_view -draw_pause -1");
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Color the matrix, i.e. determine groups of columns that share no common
  rows. These columns in the Jacobian can all be computed simultaneously.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ISColoring iscoloring;
  MatColoring coloring;

  PetscCall(MatColoringCreate(J, &coloring));
  PetscCall(MatColoringSetType(coloring, MATCOLORINGLF));
  PetscCall(MatColoringSetFromOptions(coloring));
  PetscCall(MatColoringApply(coloring, &iscoloring));
  PetscCall(MatColoringDestroy(&coloring));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve equation system
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));

  PetscCall(SNESSetType(snes, SNESNEWTONLS));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCJACOBI));

  PetscCall(SNESSetFunction(snes, Y, evaluate_RHS, &ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create the data structure that SNESComputeJacobianDefaultColor() uses
   to compute the actual Jacobians via finite differences.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatFDColoring fdcoloring;
  PetscCall(MatFDColoringCreate(J, iscoloring, &fdcoloring));
  PetscCall(MatFDColoringSetFunction(
      fdcoloring, (PetscErrorCode(*)(void))evaluate_RHS, &ctx));
  PetscCall(MatFDColoringSetFromOptions(fdcoloring));
  PetscCall(MatFDColoringSetUp(J, iscoloring, fdcoloring));
  PetscCall(ISColoringDestroy(&iscoloring));

  /*
    Tell SNES to use the routine SNESComputeJacobianDefaultColor()
    to compute Jacobians.
  */
  SNESSetJacobian(snes, J, J, SNESComputeJacobianDefaultColor, fdcoloring);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Monitor convergence
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESMonitorSet(snes, monitor_equilibrium, &ctx, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Convergence tolerances
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetTolerances(snes, abstol, rtol, stol, maxit, maxf));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Compute the trial
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(evaluate_RHS(snes, X, Y, &ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set option flags in the terminal
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetOptionsPrefix(snes, "minV_dF_"));

  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve equation system
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSolve(snes, NULL, X));

  PetscCall(SNESGetIterationNumber(snes, &SNES_iterations));
  PetscCall(SNESGetConvergedReason(snes, &SNES_reason));
  PetscCall(SNESGetConvergedReasonString(snes, &SNES_strreason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Number of SNES iterations = %" PetscInt_FMT "\n",
                        SNES_iterations));
  if (SNES_reason > 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Converged due to %s \n",
                          SNES_strreason));
  } else if (SNES_reason <= 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Did not converged due to %s \n",
                          SNES_strreason));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Update stretch tensor F and reference configuration
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#ifdef DEBUG_MODE
  PetscCall(VecView(X, PETSC_VIEWER_STDOUT_WORLD));
#endif

  const PetscScalar* X_ptr;
  PetscCall(VecGetArrayRead(X, &X_ptr));

  Eigen::Matrix3d F_relax = Eigen::Matrix3d::Identity();
  F_relax(0, 0) = X_ptr[0];
  F_relax(1, 1) = X_ptr[1];
  F_relax(2, 2) = X_ptr[2];

  Eigen::Map<MatrixType> mean_q_ref(mean_q_ptr, n_sites_local_ghosted, 3);
  mean_q_ref = (F_relax * mean_q_ref.transpose()).transpose();

  Simulation->F(0, 0) = F_relax(0, 0);
  Simulation->F(1, 1) = F_relax(1, 1);
  Simulation->F(2, 2) = F_relax(2, 2);

  lattice_x_B0 = F_relax * lattice_x_B0;
  lattice_y_B0 = F_relax * lattice_y_B0;
  lattice_z_B0 = F_relax * lattice_z_B0;

  box_origin_0 = F_relax * box_origin_0;

  local_domain_ll = F_relax * local_domain_ll;
  local_domain_ur = F_relax * local_domain_ur;


  PetscCall(VecRestoreArrayRead(X, &X_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore mean-q data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data,
                                DMSwarmPICField_coor, NULL, NULL,
                                (void**)&mean_q_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore meanfield energy density
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "mf-rho", NULL,
                                NULL, (void**)&mf_rho_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore molar fraction data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "molar-fraction",
                                NULL, NULL, (void**)&xi_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore atomic specie data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMSwarmRestoreField(Simulation->atomistic_data, "specie", NULL,
                                NULL, (void**)&specie_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore idx data
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMSwarmRestoreField(Simulation->atomistic_data, "idx", NULL, NULL,
                      (void**)&idx_q_ptr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Free work space.
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatFDColoringDestroy(&fdcoloring));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/************************************************************************/

static PetscErrorCode evaluate_RHS(SNES snes, Vec X, Vec Y, void* ctx) {

  PetscFunctionBeginUser;

  unsigned int dim = NumberDimensions;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Get user context
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  //! Get number of atoms and number of atoms per rank
  PetscInt n_sites_local = ((Equilibrium_dV_dF_ctx*)ctx)->n_sites_local;

  //! Get local size with ghost particles
  PetscInt n_sites_local_ghosted =
      ((Equilibrium_dV_dF_ctx*)ctx)->n_sites_local_ghosted;

  //! Get number of ghost particles
  PetscInt n_sites_ghost = ((Equilibrium_dV_dF_ctx*)ctx)->n_sites_ghost;

  //! Get number of mechanical sites in the local domain (excluding ghosts)
  PetscInt n_mechanical_sites_local =
      ((Equilibrium_dV_dF_ctx*)ctx)->n_mechanical_sites_local;

  //! Get the index of the particles
  PetscInt* idx_q_ptr = ((Equilibrium_dV_dF_ctx*)ctx)->idx_q_ptr;

  //! Get list of active mechanical sites in the local domain (including ghosts)
  IS active_mech_sites = ((Equilibrium_dV_dF_ctx*)ctx)->active_mech_sites;

  //! Get the list of neighbors of each site
  IS* mechanical_neighs_idx =
      ((Equilibrium_dV_dF_ctx*)ctx)->mechanical_neighs_idx;

  //! Get the reference mean value of the position
  PetscScalar* mean_q_ptr = ((Equilibrium_dV_dF_ctx*)ctx)->mean_q_ptr;
  Eigen::Map<MatrixType> mean_q_ref(mean_q_ptr, n_sites_local_ghosted, 3);

  //! Get the meanfield energy density
  PetscScalar* mf_rho_ptr = ((Equilibrium_dV_dF_ctx*)ctx)->mf_rho_ptr;
  Eigen::Map<VectorType> mf_rho(mf_rho_ptr, n_sites_local_ghosted);

  //! Get the occupancy of the sites
  PetscScalar* xi_ptr = ((Equilibrium_dV_dF_ctx*)ctx)->xi_ptr;
  Eigen::Map<VectorType> xi(xi_ptr, n_sites_local_ghosted);

  //! Get the atomis specie of the site0
  AtomicSpecie* specie_ptr = ((Equilibrium_dV_dF_ctx*)ctx)->specie_ptr;

  //! Take structure with the dmd equations
  dmd_equations system_equations =
      ((Equilibrium_dV_dF_ctx*)ctx)->system_equations;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Get index with the active mechanical sites
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInt* active_mech_sites_ptr;
  PetscCall(ISGetIndices(active_mech_sites,
                         (const PetscInt**)&active_mech_sites_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Get solution vector X = {F}
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  const PetscScalar* X_ptr;
  PetscCall(VecGetArrayRead(X, &X_ptr));

  //! Update stretch tensor F
  Eigen::Matrix3d F = Eigen::Matrix<double, 3, 3>::Identity();
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    if (X_ptr[alpha] > 0.0) {
      F(alpha, alpha) = X_ptr[alpha];
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Update current box and site position
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  MatrixType mean_q = (F * mean_q_ref.transpose()).transpose();
  lattice_x_Bn = F * lattice_x_B0;
  lattice_y_Bn = F * lattice_y_B0;
  lattice_z_Bn = F * lattice_z_B0;

  double Volume_0 = fabs(lattice_x_B0.dot(lattice_y_B0.cross(lattice_z_B0)));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Read atom topology
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  AtomTopology* atom_topology =
      (AtomTopology*)malloc(n_sites_local_ghosted * sizeof(AtomTopology));

  for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++) {

    PetscCall(read_atom_topology(&atom_topology[site_u],
                                 mechanical_neighs_idx[site_u]));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Compute energy density
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
#pragma omp parallel for schedule(runtime)
  for (PetscInt mech_site_u = 0; mech_site_u < n_mechanical_sites_local;
       mech_site_u++) {

    //! Get index of the site u
    PetscInt site_u = active_mech_sites_ptr[mech_site_u];

    //! @brief Evaluate energy density at site u
    mf_rho(site_u) = system_equations.evaluate_rho_i(
        site_u, mean_q, xi, specie_ptr, atom_topology[site_u]);
  }

  //! Migrate ghost field (energy density)
  PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
                                     &idx_q_ptr[n_sites_local], mf_rho_ptr));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Compute the local contribution of the residual equation of each rank
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  //! Initialize RHS vector Y = {DF0_DF}
  PetscCall(VecSet(Y, 0.0));

  //! Get RHS vector Y = {DV_DF}
  PetscScalar* Y_ptr;
  PetscCall(VecGetArray(Y, &Y_ptr));

  PetscScalar RHS_local[3] = {0.0, 0.0, 0.0};

#pragma omp parallel for reduction(+ : RHS_local) schedule(runtime)
  for (PetscInt mech_site_u = 0; mech_site_u < n_mechanical_sites_local;
       mech_site_u++) {

    //! Get index of the site u
    PetscInt site_u = active_mech_sites_ptr[mech_site_u];

    //! @brief Evaluate deformation gradient derivative of the potential at
    //! site i
    Eigen::Matrix3d DV_u_DF = system_equations.evaluate_DV_i_DF(
        site_u, mean_q, mean_q_ref, xi, mf_rho, specie_ptr,
        atom_topology[site_u]);

    //! @brief Add up the local contribution
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      RHS_local[alpha] += (1.0 / Volume_0) * DV_u_DF(alpha, alpha);
    }
  }

  //! Perform partial sum reduction of each MPI process and update Y_ptr
  PetscCall(
      MPIU_Allreduce(RHS_local, Y_ptr, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Restore atom topology
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++) {

    PetscCall(restore_atom_topology(&atom_topology[site_u],
                                    mechanical_neighs_idx[site_u]));
  }

  free(atom_topology);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Restore vectors
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecRestoreArrayRead(X, &X_ptr));
  PetscCall(VecRestoreArray(Y, &Y_ptr));

#ifdef DEBUG_MODE
  PetscCall(VecView(Y, PETSC_VIEWER_STDOUT_WORLD));
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Restore index with the active mechanical sites
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(ISRestoreIndices(active_mech_sites,
                             (const PetscInt**)&active_mech_sites_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/************************************************************************/

static PetscErrorCode monitor_equilibrium(SNES snes, PetscInt its,
                                          PetscReal fnorm, void* ctx) {

  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "iter = %" PetscInt_FMT "\t||dV_dF|| = %e\n", its,
                        (double)fnorm));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/************************************************************************/