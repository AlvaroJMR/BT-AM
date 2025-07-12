/**
 * @file Atoms/Atom.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <petscis.h>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include "Periodic-Boundary/boundary_conditions.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern PetscInt ndiv_mesh_X;
extern PetscInt ndiv_mesh_Y;
extern PetscInt ndiv_mesh_Z;

extern PetscInt size_MPI_X;
extern PetscInt size_MPI_Y;
extern PetscInt size_MPI_Z;

using namespace std;
extern int diffusion;

extern Eigen::Vector3d lattice_x_B0;
extern Eigen::Vector3d lattice_y_B0;
extern Eigen::Vector3d lattice_z_B0;

extern Eigen::Vector3d lattice_x_Bn;
extern Eigen::Vector3d lattice_y_Bn;
extern Eigen::Vector3d lattice_z_Bn;

extern Eigen::Vector3d local_domain_ll;
extern Eigen::Vector3d local_domain_ur;

extern double box_x_min;
extern double box_x_max;
extern double box_y_min;
extern double box_y_max;
extern double box_z_min;
extern double box_z_max;

extern Eigen::Vector3d box_origin_0;

static PetscErrorCode GetElementCoords(const PetscScalar _coords[],
                                       const PetscInt e2n[],
                                       PetscScalar el_coords[]);

static bool IsAtomElement(const Eigen::Vector3d mean_q,
                          const PetscScalar el_coords[]);

static PetscErrorCode _DMLocatePoints_DMDARegular_IS(DM dm, Vec pos,
                                                     IS *iscell);
static PetscErrorCode DMLocatePoints_DMDARegular(DM dm, Vec pos,
                                                 DMPointLocationType ltype,
                                                 PetscSF cellSF);
static PetscErrorCode DMGetNeighbors_DMDARegular(DM dm, PetscInt *nneighbors,
                                                 const PetscMPIInt **neighbors);

/*******************************************************/

/*
 Create a DMShell and attach a regularly spaced DMDA for point location
 Override methods for point location
*/
PetscErrorCode init_DMD_simulation(DMD *Simulation, dump_file Simulation_file)
{

  unsigned int dim = NumberDimensions;

  DM atomistic_data, bounding_cell, background_mesh;
  PetscMPIInt rank;
  PetscInt blocksize;

  //! Topological variables
  AO dump2petsc_mapping;

  //!
  PetscInt n_atoms_local;
  PetscInt n_atoms_global;
  PetscInt n_atoms = Simulation_file.n_atoms;

  //!
  IS beta_bcc;
  IS gamma_bcc;
  IS interstitial_sites;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create a regularly spaced DMDA */
  PetscInt overlap = 1;
  PetscInt dof = 1;

  PetscCall(DMDACreate(PETSC_COMM_WORLD, &background_mesh));
  PetscCall(DMSetDimension(background_mesh, dim));
  PetscCall(DMDASetSizes(background_mesh, ndiv_mesh_X, ndiv_mesh_Y, ndiv_mesh_Z));
  PetscCall(DMDASetNumProcs(background_mesh, size_MPI_X, size_MPI_Y, size_MPI_Z));
//  PetscCall(DMDASetBoundaryType(*background_mesh, Simulation_file.bx,
//                                Simulation_file.by, Simulation_file.bz));
  PetscCall(DMDASetDof(background_mesh, dof));
  PetscCall(DMDASetStencilType(background_mesh, DMDA_STENCIL_BOX));
  PetscCall(DMDASetStencilWidth(background_mesh, overlap));
  PetscCall(DMDASetElementType(background_mesh, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(background_mesh));
  PetscCall(DMSetUp(background_mesh));
  PetscCall(DMDASetUniformCoordinates(background_mesh, box_x_min, box_x_max, box_y_min,
                                      box_y_max, box_z_min, box_z_max));

  //! @brief Check if the local dimensions are smaller than twice the cutoff
  //! radius
  PetscReal ll_vertex[3], ur_vertex[3];
  PetscCall(DMGetLocalBoundingBox(background_mesh, ll_vertex, ur_vertex));

  PetscReal local_dx = ur_vertex[0] - ll_vertex[0];
  if (local_dx < 2 * r_cutoff_ADP_MgHx)
  {
    PetscCall(PetscError(PETSC_COMM_WORLD, __LINE__, "init_DMD_simulation",
                         __FILE__, PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
                         "The local dimension X (%f) is smaller than %f",
                         local_dx, 2 * r_cutoff_ADP_MgHx));
    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  PetscReal local_dy = ur_vertex[1] - ll_vertex[1];
  if (local_dy < 2 * r_cutoff_ADP_MgHx)
  {
    PetscCall(PetscError(PETSC_COMM_WORLD, __LINE__, "init_DMD_simulation",
                         __FILE__, PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
                         "The local dimension Y (%f) is smaller than %f",
                         local_dy, 2 * r_cutoff_ADP_MgHx));
    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  PetscReal local_dz = ur_vertex[2] - ll_vertex[2];
  if (local_dz < 2 * r_cutoff_ADP_MgHx)
  {
    PetscCall(PetscError(PETSC_COMM_WORLD, __LINE__, "init_DMD_simulation",
                         __FILE__, PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
                         "The local dimension Z (%f) is smaller than %f",
                         local_dz, 2 * r_cutoff_ADP_MgHx));
    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  PetscCall(DMShellCreate(PETSC_COMM_WORLD, &bounding_cell));
  PetscCall(DMSetApplicationContext(bounding_cell, background_mesh));
  bounding_cell->ops->locatepoints = DMLocatePoints_DMDARegular;
  bounding_cell->ops->getneighbors = DMGetNeighbors_DMDARegular;

  /* Create the swarm */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &atomistic_data));
  PetscCall(DMSetType(atomistic_data, DMSWARM));
  PetscCall(DMSetDimension(atomistic_data, dim));
  PetscCall(PetscObjectSetName((PetscObject)atomistic_data, "Atoms"));

  PetscCall(DMSwarmSetType(atomistic_data, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(atomistic_data, bounding_cell));

  //! Apply bcc in the DMSwarm
  PetscCall(set_periodic_boundary_conditions(Simulation_file.bx, //!
                                             Simulation_file.by, //!
                                             Simulation_file.bz));

  //! Specie of the atom (H, Mg, Al, Cu, ...)
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "specie", 1,
                                              PETSC_INT));

  //! Molar fraction: Xi := <n>_0
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "molar-fraction",
                                              1, PETSC_REAL));

  //! Energy density: mf_rho := <rho>_0
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "mf-rho", 1,
                                              PETSC_REAL));

  //! Thermal multiplier: Beta := 1/(kb * T)
  PetscCall(
      DMSwarmRegisterPetscDatatypeField(atomistic_data, "beta", 1, PETSC_REAL));

  //! Chemical multiplier: Gamma
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "gamma", 1,
                                              PETSC_REAL));

  //! Standard desviation of the position: stdv-q
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "stdv-q", 1,
                                              PETSC_REAL));

  //! First Piola-Kirchoff stress tensor: FPK
  PetscCall(
      DMSwarmRegisterPetscDatatypeField(atomistic_data, "FPK", 9, PETSC_REAL));

  //! Site ghost
  PetscCall(
      DMSwarmRegisterPetscDatatypeField(atomistic_data, "ghost", 1, PETSC_INT));

  //! Site idx
  PetscCall(
      DMSwarmRegisterPetscDatatypeField(atomistic_data, "idx", 1, PETSC_INT));

  //! MPI rank
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "MPI-rank", 1,
                                              PETSC_INT));

  //! List with diffusive sites
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "idx-diff", 1,
                                              PETSC_INT));

  //! Boundary condition idex for the beta thermal multiplier
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "idx-bcc-beta", 1,
                                              PETSC_INT));

  //! Boundary condition idex for the gamma chemical multiplier
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "idx-bcc-gamma",
                                              1, PETSC_INT));

  //!
  PetscCall(DMSwarmRegisterPetscDatatypeField(atomistic_data, "idx-bcc-fix", 1,
                                              PETSC_INT));

  PetscCall(DMSwarmFinalizeFieldRegister(atomistic_data));
  PetscCall(DMSwarmSetLocalSizes(atomistic_data, n_atoms, BufferLenght));

  //! @brief Get the bounding box of the background mesh
  PetscCall(DMGetLocalBoundingBox(background_mesh, local_domain_ll.data(),
                                  local_domain_ur.data()));

  //! Lower left coordinates of the brick
  PetscReal xI_lw, yI_lw, zI_lw;
  xI_lw = local_domain_ll(0);
  yI_lw = local_domain_ll(1);
  zI_lw = local_domain_ll(2);

  //! Upper right coordinates of the brick
  PetscReal xI_up, yI_up, zI_up;
  xI_up = local_domain_ur(0);
  yI_up = local_domain_ur(1);
  zI_up = local_domain_ur(2);

  //!
  local_dx = xI_up - xI_lw;
  local_dy = yI_up - yI_lw;
  local_dz = zI_up - zI_lw;

  //! @brief Check if the local dimensions are smaller than twice the cutoff radius
  if (local_dx < 2.0 * r_cutoff_ADP_MgHx)
  {
    PetscCall(PetscError(
        PETSC_COMM_WORLD, __LINE__, "init_DMD_simulation", __FILE__,
        PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
        "The local dimension X (%f) is smaller than twice the cutoff radius (%f)",
        local_dx, 2.0 * r_cutoff_ADP_MgHx));
    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  if (local_dy < 2.0 * r_cutoff_ADP_MgHx)
  {
    PetscCall(PetscError(
        PETSC_COMM_WORLD, __LINE__, "init_DMD_simulation", __FILE__,
        PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
        "The local dimension Y (%f) is smaller than twice the cutoff radius (%f)",
        local_dy, 2.0 * r_cutoff_ADP_MgHx));
    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  if (local_dz < 2.0 * r_cutoff_ADP_MgHx)
  {
    PetscCall(PetscError(
        PETSC_COMM_WORLD, __LINE__, "init_DMD_simulation", __FILE__,
        PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
        "The local dimension Z (%f) is smaller than twice the cutoff radius (%f)",
        local_dz, 2.0 * r_cutoff_ADP_MgHx));
    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  //! @brief mean_q: Mean value of each atomic position
  double *mean_q_ptr;
  PetscCall(DMSwarmGetField(atomistic_data, DMSwarmPICField_coor, &blocksize,
                            NULL, (void **)&mean_q_ptr));
  Eigen::Map<MatrixType> mean_q(mean_q_ptr, n_atoms, blocksize);

  //! @brief idx_ptr: Global index of each atomic position
  int *idx_ptr;
  PetscCall(
      DMSwarmGetField(atomistic_data, "idx", NULL, NULL, (void **)&idx_ptr));

  //! @brief Loop in the elemnts of the background mesh and find atoms in the
  //! mesh
  n_atoms_local = 0;
  for (PetscInt site_i = 0; site_i < n_atoms; site_i++)
  {

    Eigen::Map<const Eigen::Vector3d> mean_q_i(
        &(Simulation_file.mean_q[site_i * dim]), dim);

    if ((mean_q_i(0) >= xI_lw) && //!
        (mean_q_i(0) < xI_up) &&  //!
        (mean_q_i(1) >= yI_lw) && //!
        (mean_q_i(1) < yI_up) &&  //!
        (mean_q_i(2) >= zI_lw) && //!
        (mean_q_i(2) < zI_up))
    {
      idx_ptr[n_atoms_local] = site_i;
      mean_q.row(n_atoms_local) = mean_q_i;
      n_atoms_local++;
    }
  }

  PetscCall(DMSwarmRestoreField(atomistic_data, DMSwarmPICField_coor,
                                &blocksize, NULL, (void **)&mean_q_ptr));
  PetscCall(
      DMSwarmRestoreField(atomistic_data, "idx", NULL, NULL, (void **)&idx_ptr));

  //! @brief Set local sizes
  PetscCall(DMSwarmSetLocalSizes(atomistic_data, n_atoms_local, BufferLenght));

  //! @brief specie: Integer which defines the atomic specie
  AtomicSpecie *specie;
  PetscCall(DMSwarmGetField(atomistic_data, "specie", &blocksize, NULL,
                            (void **)&specie));

  //! @brief stdv_q: Standard desviation of each atomic position
  double *stdv_q;
  PetscCall(DMSwarmGetField(atomistic_data, "stdv-q", &blocksize, NULL,
                            (void **)&stdv_q));

  //! @brief mf-rho: Meanfield energy density
  double *mf_rho;
  PetscCall(DMSwarmGetField(atomistic_data, "mf-rho", &blocksize, NULL,
                            (void **)&mf_rho));

  //! @brief xi: Molar fraction (mean occupancy)
  double *xi;
  PetscCall(DMSwarmGetField(atomistic_data, "molar-fraction", &blocksize, NULL,
                            (void **)&xi));

  //! @brief beta: Thermal Lagrange multiplier
  double *beta;
  PetscCall(
      DMSwarmGetField(atomistic_data, "beta", &blocksize, NULL, (void **)&beta));

  //! @brief gamma: Chemical Lagrange multiplier
  double *gamma;
  PetscCall(DMSwarmGetField(atomistic_data, "gamma", &blocksize, NULL,
                            (void **)&gamma));

  //! @brief idx_ptr: index of the site
  PetscCall(DMSwarmGetField(atomistic_data, "idx", &blocksize, NULL,
                            (void **)&idx_ptr));

  //! @brief ghost_ptr: index if the site is a ghost atom or not
  PetscInt *ghost_ptr;
  PetscCall(DMSwarmGetField(atomistic_data, "ghost", &blocksize, NULL,
                            (void **)&ghost_ptr));

  //! @brief site_mpi_rank: Integer which defines MPI rank location
  PetscInt *site_mpi_rank;
  PetscCall(DMSwarmGetField(atomistic_data, "MPI-rank", &blocksize, NULL,
                            (void **)&site_mpi_rank));

  //! @brief diff_idx_ptr: index of the interstitial
  PetscInt *diff_idx_ptr;
  PetscCall(DMSwarmGetField(atomistic_data, "idx-diff", &blocksize, NULL,
                            (void **)&diff_idx_ptr));

  //! @brief Pointers with the information of the boundary condition
  PetscInt *beta_bcc_ptr;
  PetscCall(DMSwarmGetField(atomistic_data, "idx-bcc-beta", &blocksize, NULL,
                            (void **)&beta_bcc_ptr));

  PetscInt *gamma_bcc_ptr;
  PetscCall(DMSwarmGetField(atomistic_data, "idx-bcc-gamma", &blocksize, NULL,
                            (void **)&gamma_bcc_ptr));

  //! Set information from the .dump file
  for (PetscInt local_site_i = 0; local_site_i < n_atoms_local;
       local_site_i++)
  {
    PetscInt site_i = idx_ptr[local_site_i];
    site_mpi_rank[local_site_i] = (PetscInt)rank;
    ghost_ptr[local_site_i] = 0;
    specie[local_site_i] = Simulation_file.specie[site_i];
    stdv_q[local_site_i] = Simulation_file.stdv_q[site_i];
    xi[local_site_i] = Simulation_file.xi[site_i];
    mf_rho[local_site_i] = 0.0;
    beta[local_site_i] = Simulation_file.beta[site_i];
    gamma[local_site_i] = Simulation_file.gamma[site_i];
    beta_bcc_ptr[local_site_i] = Simulation_file.beta_bcc[site_i];
    gamma_bcc_ptr[local_site_i] = Simulation_file.gamma_bcc[site_i];
    diff_idx_ptr[local_site_i] = Simulation_file.diffusive_idx[site_i];
  }

  //! Create a context to relate petsc numbering and dump numbering
  IS dump_ordering;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, n_atoms_local, idx_ptr,
                            PETSC_COPY_VALUES, &dump_ordering));
  PetscCall(AOCreateBasicIS(dump_ordering, NULL, &dump2petsc_mapping));
  PetscCall(ISDestroy(&dump_ordering));

  //! Turn dump numbering of "idx" into petsc numbering
  PetscCall(AOApplicationToPetsc(dump2petsc_mapping, n_atoms_local, idx_ptr));

  //! Restore particle fields
  PetscCall(DMSwarmRestoreField(atomistic_data, "specie", &blocksize, NULL,
                                (void **)&specie));
  PetscCall(DMSwarmRestoreField(atomistic_data, "stdv-q", &blocksize, NULL,
                                (void **)&stdv_q));
  PetscCall(DMSwarmRestoreField(atomistic_data, "mf-rho", &blocksize, NULL,
                                (void **)&mf_rho));
  PetscCall(DMSwarmRestoreField(atomistic_data, "molar-fraction", &blocksize,
                                NULL, (void **)&xi));
  PetscCall(DMSwarmRestoreField(atomistic_data, "beta", &blocksize, NULL,
                                (void **)&beta));
  PetscCall(DMSwarmRestoreField(atomistic_data, "gamma", &blocksize, NULL,
                                (void **)&gamma));
  PetscCall(DMSwarmRestoreField(atomistic_data, "idx", &blocksize, NULL,
                                (void **)&idx_ptr));
  PetscCall(DMSwarmRestoreField(atomistic_data, "ghost", &blocksize, NULL,
                                (void **)&ghost_ptr));
  PetscCall(DMSwarmRestoreField(atomistic_data, "MPI-rank", &blocksize, NULL,
                                (void **)&site_mpi_rank));
  PetscCall(DMSwarmRestoreField(atomistic_data, "idx-diff", &blocksize, NULL,
                                (void **)&diff_idx_ptr));
  PetscCall(DMSwarmRestoreField(atomistic_data, "idx-bcc-beta", &blocksize,
                                NULL, (void **)&beta_bcc_ptr));
  PetscCall(DMSwarmRestoreField(atomistic_data, "idx-bcc-gamma", &blocksize,
                                NULL, (void **)&gamma_bcc_ptr));

  //! Check global size
  DMSwarmGetSize(atomistic_data, &n_atoms_global);
  if (n_atoms_global != Simulation_file.n_atoms)
  {
    PetscCall(PetscError(
        PETSC_COMM_WORLD, __LINE__, "init_DMD_simulation", __FILE__,
        PETSC_ERR_RETURN, PETSC_ERROR_INITIAL,
        "The global size (%i) of the DMSwarm does not match the input (%i) !",
        n_atoms_global, Simulation_file.n_atoms));
    PetscFunctionReturn(PETSC_ERR_RETURN);
  }

  //! Generate output structure
  PetscCall(DMSwarmGetSize(atomistic_data, &Simulation->n_sites_global));
  PetscCall(DMSwarmGetLocalSize(atomistic_data, &Simulation->n_sites_local));
  Simulation->Temperature_env = 0.0;
  Simulation->ChemicalPotential_env = 0.0;
  Simulation->Pressure_env = 0.0;
  Simulation->F = Eigen::Matrix3d::Identity();
  Simulation->atomistic_data = atomistic_data;
  Simulation->background_mesh = background_mesh;
  Simulation->bounding_cell = bounding_cell;
  Simulation->dump2petsc_mapping = dump2petsc_mapping;

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*******************************************************/

PetscErrorCode destroy_DMD_simulation(DMD *Simulation)
{

  PetscFunctionBegin;

  //! @brief Atomistic variables
  PetscCall(DMDestroy(&Simulation->atomistic_data));
  PetscCall(DMDestroy(&Simulation->background_mesh));
  PetscCall(DMDestroy(&Simulation->bounding_cell));

  //! @brief Topological variables
  PetscCall(AODestroy(&Simulation->dump2petsc_mapping));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*******************************************************/

static PetscErrorCode GetElementCoords(const PetscScalar _coords[],
                                       const PetscInt e2n[],
                                       PetscScalar el_coords[])
{
  PetscInt dim = NumberDimensions;
  PetscInt n_nodes = 8;

  PetscFunctionBeginUser;
  /* get coords for the element */
  for (PetscInt i = 0; i < n_nodes; i++)
  {
    for (PetscInt d = 0; d < dim; d++)
      el_coords[dim * i + d] = _coords[dim * e2n[i] + d];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*******************************************************/

static bool IsAtomElement(const Eigen::Vector3d mean_q,
                          const PetscScalar el_coords[])
{

  unsigned int dim = NumberDimensions;

  //! Lower left coordinates of the brick
  PetscReal xI_lw, yI_lw, zI_lw;
  xI_lw = el_coords[0 * dim + 0];
  yI_lw = el_coords[0 * dim + 1];
  zI_lw = el_coords[0 * dim + 2];

  //! Upper right coordinates of the brick
  PetscReal xI_up, yI_up, zI_up;
  xI_up = el_coords[6 * dim + 0];
  yI_up = el_coords[6 * dim + 1];
  zI_up = el_coords[6 * dim + 2];

  if ((mean_q(0) >= xI_lw) && //!
      (mean_q(0) < xI_up) &&  //!
      (mean_q(1) >= yI_lw) && //!
      (mean_q(1) < yI_up) &&  //!
      (mean_q(2) >= zI_lw) && //!
      (mean_q(2) < zI_up))
  {
    return true;
  }
  else
  {
    return false;
  }
}

/*******************************************************/

static PetscErrorCode _DMLocatePoints_DMDARegular_IS(DM dm, Vec mean_q_petsc,
                                                     IS *iscell)
{

  PetscInt p, n, bs, si, sj, sk, milocal, mjlocal, mklocal, mx, my, mz;
  DM background_mesh;
  PetscInt *cellidx;
  const PetscScalar *mean_q_ptr;

  unsigned int dim = NumberDimensions;

  Vec coords;
  PetscInt nel, npe;
  const PetscScalar *_coords;
  const PetscInt *element_list;
  PetscScalar el_coords[24];

  PetscFunctionBegin;

  //! @brief Get particle coordinates
  PetscCall(VecGetLocalSize(mean_q_petsc, &n));
  PetscCall(VecGetBlockSize(mean_q_petsc, &bs));
  PetscInt npoints = n / bs;
  PetscCall(VecGetArrayRead(mean_q_petsc, &mean_q_ptr));
  Eigen::Map<const MatrixType> mean_q(mean_q_ptr, npoints, bs);

  PetscCall(PetscMalloc1(npoints, &cellidx));

  //! @brief Get mesh information
  PetscCall(DMGetApplicationContext(dm, &background_mesh));
  PetscCall(DMGetCoordinatesLocal(background_mesh, &coords));
  PetscCall(VecGetArrayRead(coords, &_coords));
  PetscCall(DMDAGetElements(background_mesh, &nel, &npe, &element_list));

  for (p = 0; p < npoints; p++)
  {
    PetscReal coorx, coory, coorz;
    PetscInt mi, mj, mk;

    Eigen::Vector3d mean_q_i = mean_q.row(p);

    cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;

    for (PetscInt eidx = 0; eidx < nel; eidx++)
    {

      /* get coords for the element */
      const PetscInt *element = &element_list[npe * eidx];
      PetscCall(GetElementCoords(_coords, element, el_coords));

      if (IsAtomElement(mean_q_i, el_coords))
      { //!
        cellidx[p] = eidx;
      }
    }

    if (mean_q_i(0) < box_x_min)
      cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (mean_q_i(0) > box_x_max)
      cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (mean_q_i(1) < box_y_min)
      cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (mean_q_i(1) > box_y_max)
      cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (mean_q_i(2) < box_z_min)
      cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
    if (mean_q_i(2) > box_z_max)
      cellidx[p] = DMLOCATEPOINT_POINT_NOT_FOUND;
  }

  PetscCall(VecRestoreArrayRead(mean_q_petsc, &mean_q_ptr));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, npoints, cellidx,
                            PETSC_OWN_POINTER, iscell));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*******************************************************/

static PetscErrorCode DMLocatePoints_DMDARegular(DM dm, Vec pos,
                                                 DMPointLocationType ltype,
                                                 PetscSF cellSF)
{
  IS iscell;
  PetscSFNode *cells;
  PetscInt p, bs, npoints, nfound;
  const PetscInt *boxCells;

  PetscFunctionBegin;
  PetscCall(_DMLocatePoints_DMDARegular_IS(dm, pos, &iscell));
  PetscCall(VecGetLocalSize(pos, &npoints));
  PetscCall(VecGetBlockSize(pos, &bs));
  npoints = npoints / bs;

  PetscCall(PetscMalloc1(npoints, &cells));
  PetscCall(ISGetIndices(iscell, &boxCells));

  for (p = 0; p < npoints; p++)
  {
    cells[p].rank = 0;
    cells[p].index = DMLOCATEPOINT_POINT_NOT_FOUND;
    cells[p].index = boxCells[p];
  }

  PetscCall(ISRestoreIndices(iscell, &boxCells));
  PetscCall(ISDestroy(&iscell));
  nfound = npoints;
  PetscCall(PetscSFSetGraph(cellSF, npoints, nfound, NULL, PETSC_OWN_POINTER,
                            cells, PETSC_OWN_POINTER));
  PetscCall(ISDestroy(&iscell));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*******************************************************/

static PetscErrorCode DMGetNeighbors_DMDARegular(
    DM dm, PetscInt *nneighbors, const PetscMPIInt **neighbors)
{
  DM background_mesh;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &background_mesh));
  PetscCall(DMGetNeighbors(background_mesh, nneighbors, neighbors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*******************************************************/