
/**
 * @file MgHx-mf-V-bulk.cpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief Compute meanfield-variables for the Mg-Hx bulk
 * @version 0.1
 * @date 2023-05-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "ADP/density.hpp"
#include "Atoms/Atom.hpp"
#include "Atoms/Topology.hpp"
#include "Macros.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <Utils/KokkosUtils.hpp>

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern adpPotential adp_MgMg;
extern adpPotential adp_HH;
extern adpPotential adp_MgH;

extern double element_mass[112];

/********************************************************************************/

KOKKOS_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos_Device(unsigned int site_i,           //!
  const PetscScalar_Matrix_Default  &mean_q, //! Mean q
  const PetscScalar_Vector_Default &xi,     //! Molar fraction
  const AtomSpecie_Default &specie,    //! Atom
  const AtomTopology atom_topology_i,
  const View_Double_Vector_Device mean_q_ij1) {

  //! @brief Auxiliar variables
  double rho_i = 0.0;

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt *mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie(site_i);
  double xi_i = xi(site_i);
  auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 2);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return 0.0;
  }

  //! @brief Compute the gradient of the potential with respect the
  //! mean value of the position at site i
  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

    //! @brief Get atomistic information of site j
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = specie(site_j1);
    double xi_j1 = xi(site_j1);
    auto mean_q_j1 = extractRowBlock(mean_q, site_i, 0, 2);

    //! If the site is empty, skip from the evaluation
    if (xi_j1 < min_occupancy) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)

    concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
    std::array<double, 2>  xi_ij1 = concatenate(xi_i, xi_j1);
    std::array<unsigned int, 2>  sites_ij1 = concatenate(site_i, site_j1);
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    std::cout << "Estoy en el constructor: " << idx_j1 << std::endl;

    //! Create functions
    potential_function functions_rho_ij = rho_ij_adp_MgHx_constructor();

    //! @brief Compute energy density
    double rho_ij = 0.0;
    std::cout << "Después constructor: " << idx_j1 << std::endl;
    functions_rho_ij.FK(&rho_ij, xi_ij1.data(), mean_q_ij1.data(), spc_ij1);
    rho_i += rho_ij;
  }

  return rho_i;
}

double evaluate_rho_i_adp_MgHx_kokkos(unsigned int site_i,           //!
  const Eigen::MatrixXd &mean_q, //! Mean q
  const Eigen::VectorXd &xi,     //! Molar fraction
  const AtomicSpecie *specie,    //! Atom
  const AtomTopology atom_topology_i) {

//! @brief Auxiliar variables
double rho_i = 0.0;

//! @brief Get topologic information of site i
unsigned int numneigh_site_i = atom_topology_i.numneigh;
const PetscInt *mech_neighs_i = atom_topology_i.mech_neighs_ptr;

//! @brief Get atomistic information of site i
AtomicSpecie spc_i = specie[site_i];
double xi_i = xi(site_i);
Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

//! If the site is empty, skip from the evaluation
if (xi_i < min_occupancy) {
return 0.0;
}

//! @brief Compute the gradient of the potential with respect the
//! mean value of the position at site i
for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

//! @brief Get atomistic information of site j
unsigned int site_j1 = mech_neighs_i[idx_j1];
AtomicSpecie spc_j1 = specie[site_j1];
double xi_j1 = xi(site_j1);
Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

//! If the site is empty, skip from the evaluation
if (xi_j1 < min_occupancy) {
continue;
}

//! @brief Create dof table
int dof_table_ij[4] = {1, 0, 0, 1};

//! @brief Fill data for the measure (i,j1)
Eigen::VectorXd mean_q_ij1(6);
mean_q_ij1 << mean_q_i, mean_q_j1;
Eigen::VectorXd xi_ij1(2);
xi_ij1 << xi_i, xi_j1;
Eigen::VectorXd sites_ij1(2);
sites_ij1 << site_i, site_j1;
AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

//! Create functions
potential_function functions_rho_ij = rho_ij_adp_MgHx_constructor();

//! @brief Compute energy density
double rho_ij = 0.0;
functions_rho_ij.F(&rho_ij, xi_ij1.data(), mean_q_ij1.data(), spc_ij1);
rho_i += rho_ij;
}

return rho_i;
}

/********************************************************************************/