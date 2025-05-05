
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
#include "Numerical/Quadrature-Multipole.hpp"
#include "Numerical/Quadrature-Hermitian-3th.hpp"


extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern adpPotential adp_MgMg;
extern adpPotential adp_HH;
extern adpPotential adp_MgH;

extern double element_mass[112];

/********************************************************************************/

KOKKOS_INLINE_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos_Device(unsigned int site_i,
  const PetscScalar_Matrix_Default  &mean_q,
  const PetscScalar_Vector_Default &xi,
  const AtomSpecie_Default &specie,
  const AtomTopologyKokkos atom_topology_i,
  const View_Double_Vector_Device mean_q_ij1,
  const AdpPotencial_Device adp_Device_Default) {

  //! @brief Auxiliar variables
  double rho_i = 0.0;

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  PetscInt_Vector_Default mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie(site_i);
  double xi_i = xi(site_i);
  auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return 0.0;
  }

  //! @brief Compute the gradient of the potential with respect the
  //! mean value of the position at site i
  for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {
    //! @brief Get atomistic information of site j

    unsigned int site_j1 = mech_neighs_i(idx_j1);
    AtomicSpecie spc_j1 = specie(site_j1);
    double xi_j1 = xi(site_j1);
    auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);

    //! If the site is empty, skip from the evaluation
    if (xi_j1 < min_occupancy) {
      continue;
    }

    //! @brief Create dof table
    int dof_table_ij[4] = {1, 0, 0, 1};

    //! @brief Fill data for the measure (i,j1)
    concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
    std::array<double, 2>  xi_ij1 = std::array<double, 2> {xi_i, xi_j1};
    std::array<unsigned int, 2>  sites_ij1 = std::array<unsigned int, 2> {site_i, site_j1};
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    //! Create functions
    potential_function functions_rho_ij = rho_ij_adp_MgHx_constructor();

    //! @brief Compute energy density
    double rho_ij = 0.0;

    functions_rho_ij.FK(&rho_ij, xi_ij1.data(), mean_q_ij1.data(), spc_ij1, adp_Device_Default );
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

double evaluate_V_i_adp_MgHx(unsigned int site_i,                 //!
  const Eigen::MatrixXd& mean_q,       //!
  const Eigen::VectorXd& xi,           //!
  const Eigen::VectorXd& rho,          //!
  const AtomicSpecie* specie,          //!
  const AtomTopology atom_topology_i)  //!
{

  unsigned int dim = NumberDimensions;

  //! Local variables
  double rho_i = rho(site_i);
  double V_embed_i = 0.0;  //! Embedded forces term
  double V_pair_i = 0.0;   //! Pairing forces term
  double V_dip_i = 0.0;    //! Dipole distortion term
  double V_quad_i = 0.0;   //! Quadrupole distortion term
  double V_i = 0.0;        //! Total potential

  //! @brief Get topologic information of site i
  unsigned int numneigh_site_i = atom_topology_i.numneigh;
  const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

  //! @brief Get atomistic information of site i
  AtomicSpecie spc_i = specie[site_i];
  double xi_i = xi(site_i);
  Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

  //! If the site is empty, skip from the evaluation
  if (xi_i < min_occupancy) {
    return 0.0;
  }

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

    //! @brief Fill data for the measure
    Eigen::VectorXd mean_q_ij1(6);
    mean_q_ij1 << mean_q_i, mean_q_j1;
    Eigen::VectorXd xi_ij1(2);
    xi_ij1 << xi_i, xi_j1;
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

    potential_function function_V_pair_ij = V_pair_ij_adp_MgHx_constructor();

    //! @brief Compute meanfield pairing term
    double V_pair_ij = 0.0;
    function_V_pair_ij.F(&V_pair_ij, xi_ij1.data(), mean_q_ij1.data(), spc_ij1);
    V_pair_i += V_pair_ij;

    //! @brief Loop in the neighborhood considering the simmetry of the
    //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
    for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

      //! @brief Compute atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie[site_j2];
      double xi_j2 = xi(site_j2);
      Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

      //! If the site is empty, skip from the evaluation
      if (xi_j2 < min_occupancy) {
        continue;
      }

      //! @brief Fill data for the measure
      Eigen::VectorXd mean_q_ij1j2(9);
      mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
      Eigen::VectorXd xi_ij1j2(3);
      xi_ij1j2 << xi_i, xi_j1, xi_j2;
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

      //! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
      //! a_ik*a_ij
      double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

      potential_function function_V_dipole_ij1j2 =
          V_dipole_ij1j2_adp_MgHx_constructor();
      potential_function function_V_quadrupole_ij1j2 =
          V_quadrupole_ij1j2_adp_MgHx_constructor();

      //! Compute meanfield dipole angular term
      double V_dip_ij1j2 = 0.0;
      function_V_dipole_ij1j2.F(&V_dip_ij1j2, xi_ij1.data(), mean_q_ij1.data(),
                                spc_ij1);
      V_dip_i += factor_j1j2 * V_dip_ij1j2;

      //! Compute meanfield quadrupole angular term
      double V_quad_ij1j2 = 0.0;
      function_V_quadrupole_ij1j2.F(&V_quad_ij1j2, xi_ij1.data(),
                                    mean_q_ij1.data(), spc_ij1);
      V_quad_i += factor_j1j2 * V_quad_ij1j2;
    }
  }

  //! @brief Add the contribution of the embedded forces
  CubicSpline embed_ii;
  if (spc_i == Mg) {
    embed_ii = adp_MgMg.embed;
  } else if (spc_i == H) {
    embed_ii = adp_HH.embed;
  }
  V_embed_i = xi_i * cubic_spline(&embed_ii, rho_i);

  //! @brief Add up each contribution to the meanfield potential
  V_i = V_embed_i + V_pair_i + V_dip_i + V_quad_i;

  return V_i;
}

KOKKOS_FUNCTION double evaluate_V_i_adp_MgHx_Kokkos(unsigned int site_i,                 //!
  const PetscScalar_Matrix_Default  &mean_q,
  const PetscScalar_Vector_Default &xi,
  const PetscScalar_Vector_Default& rho,
  const AtomSpecie_Default &specie,
  const AtomTopologyKokkos atom_topology_i,
  const AdpPotencial_Device adp_Device_Default,
  const View_Double_Vector_Device mean_q_ij1)  //!
{

unsigned int dim = NumberDimensions;

//! Local variables
double rho_i = rho(site_i);
double V_embed_i = 0.0;  //! Embedded forces term
double V_pair_i = 0.0;   //! Pairing forces term
double V_dip_i = 0.0;    //! Dipole distortion term
double V_quad_i = 0.0;   //! Quadrupole distortion term
double V_i = 0.0;        //! Total potential

//! @brief Get topologic information of site i
unsigned int numneigh_site_i = atom_topology_i.numneigh;
PetscInt_Vector_Default mech_neighs_i = atom_topology_i.mech_neighs_ptr;

//! @brief Get atomistic information of site i
AtomicSpecie spc_i = specie(site_i);
double xi_i = xi(site_i);
auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
//! If the site is empty, skip from the evaluation
if (xi_i < min_occupancy) {
return 0.0;
}

for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

//! @brief Get atomistic information of site j
unsigned int site_j1 = mech_neighs_i(idx_j1);
AtomicSpecie spc_j1 = specie(site_j1);
double xi_j1 = xi(site_j1);
auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);

//! If the site is empty, skip from the evaluation
if (xi_j1 < min_occupancy) {
continue;
}

//! @brief Fill data for the measure

concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
std::array<double, 2>  xi_ij1 = std::array<double, 2> {xi_i, xi_j1};
AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

potential_function function_V_pair_ij = V_pair_ij_adp_MgHx_constructor();

//! @brief Compute meanfield pairing term
double V_pair_ij = 0.0;
function_V_pair_ij.FK(&V_pair_ij, xi_ij1.data(), mean_q_ij1.data(), spc_ij1, adp_Device_Default);
V_pair_i += V_pair_ij;

//! @brief Loop in the neighborhood considering the simmetry of the
//! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

//! @brief Compute atomistic information of site j2
unsigned int site_j2 = mech_neighs_i(idx_j2);
AtomicSpecie spc_j2 = specie(site_j2);
double xi_j2 = xi(site_j2);
auto mean_q_j2 = extractRowBlock(mean_q, site_j2, 0, 3);

//! If the site is empty, skip from the evaluation
if (xi_j2 < min_occupancy) {
continue;
}

//! @brief Fill data for the measure
auto mean_q_ij1j2 = concatenateToArray<Kokkos::View<double*>, 9>(mean_q_i, mean_q_j1, mean_q_j2);
std::array<double, 3>  xi_ij1j2 = std::array<double, 3> {xi_i, xi_j1, xi_j2};
AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

//! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
//! a_ik*a_ij
double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

potential_function function_V_dipole_ij1j2 =
V_dipole_ij1j2_adp_MgHx_constructor();
potential_function function_V_quadrupole_ij1j2 =
V_quadrupole_ij1j2_adp_MgHx_constructor();

//! Compute meanfield dipole angular term
double V_dip_ij1j2 = 0.0;
function_V_dipole_ij1j2.FK(&V_dip_ij1j2, xi_ij1.data(), mean_q_ij1.data(), spc_ij1,
adp_Device_Default);
V_dip_i += factor_j1j2 * V_dip_ij1j2;

//! Compute meanfield quadrupole angular term
double V_quad_ij1j2 = 0.0;

function_V_quadrupole_ij1j2.FK(&V_quad_ij1j2, xi_ij1.data(),
         mean_q_ij1.data(), spc_ij1, adp_Device_Default);
V_quad_i += factor_j1j2 * V_quad_ij1j2;
}
}

//! @brief Add the contribution of the embedded forces
CubicSpline embed_ii;
if (spc_i == Mg) {
embed_ii = adp_Device_Default(0).embed;
} else if (spc_i == H) {
embed_ii = adp_Device_Default(1).embed;
}
V_embed_i = xi_i * cubic_spline(&embed_ii, rho_i);

//! @brief Add up each contribution to the meanfield potential
V_i = V_embed_i + V_pair_i + V_dip_i + V_quad_i;

return V_i;
}

double evaluate_mf_rho_i_adp_MgHx(unsigned int site_i,            //!
  const Eigen::MatrixXd& mean_q,  //!
  const Eigen::VectorXd& stdv_q,  //!
  const Eigen::VectorXd& xi,      //!
  const AtomicSpecie* specie,     //!
  const AtomTopology atom_topology_i) {

//! Define Integration rule
void (*meanfield_integral)(double* integral_f, potential_function function,
void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

//! Local variables
double mf_rho_i = 0.0;  //! Meanfield Energy density term

//! @brief Get topologic information of site i
unsigned int numneigh_site_i = atom_topology_i.numneigh;
const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

//! @brief Get atomistic information of site i
AtomicSpecie spc_i = specie[site_i];
double xi_i = xi(site_i);
double stdv_q_i = stdv_q(site_i);
Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);

//! If the site is empty, skip from the evaluation
if (xi_i < min_occupancy) {
return 0.0;
}

for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

//! @brief Get atomistic information of site j
unsigned int site_j1 = mech_neighs_i[idx_j1];
AtomicSpecie spc_j1 = specie[site_j1];
double xi_j1 = xi(site_j1);
double stdv_q_j1 = stdv_q(site_j1);
Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

//! If the site is empty, skip from the evaluation
if (xi_j1 < min_occupancy) {
continue;
}

//! @brief Fill data for the measure
Eigen::VectorXd mean_q_ij1(6);
mean_q_ij1 << mean_q_i, mean_q_j1;
Eigen::VectorXd stdv_q_ij1(2);
stdv_q_ij1 << stdv_q_i, stdv_q_j1;
Eigen::VectorXd xi_ij1(2);
xi_ij1 << xi_i, xi_j1;
AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

//! @brief Create dof table for i-j par
int dof_table_ij[4] = {1, 0, 0, 1};

//! @brief Create measure/functions
gaussian_measure_ctx measure_ij =
fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

potential_function rho_ij = rho_ij_adp_MgHx_constructor();

//! @brief Compute meanfield energy density to evaluate the embeded energy
double mf_rho_ij = 0.0;
meanfield_integral(&mf_rho_ij, rho_ij, &measure_ij);
mf_rho_i += mf_rho_ij;

//! @brief Remove measure
destroy_gaussian_measure(&measure_ij);
}

return mf_rho_i;
}



KOKKOS_FUNCTION double evaluate_mf_rho_i_adp_MgHx_Kokkos(unsigned int site_i,                 //!
  const PetscScalar_Matrix_Default  &mean_q,
  const PetscScalar_Vector_Default &stdv_q,
  const PetscScalar_Vector_Default &xi,
  const AtomSpecie_Default &specie,
  const AtomTopologyKokkos atom_topology_i,
  const View_Double_Vector_Device mean_q_ij1,
  const AdpPotencial_Device adp_Device_Default
  ) {

//! Define Integration rule
void (*meanfield_integral)(double* integral_f, potential_function function,
void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

//! Local variables
double mf_rho_i = 0.0;  //! Meanfield Energy density term

//! @brief Get topologic information of site i
unsigned int numneigh_site_i = atom_topology_i.numneigh;
const PetscInt_Vector_Default mech_neighs_i = atom_topology_i.mech_neighs_ptr;

//! @brief Get atomistic information of site i
AtomicSpecie spc_i = specie(site_i);
double xi_i = xi(site_i);
double stdv_q_i = stdv_q(site_i);
auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);

//! If the site is empty, skip from the evaluation
if (xi_i < min_occupancy) {
return 0.0;
}

for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

//! @brief Get atomistic information of site j
unsigned int site_j1 = mech_neighs_i[idx_j1];
AtomicSpecie spc_j1 = specie(site_j1);
double xi_j1 = xi(site_j1);
double stdv_q_j1 = stdv_q(site_j1);
auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);
//! If the site is empty, skip from the evaluation
if (xi_j1 < min_occupancy) {
continue;
}

//! @brief Fill data for the measure

concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
std::array<double, 2>  xi_ij1 = std::array<double, 2> {xi_i, xi_j1};
std::array<unsigned int, 2>  sites_ij1 = std::array<unsigned int, 2> {site_i, site_j1};
double stdv_q_ij1[2] = {stdv_q_i, stdv_q_j1};
AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

//! @brief Create dof table for i-j par
int dof_table_ij[4] = {1, 0, 0, 1};

//! @brief Create measure/functions
gaussian_measure_ctx measure_ij =
fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1,
  xi_ij1.data(), spc_ij1, dof_table_ij, 2);

potential_function rho_ij = rho_ij_adp_MgHx_constructor();

//! @brief Compute meanfield energy density to evaluate the embeded energy
double mf_rho_ij = 0.0;
meanfield_integral(&mf_rho_ij, rho_ij, &measure_ij);
mf_rho_i += mf_rho_ij;

//! @brief Remove measure
destroy_gaussian_measure(&measure_ij);
}

return mf_rho_i;
}


double evaluate_S0_i_adp_MgHx(unsigned int site_i,                 //!
  const Eigen::MatrixXd& mean_q,       //!
  const Eigen::VectorXd& stdv_q,       //!
  const Eigen::VectorXd& xi,           //!
  const Eigen::VectorXd& mf_rho,       //!
  const Eigen::VectorXd& beta,         //!
  const Eigen::VectorXd& gamma,        //!
  const AtomicSpecie* specie,          //!
  const AtomTopology atom_topology_i)  //!
{

unsigned int dim = NumberDimensions;

//! Define Integration rule
void (*meanfield_integral)(double* integral_f, potential_function function,
 void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

//! Local variables
double mf_rho_i = mf_rho(site_i);
double V0_embed_i = 0.0;  //! Meanfield Embedded forces term
double V0_pair_i = 0.0;   //! Meanfield Pairing forces term
double V0_dip_i = 0.0;    //! Meanfield Dipole distortion term
double V0_quad_i = 0.0;   //! Meanfield Quadrupole distortion term
double V0_i = 0.0;        //! Total potential

//! @brief Get topologic information of site i
unsigned int numneigh_site_i = atom_topology_i.numneigh;
const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;

//! @brief Get atomistic information of site i
Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);
double stdv_q_i = stdv_q(site_i);
double xi_i = xi(site_i);
double beta_i = beta(site_i);
double gamma_i = gamma(site_i);
AtomicSpecie spc_i = specie[site_i];
double m_i = unit_change_uma * xi_i * element_mass[spc_i];

//! If the site is empty, skip from the evaluation
if (xi_i < min_occupancy) {
return 0.0;
}

for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

//! @brief Get atomistic information of site j
unsigned int site_j1 = mech_neighs_i[idx_j1];
AtomicSpecie spc_j1 = specie[site_j1];
double xi_j1 = xi(site_j1);
double stdv_q_j1 = stdv_q(site_j1);
Eigen::Vector3d mean_q_j1 = mean_q.block<1, 3>(site_j1, 0);

//! If the site is empty, skip from the evaluation
if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
continue;
}

//! @brief Fill data for the measure
Eigen::VectorXd mean_q_ij1(6);
mean_q_ij1 << mean_q_i, mean_q_j1;
Eigen::VectorXd stdv_q_ij1(2);
stdv_q_ij1 << stdv_q_i, stdv_q_j1;
Eigen::VectorXd xi_ij1(2);
xi_ij1 << xi_i, xi_j1;
AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

//! @brief Create dof table for i-j par
int dof_table_ij[4] = {1, 0, 0, 1};

//! @brief Create measure/functions
gaussian_measure_ctx measure_ij =
fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1.data(),
      xi_ij1.data(), spc_ij1, dof_table_ij, 2);

potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

//! @brief Compute meanfield pairing term
double V0_pair_ij = 0.0;
meanfield_integral(&V0_pair_ij, V_pair_ij, &measure_ij);
V0_pair_i += V0_pair_ij;

//! @brief Remove measure
destroy_gaussian_measure(&measure_ij);

//! @brief Loop in the neighborhood considering the simmetry of the
//! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

//! @brief Compute atomistic information of site j2
unsigned int site_j2 = mech_neighs_i[idx_j2];
AtomicSpecie spc_j2 = specie[site_j2];
double xi_j2 = xi(site_j2);
double stdv_q_j2 = stdv_q(site_j2);
Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);

//! If the site is empty, skip from the evaluation
if (xi_j2 < min_occupancy) {
continue;
}

//! @brief Create dof table for i-j1-j2
int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

//! If the j1 match j2, avoid integration duplic
if (site_j1 == site_j2) {
dof_table_ij1j2[5] = 1;
dof_table_ij1j2[7] = 1;
}

//! @brief Fill data for the measure
Eigen::VectorXd mean_q_ij1j2(9);
mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
Eigen::VectorXd stdv_q_ij1j2(3);
stdv_q_ij1j2 << stdv_q_i, stdv_q_j1, stdv_q_j2;
Eigen::VectorXd xi_ij1j2(3);
xi_ij1j2 << xi_i, xi_j1, xi_j2;
AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};

//! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
//! a_ik*a_ij
double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

//! Create measure/functions
gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
mean_q_ij1j2.data(), stdv_q_ij1j2.data(), xi_ij1j2.data(), spc_ij1j2,
dof_table_ij1j2, 3);

potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
potential_function V_quadrupole_ij1j2 =
V_quadrupole_ij1j2_adp_MgHx_constructor();

//! Compute meanfield dipole angular term
double V0_dip_ij1j2 = 0.0;
meanfield_integral(&V0_dip_ij1j2, V_dipole_ij1j2, &measure_ij1j2);
V0_dip_i += factor_j1j2 * V0_dip_ij1j2;

//! Compute meanfield quadrupole angular term
double V0_quad_ij1j2 = 0.0;
meanfield_integral(&V0_quad_ij1j2, V_quadrupole_ij1j2, &measure_ij1j2);
V0_quad_i += factor_j1j2 * V0_quad_ij1j2;

//! Remove measure
destroy_gaussian_measure(&measure_ij1j2);
}
}

//! @brief Add the contribution of the embedded forces
CubicSpline embed_ii;
if (spc_i == Mg) {
embed_ii = adp_MgMg.embed;
} else if (spc_i == H) {
embed_ii = adp_HH.embed;
}
double mf_F_i = cubic_spline(&embed_ii, mf_rho_i);
V0_embed_i = xi_i * mf_F_i;

//! @brief Add up each contribution to the meanfield potential
V0_i = V0_embed_i + V0_pair_i + V0_dip_i + V0_quad_i;

//! @brief Compute mean meanfield Hamiltonian
double H0_i = 1.0 / (2.0 * beta_i) + V0_i;

//! @brief Compute log of the grand-cannonical partition function
double log_Z0_i = 3.0 * log((stdv_q_i * sqrt(m_i / beta_i)) / h_planck);
if (xi_i < 1.0) {
log_Z0_i += log(1.0 / (1.0 - xi_i));
}

//! @brief Chemical multiplier
double gamma_xi_i = 0.0;
if (spc_i == H) {
gamma_xi_i = gamma_i * (xi_i > 0.9999 ? 0.9999 : xi_i);
}

//! @brief Compute the meanfield Entropy
double S0_i = -log_Z0_i + beta_i * H0_i - gamma_xi_i;

return S0_i;
}


KOKKOS_FUNCTION double evaluate_S0_i_adp_MgHx_Kokkos(unsigned int site_i, 
  const PetscScalar_Matrix_Default  &mean_q,
  const PetscScalar_Vector_Default &stdv_q,
  const PetscScalar_Vector_Default &xi,
  const PetscScalar_Vector_Default &mf_rho,       //!
  const PetscScalar_Vector_Default &beta,         //!
  const PetscScalar_Vector_Default &gamma,        //!
  const AtomSpecie_Default &specie,
  const AtomTopologyKokkos atom_topology_i,
  const View_Double_Vector_Device mean_q_ij1,
  const AdpPotencial_Device adp_Device_Default,
  const View_Double_Vector_Device element_mass)  //!
{

unsigned int dim = NumberDimensions;

//! Define Integration rule
void (*meanfield_integral)(double* integral_f, potential_function function,
 void* ctx_measure);
#if defined(MULTIPOLE_INTEGRAL)
meanfield_integral = meanfield_integral_mp;
#elif defined(GH3TH_INTEGRAL)
meanfield_integral = meanfield_integral_gh3th;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

//! Local variables
double mf_rho_i = mf_rho(site_i);
double V0_embed_i = 0.0;  //! Meanfield Embedded forces term
double V0_pair_i = 0.0;   //! Meanfield Pairing forces term
double V0_dip_i = 0.0;    //! Meanfield Dipole distortion term
double V0_quad_i = 0.0;   //! Meanfield Quadrupole distortion term
double V0_i = 0.0;        //! Total potential

//! @brief Get topologic information of site i
unsigned int numneigh_site_i = atom_topology_i.numneigh;
const PetscInt_Vector_Default mech_neighs_i = atom_topology_i.mech_neighs_ptr;

//! @brief Get atomistic information of site i
auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
double stdv_q_i = stdv_q(site_i);
double xi_i = xi(site_i);
double beta_i = beta(site_i);
double gamma_i = gamma(site_i);
AtomicSpecie spc_i = specie(site_i);
double m_i = unit_change_uma * xi_i * element_mass(spc_i);

//! If the site is empty, skip from the evaluation
if (xi_i < min_occupancy) {
return 0.0;
}

for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {

//! @brief Get atomistic information of site j
unsigned int site_j1 = mech_neighs_i(idx_j1);
AtomicSpecie spc_j1 = specie(site_j1);
double xi_j1 = xi(site_j1);
double stdv_q_j1 = stdv_q(site_j1);
auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);

//! If the site is empty, skip from the evaluation
if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
continue;
}

//! @brief Fill data for the measure
concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
std::array<double, 2>  xi_ij1 = std::array<double, 2> {xi_i, xi_j1};
double stdv_q_ij1[2] = {stdv_q_i, stdv_q_j1};
AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};


//! @brief Create dof table for i-j par
int dof_table_ij[4] = {1, 0, 0, 1};

//! @brief Create measure/functions
gaussian_measure_ctx measure_ij =
fill_out_gaussian_measure(mean_q_ij1.data(), stdv_q_ij1,
      xi_ij1.data(), spc_ij1, dof_table_ij, 2);

potential_function V_pair_ij = V_pair_ij_adp_MgHx_constructor();

//! @brief Compute meanfield pairing term
double V0_pair_ij = 0.0;
meanfield_integral(&V0_pair_ij, V_pair_ij, &measure_ij);
V0_pair_i += V0_pair_ij;

//! @brief Remove measure
destroy_gaussian_measure(&measure_ij);

//! @brief Loop in the neighborhood considering the simmetry of the
//! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

//! @brief Compute atomistic information of site j2
unsigned int site_j2 = mech_neighs_i(idx_j2);
AtomicSpecie spc_j2 = specie(site_j2);
double xi_j2 = xi(site_j2);
double stdv_q_j2 = stdv_q(site_j2);
auto mean_q_j2 = extractRowBlock(mean_q, site_j2, 0, 3);

//! If the site is empty, skip from the evaluation
if (xi_j2 < min_occupancy) {
continue;
}

//! @brief Create dof table for i-j1-j2
int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

//! If the j1 match j2, avoid integration duplic
if (site_j1 == site_j2) {
dof_table_ij1j2[5] = 1;
dof_table_ij1j2[7] = 1;
}

//! @brief Fill data for the measure
auto mean_q_ij1j2 = concatenateToArray<Kokkos::View<double*>, 9>(mean_q_i, mean_q_j1, mean_q_j2);
std::array<double, 3>  xi_ij1j2 = std::array<double, 3> {xi_i, xi_j1, xi_j2};
AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};
double stdv_q_ij1j2[3] = {spc_i, spc_j1, spc_j2};

//! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
//! a_ik*a_ij
double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

//! Create measure/functions
gaussian_measure_ctx measure_ij1j2 = fill_out_gaussian_measure(
mean_q_ij1j2.data(), stdv_q_ij1j2, xi_ij1j2.data(), spc_ij1j2,
dof_table_ij1j2, 3);

potential_function V_dipole_ij1j2 = V_dipole_ij1j2_adp_MgHx_constructor();
potential_function V_quadrupole_ij1j2 =
V_quadrupole_ij1j2_adp_MgHx_constructor();

//! Compute meanfield dipole angular term
double V0_dip_ij1j2 = 0.0;
meanfield_integral(&V0_dip_ij1j2, V_dipole_ij1j2, &measure_ij1j2);
V0_dip_i += factor_j1j2 * V0_dip_ij1j2;

//! Compute meanfield quadrupole angular term
double V0_quad_ij1j2 = 0.0;
meanfield_integral(&V0_quad_ij1j2, V_quadrupole_ij1j2, &measure_ij1j2);
V0_quad_i += factor_j1j2 * V0_quad_ij1j2;

//! Remove measure
destroy_gaussian_measure(&measure_ij1j2);
}
}

//! @brief Add the contribution of the embedded forces
CubicSpline embed_ii;
if (spc_i == Mg) {
embed_ii = adp_Device_Default(0).embed;
} else if (spc_i == H) {
embed_ii = adp_Device_Default(1).embed;
}
double mf_F_i = cubic_spline(&embed_ii, mf_rho_i);
V0_embed_i = xi_i * mf_F_i;

//! @brief Add up each contribution to the meanfield potential
V0_i = V0_embed_i + V0_pair_i + V0_dip_i + V0_quad_i;

//! @brief Compute mean meanfield Hamiltonian
double H0_i = 1.0 / (2.0 * beta_i) + V0_i;

//! @brief Compute log of the grand-cannonical partition function
double log_Z0_i = 3.0 * log((stdv_q_i * sqrt(m_i / beta_i)) / h_planck);
if (xi_i < 1.0) {
log_Z0_i += log(1.0 / (1.0 - xi_i));
}

//! @brief Chemical multiplier
double gamma_xi_i = 0.0;
if (spc_i == H) {
gamma_xi_i = gamma_i * (xi_i > 0.9999 ? 0.9999 : xi_i);
}

//! @brief Compute the meanfield Entropy
double S0_i = -log_Z0_i + beta_i * H0_i - gamma_xi_i;

return S0_i;
}

/********************************************************************************/