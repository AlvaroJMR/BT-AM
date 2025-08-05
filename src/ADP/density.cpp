
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
       function_V_dipole_ij1j2.F(&V_dip_ij1j2, xi_ij1j2.data(), mean_q_ij1j2.data(),
                                 spc_ij1j2);
       V_dip_i += factor_j1j2 * V_dip_ij1j2;
 
       //! Compute meanfield quadrupole angular term
       double V_quad_ij1j2 = 0.0;
       function_V_quadrupole_ij1j2.F(&V_quad_ij1j2, xi_ij1j2.data(),
                                     mean_q_ij1j2.data(), spc_ij1j2);                               
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
   const AtomTopology atom_topology_i,
   const DevSnapUnmanaged &soADevice)  //!
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
 const PetscInt *mech_neighs_i = atom_topology_i.mech_neighs_ptr;
 
 double mean_q_ij1[9]; 

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
 unsigned int site_j1 = mech_neighs_i[idx_j1];
 AtomicSpecie spc_j1 = specie(site_j1);
 double xi_j1 = xi(site_j1);
 auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);
 
 //! If the site is empty, skip from the evaluation
 if (xi_j1 < min_occupancy) {
 continue;
 }
 
 //! @brief Fill data for the measure
 
 concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
 Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
 AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
  
 //! @brief Compute meanfield pairing term
 double V_pair_ij = 0.0;
 V_pair_ij_adp_MgHx_dispatcher function_V_pair_ij { Functions_Enum::FK, &V_pair_ij, xi_ij1.data(), mean_q_ij1, spc_ij1, soADevice.data() };
 function_V_pair_ij();
 V_pair_i += V_pair_ij;
 
 //! @brief Loop in the neighborhood considering the simmetry of the
 //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
 for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {

 //! @brief Compute atomistic information of site j2
 unsigned int site_j2 = mech_neighs_i[idx_j2];
 AtomicSpecie spc_j2 = specie(site_j2);
 double xi_j2 = xi(site_j2);
 auto mean_q_j2 = extractRowBlock(mean_q, site_j2, 0, 3);
 
 //! If the site is empty, skip from the evaluation
 if (xi_j2 < min_occupancy) {
 continue;
 }
 
 //! @brief Fill data for the measure
 auto mean_q_ij1j2 = concatenateToArray<Kokkos::View<double*>, 9>(mean_q_i, mean_q_j1, mean_q_j2);
 Kokkos::Array<double, 3>  xi_ij1j2 = Kokkos::Array<double, 3> {xi_i, xi_j1, xi_j2};
 AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};
 
 //! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
 //! a_ik*a_ij
 double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;
 
 //! Compute meanfield dipole angular term
 double V_dip_ij1j2 = 0.0;
 
 V_dipole_ij1j2_dispatcher function_V_dipole_ij1j2 { Functions_Enum::FK, &V_dip_ij1j2, xi_ij1j2.data(), mean_q_ij1j2.data(), spc_ij1j2, soADevice.data() };
 function_V_dipole_ij1j2();
 
 V_dip_i += factor_j1j2 * V_dip_ij1j2; 
 //! Compute meanfield quadrupole angular term
 double V_quad_ij1j2 = 0.0;
 
 V_quadrupole_ij1j2_dispatcher function_V_quadrupole_ij1j2 { Functions_Enum::FK, &V_quad_ij1j2, xi_ij1j2.data(), mean_q_ij1j2.data(), spc_ij1j2, soADevice.data() };
 function_V_quadrupole_ij1j2();  
 
 V_quad_i += factor_j1j2 * V_quad_ij1j2;
 }
 }
 
 //! @brief Add the contribution of the embedded forces
 CubicSpline embed_ii;
 if (spc_i == Mg) {
 embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, soADevice.data());
 } else if (spc_i == H) {
 embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, soADevice.data());
 }
 V_embed_i = xi_i * cubic_spline_Kokkos(&embed_ii, rho_i);
 
 //! @brief Add up each contribution to the meanfield potential
 V_i = V_embed_i + V_pair_i + V_dip_i + V_quad_i;

 return V_i;
 }
 
 KOKKOS_FUNCTION double evaluate_V_i_adp_MgHx_Kokkos_SIMD(unsigned int site_i,                 //!
   const PetscScalar_Matrix_Default  &mean_q,
   const PetscScalar_Vector_Default &xi,
   const PetscScalar_Vector_Default& rho,
   const AtomSpecie_Default &specie,
   const AtomTopology atom_topology_i,
   const DevSnapUnmanaged &soADevice)  //!
 {
 
 unsigned int dim = NumberDimensions;
 using simd_t = Kokkos::Experimental::native_simd<double>;
 using simd_int_t = Kokkos::Experimental::native_simd<int>;
 using mask_t  = typename simd_t::mask_type; 
 constexpr int W   = simd_t::size();
 
 double rho_i = rho(site_i);
 double V_embed_i = 0.0;  //! Embedded forces term
 double V_pair_i = 0.0;   //! Pairing forces term
 double V_dip_i = 0.0;    //! Dipole distortion term
 double V_quad_i = 0.0;   //! Quadrupole distortion term
 double V_i = 0.0;        //! Total potential
 
 unsigned int numneigh_site_i = atom_topology_i.numneigh;
 const PetscInt *mech_neighs_i = atom_topology_i.mech_neighs_ptr;

 CubicSpline u_iju_Mg = getSpline(AdpType::MgMg, SplineType::u, CubicSpline(), soADevice.data());
 CubicSpline u_iju_H  = getSpline(AdpType::HH,  SplineType::u, CubicSpline(), soADevice.data());
 CubicSpline u_iju_MgH  = getSpline(AdpType::MgH,  SplineType::u, CubicSpline(), soADevice.data());
 CubicSpline u_ijw_Mg = getSpline(AdpType::MgMg, SplineType::w, CubicSpline(), soADevice.data());
 CubicSpline u_ijw_H  = getSpline(AdpType::HH,  SplineType::w, CubicSpline(), soADevice.data());
 CubicSpline u_ijw_MgH  = getSpline(AdpType::MgH,  SplineType::w, CubicSpline(), soADevice.data());
 
 double mean_q_ij1[9]; 

 AtomicSpecie spc_i = specie(site_i);
 double xi_i = xi(site_i);
 auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
 if (xi_i < min_occupancy) {
 return 0.0;
 }
 for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {
 
 unsigned int site_j1 = mech_neighs_i[idx_j1];
 AtomicSpecie spc_j1 = specie(site_j1);
 auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);

 double xi_j1 = xi(site_j1);
 double q_i0 = mean_q(site_i,0);
 double q_i1 = mean_q(site_i,1);
 double q_i2 = mean_q(site_i,2);
 simd_t qi0(q_i0);
 simd_t qi1(q_i1);
 simd_t qi2(q_i2);


 if (xi_j1 < min_occupancy) {
 continue;
 }
 
 
 concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
 Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
 AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
  
 double V_pair_ij = 0.0;
 V_pair_ij_adp_MgHx_dispatcher function_V_pair_ij { Functions_Enum::FK, &V_pair_ij, xi_ij1.data(), mean_q_ij1, spc_ij1, soADevice.data() };
 function_V_pair_ij();
 V_pair_i += V_pair_ij;
 
 unsigned int idx_j2 = idx_j1;
 for (; idx_j2 + W < 1; idx_j2 += W) {


 simd_t qj10, qj11, qj12, qj20, qj21, qj22, n1, n2;
 simd_int_t spc_j1, spc_j2;
 simd_int_t sites_j1, sites_j2;

 sites_j1.copy_from(mech_neighs_i + idx_j1, Kokkos::Experimental::simd_flag_default);
 sites_j2.copy_from(mech_neighs_i + idx_j2, Kokkos::Experimental::simd_flag_default);

 for (int lane = 0; lane < W; ++lane) {
  qj10[lane] = mean_q(sites_j1[lane],0);
  qj11[lane] = mean_q(sites_j1[lane],1);
  qj12[lane] = mean_q(sites_j1[lane],2);
  n1[lane]   = xi(sites_j1[lane]);
  spc_j1[lane] = specie(sites_j1[lane]);
  qj20[lane] = mean_q(sites_j2[lane],0);
  qj21[lane] = mean_q(sites_j2[lane],1);
  qj22[lane] = mean_q(sites_j2[lane],2);
  n2[lane]   = xi(sites_j2[lane]);
  spc_j2[lane] = specie(sites_j2[lane]);
  }

 simd_t factor_j1j2;
 for (int lane = 0; lane < W; ++lane) {
   factor_j1j2[lane] = (idx_j2 + lane == idx_j1) ? 1.0 : 2.0;
 }
 
 simd_t dr10 = qi0 - qj10;
 simd_t dr11 = qi1 - qj11;
 simd_t dr12 = qi2 - qj12;

 simd_t r1  = Kokkos::fma(dr10, dr10,
              Kokkos::fma(dr11, dr11, dr12*dr12));

 simd_t dr20 = qi0 - qj20;
 simd_t dr21 = qi1 - qj21;
 simd_t dr22 = qi2 - qj22;

 simd_t r2  = Kokkos::fma(dr20, dr20,
              Kokkos::fma(dr21, dr21, dr22*dr22));

 simd_t r1_r2 = Kokkos::fma(dr10, dr20,
              Kokkos::fma(dr11, dr21, dr12*dr22));

 simd_t norm1 = Kokkos::sqrt(r1);
 simd_t norm2 = Kokkos::sqrt(r2);

 CubicSpline u_ij1[W], u_ij2[W], w_ij1[W], w_ij2[W];
 for (int lane = 0; lane < W; ++lane) {
  u_ij1[lane] = (spc_i==Mg && spc_j1[lane]==Mg) ? u_iju_Mg  :
                 (spc_i==H  && spc_j1[lane]==H ) ? u_iju_H   : u_iju_MgH;
  u_ij2[lane] = (spc_i==Mg && spc_j2[lane]==Mg) ? u_iju_Mg  :
                 (spc_i==H  && spc_j2[lane]==H ) ? u_iju_H   : u_iju_MgH;
 }
  simd_t nn_u_ij1 =  xi_i * n1 * cubic_spline_Kokkos_SIMD<W>(u_ij1, norm1);
  simd_t nn_u_ij2 =  xi_i * n2 * cubic_spline_Kokkos_SIMD<W>(u_ij2, norm2);
  simd_t val = 0.5 * nn_u_ij1 * nn_u_ij2 * r1_r2;
  simd_t result = factor_j1j2 * val;
  for (int lane = 0; lane < W; ++lane) {
  V_dip_i += result[lane];
  }


 for (int lane = 0; lane < W; ++lane) {
  w_ij1[lane] = (spc_i==Mg && spc_j1[lane]==Mg) ? u_ijw_Mg  :
                 (spc_i==H  && spc_j1[lane]==H ) ? u_ijw_H   : u_ijw_MgH;
  w_ij2[lane] = (spc_i==Mg && spc_j2[lane]==Mg) ? u_ijw_Mg  :
                 (spc_i==H  && spc_j2[lane]==H ) ? u_ijw_H   : u_ijw_MgH;
 }
  simd_t nn_w_ij1 =  xi_i * n1 * cubic_spline_Kokkos_SIMD<W>(w_ij1, norm1);
  simd_t nn_w_ij2 =  xi_i * n2 * cubic_spline_Kokkos_SIMD<W>(w_ij2, norm2);
  val = 0.5 * nn_w_ij1 * nn_w_ij2 * r1_r2 * r1_r2 - (nn_w_ij1 * nn_w_ij2 * r1 * r2) / simd_t(6.0);
  result = factor_j1j2 * val;
  for (int lane = 0; lane < W; ++lane) {
  V_quad_i += result[lane];
  }
 }
 for (; idx_j2 < numneigh_site_i; ++idx_j2) {
  unsigned site_j1 = mech_neighs_i[idx_j1];
  unsigned site_j2 = mech_neighs_i[idx_j2];

  double qj10 = mean_q(site_j1, 0);
  double qj11 = mean_q(site_j1, 1);
  double qj12 = mean_q(site_j1, 2);
  double n1   = xi(site_j1);
  AtomicSpecie spc1 = specie(site_j1);

  double qj20 = mean_q(site_j2, 0);
  double qj21 = mean_q(site_j2, 1);
  double qj22 = mean_q(site_j2, 2);
  double n2   = xi(site_j2);
  AtomicSpecie spc2 = specie(site_j2);

  double dr10 = q_i0 - qj10;
  double dr11 = q_i1 - qj11;
  double dr12 = q_i2 - qj12;
  double r1 = dr10*dr10 + dr11*dr11 + dr12*dr12;

  double dr20 = q_i0 - qj20;
  double dr21 = q_i1 - qj21;
  double dr22 = q_i2 - qj22;
  double r2 = dr20*dr20 + dr21*dr21 + dr22*dr22;

  double r1_r2 = dr10 * dr20 + dr11 * dr21 + dr12 * dr22;
  double norm1 = Kokkos::sqrt(r1);
  double norm2 = Kokkos::sqrt(r2);

  double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

  CubicSpline u_ij1 = (spc_i==Mg && spc1==Mg) ? u_iju_Mg  :
                      (spc_i==H  && spc1==H ) ? u_iju_H   : u_iju_MgH;
  CubicSpline u_ij2 = (spc_i==Mg && spc2==Mg) ? u_iju_Mg  :
                      (spc_i==H  && spc2==H ) ? u_iju_H   : u_iju_MgH;
  double nn_u_ij1 = xi_i * n1 * cubic_spline_Kokkos(&u_ij1, norm1);
  double nn_u_ij2 = xi_i * n2 * cubic_spline_Kokkos(&u_ij2, norm2);
  double val_dip = 0.5 * nn_u_ij1 * nn_u_ij2 * r1_r2;
  V_dip_i += factor_j1j2 * val_dip;

  CubicSpline w_ij1 = (spc_i==Mg && spc1==Mg) ? u_ijw_Mg  :
                      (spc_i==H  && spc1==H ) ? u_ijw_H   : u_ijw_MgH;
  CubicSpline w_ij2 = (spc_i==Mg && spc2==Mg) ? u_ijw_Mg  :
                      (spc_i==H  && spc2==H ) ? u_ijw_H   : u_ijw_MgH;
  double nn_w_ij1 = xi_i * n1 * cubic_spline_Kokkos(&w_ij1, norm1);
  double nn_w_ij2 = xi_i * n2 * cubic_spline_Kokkos(&w_ij2, norm2);
  double val_quad = 0.5 * nn_w_ij1 * nn_w_ij2 * dsqr(r1_r2) - (nn_w_ij1 * nn_w_ij2 * r1 * r2) / 6.0;  
  V_quad_i += factor_j1j2 * val_quad;
 }
 }
 
 //! @brief Add the contribution of the embedded forces
 CubicSpline embed_ii;
 if (spc_i == Mg) {
 embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, soADevice.data());
 } else if (spc_i == H) {
 embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, soADevice.data());
 }
 V_embed_i = xi_i * cubic_spline_Kokkos(&embed_ii, rho_i);
 
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
   const AtomTopology atom_topology_i,
   const DevSnapUnmanaged &soADevice,
   const gaussian_measure_ctx_kokkos &ctx,
   const bool multipole_integral  //! If the integral is multipole or GH3TH      
   ) {
 
 //! Define Integration rule

 //! Local variables
 double mf_rho_i = 0.0;  //! Meanfield Energy density term
 
 //! @brief Get topologic information of site i
 unsigned int numneigh_site_i = atom_topology_i.numneigh;
 const PetscInt *mech_neighs_i = atom_topology_i.mech_neighs_ptr;

 double mean_q_ij1[6]; 

  //! @brief Get atomistic information of site i
 AtomicSpecie spc_i = specie(site_i);
 double xi_i = xi(site_i);
 double stdv_q_i = stdv_q(site_i);
 auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
 
 //! If the site is empty, skip from the evaluation
 if (xi_i < min_occupancy) {
 return 0.0;
 }
 
 // numneigh_site_i
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
 Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
 Kokkos::Array<unsigned int, 2>  sites_ij1 = Kokkos::Array<unsigned int, 2> {site_i, site_j1};
 double stdv_q_ij1[2] = {stdv_q_i, stdv_q_j1};
 AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
 
 //! @brief Create dof table for i-j par
 int dof_table_ij[4] = {1, 0, 0, 1};
 
 //! @brief Create measure/functions
 fill_out_gaussian_measure_Kokkos(mean_q_ij1, stdv_q_ij1,
   xi_ij1.data(), spc_ij1, dof_table_ij, 2, &ctx);
 
 //! @brief Compute meanfield energy density to evaluate the embeded energy
 double mf_rho_ij = 0.0;
 {

 rho_ij_adp_MgHx_dispatcher function;  
 if (multipole_integral) {
   meanfield_integral_mp_Kokkos<rho_ij_adp_MgHx_dispatcher>(&mf_rho_ij, &ctx, soADevice.data(), function);
 } else {
   meanfield_integral_gh3th_Kokkos<rho_ij_adp_MgHx_dispatcher>(&mf_rho_ij, &ctx, soADevice.data(), function);
 }

 }
 mf_rho_i += mf_rho_ij;
 
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
 
 
 //std::cout << "V0_dip_ij1j2 sin Kokkos = " << V0_dip_ij1j2 << std::endl;
 //std::cout << "V0_dip_i sin Kokkos = " << V0_dip_i << std::endl;
 
 
 //! Compute meanfield quadrupole angular term
 double V0_quad_ij1j2 = 0.0;
 meanfield_integral(&V0_quad_ij1j2, V_quadrupole_ij1j2, &measure_ij1j2);
 V0_quad_i += factor_j1j2 * V0_quad_ij1j2;
 
 //std::cout << "V0_quad_ij1j2 sin Kokkos = " << V0_quad_ij1j2 << std::endl;
 //std::cout << "V0_quad_i sin Kokkos = " << V0_quad_i << std::endl;
 
 //! Remove measure
 destroy_gaussian_measure(&measure_ij1j2);
 }
 }
// Kokkos::printf("Time for loop in the neighborhood no Kokkos= %f\n", timer11.seconds());
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
 
 //Kokkos::printf("V0_i sin Kokkos= %f\n", V0_i);
 
 //! @brief Compute mean meanfield Hamiltonian
 double H0_i = 1.0 / (2.0 * beta_i) + V0_i;
 
 //Kokkos::printf("H0_i sin Kokkos= %f\n", H0_i);
 //Kokkos::printf("beta_i i=%d sin Kokkos = %f\n", site_i, beta_i);
 
 
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
  
 
 Eigen::Vector3d evaluate_DV_i_Dq_u_adp_MgHx(
   unsigned int site_i_star,            //!
   unsigned int site_i,                 //!
   const Eigen::MatrixXd& mean_q,       //! Mean value of q
   const Eigen::VectorXd& xi,           //! Molar fraction
   const Eigen::VectorXd& rho,          //! Energy density
   const AtomicSpecie* specie,          //! Atom
   const AtomTopology atom_topology_i)  //!
 {
 
   unsigned int dim = NumberDimensions;
 
   //! @brief If we are in a hydrogen site with a occupancy below a certain
   //! thereshold, skip the evaluation of this equation
   double xi_i_star = xi(site_i_star);
   AtomicSpecie spc_i_star = specie[site_i_star];
   if (xi_i_star < min_occupancy) {
     return Eigen::Vector3d::Zero();
   }
 
   //! @brief Auxiliar variables
   double rho_i = rho(site_i);
   Eigen::Vector3d D_V_i_embed_Dq = Eigen::Vector3d::Zero();
   Eigen::Vector3d D_rho_i_Dq = Eigen::Vector3d::Zero();
   Eigen::Vector3d D_V_i_pair_Dq = Eigen::Vector3d::Zero();
   Eigen::Vector3d D_V_i_dipol_Dq = Eigen::Vector3d::Zero();
   Eigen::Vector3d D_V_i_quad_Dq = Eigen::Vector3d::Zero();
 
   //! @brief Get topologic information of site i
   unsigned int numneigh_site_i = atom_topology_i.numneigh;
   const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;
 
   //! @brief Get atomistic information of site i
   AtomicSpecie spc_i = specie[site_i];
   double xi_i = xi(site_i);
   Eigen::Vector3d mean_q_i = mean_q.block<1, 3>(site_i, 0);
 
   //! If the site is empty, skip from the evaluation
   if (xi_i < min_occupancy) {
     return Eigen::Vector3d::Zero();
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
     potential_function functions_pair_ij = V_pair_ij_adp_MgHx_constructor();
 
     //! @brief Compute gradient terms
     Eigen::Vector3d d_rho_ij1_dq;
     Eigen::Vector3d d_V_pair_ij1_dq;
 
     for (unsigned int direction = 0; direction < 2; direction++) {
       if (sites_ij1(direction) == site_i_star) {
 
         //! @brief Compute q-grad energy density
         d_rho_ij1_dq.setZero();
         functions_rho_ij.dF_dq(direction, d_rho_ij1_dq.data(), xi_ij1.data(),
                                mean_q_ij1.data(), spc_ij1);
 
         //! @brief Compute q-grad pairing term
         d_V_pair_ij1_dq.setZero();
         functions_pair_ij.dF_dq(direction, d_V_pair_ij1_dq.data(),
                                 xi_ij1.data(), mean_q_ij1.data(), spc_ij1);
 
         //! @brief Add up partial contributions
         D_rho_i_Dq += d_rho_ij1_dq;        
         D_V_i_pair_Dq += d_V_pair_ij1_dq;
       }
     }
 
 #if COMPUTE_ANGULAR_TERMS == 1
     //! @brief Loop in the neighborhood considering the simmetry of the
     //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
     for (unsigned int idx_j2 = 0; idx_j2 < numneigh_site_i; idx_j2++) {
 
       //! @brief Compute atomistic information of site j2
       unsigned int site_j2 = mech_neighs_i[idx_j2];
       AtomicSpecie spc_j2 = specie[site_j2];
       double xi_j2 = xi(site_j2);
       Eigen::Vector3d mean_q_j2 = mean_q.block<1, 3>(site_j2, 0);
 
       //! If the site is empty, skip from the evaluation
       if (xi_j2 < min_occupancy) {
         continue;
       }
 
       //! Check if sites j1 and j2 are in the first lasyer of neighs
       double norm_r_ij1 = (mean_q_i - mean_q_j1).norm();
       double norm_r_ij2 = (mean_q_i - mean_q_j2).norm();
       if ((norm_r_ij1 > r_cutoff_Angular_ADP_MgHx) ||
           (norm_r_ij2 > r_cutoff_Angular_ADP_MgHx)) {
         continue;
       }
 
       //! @brief Factor to consider the simmetry of the opration
       //! a_ij1*a_ij2 = a_ij2*a_ij1
       double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;
 
       //! @brief Fill data for the measure (i,j1,j2)
       Eigen::VectorXd mean_q_ij1j2(9);
       mean_q_ij1j2 << mean_q_i, mean_q_j1, mean_q_j2;
       Eigen::VectorXd xi_ij1j2(3);
       xi_ij1j2 << xi_i, xi_j1, xi_j2;
       Eigen::VectorXd sites_ij1j2(3);
       sites_ij1j2 << site_i, site_j1, site_j2;
       AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};
 
       //! Create functions
       potential_function functions_dipole_ij1j2 =
           V_dipole_ij1j2_adp_MgHx_constructor();
       potential_function functions_quadrupole_ij1j2 =
           V_quadrupole_ij1j2_adp_MgHx_constructor();
 
       //! @brief Compute gradient terms
       Eigen::Vector3d dV_dipole_ij1j2_dq;
       Eigen::Vector3d dV_quadrupole_ij1j2_dq;
 
       for (unsigned int direction = 0; direction < 3; direction++) {
         if (sites_ij1j2(direction) == site_i_star) {
 
           //! @brief Compute q-grad dipole angular term
           dV_dipole_ij1j2_dq.setZero();
           functions_dipole_ij1j2.dF_dq(direction, dV_dipole_ij1j2_dq.data(),
                                        xi_ij1j2.data(), mean_q_ij1j2.data(),
                                        spc_ij1j2);
 
           //! @brief Compute q-grad quadrupole angular term
           dV_quadrupole_ij1j2_dq.setZero();
           functions_quadrupole_ij1j2.dF_dq(
               direction, dV_quadrupole_ij1j2_dq.data(), xi_ij1j2.data(),
               mean_q_ij1j2.data(), spc_ij1j2);
 
           //! @brief Add up partial contributions
           D_V_i_dipol_Dq += factor_j1j2 * dV_dipole_ij1j2_dq;
           D_V_i_quad_Dq += factor_j1j2 * dV_quadrupole_ij1j2_dq;
         }
       }
     }
 #endif
   }
 
   //! @brief Compute embedding forces
   CubicSpline embed_ii;
   if (spc_i == Mg) {
     embed_ii = adp_MgMg.embed;
   } else if (spc_i == H) {
     embed_ii = adp_HH.embed;
   }
   double d_F_embed_i = d_cubic_spline(&embed_ii, rho_i);
   D_V_i_embed_Dq = xi_i * d_F_embed_i * D_rho_i_Dq;
 
   //! @brief Assembly the gradient of the potential V0 at the site i with
   //! respect the stretch tensor U
   Eigen::Vector3d D_V_i_Dq_i_star =
       D_V_i_embed_Dq + D_V_i_pair_Dq + D_V_i_dipol_Dq + D_V_i_quad_Dq;
 
   return D_V_i_Dq_i_star;
 }
 
 
 KOKKOS_FUNCTION aux_Vector evaluate_DV_i_Dq_u_adp_MgHx_Kokkos(
   unsigned int site_i_star,            //!
   unsigned int site_i,                 //!
   const PetscScalar_Matrix_Default  &mean_q,       //! Mean value of q
   const PetscScalar_Vector_Default &xi,           //! Molar fraction
   const PetscScalar_Vector_Default &mf_rho,          //! Energy density
   const AtomSpecie_Default &specie,          //! Atom
   const AtomTopology atom_topology_i,
   const aux_Vector mean_q_ij1, 
   const ThreeD_Double_View aux_view,
   const DevSnapUnmanaged &soADevice)  //!
 {
 
 unsigned int dim = NumberDimensions;
 
 //! @brief If we are in a hydrogen site with a occupancy below a certain
 //! thereshold, skip the evaluation of this equation
 double xi_i_star = xi(site_i_star);
 AtomicSpecie spc_i_star = specie(site_i_star);
 if ((spc_i_star == H) && (xi_i_star < min_occupancy)) {
   auto sub_vec = Kokkos::subview(aux_view, Kokkos::ALL(), 0, site_i_star);
   return sub_vec;
 }
 
 //! @brief Auxiliar variables
 double rho_i = mf_rho(site_i);
 auto D_V_i_embed_Dq = Kokkos::subview(aux_view, Kokkos::ALL(), 1, site_i_star);
 auto D_rho_i_Dq = Kokkos::subview(aux_view, Kokkos::ALL(), 2, site_i_star);
 auto D_V_i_pair_Dq = Kokkos::subview(aux_view, Kokkos::ALL(), 3, site_i_star);
 auto D_V_i_dipol_Dq = Kokkos::subview(aux_view, Kokkos::ALL(), 4, site_i_star);
 auto D_V_i_quad_Dq = Kokkos::subview(aux_view, Kokkos::ALL(), 5, site_i_star);
 
 D_rho_i_Dq(0) = 0.0;
 D_rho_i_Dq(1) = 0.0;
 D_rho_i_Dq(2) = 0.0;
 
 D_V_i_embed_Dq(0) = 0.0;
 D_V_i_embed_Dq(1) = 0.0;
 D_V_i_embed_Dq(2) = 0.0;
 
 D_V_i_pair_Dq(0) = 0.0;
 D_V_i_pair_Dq(1) = 0.0;
 D_V_i_pair_Dq(2) = 0.0;
 
 D_V_i_pair_Dq(0) = 0.0;
 D_V_i_pair_Dq(1) = 0.0; 
 D_V_i_pair_Dq(2) = 0.0;
 
 //! @brief Get topologic information of site i
 unsigned int numneigh_site_i = atom_topology_i.numneigh;
 const PetscInt *mech_neighs_i = atom_topology_i.mech_neighs_ptr;
 
 //! @brief Get atomistic information of site i
 AtomicSpecie spc_i = specie(site_i);
 double xi_i = xi(site_i);
 auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
 
 //! If the site is empty, skip from the evaluation
 if ((spc_i == H) && (xi_i < min_occupancy)) {
   auto sub_vec = Kokkos::subview(aux_view, Kokkos::ALL(), 0, site_i);
   return sub_vec;
 }
 
 
 //! @brief Compute the gradient of the potential with respect the
 //! mean value of the position at site i
 for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {
   //! @brief Get atomistic information of site j
   unsigned int site_j1 = mech_neighs_i[idx_j1];
   AtomicSpecie spc_j1 = specie(site_j1);
   double xi_j1 = xi(site_j1);
   auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);
 
   //! If the site is empty, skip from the evaluation
   if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
     continue;
   }
 
   //! @brief Create dof table
   int dof_table_ij[4] = {1, 0, 0, 1};
 
   //! @brief Fill data for the measure (i,j1)
 
   concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
   Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
   Kokkos::Array<int, 2>  sites_ij1 = Kokkos::Array<int, 2> {site_i, site_j1};
   AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
   
   //! @brief Compute gradient terms
   auto d_rho_ij1_dq = Kokkos::subview(aux_view, Kokkos::ALL(), 6, site_i_star);
   auto d_V_pair_ij1_dq = Kokkos::subview(aux_view, Kokkos::ALL(), 7, site_i_star);
 
   rho_ij_adp_MgHx_dispatcher functions_rho_ij { Functions_Enum::dF_dq_k, nullptr, xi_ij1.data(), mean_q_ij1.data(), spc_ij1, soADevice.data(), 0,  d_rho_ij1_dq };
   V_pair_ij_adp_MgHx_dispatcher functions_pair_ij { Functions_Enum::dF_dq_k, nullptr, xi_ij1.data(), mean_q_ij1.data(), spc_ij1, soADevice.data(), 0, d_V_pair_ij1_dq };
 
   for (unsigned int direction = 0; direction < 2; direction++) {
     if (sites_ij1[direction] == site_i_star) {
       //! @brief Compute q-grad energy density
       functions_rho_ij.direction = direction;
       functions_rho_ij();
 
       //! @brief Compute q-grad pairing term
       functions_pair_ij.direction = direction;
       functions_pair_ij();
 
       //! @brief Add up partial contributions
       multi_sum_scaled(D_V_i_pair_Dq, 1, d_V_pair_ij1_dq);
       multi_sum_scaled(D_rho_i_Dq, 1, d_rho_ij1_dq);
     }
   }
 
 #if COMPUTE_ANGULAR_TERMS == 1
     //! @brief Loop in the neighborhood considering the simmetry of the
     //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
     for (unsigned int idx_j2 = 0; idx_j2 < numneigh_site_i; idx_j2++) {
 
       //! @brief Compute atomistic information of site j2
       unsigned int site_j2 = mech_neighs_i[idx_j2];
       AtomicSpecie spc_j2 = specie(site_j2);
       double xi_j2 = xi(site_j2);
       auto mean_q_j2 = extractRowBlock(mean_q, site_j2, 0, 3);
 
       //! If the site is empty, skip from the evaluation
       if ((spc_j2 == H) && (xi_j2 < min_occupancy)) {
         continue;
       }
 
       //! Check if sites j1 and j2 are in the first lasyer of neighs
       double norm_r_ij1 = (mean_q_i - mean_q_j1).norm();
       double norm_r_ij2 = (mean_q_i - mean_q_j2).norm();
       if ((norm_r_ij1 > r_cutoff_Angular_ADP_MgHx) ||
           (norm_r_ij2 > r_cutoff_Angular_ADP_MgHx)) {
         continue;
       }
 
       //! @brief Factor to consider the simmetry of the opration
       //! a_ij1*a_ij2 = a_ij2*a_ij1
       double factor_j1j2 = idx_j1 == idx_j2 ? 1.0 : 2.0;
 
       //! @brief Fill data for the measure (i,j1,j2)
 
       auto mean_q_ij1j2 = concatenateToArray<Kokkos::View<double*>, 9>(mean_q_i, mean_q_j1, mean_q_j2);
       Kokkos::Array<double, 3>  xi_ij1j2 = Kokkos::Array<double, 3> {xi_i, xi_j1, xi_j2};
       Kokkos::Array<double, 3>  sites_ij1j2 = Kokkos::Array<double, 3> {site_i, site_j1, site_j2};
       AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};
 
       //! @brief Compute gradient terms
       auto dV_dipole_ij1j2_dq = Kokkos::subview(aux_view, Kokkos::ALL(), 8, site_i_star);
       auto dV_quadrupole_ij1j2_dq = Kokkos::subview(aux_view, Kokkos::ALL(), 9, site_i_star);
 
       V_dipole_ij1j2_dispatcher functions_dipole_ij1j2 { Functions_Enum::dF_dq_k, dV_dipole_ij1j2_dq.data(), xi_ij1j2.data(), mean_q_ij1j2.data(), spc_ij1j2, soADevice.data() };    
       V_quadrupole_ij1j2_dispatcher functions_quadrupole_ij1j2 { Functions_Enum::dF_dq_k, dV_quadrupole_ij1j2_dq.data(), xi_ij1j2.data(), mean_q_ij1j2.data(), spc_ij1j2, soADevice.data() };
 
       for (unsigned int direction = 0; direction < 3; direction++) {
         if (sites_ij1j2[direction] == site_i_star) {
 
           //! @brief Compute q-grad dipole angular term
           functions_dipole_ij1j2.direction = direction;
           functions_dipole_ij1j2();
 
           //! @brief Compute q-grad quadrupole angular term
           functions_quadrupole_ij1j2.direction = direction;
           functions_quadrupole_ij1j2();
 
           //! @brief Add up partial contributions
           multi_sum_scaled(D_V_i_dipol_Dq, factor_j1j2, dV_dipole_ij1j2_dq);
           multi_sum_scaled(D_V_i_quad_Dq, factor_j1j2, dV_quadrupole_ij1j2_dq);
         }
       }
     }
 #endif
 }
 
 //! @brief Compute embedding forces
 CubicSpline embed_ii;
 if (spc_i == Mg) {
   embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, soADevice.data());
 } else if (spc_i == H) {
   embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, soADevice.data());
 }
 
 
 double d_F_embed_i = d_cubic_spline_Kokkos(&embed_ii, rho_i);
 
 
 multi_sum_scaled(D_V_i_embed_Dq, (xi_i * d_F_embed_i), D_rho_i_Dq);
 
 //! @brief Assembly the gradient of the potential V0 at the site i with
 //! respect the stretch tensor U
 
 auto D_V_i_Dq_i_star = Kokkos::subview(aux_view, Kokkos::ALL(), 10, site_i_star);
 
 D_V_i_Dq_i_star(0) = 0.0;
 D_V_i_Dq_i_star(1) = 0.0; 
 D_V_i_Dq_i_star(2) = 0.0;
 
 multi_sum_scaled(D_V_i_Dq_i_star, 1, D_V_i_embed_Dq, D_V_i_pair_Dq, D_V_i_dipol_Dq, D_V_i_quad_Dq);

 return D_V_i_Dq_i_star;
 
 }
 /********************************************************************************/
 
  KOKKOS_FUNCTION Kokkos::Array<double, 3> evaluate_DV_i_Dq_u_adp_MgHx_Kokkos_SIMD(
   unsigned int site_i_star,            //!
   unsigned int site_i,                 //!
   const PetscScalar_Matrix_Default  &mean_q,       //! Mean value of q
   const PetscScalar_Vector_Default &xi,           //! Molar fraction
   const PetscScalar_Vector_Default &mf_rho,          //! Energy density
   const AtomSpecie_Default &specie,          //! Atom
   const AtomTopology atom_topology_i,
   const DevSnapUnmanaged &soADevice)  //!
 {
 using simd_t = Kokkos::Experimental::native_simd<double>;
 using simd_int_t = Kokkos::Experimental::native_simd<int>;
 using mask_t     = typename simd_int_t::mask_type;
 constexpr int W   = simd_t::size();
 unsigned int dim = NumberDimensions;
 Kokkos::Array<double, 3>  D_V_i_Dq_i_star = {0.0, 0.0, 0.0};
 Kokkos::Array<double,3> result_d_rho_ij_dq = {0.0, 0.0, 0.0};
 Kokkos::Array<double,3> result_dV_pair_ij_dq = {0.0, 0.0, 0.0};
 CubicSpline rho_j_Mg = getSpline(AdpType::MgMg, SplineType::rho, CubicSpline(), soADevice.data());
 CubicSpline rho_j_H  = getSpline(AdpType::HH,  SplineType::rho, CubicSpline(), soADevice.data());
 CubicSpline pair_ij_MgMg = getSpline(AdpType::MgMg, SplineType::pair, CubicSpline(), soADevice.data());
 CubicSpline pair_ij_HH = getSpline(AdpType::HH, SplineType::pair, CubicSpline(), soADevice.data());
 CubicSpline pair_ij_MgH = getSpline(AdpType::MgH, SplineType::pair, CubicSpline(), soADevice.data());

 double xi_i_star = xi(site_i_star);
 double q_i0 = mean_q(site_i,0);
 double q_i1 = mean_q(site_i,1);
 double q_i2 = mean_q(site_i,2);
 bool isStar = (site_i_star == site_i);
 simd_int_t v_site(site_i_star);
 simd_t v_xi(xi(site_i_star));
 simd_t qi0(q_i0);
 simd_t qi1(q_i1);
 simd_t qi2(q_i2); 
 AtomicSpecie spc_i_star = specie(site_i_star);
 if ((spc_i_star == H) && (xi_i_star < min_occupancy)) {
   return D_V_i_Dq_i_star;
 }
 
 double rho_i = mf_rho(site_i);
 
 unsigned int numneigh_site_i = atom_topology_i.numneigh;
 const PetscInt *mech_neighs_i = atom_topology_i.mech_neighs_ptr;
 
 AtomicSpecie spc_i = specie(site_i);
 simd_int_t v_spc_i(static_cast<int>(spc_i));
 double xi_i = xi(site_i);
 auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
 
 if ((spc_i == H) && (xi_i < min_occupancy)) {
   return D_V_i_Dq_i_star;
 }
 
 unsigned int idx_j1 = 0;
 for (; idx_j1 + W < 1; idx_j1+= W) {

      simd_t qj0;
      simd_t qj1;
      simd_t qj2;
      simd_t n;
      simd_int_t spc_simd;
      simd_int_t site_simd;
      site_simd.copy_from(mech_neighs_i + idx_j1, Kokkos::Experimental::simd_flag_default);

      for (int lane = 0; lane < W; ++lane) {
        qj0[lane] = mean_q(site_simd[lane],0);
        qj1[lane] = mean_q(site_simd[lane],1);
        qj2[lane] = mean_q(site_simd[lane],2);
        n[lane]   = xi(site_simd[lane]);
        spc_simd[lane] = int(specie(site_simd[lane]));
      }
  
      simd_t dr0 = qi0 - qj0;
      simd_t dr1 = qi1 - qj1;
      simd_t dr2 = qi2 - qj2;

      simd_t r2  = Kokkos::fma(dr0, dr0,
                   Kokkos::fma(dr1, dr1, dr2*dr2));
      simd_t norm = Kokkos::sqrt(r2);

      mask_t m_j = (site_simd == v_site);
      mask_t nz = (norm > simd_t(0.0));

      simd_t inv_norm_raw = simd_t(1.0) / norm;

      simd_t inv_norm = Kokkos::Experimental::condition(nz, inv_norm_raw, simd_t(0.0));
      simd_t d_pair;

      simd_t d_rho_Mg = d_cubic_spline_Kokkos_SIMD(&rho_j_Mg, norm);
      simd_t d_rho_H  = d_cubic_spline_Kokkos_SIMD(&rho_j_H,  norm);

      mask_t is_Mg = (spc_simd == int(Mg));

      simd_t d_rho = Kokkos::Experimental::condition(is_Mg, d_rho_Mg, d_rho_H);

      simd_t fact_rho = n * d_rho * inv_norm;
      simd_t c0 = dr0 * fact_rho;
      simd_t c1 = dr1 * fact_rho;
      simd_t c2 = dr2 * fact_rho;
 
    for (unsigned int direction = 0; direction < 2; direction++) {
     for (int lane = 0; lane < W; ++lane) {
        unsigned site_sel = direction == 0 ? site_i : site_simd[lane];
        if (site_sel != site_i_star) continue;
        if (direction == 0) {
          result_d_rho_ij_dq[0] += c0[lane];
          result_d_rho_ij_dq[1] += c1[lane];
          result_d_rho_ij_dq[2] += c2[lane];
        } else {
          result_d_rho_ij_dq[0] -= c0[lane];
          result_d_rho_ij_dq[1] -= c1[lane];
          result_d_rho_ij_dq[2] -= c2[lane];
        }
   }
  }

    mask_t mask_MgMg = ((spc_simd == int(Mg)) && (v_spc_i == int(Mg)));
    mask_t mask_HH   = ((spc_simd == int(H)) && (v_spc_i == int(H)));

    simd_int_t any_int = Kokkos::Experimental::condition(mask_MgMg, simd_int_t(1), simd_int_t(0)); 
    any_int = Kokkos::Experimental::condition(mask_HH, simd_int_t(1), any_int);    

    mask_t mask_else = any_int == simd_int_t(0);

    simd_t d_pair_MgMg = d_cubic_spline_Kokkos_SIMD(&pair_ij_MgMg, norm);
    simd_t d_pair_HH   = d_cubic_spline_Kokkos_SIMD(&pair_ij_HH,   norm);
    simd_t d_pair_MgH  = d_cubic_spline_Kokkos_SIMD(&pair_ij_MgH,  norm);

    d_pair = Kokkos::Experimental::condition(mask_MgMg, d_pair_MgMg, simd_t(0.0));
    d_pair = Kokkos::Experimental::condition(mask_HH, d_pair_HH, d_pair);
    d_pair = Kokkos::Experimental::condition(mask_else, d_pair_MgH, d_pair);

    simd_t fact_pair = v_xi * n * d_pair * inv_norm * simd_t(0.5);
    simd_t p0 = dr0 * fact_pair;
    simd_t p1 = dr1 * fact_pair;
    simd_t p2 = dr2 * fact_pair;
    
    for (unsigned int direction = 0; direction < 2; direction++) {
     for (int lane = 0; lane < W; ++lane) {
        unsigned site_sel = direction == 0 ? site_i : site_simd[lane];
        if (site_sel != site_i_star) continue;
        if (direction == 0) {
          result_dV_pair_ij_dq[0] += p0[lane];
          result_dV_pair_ij_dq[1] += p1[lane];
          result_dV_pair_ij_dq[2] += p2[lane];
        } else {
          result_dV_pair_ij_dq[0] -= p0[lane];
          result_dV_pair_ij_dq[1] -= p1[lane];
          result_dV_pair_ij_dq[2] -= p2[lane];
        }
   }
  }   
 }
 for (; idx_j1 < numneigh_site_i; ++idx_j1) {
  unsigned site_j1 = mech_neighs_i[idx_j1];
  Kokkos::Array<int, 2>  sites_ij1 = Kokkos::Array<int, 2> {site_i, site_j1};
  double d0 = q_i0 - mean_q(site_j1,0);
  double d1 = q_i1 - mean_q(site_j1,1);
  double d2 = q_i2 - mean_q(site_j1,2);
  double r2 = d0*d0 + d1*d1 + d2*d2;
  if (r2==0.0) continue;
  double r    = Kokkos::sqrt(r2);
  double invr = 1.0/r;
  AtomicSpecie spcj = specie(site_j1);
  double nij  = xi(site_j1);
  double dpair = nij * d_cubic_spline_Kokkos(
                    (spcj==Mg? &rho_j_Mg : &rho_j_H), r) * invr;
  for (int dir=0; dir<2; ++dir) {
    if (sites_ij1[dir] == site_i_star) {
      double direction = dir==0?+1.0:-1.0;
      result_d_rho_ij_dq[0] += direction * dpair * d0;
      result_d_rho_ij_dq[1] += direction * dpair * d1;
      result_d_rho_ij_dq[2] += direction * dpair * d2;
    }
  }
  double factor = xi_i * nij * d_cubic_spline_Kokkos(
                    (spcj==Mg && spc_i==Mg)
                       ? &pair_ij_MgMg
                       : (spcj==H && spc_i==H)
                         ? &pair_ij_HH
                         : &pair_ij_MgH, r) * invr * 0.5;
  for (int dir=0; dir<2; ++dir) {
    if (sites_ij1[dir] == site_i_star) {
      double direction = dir==0?+1.0:-1.0;
      result_dV_pair_ij_dq[0] +=  direction * factor * d0;
      result_dV_pair_ij_dq[1] +=  direction * factor * d1;
      result_dV_pair_ij_dq[2] +=  direction * factor * d2;
    }
  }
} 
 CubicSpline embed_ii;
 if (spc_i == Mg) {
   embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, soADevice.data());
 } else if (spc_i == H) {
   embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, soADevice.data());
 }

 double d_F_embed_i = d_cubic_spline_Kokkos(&embed_ii, rho_i);

 D_V_i_Dq_i_star[0] += xi_i * d_F_embed_i * result_d_rho_ij_dq[0] + result_dV_pair_ij_dq[0];
 D_V_i_Dq_i_star[1] += xi_i * d_F_embed_i * result_d_rho_ij_dq[1] + result_dV_pair_ij_dq[1];
 D_V_i_Dq_i_star[2] += xi_i * d_F_embed_i * result_d_rho_ij_dq[2] + result_dV_pair_ij_dq[2];

 return D_V_i_Dq_i_star;

}




