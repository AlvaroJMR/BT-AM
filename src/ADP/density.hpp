/**
 * @file MgHx-mf-V-bulk.hpp
 * @author Miguel Molinos (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-05-19
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef mf_ADP_MgHx_kokkos_HPP
#define mf_ADP_MgHx_kokkos_HPP

#include "ADP/MgHx-ADP.hpp"
#include "Macros.hpp"
#include <Eigen/Dense>
#include <Kokkos_Core.hpp>

/**
 * @brief
 *
 * @param site_i
 * @param mean_q:Mean value of q
 * @param xi Molar fraction
 * @param specie Atomic specie
 * @param atom_topology_i List of neighs
 * @return double
 */
KOKKOS_INLINE_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos_Device(unsigned int site_i,
    const PetscScalar_Matrix_Default  &mean_q,
    const PetscScalar_Vector_Default &xi,
    const AtomSpecie_Default &specie,
    const AtomTopologyKokkos atom_topology_i,
    const View_Double_Vector_Device mean_q_ij1,
    const AdpPotencial_Device adp_Device_Default);

KOKKOS_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos(unsigned int site_i,           //!
    const Eigen::MatrixXd &mean_q, //!
    const Eigen::VectorXd &xi,     //!
    const AtomicSpecie *specie,    //!
    const AtomTopology atom_topology_i);

double evaluate_V_i_adp_MgHx(unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //!
    const Eigen::VectorXd& xi,           //!
    const Eigen::VectorXd& rho,          //!
    const AtomicSpecie* specie,          //!
    const AtomTopology atom_topology_i);

KOKKOS_FUNCTION double evaluate_V_i_adp_MgHx_Kokkos(unsigned int site_i,                 //!
    const PetscScalar_Matrix_Default  &mean_q,
    const PetscScalar_Vector_Default &xi,
    const PetscScalar_Vector_Default& rho,
    const AtomSpecie_Default &specie,
    const AtomTopologyKokkos atom_topology_i,
    const AdpPotencial_Device adp_Device_Default,
    const View_Double_Vector_Device mean_q_ij1);
      

double evaluate_mf_rho_i_adp_MgHx(unsigned int site_i,            //!
    const Eigen::MatrixXd& mean_q,  //!
    const Eigen::VectorXd& stdv_q,  //!
    const Eigen::VectorXd& xi,      //!
    const AtomicSpecie* specie,     //!
    const AtomTopology atom_topology_i);

KOKKOS_FUNCTION double evaluate_mf_rho_i_adp_MgHx_Kokkos(unsigned int site_i,                 //!
    const PetscScalar_Matrix_Default  &mean_q,
    const PetscScalar_Vector_Default &stdv_q,
    const PetscScalar_Vector_Default &xi,
    const AtomSpecie_Default &specie,
    const AtomTopologyKokkos atom_topology_i,
    const View_Double_Vector_Device mean_q_ij1,
    const AdpPotencial_Device adp_Device_Default);
    
double evaluate_S0_i_adp_MgHx(unsigned int site_i,                 //!
    const Eigen::MatrixXd& mean_q,       //!
    const Eigen::VectorXd& stdv_q,       //!
    const Eigen::VectorXd& xi,           //!
    const Eigen::VectorXd& mf_rho,       //!
    const Eigen::VectorXd& beta,         //!
    const Eigen::VectorXd& gamma,        //!
    const AtomicSpecie* specie,          //!
    const AtomTopology atom_topology_i); 
    

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
    const View_Double_Vector_Device element_mass);     
#endif /* mf_ADP_MgHx_kokkos_HPP */