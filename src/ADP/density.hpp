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
KOKKOS_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos_Device(unsigned int site_i,
    const PetscScalar_Matrix_Default  &mean_q,
    const PetscScalar_Vector_Default &xi,
    const AtomSpecie_Default &specie,
    const AtomTopology atom_topology_i,
    const View_Double_Vector_Device mean_q_ij1);

double evaluate_rho_i_adp_MgHx_kokkos(unsigned int site_i,           //!
                                      const Eigen::MatrixXd &mean_q, //!
                                      const Eigen::VectorXd &xi,     //!
                                      const AtomicSpecie *specie,    //!
                                      const AtomTopology atom_topology_i);

#endif /* mf_ADP_MgHx_kokkos_HPP */