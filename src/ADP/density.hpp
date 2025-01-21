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
double evaluate_rho_i_adp_MgHx_kokkos(unsigned int site_i,           //!
                                      const Eigen::MatrixXd &mean_q, //!
                                      const Eigen::VectorXd &xi,     //!
                                      const AtomicSpecie *specie,    //!
                                      const AtomTopology atom_topology_i);

#endif /* mf_ADP_MgHx_kokkos_HPP */