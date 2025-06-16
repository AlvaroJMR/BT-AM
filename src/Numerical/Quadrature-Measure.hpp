/**
 * @file Quadrature-Measure.hpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief Gaussian meassure
 * @version 0.1
 * @date 2023-01-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef Quadrature_Measure_HPP
#define Quadrature_Measure_HPP

// clang-format off
#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
// clang-format on

#include <Kokkos_Core.hpp>

/**
 * @brief Arguments of the integral
 *
 */
typedef struct {

  /*! @param num_sites: Number of sites */
  int num_sites;

  /*! @param intergal_dim: Number of dimensions of the integral */
  int intergal_dim;

  /*! @param dof_table: Table with the active degree of freedom */
  int *dof_table;

  /*! @param gp_board: Integration order */
  int *gp_board;

  /*! @param mean_q_ij: Mean value of q at site i and j */
  double *mean_q_ij;

  /*! @param stddev_q_ij: Standard desviation of q at site i and j */
  double *stddev_q_ij;

  /*! @param xi_ij: Molar fraction of sites i and j */
  double *xi_ij;

  /*! @param AtomicSpecie: List of atomic species of each site */
  AtomicSpecie *spc;

} gaussian_measure_ctx;

/* Gaussian measure for Kokkos*/

typedef struct {

  int num_sites;

  int intergal_dim;

  Int_Vector_Default dof_table;

  Int_Vector_Default dof_table_aux;

  Int_Vector_Default gp_board;

  Int_Vector_Default active_dof;

  double *mean_q_ij;

  double *stddev_q_ij;

  double *xi_ij;

  AtomicSpecie *spc;

} gaussian_measure_ctx_kokkos;

/**
 * @brief Fill out integral context
 *
 * @param mean_q_ij Mean value of q at site i and j
 * @param stddev_q_ij Standard desviation of q at sites i and j
 * @param xi_ij Molar fraction of sites i and j
 * @param AtomicSpecie
 * @param dof_table Table with the relation of i, j and k
 * @param NumSites Number of sites to evaluate the integral
 */
gaussian_measure_ctx fill_out_gaussian_measure(double *mean_q_ij,   //!
                                               double *stddev_q_ij, //!
                                               double *xi_ij,       //!
                                               AtomicSpecie *spc,   //!
                                               int *dof_table,      //!
                                               unsigned int NumSites);

/**
 * @brief Free memory inside of the gaussian measure variable
 *
 * @param measure
 */
void destroy_gaussian_measure(gaussian_measure_ctx *measure);

KOKKOS_FUNCTION void fill_out_gaussian_measure_Kokkos (double* mean_q_ij,
                                               double* stddev_q_ij,
                                               double* xi_ij, AtomicSpecie* spc,
                                               int* dof_table,
                                               unsigned int NumSites,
                                               gaussian_measure_ctx_kokkos* ctx); 

typedef Kokkos::View<gaussian_measure_ctx_kokkos*, DefaultLayout, HostMemorySpace> gaussian_measure_ctx_Host;
typedef Kokkos::View<gaussian_measure_ctx_kokkos*, DefaultLayout, DefaultMemorySpace> gaussian_measure_ctx_Default;                                               

#endif // Quadrature_Measure_HPP