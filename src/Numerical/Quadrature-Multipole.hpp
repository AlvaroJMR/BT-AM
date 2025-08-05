/**
 * @file Quadrature-Multipole.hpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief
 * @version 0.1
 * @date 2023-03-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef quadrature_hermite_mp_HPP
#define quadrature_hermite_mp_HPP

#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include "ADP/MgHx-ADP.hpp" 

/**
 * @brief Multipole quadrature of the mean-field integral of a n-d function
 *
 * @param integral_f Value of the integral
 * @param function structure which contain function to pointer
 * @param ctx Integral auxiliar variables
 */
void meanfield_integral_mp(double *integral_f, potential_function function,
                           void *ctx);

/**
 * @brief Multipole quadrature of the mean-field integral of a n-d function
 *
 * @param direction Direction of the n-d function
 * @param integral_f_ds Value of the integral
 * @param function structure which contain function to pointer
 * @param ctx Integral auxiliar variables
 */
void meanfield_integral_mp_dsq(int direction, double *integral_f_ds,
                               potential_function function, void *ctx);

/**
 * @brief Multipole quadrature of the mean-field integral of a n-d grad-q
 * function
 *
 * @param direction Direction of the n-d function
 * @param integral_grad_f Integral of the gradient function
 * @param function structure which contain function to pointer
 * @param ctx Integral auxiliar variables
 */
void meanfield_integral_mp_dmq(int direction, double *integral_grad_f,
                               potential_function function, void *ctx_measure);

template<typename Dispatcher>
KOKKOS_INLINE_FUNCTION void meanfield_integral_mp_Kokkos(double* integral_f,
                           void* ctx_measure, SoADevice *soADevice, Dispatcher function) {

  *integral_f = 0.0;

  unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx*)ctx_measure)->intergal_dim;
  const int* dof_table = ((gaussian_measure_ctx*)ctx_measure)->dof_table;
  double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;
  // clang-format on

  //! 0 contribution
  double c_0 = 1.0;
  double f_0 = 0.0;

  function.functions_Enum = Functions_Enum::FK;
  function.value = &f_0;
  function.n = xi_ij;
  function.q = mean_q_ij;
  function.spc = spc;
  function.soADevice = soADevice;
  function();
  

  function();

  *integral_f += c_0 * f_0;

  //! 2 contribution
  double hess_f_0[dim * dim];
  for (unsigned int site_idx_i = 0; site_idx_i < num_sites; site_idx_i++) {

    double c_2_idx_i = dsqr(sigma[site_idx_i]) / 2.0;

    for (unsigned int site_idx_j = 0; site_idx_j < num_sites; site_idx_j++) {

      int direction = site_idx_i * num_sites + site_idx_j;

      if (dof_table[direction] == 1) {

#ifdef NUMERICAL_DERIVATIVES
        function.functions_Enum = Functions_Enum::d2F_dq2_FD;
        function.direction = direction;
        function.value = hess_f_0;
        function();
        // function.d2F_dq2_FD(direction, hess_f_0, xi_ij, mean_q_ij, spc);
#else
        function.functions_Enum = Functions_Enum::d2F_dq2;
        function.direction = direction;
        function.value = hess_f_0;
        function();
        // function.d2F_dq2(direction, hess_f_0, xi_ij, mean_q_ij, spc);
#endif

        for (unsigned int alpha = 0; alpha < dim; alpha++) {
          *integral_f += c_2_idx_i * hess_f_0[alpha * dim + alpha];
        }
      }
    }
  }

}
                           
#endif // quadrature_hermite_mp_HPP