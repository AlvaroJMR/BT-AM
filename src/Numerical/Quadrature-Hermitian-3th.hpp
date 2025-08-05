/**
 * @file Quadrature-Hermitian-3th.hpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief Hermitian quadrature of a 6d/9d function using a third order
 * quadrature
 * @version 0.1
 * @date 2022-11-08
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef quadrature_hermite_3th_HPP
#define quadrature_hermite_3th_HPP

#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include "ADP/MgHx-ADP.hpp" 

/**
 * @brief Hermitian quadrature of a n-d function using a third order quadrature
 *
 * @param integral_f Value of the integral
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 * @return Value of the integral
 */
void meanfield_integral_gh3th(double *integral_f, potential_function function,
                              void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function using a third order quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_f Value of the integral
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 * @return Value of the integral
 */
void meanfield_integral_gh3th_dsq(int direction, double *integral_f,
                                  potential_function function,
                                  void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function gradient using a third order
 * quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_grad_f Gradient of the function f to integrate
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 */
void meanfield_integral_gh_3th_dmq(int direction, double *integral_grad_f,
                                   potential_function function,
                                   void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function gradient using a third order
 * quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_grad_f Gradient of the function f to integrate
 * @param function structure which contain function to pointer
 * @param ctx_measure Integral auxiliar variables
 */
void meanfield_integral_gh_3th_dxi(int direction, double *integral_grad_f,
                                   potential_function function,
                                   void *ctx_measure);

/**
 * @brief Hermitian quadrature of a n-d function hessian using a third order
 * quadrature
 *
 * @param direction Direction of the n-d function
 * @param integral_hess_f Hessian of the function f to integrate
 * @param function structure which contain function to pointer
 * @param ctx Integral auxiliar variables
 */
void meanfield_integral_gh_3th_d2mq(int direction, double *integral_hess_f,
                                    potential_function function,
                                    void *ctx_measure);

/********************************************************************************/
template<typename Dispatcher>
KOKKOS_INLINE_FUNCTION void meanfield_integral_gh3th_Kokkos(double* integral_f, 
                              void* ctx_measure, SoADevice *soADevice, Dispatcher function) {

  //! Set to zero the value of the integral
  *integral_f = 0.0;

  constexpr unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx_kokkos*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx_kokkos*)ctx_measure)->intergal_dim;
  const int* gp_board = ((gaussian_measure_ctx_kokkos*)ctx_measure)->gp_board.data();
  const double* mean_q_ij = ((gaussian_measure_ctx_kokkos*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx_kokkos*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx_kokkos*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx_kokkos*)ctx_measure)->spc;
  // clang-format on

  //! Weight
  double W_gp = 1.0 / (2.0 * intergal_dim);

  //! Quadrature points
  //! zeta = (+,-) sqrt(intergal_dim/2)
  double zeta_l = sqrt((double)intergal_dim) / sqrt_2;

  //! In this integration rule we use 2 gp's for each dof
  double q_l[dim * 3];
  double f_gp;

  //! Positive contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {
        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] +
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }
  
    //! Evaluate the function using the modified position of the
    //! integration space
  function.functions_Enum = Functions_Enum::FK;
  function.value = &f_gp;
  function.n = xi_ij;
  function.q = q_l;
  function.spc = spc;
  function.soADevice = soADevice;
  function();  

  // function.F(&f_gp, xi_ij, q_l, spc);

    *integral_f += f_gp * W_gp;
  }


  //! Negative contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] -
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function();
    //function.F(&f_gp, xi_ij, q_l, spc);
    *integral_f += f_gp * W_gp;
  }

}


template<typename Dispatcher>
KOKKOS_INLINE_FUNCTION void meanfield_integral_gh3th_Kokkos_s(double* integral_f, 
                              void* ctx_measure, SoADevice *soADevice, Dispatcher function) {

  //! Set to zero the value of the integral
  *integral_f = 0.0;

  constexpr unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx_kokkos_s*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx_kokkos_s*)ctx_measure)->intergal_dim;
  const int* gp_board = ((gaussian_measure_ctx_kokkos_s*)ctx_measure)->gp_board;
  const double* mean_q_ij = ((gaussian_measure_ctx_kokkos_s*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx_kokkos_s*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx_kokkos_s*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx_kokkos_s*)ctx_measure)->spc;
  // clang-format on

  //! Weight
  double W_gp = 1.0 / (2.0 * intergal_dim);

  //! Quadrature points
  //! zeta = (+,-) sqrt(intergal_dim/2)
  double zeta_l = sqrt((double)intergal_dim) / sqrt_2;

  //! In this integration rule we use 2 gp's for each dof
  double q_l[dim * 3];
  double f_gp;

  //! Positive contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {
        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] +
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }
  
    //! Evaluate the function using the modified position of the
    //! integration space
  function.functions_Enum = Functions_Enum::FK;
  function.value = &f_gp;
  function.n = xi_ij;
  function.q = q_l;
  function.spc = spc;
  function.soADevice = soADevice;
  function();  
  // function.F(&f_gp, xi_ij, q_l, spc);

    *integral_f += f_gp * W_gp;
  }


  //! Negative contribution to the integral
  for (unsigned int l = 0; l < intergal_dim; l++) {

    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int alpha = 0; alpha < dim; alpha++) {

        q_l[site_idx * dim + alpha] =
            mean_q_ij[site_idx * dim + alpha] -
            sqrt_2 * sigma[site_idx] * zeta_l *
                gp_board[l * num_sites * dim + site_idx * dim + alpha];
      }
    }

    //! Evaluate the function using the modified position of the
    //! integration space
    function();
    //function.F(&f_gp, xi_ij, q_l, spc);
    *integral_f += f_gp * W_gp;
  }

}


#endif // quadrature_hermite_3th_HPP