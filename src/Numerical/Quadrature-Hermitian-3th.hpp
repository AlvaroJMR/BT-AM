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


template<typename Dispatcher_SIMD, typename Dispatcher>
KOKKOS_INLINE_FUNCTION
void meanfield_integral_gh3th_Kokkos_SIMD(
  double                              integral_f[W],
  gaussian_measure_ctx_kokkos_s ctxs[W],
  SoADevice*                          soADevice,
  Dispatcher_SIMD                      function_SIMD,
  Dispatcher                           function)
{
  using simd_t = Kokkos::Experimental::native_simd<double>;
  constexpr int W   = simd_t::size();
  constexpr int dim       = NumberDimensions;
  constexpr double inv_sqrt2 = 1.0 / sqrt_2;

  for (int lane = 0; lane < W; ++lane) {
    integral_f[lane] = 0.0;
  }

  int dims[W];
  for (int lane = 0; lane < W; ++lane) {
    dims[lane] = ctxs[lane].intergal_dim;
  }
  int min_dim    = dims[0];
  int lane_max   = 0;
  for (int lane = 1; lane < W; ++lane) {
    if (dims[lane] < min_dim)       min_dim  = dims[lane];
    if (dims[lane] > dims[lane_max]) lane_max = lane;
  }

  static double q_l_arr[W][dim * 3];
  double        f_gp[W];
  double        Wgp[W];
  const double* n_ptrs[W];
  const double* q_ptrs[W];
  const AtomicSpecie* spc_ptrs[W];

  for (int lane = 0; lane < W; ++lane) {
    n_ptrs[lane]   = ctxs[lane].xi_ij;
    spc_ptrs[lane] = ctxs[lane].spc;
    Wgp[lane]      = 1.0 / (2.0 * dims[lane]);
  }

  for (int l = 0; l < min_dim; ++l) {
    for (int lane = 0; lane < W; ++lane) {
      auto const& ctx       = ctxs[lane];
      int          num_sites= ctx.num_sites;
      const int*   gp       = ctx.gp_board;
      const double* mq      = ctx.mean_q_ij;
      const double* sd      = ctx.stddev_q_ij;
      double        zeta    = Kokkos::sqrt((double)dims[lane]) * inv_sqrt2;

      for (int site = 0; site < num_sites; ++site) {
        for (int a = 0; a < dim; ++a) {
          q_l_arr[lane][site*dim + a] =
            mq[site*dim + a]
            + sqrt_2 * sd[site] * zeta
              * gp[l*num_sites*dim + site*dim + a];
        }
      }
      q_ptrs[lane] = q_l_arr[lane];
    }

    function_SIMD.value = f_gp;
    function_SIMD.n_ptrs = n_ptrs;
    function_SIMD.q_ptrs = q_ptrs;
    function_SIMD.spc_ptrs = spc_ptrs;
    function_SIMD.soADevice = soADevice;
    function_SIMD();

    for (int lane = 0; lane < W; ++lane) {
      integral_f[lane] += f_gp[lane] * Wgp[lane];
    }

  }

  for (int l = 0; l < min_dim; ++l) {
    for (int lane = 0; lane < W; ++lane) {
      auto const& ctx       = ctxs[lane];
      int          num_sites= ctx.num_sites;
      const int*   gp       = ctx.gp_board;
      const double* mq      = ctx.mean_q_ij;
      const double* sd      = ctx.stddev_q_ij;
      double        zeta    = Kokkos::sqrt((double)dims[lane]) * inv_sqrt2;

      for (int site = 0; site < num_sites; ++site) {
        for (int a = 0; a < dim; ++a) {
          q_l_arr[lane][site*dim + a] =
            mq[site*dim + a]
            - sqrt_2 * sd[site] * zeta
              * gp[l*num_sites*dim + site*dim + a];
        }
      }
      q_ptrs[lane] = q_l_arr[lane];
    }

    function_SIMD.value = f_gp;
    function_SIMD.n_ptrs = n_ptrs;
    function_SIMD.q_ptrs = q_ptrs;
    function_SIMD.spc_ptrs = spc_ptrs;
    function_SIMD.soADevice = soADevice;
    function_SIMD();

    for (int lane = 0; lane < W; ++lane) {
      integral_f[lane] += f_gp[lane] * Wgp[lane];
    }
  }

    if (dims[lane_max] != min_dim) {
    auto const& ctx        = ctxs[lane_max];
    int          num_sites = ctx.num_sites;
    const int*   gp        = ctx.gp_board;
    const double* mq       = ctx.mean_q_ij;
    const double* sd       = ctx.stddev_q_ij;
    const double* n_ptr    = ctx.xi_ij;
    const AtomicSpecie* spc_ptr = ctx.spc;
    double        Wg       = Wgp[lane_max];
    double        f_val;

    for (int sign_z : {+1, -1}) {
      double zeta = std::sqrt((double)dims[lane_max]) * inv_sqrt2;
      for (int l = min_dim; l < dims[lane_max]; ++l) {
        for (int site = 0; site < num_sites; ++site) {
          for (int a = 0; a < dim; ++a) {
            q_l_arr[lane_max][site*dim + a] =
              mq[site*dim + a]
              + sign_z * sqrt_2 * sd[site] * zeta
                * gp[l*num_sites*dim + site*dim + a];
          }
        }
        // prepara y llama la versión escalar
        function.value = &f_val;
        function.n                 = n_ptr;
        function.q                 = q_l_arr[lane_max];
        function.spc               = spc_ptr;
        function.soADevice         = soADevice;
        function();

        integral_f[lane_max] += f_val * Wg;
      }
    }
  }
}

#endif // quadrature_hermite_3th_HPP