
#include <cstdlib>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include "Numerical/Quadrature-Multipole.hpp"
#include <fstream>
#include <iostream>  //std::cout//std::cin
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "ADP/MgHx-ADP.hpp"
#include "Utils/KokkosUtils.hpp" 


using namespace std;

/**
 * @file Quadrature-Multipole.cpp
 * @brief Numerical quadrature of nd function using a multipole-based quadrature
 A general scheme for the approximation of integrals that is well-suited to the
 evaluation of phase-space, \f$\{{z}\} \equiv (\{{q}\}, \{{p}\})\f$, averages is
 multipole expansion.  Let \f$\mu\f$ be  a bounded measure over
 \f$\mathbb{R}^n\f$ and let \f$f\ \in\ C(\mathbb{R}^n)\f$.  We write the action
of the measure \f$\mu\f$ on \f$f\f$ as:
\f[
\mu(f)\ =\ \int_{-\infty}^{+\infty} f(\{z\}) d\mu(\{z\}).
\f]
with \f$z_i \equiv (q_i,p_i)\f$, \f$\bar{z}_i = (\bar{q}_i,\bar{p}_i)\f$,
\f$z_i' = z_i - \bar{z}_i\f$. Conveniently, the m\f$^{th}\f$ order multipole
expansion of the measure is:
\f[
    \mu_m(f)\ =\ \int_{-\infty}^{+\infty} P_m(\{z'\})  f(\{z'\})
d\mu(\{z'\}).
\f]
Where \f$P_m(\{{z}\})\f$ is defined as a linear
combination of Hermite polynomials of degree less or equal m
\f[
        P_m(\{{z}\})\
        =\
        \sum_{|\alpha|\leq m}
        c_\alpha
        \exp\left( \frac{1}{2} \{{z}\}^T Q \{{z}\} \right)
        D^\alpha
        \exp\left( - \frac{1}{2} \{{z}\}^T Q \{{z}\} \right)\
        =\
        \sum_{|\alpha|\leq m}
        c_\alpha
        H_\alpha(\{{z}\}),
\f]
where \f$\alpha\f$ is a multiindex in \f$\mathbb{N}^n_0\f$, and
\f[
    \{z'\}^T Q \{z'\}\
    \equiv\
    \sum_{i=1}^N \frac{1}{2\bar{\sigma}_i^2}|{q}_i - \bar{{q}}_i|^2\
    +\
    \sum_{i=1}^N \frac{1}{2k_{\text{B}}T m_i}|{p}_i - \bar{{p}}_i|^2 .
\f]
\f$D^\alpha\f$ is the multiindex derivative. The \cref{eq:Rodriges-formula-1D}
can be easily extended to higher dimensions when the varibles are independent.
For the particular case \f$z_i = q_i\f$,
\cref{eq:multipole-expansion-polynomial} reduces to \f[ P_m(\{{z}\})
    =
    \sum_{|\alpha|\leq m}
    c_\alpha
    \prod_{i=1}^{N} \exp\left(\frac{1}{2\bar{\sigma}_i^2}|{q}_i - \bar{{q}}_i|^2
\right)
    D^\alpha \exp\left(- \frac{1}{2\bar{\sigma}_i^2}|{q}_i - \bar{{q}}_i|^2
\right)\
    =\
    \sum_{|\alpha|\leq m}
    c_\alpha
    \prod_{i=1}^{N} H_\alpha(q_i),
\f]
The multipole approximation of degree \f$k\f$ of \f$\mu\f$ is the measure
\f$\mu_k\f$ such that: \f[ \mu_k(f)\ =\ \sum_{|\alpha| \leq k} c_{\alpha} \,
D^{\alpha} \, f(0) \f] where the coefficients \f$\{c_{\alpha},  |\alpha| \leq k
\}\f$ of the approximation are chosen such that the approximation is exact for
polynomials of degree \f$\leq k\f$,  {\it i.e.},  such that: \f[ \mu_k
(x^{\alpha})\ =\ \mu (x^{\alpha}), \quad |\alpha| \leq k \f] This requirement
gives \f[ c_{\alpha}\ =\ \frac{1}{\alpha !} \int_{-\infty}^{+\infty}  x^{\alpha}
d\mu(x). \f]
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper)), Pilar
Ariza ([mpariza](https://github.com/mpariza)), Miguel Ortiz
([mortizcaltech](https://github.com/mortizcaltech))
 * @version 0.1
 * @date 2023-06-19
 *
 * @copyright Copyright (c) 2023
 *
 */

/********************************************************************************/

void meanfield_integral_mp(double* integral_f, potential_function function,
                           void* ctx_measure) {

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
  function.F(&f_0, xi_ij, mean_q_ij, spc);

  *integral_f += c_0 * f_0;

  //! 2 contribution
  double* hess_f_0 = (double*)calloc(dim * dim, sizeof(double));
  for (unsigned int site_idx_i = 0; site_idx_i < num_sites; site_idx_i++) {

    double c_2_idx_i = DSQR(sigma[site_idx_i]) / 2.0;

    for (unsigned int site_idx_j = 0; site_idx_j < num_sites; site_idx_j++) {

      int direction = site_idx_i * num_sites + site_idx_j;

      if (dof_table[direction] == 1) {

#ifdef NUMERICAL_DERIVATIVES
        function.d2F_dq2_FD(direction, hess_f_0, xi_ij, mean_q_ij, spc);
#else
        function.d2F_dq2(direction, hess_f_0, xi_ij, mean_q_ij, spc);
#endif

        for (unsigned int alpha = 0; alpha < dim; alpha++) {
          *integral_f += c_2_idx_i * hess_f_0[alpha * dim + alpha];
        }
      }
    }
  }

  //! Free memory
  free(hess_f_0);
}

/********************************************************************************/

void meanfield_integral_mp_dsq(int direction, double* integral_f_ds,
                               potential_function function, void* ctx_measure) {

  *integral_f_ds = 0.0;

  unsigned int dim = NumberDimensions;

  //! Read integral context
  // clang-format off
  unsigned int num_sites = ((gaussian_measure_ctx*)ctx_measure)->num_sites;
  unsigned int intergal_dim = ((gaussian_measure_ctx*)ctx_measure)->intergal_dim;
  const int* dof_table = ((gaussian_measure_ctx*)ctx_measure)->dof_table;
  const int* gp_board = ((gaussian_measure_ctx*)ctx_measure)->gp_board;
  double* mean_q_ij = ((gaussian_measure_ctx*)ctx_measure)->mean_q_ij;
  double* sigma = ((gaussian_measure_ctx*)ctx_measure)->stddev_q_ij;
  double* xi_ij = ((gaussian_measure_ctx*)ctx_measure)->xi_ij;
  AtomicSpecie* spc = ((gaussian_measure_ctx*)ctx_measure)->spc;
  // clang-format on

  //! 2 contribution
  double* hess_f_0 = (double*)calloc(dim * dim, sizeof(double));
  for (unsigned int site_idx_i = 0; site_idx_i < num_sites; site_idx_i++) {

    if (site_idx_i == direction) {
      double c_2_idx_i_ds = sigma[site_idx_i];

      for (unsigned int site_idx_j = 0; site_idx_j < num_sites; site_idx_j++) {

        int dof_idx = site_idx_i * num_sites + site_idx_j;

        if (dof_table[dof_idx] == 1) {

#ifdef NUMERICAL_DERIVATIVES
          function.d2F_dq2_FD(
              dof_idx, hess_f_0, xi_ij, mean_q_ij, spc);
#else
          function.d2F_dq2(dof_idx, hess_f_0, xi_ij, mean_q_ij, spc);
#endif

          for (unsigned int alpha = 0; alpha < dim; alpha++) {
            *integral_f_ds += c_2_idx_i_ds * hess_f_0[alpha * dim + alpha];
          }
        }
      }
    }
  }
  //! Free memory
  free(hess_f_0);
}

/********************************************************************************/

void meanfield_integral_mp_dmq(int direction, double* integral_grad_f,
                               potential_function function, void* ctx_measure) {

  unsigned int dim = NumberDimensions;

  //! Set to zero the integral
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    integral_grad_f[alpha] = 0.0;
  }

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
  double grad_f[3] = {0.0, 0.0, 0.0};

  function.dF_dq(direction, grad_f, xi_ij, mean_q_ij, spc);

  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    integral_grad_f[alpha] += c_0 * grad_f[alpha];
  }

  //! 2 contribution
  double dr = 0.01;  // *rc;

  //! Create auxliar vector with the mean coordinates
  double* mean_q_ij_k = (double*)calloc(dim * num_sites, sizeof(double));
  for (unsigned int dof = 0; dof < dim * num_sites; dof++) {
    mean_q_ij_k[dof] = mean_q_ij[dof];
  }

  for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

    double c_2_site_idx = DSQR(sigma[site_idx]) / 2.0;

    for (unsigned int alpha = 0; alpha < dim; alpha++) {

      unsigned int dof_idx_aux = site_idx * dim + alpha;

      //! Evaluate functions at the auxiliar points
      // f(... , x + 2*Dx , ...)
      mean_q_ij_k[dof_idx_aux] = mean_q_ij[dof_idx_aux] + dr * 2;
      double grad_f_pp[3] = {0.0, 0.0, 0.0};
      function.dF_dq(direction, grad_f_pp, xi_ij, mean_q_ij_k, spc);
      // f(... , x + Dx , ...)
      mean_q_ij_k[dof_idx_aux] = mean_q_ij[dof_idx_aux] + dr;
      double grad_f_p[3] = {0.0, 0.0, 0.0};
      function.dF_dq(direction, grad_f_p, xi_ij, mean_q_ij_k, spc);
      // f(... , x - Dx , ...)
      mean_q_ij_k[dof_idx_aux] = mean_q_ij[dof_idx_aux] - dr;
      double grad_f_m[3] = {0.0, 0.0, 0.0};
      function.dF_dq(direction, grad_f_m, xi_ij, mean_q_ij_k, spc);
      // f(... , x - 2*Dx , ...)
      mean_q_ij_k[dof_idx_aux] = mean_q_ij[dof_idx_aux] - dr * 2;
      double grad_f_mm[3] = {0.0, 0.0, 0.0};
      function.dF_dq(direction, grad_f_mm, xi_ij, mean_q_ij_k, spc);
      // f(... , x , ...)
      mean_q_ij_k[dof_idx_aux] = mean_q_ij[dof_idx_aux];

      //! Evaluate derivative
      for (unsigned int beta = 0; beta < dim; beta++) {
        double D3_f_ijj =
            (-grad_f_pp[beta] + 16.0 * grad_f_p[beta] - 30.0 * grad_f[beta] +
             16.0 * grad_f_m[beta] - grad_f_mm[beta]) /
            (12.0 * dr * dr);

        integral_grad_f[beta] += c_2_site_idx * D3_f_ijj;
      }
    }
  }

  //! Free memory
  free(mean_q_ij_k);
}

/********************************************************************************/
template<typename Dispatcher>
KOKKOS_FUNCTION void meanfield_integral_mp_Kokkos(double* integral_f,
                           void* ctx_measure, SoADevice *soADevice, Dispatcher function) {

  *integral_f = 0.0;

  unsigned int dim = NumberDimensions;

  Kokkos::printf("meanfield_integral_mp_Kokkos");

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

  // rho_ij_adp_MgHx_dispatcher function { Functions_Enum::FK, &f_0, xi_ij, mean_q_ij1, spc, soADevice.data() };


  //! 0 contribution
  double c_0 = 1.0;
  double f_0 = 0.0;

  function.function_rho_ij_Enum = Functions_Enum::FK;
  function.value = &f_0;
  function.n = xi_ij;
  function.q = mean_q_ij;
  function.spc = spc;
  function.soADevice = soADevice;
  function();
  

  function.F();

  *integral_f += c_0 * f_0;

  //! 2 contribution
  double hess_f_0[dim * dim];
  for (unsigned int site_idx_i = 0; site_idx_i < num_sites; site_idx_i++) {

    double c_2_idx_i = dsqr(sigma[site_idx_i]) / 2.0;

    for (unsigned int site_idx_j = 0; site_idx_j < num_sites; site_idx_j++) {

      int direction = site_idx_i * num_sites + site_idx_j;

      if (dof_table[direction] == 1) {

#ifdef NUMERICAL_DERIVATIVES
        function.function_rho_ij_Enum = Functions_Enum::d2F_dq2_FD;
        function.direction = direction;
        function.value = hess_f_0;
        function();
        // function.d2F_dq2_FD(direction, hess_f_0, xi_ij, mean_q_ij, spc);
#else
        function.function_rho_ij_Enum = Functions_Enum::d2F_dq2;
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


/********************************************************************************/