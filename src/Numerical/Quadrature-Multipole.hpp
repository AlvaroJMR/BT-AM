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
KOKKOS_FUNCTION void meanfield_integral_mp_Kokkos(double* integral_f,
                           void* ctx_measure, SoADevice *soADevice, Dispatcher function);
                           
#endif // quadrature_hermite_mp_HPP