/**
 * @file cubic-spline.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-07-16
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef CUBIC_SPLINE_HPP
#define CUBIC_SPLINE_HPP


#include "Macros.hpp"

using namespace std;

/**
 * @brief
 *
 * @param cs
 * @param n
 * @param x
 * @return STATUS
 */
int init_spline(CubicSpline *cs, int n, double x);

/**
 * @brief
 *
 * @param cs
 * @return STATUS
 */
int destroy_spline(CubicSpline *cs);

/**
 * @brief
 *
 * @param cs
 * @param x
 * @return double
 */
KOKKOS_FUNCTION double cubic_spline(CubicSpline *cs, double x);

/**
 * @brief
 *
 * @param cs
 * @param x
 * @return double
 */
KOKKOS_FUNCTION double d_cubic_spline(CubicSpline *cs, double x);

/**
 * @brief
 *
 * @param cs
 * @param x
 * @return double
 */
KOKKOS_FUNCTION double d2_cubic_spline(CubicSpline *cs, double x);


KOKKOS_INLINE_FUNCTION double cubic_spline_Kokkos(CubicSpline* cs, double x) {

    double p;  // This variable indicates the relative position in the segment of
               // the cubic spline: x-x_m
    int m;     // This variable indicates the segment of the cubic spline S_m(x)
    m = static_cast<int>(x / cs->dx);
    m = min(m, cs->n - 1);  // comprobation to know if m>m_max; m_max=n-1
    p = m * cs->dx;         // x_m=m*dx
    p = x - p;              // p=x-x_m=x-m*dx
    p = min(p, cs->dx);     // comprobation to know if p>dx
  
      return cs->a_d[m] + (cs->b_d[m] + (cs->c_d[m] + cs->d_d[m] * p) * p) * p;
  }

KOKKOS_INLINE_FUNCTION double d_cubic_spline_Kokkos(CubicSpline* cs, double x) {

    double p;  // This variable indicates the relative position in the segment of
               // the cubic spline: x-x_m
    int m;     // This variable indicates the segment of the cubic spline S_m(x)
    m = static_cast<int>(x / cs->dx);
    m = min(m, cs->n - 1);  // comprobation to konw if m>m_max; m_max=n-1
    p = m * cs->dx;         // x_m=m*dx
    p = x - p;              // p=x-x_m=x-m*dx
    p = min(p, cs->dx);     // comprobation to know if p>dx
  
  

      return cs->db_d[m] + (cs->dc_d[m] + cs->dd_d[m] * p) * p;

  }
  
  /********************************************************************************/  
#endif
