/**
 * @file cubic-spline.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-07-23
 *
 * @copyright Copyright (c) 2022
 *
 */

// clang-format off
#include <cstdlib>
#include <stdio.h>
#include <iostream> //std::cout//std::cin
#include "Macros.hpp"
#include <math.h>
#include <stdlib.h>
#include "Numerical/cubic-spline.hpp"
#include <Kokkos_Core.hpp>
// clang-format on

using namespace std;

/********************************************************************************/

KOKKOS_FUNCTION int init_spline(CubicSpline* cs, int n, double dx) {

  cs->dx = dx;

  cs->n = n;

  cs->x = (double*)calloc((1), sizeof(double));

  cs->a = (double*)calloc((n), sizeof(double));

  cs->b = (double*)calloc((n), sizeof(double));

  cs->c = (double*)calloc((n), sizeof(double));

  cs->d = (double*)calloc((n), sizeof(double));

  cs->db = (double*)calloc((n), sizeof(double));

  cs->dc = (double*)calloc((n), sizeof(double));

  cs->dd = (double*)calloc((n), sizeof(double));

  cs->ddc = (double*)calloc((n), sizeof(double));

  cs->ddd = (double*)calloc((n), sizeof(double));

  return EXIT_SUCCESS;
}

/********************************************************************************/

KOKKOS_FUNCTION int destroy_spline(CubicSpline* cs) {

  free(cs->x);

  free(cs->a);

  free(cs->b);

  free(cs->c);

  free(cs->d);

  free(cs->db);

  free(cs->dc);

  free(cs->dd);

  free(cs->ddc);

  free(cs->ddd);

  return EXIT_SUCCESS;
}

/********************************************************************************/

KOKKOS_FUNCTION double cubic_spline(CubicSpline* cs, double x) {

  double p;  // This variable indicates the relative position in the segment of
             // the cubic spline: x-x_m
  int m;     // This variable indicates the segment of the cubic spline S_m(x)
  m = static_cast<int>(x / cs->dx);
  m = min(m, cs->n - 1);  // comprobation to konw if m>m_max; m_max=n-1
  p = m * cs->dx;         // x_m=m*dx
  p = x - p;              // p=x-x_m=x-m*dx
  p = min(p, cs->dx);     // comprobation to know if p>dx

  /* std::cout << "Antes de probar el spline: " << std::endl;
  double test = cs->a[m];    
  std::cout << "Probar el spline: " << test << std::endl; */
  return cs->a[m] + (cs->b[m] + (cs->c[m] + cs->d[m] * p) * p) * p;
}

/********************************************************************************/

KOKKOS_FUNCTION double d_cubic_spline(CubicSpline* cs, double x) {

  double p;  // This variable indicates the relative position in the segment of
             // the cubic spline: x-x_m
  int m;     // This variable indicates the segment of the cubic spline S_m(x)
  m = static_cast<int>(x / cs->dx);
  m = min(m, cs->n - 1);  // comprobation to konw if m>m_max; m_max=n-1
  p = m * cs->dx;         // x_m=m*dx
  p = x - p;              // p=x-x_m=x-m*dx
  p = min(p, cs->dx);     // comprobation to know if p>dx

  return cs->db[m] + (cs->dc[m] + cs->dd[m] * p) * p;
}

/********************************************************************************/

KOKKOS_FUNCTION double d2_cubic_spline(CubicSpline* cs, double x) {

  double p;  // This variable indicates the relative position in the segment of
             // the cubic spline: x-x_m
  int m;     // This variable indicates the segment of the cubic spline S_m(x)
  m = static_cast<int>(x / cs->dx);
  m = min(m, cs->n - 1);  // comprobation to konw if m>m_max; m_max=n-1
  p = m * cs->dx;         // x_m=m*dx
  p = x - p;              // p=x-x_m=x-m*dx
  p = min(p, cs->dx);     // comprobation to know if p>dx

  return cs->ddc[m] + cs->ddd[m] * p;
}

/********************************************************************************/