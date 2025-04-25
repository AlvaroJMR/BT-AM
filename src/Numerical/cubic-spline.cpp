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

#if USE_KOKKOS == 1
  #define SPLINE_FUNCTION KOKKOS_FUNCTION
#else
  #define SPLINE_FUNCTION
#endif

SPLINE_FUNCTION int init_spline(CubicSpline* cs, int n, double dx) {

  cs->dx = dx;

  cs->n = n;

  #if USE_KOKKOS == 1
  cs->x = View_Double_Vector_Device("cs->x", n + 1);
  Kokkos::deep_copy(cs->x, 0.0);

  cs->a = View_Double_Vector_Device("cs->a", n + 1);
  Kokkos::deep_copy(cs->a, 0.0);

  cs->b = View_Double_Vector_Device("cs->b", n + 1);
  Kokkos::deep_copy(cs->b, 0.0);

  cs->c = View_Double_Vector_Device("cs->c", n + 1);
  Kokkos::deep_copy(cs->c, 0.0);

  cs->d = View_Double_Vector_Device("cs->d", n + 1);
  Kokkos::deep_copy(cs->d, 0.0);

  cs->db = View_Double_Vector_Device("cs->db", n + 1);
  Kokkos::deep_copy(cs->db, 0.0);

  cs->dc = View_Double_Vector_Device("cs->dc", n + 1);
  Kokkos::deep_copy(cs->dc, 0.0);

  cs->dd = View_Double_Vector_Device("cs->dd", n + 1);
  Kokkos::deep_copy(cs->dd, 0.0);

  cs->ddc = View_Double_Vector_Device("cs->ddc", n + 1);
  Kokkos::deep_copy(cs->ddc, 0.0);

  cs->ddd = View_Double_Vector_Device("cs->ddd", n + 1);
  Kokkos::deep_copy(cs->ddd, 0.0);

  #endif

  #if USE_KOKKOS == 0

  cs->x = Eigen::VectorXd::Zero(n + 1);

  cs->a = Eigen::VectorXd::Zero(n + 1);

  cs->b = Eigen::VectorXd::Zero(n + 1);

  cs->c = Eigen::VectorXd::Zero(n + 1);

  cs->d = Eigen::VectorXd::Zero(n + 1);

  cs->db = Eigen::VectorXd::Zero(n + 1);

  cs->dc = Eigen::VectorXd::Zero(n + 1);

  cs->dd = Eigen::VectorXd::Zero(n + 1);

  cs->ddc = Eigen::VectorXd::Zero(n + 1);

  cs->ddd = Eigen::VectorXd::Zero(n + 1);
  #endif

  return EXIT_SUCCESS;
}

/********************************************************************************/


/********************************************************************************/

SPLINE_FUNCTION double cubic_spline(CubicSpline* cs, double x) {

  double p;  // This variable indicates the relative position in the segment of
             // the cubic spline: x-x_m
  int m;     // This variable indicates the segment of the cubic spline S_m(x)
  m = static_cast<int>(x / cs->dx);
  m = min(m, cs->n - 1);  // comprobation to know if m>m_max; m_max=n-1
  p = m * cs->dx;         // x_m=m*dx
  p = x - p;              // p=x-x_m=x-m*dx
  p = min(p, cs->dx);     // comprobation to know if p>dx

  double result = cs->a(m) + (cs->b(m) + (cs->c(m) + cs->d(m) * p) * p) * p;
  return result;
}

/********************************************************************************/

SPLINE_FUNCTION double d_cubic_spline(CubicSpline* cs, double x) {

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

SPLINE_FUNCTION double d2_cubic_spline(CubicSpline* cs, double x) {

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