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

  cs->isKernel = false;

  #if USE_KOKKOS == 1
  /* cs->x   = DualView_Vector_Double("cs_x",   n + 1);
  cs->a   = DualView_Vector_Double("cs_a",   n + 1);
  cs->b   = DualView_Vector_Double("cs_b",   n + 1);
  cs->c   = DualView_Vector_Double("cs_c",   n + 1);
  cs->d   = DualView_Vector_Double("cs_d",   n + 1);
  cs->db  = DualView_Vector_Double("cs_db",  n + 1);  
  cs->dc  = DualView_Vector_Double("cs_dc",  n + 1);
  cs->dd  = DualView_Vector_Double("cs_dd",  n + 1);
  cs->ddc = DualView_Vector_Double("cs_ddc", n + 1);
  cs->ddd = DualView_Vector_Double("cs_ddd", n + 1);
  
  cs->x .modify_host();   Kokkos::deep_copy(cs->x .view_host(), 0.0);   cs->x .sync_device();
  cs->a .modify_host();   Kokkos::deep_copy(cs->a .view_host(), 0.0);   cs->a .sync_device();
  cs->b .modify_host();   Kokkos::deep_copy(cs->b .view_host(), 0.0);   cs->b .sync_device();
  cs->c .modify_host();   Kokkos::deep_copy(cs->c .view_host(), 0.0);   cs->c .sync_device();
  cs->d .modify_host();   Kokkos::deep_copy(cs->d .view_host(), 0.0);   cs->d .sync_device();
  cs->db.modify_host();   Kokkos::deep_copy(cs->db.view_host(), 0.0);   cs->db.sync_device();
  cs->dc.modify_host();   Kokkos::deep_copy(cs->dc.view_host(), 0.0);   cs->dc.sync_device();
  cs->dd.modify_host();   Kokkos::deep_copy(cs->dd.view_host(), 0.0);   cs->dd.sync_device();
  cs->ddc.modify_host();  Kokkos::deep_copy(cs->ddc.view_host(),0.0);   cs->ddc.sync_device();
  cs->ddd.modify_host();  Kokkos::deep_copy(cs->ddd.view_host(),0.0);   cs->ddd.sync_device(); */

  cs->x   = View_Double_Vector_Host("cs_x",   n + 1);
  cs->a   = View_Double_Vector_Host("cs_a",   n + 1);
  cs->b   = View_Double_Vector_Host("cs_b",   n + 1);
  cs->c   = View_Double_Vector_Host("cs_c",   n + 1);
  cs->d   = View_Double_Vector_Host("cs_d",   n + 1);
  cs->db  = View_Double_Vector_Host("cs_db",  n + 1);  
  cs->dc  = View_Double_Vector_Host("cs_dc",  n + 1);
  cs->dd  = View_Double_Vector_Host("cs_dd",  n + 1);
  cs->ddc = View_Double_Vector_Host("cs_ddc", n + 1);
  cs->ddd = View_Double_Vector_Host("cs_ddd", n + 1);
  
  Kokkos::deep_copy(cs->x, 0.0);
  Kokkos::deep_copy(cs->a, 0.0);
  Kokkos::deep_copy(cs->b, 0.0);
  Kokkos::deep_copy(cs->c, 0.0);
  Kokkos::deep_copy(cs->d, 0.0);
  Kokkos::deep_copy(cs->db, 0.0);
  Kokkos::deep_copy(cs->dc, 0.0);
  Kokkos::deep_copy(cs->dd, 0.0);
  Kokkos::deep_copy(cs->ddc,0.0);
  Kokkos::deep_copy(cs->ddd,0.0);

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

KOKKOS_FUNCTION double cubic_spline(CubicSpline* cs, double x) {

  double p;  // This variable indicates the relative position in the segment of
             // the cubic spline: x-x_m
  int m;     // This variable indicates the segment of the cubic spline S_m(x)
  m = static_cast<int>(x / cs->dx);
  m = min(m, cs->n - 1);  // comprobation to know if m>m_max; m_max=n-1
  p = m * cs->dx;         // x_m=m*dx
  p = x - p;              // p=x-x_m=x-m*dx
  p = min(p, cs->dx);     // comprobation to know if p>dx

  if (cs->isKernel) {
    return cs->a_d[m] + (cs->b_d[m] + (cs->c_d[m] + cs->d_d[m] * p) * p) * p;
  } else {
    return cs->a[m] + (cs->b[m] + (cs->c[m] + cs->d[m] * p) * p) * p;
  }
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


    if (cs->isKernel) {
    return cs->db_d[m] + (cs->dc_d[m] + cs->dd_d[m] * p) * p;
  } else {
     return cs->db[m] + (cs->dc[m] + cs->dd[m] * p) * p;
  }
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


  if (cs->isKernel) {
    return cs->ddc_d[m] +  cs->ddd_d[m]* p;
  } else {
    return cs->ddc[m] + cs->ddd[m] * p;
  }
 
}

/********************************************************************************/