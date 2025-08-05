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

template<int W>
KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::native_simd<double>
cubic_spline_Kokkos_SIMD(const CubicSpline splines[W],
                         Kokkos::Experimental::native_simd<double> x)
{
  using simd_t     = Kokkos::Experimental::native_simd<double>;
  using simd_int_t = Kokkos::Experimental::native_simd<int>;

  int    max_idx_arr[W];
  double dx_arr[W];
  for(int lane=0; lane<W; ++lane){
    dx_arr[lane]      = splines[lane].dx;
    max_idx_arr[lane] = splines[lane].n - 1;
  }

  simd_t dx; 
  dx.copy_from(dx_arr, Kokkos::Experimental::simd_flag_default);
  simd_int_t max_idx; 
  max_idx.copy_from(max_idx_arr, Kokkos::Experimental::simd_flag_default);
  simd_t div     = x / dx;
  simd_t floored = Kokkos::floor(div);

  simd_int_t m_simd;
  for (int lane = 0; lane < W; ++lane) {
    m_simd[lane] = static_cast<int>(floored[lane]);
  }
  simd_int_t zero    = simd_int_t(0);
  m_simd = Kokkos::Experimental::condition(m_simd < zero, zero, m_simd);
  m_simd = Kokkos::Experimental::condition(m_simd > max_idx, max_idx, m_simd);

  simd_t a_m, b_m, c_m, d_m;
  for (int lane = 0; lane < W; ++lane) {
    int idx = m_simd[lane];
    a_m[lane] = splines[lane].a_d[idx];
    b_m[lane] = splines[lane].b_d[idx];
    c_m[lane] = splines[lane].c_d[idx];
    d_m[lane] = splines[lane].d_d[idx];
  }

  simd_t idx_d;
  for (int lane = 0; lane < W; ++lane) {
    idx_d[lane] = static_cast<double>(m_simd[lane]);
  }
  simd_t p = x - idx_d * dx;
  p = Kokkos::Experimental::condition(p < dx, p, dx);

  simd_t result = a_m + (b_m + (c_m + d_m * p) * p) * p;
  return result;
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

KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::native_simd<double> d_cubic_spline_Kokkos_SIMD(CubicSpline* cs,
                                                                    Kokkos::Experimental::native_simd<double> x) {
  using simd_t     = Kokkos::Experimental::native_simd<double>;
  using simd_int_t = Kokkos::Experimental::native_simd<int>;
  using mask_int_t = typename simd_int_t::mask_type;
  constexpr int W = simd_t::size();

  simd_t dx = simd_t(cs->dx);
  simd_t div = x / dx;
  simd_t floored = Kokkos::floor(div);

  simd_int_t m_simd;
  for (int lane = 0; lane < W; ++lane) {
    m_simd[lane] = static_cast<int>(floored[lane]);
  }

  simd_int_t zero = simd_int_t(0);
  simd_int_t max_idx = simd_int_t(cs->n - 1);
  m_simd = Kokkos::Experimental::condition(m_simd < zero, zero, m_simd);
  m_simd = Kokkos::Experimental::condition(m_simd > max_idx, max_idx, m_simd);

  simd_t db_d_m, dc_d_m, dd_d_m;

  for (int lane = 0; lane < W; ++lane) {
    int m = m_simd[lane];
    db_d_m[lane] = cs->db_d[m];
    dc_d_m[lane] = cs->dc_d[m];
    dd_d_m[lane] = cs->dd_d[m];
  }

  simd_t m_double;
  for (int lane = 0; lane < W; ++lane) {
    m_double[lane] = static_cast<double>(m_simd[lane]);
  }

  simd_t p = x - m_double * dx;
  p = Kokkos::Experimental::condition(p < dx, p, dx);

  simd_t result = db_d_m + (dc_d_m + dd_d_m * p) * p;
  return result;
}

  /********************************************************************************/    


  /*KOKKOS_INLINE_FUNCTION Kokkos::Experimental::native_simd<double> d_cubic_spline_Kokkos_SIMD(CubicSpline* cs, Kokkos::Experimental::native_simd<double> x) {

    using simd_t = Kokkos::Experimental::native_simd<double>;
    constexpr int W = simd_t::size();
    simd_t result(0.0);

        for (int lane = 0; lane < W; ++lane) {
        double x_lane = x[lane];
        int m = static_cast<int>(x_lane / cs->dx);
        m = (m < cs->n - 1) ? m : (cs->n - 1);
        double p = m * cs->dx;
        p = x_lane - p;
        p = (p < cs->dx) ? p : cs->dx;
        result[lane] = cs->db_d[m] + (cs->dc_d[m] + cs->dd_d[m] * p) * p;
    }
    return result;

  }*/


  /*KOKKOS_INLINE_FUNCTION
Kokkos::Experimental::native_simd<double>
cubic_spline_Kokkos_SIMD(CubicSpline* cs, Kokkos::Experimental::native_simd<double> x)
{
    using simd_t = Kokkos::Experimental::native_simd<double>;
    constexpr int W = simd_t::size();

    simd_t result(0.0);
    for (int lane = 0; lane < W; ++lane) {
        double x_lane = x[lane];
        int m = static_cast<int>(x_lane / cs->dx);
        m = (m < cs->n - 1) ? m : (cs->n - 1);
        double p = m * cs->dx;
        p = x_lane - p;
        p = (p < cs->dx) ? p : cs->dx;
        result[lane] = cs->a_d[m] + (cs->b_d[m] + (cs->c_d[m] + cs->d_d[m] * p) * p) * p;
    }
    return result;
}
  */
#endif
