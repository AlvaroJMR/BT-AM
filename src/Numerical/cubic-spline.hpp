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

/**
 * @brief
 *
 * @param cs
 * @param x
 * @return double
 */
double cubic_spline(CubicSpline *cs, double x);

/**
 * @brief
 *
 * @param cs
 * @param x
 * @return double
 */
double d_cubic_spline(CubicSpline *cs, double x);

/**
 * @brief
 *
 * @param cs
 * @param x
 * @return double
 */
double d2_cubic_spline(CubicSpline *cs, double x);

#endif
