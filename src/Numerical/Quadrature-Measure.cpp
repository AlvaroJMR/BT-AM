/**
 * @file Quadrature-Meassure.cpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief Gaussian meassure
 * @version 0.1
 * @date 2023-01-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cstdlib>
#if __APPLE__
#include <malloc/_malloc.h>
#endif
#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include <fstream>
#include <iostream>  //std::cout//std::cin
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/********************************************************************************/

gaussian_measure_ctx fill_out_gaussian_measure(double* mean_q_ij,
                                               double* stddev_q_ij,
                                               double* xi_ij, AtomicSpecie* spc,
                                               int* dof_table,
                                               unsigned int NumSites) {

  unsigned int dim = NumberDimensions;

  gaussian_measure_ctx ctx;
  ctx.num_sites = NumSites;
  ctx.mean_q_ij = mean_q_ij;
  ctx.stddev_q_ij = stddev_q_ij;
  ctx.xi_ij = xi_ij;
  ctx.spc = spc;

  //! Copy dof table
  ctx.dof_table = (int*)calloc(NumSites * NumSites, sizeof(int));
  int* dof_table_aux = (int*)calloc(NumSites * NumSites, sizeof(int));
  for (unsigned int i = 0; i < NumSites; i++) {
    for (unsigned int j = 0; j < NumSites; j++) {
      ctx.dof_table[i * NumSites + j] = dof_table[i * NumSites + j];
      dof_table_aux[i * NumSites + j] = dof_table[i * NumSites + j];
    }
  }

  //! Remove redundant dofs in dof table
  for (unsigned int i = 0; i < NumSites; i++) {

    for (unsigned int j = 0; j < NumSites; j++) {

      if (j != i) {
        for (unsigned int k = 0; k < NumSites; k++) {
          if (dof_table_aux[j * NumSites + k] ==
              dof_table_aux[i * NumSites + k]) {
            dof_table_aux[j * NumSites + k] = 0;
          }
        }
      }
    }
  }

  int* active_dof = (int*)calloc(NumSites, sizeof(int));
  int counter = 0;
  for (unsigned int i = 0; i < NumSites; i++) {
    int counter_i = 0;
    for (unsigned int j = 0; j < NumSites; j++) {
      counter_i += dof_table_aux[i * NumSites + j];
    }
    if (counter_i > 0) {
      active_dof[i] = 1;
      counter++;
    }
  }

  unsigned int NumRows = counter * dim;
  unsigned int NumCols = NumSites * dim;

  //! Create table with active dofs
  ctx.gp_board = (int*)calloc(NumRows * NumCols, sizeof(int));
  int i_new = 0;
  for (unsigned int i = 0; i < NumSites; i++) {
    if (active_dof[i]) {
      for (unsigned int j = 0; j < NumSites; j++) {
        if (dof_table_aux[i * NumSites + j]) {
          for (unsigned int k = 0; k < dim; k++) {
            ctx.gp_board[i_new * dim * NumCols + j * dim + k * NumCols + k] = 1;
          }
        }
      }
      i_new++;
    }
  }

  ctx.intergal_dim = counter * dim;

  free(active_dof);
  free(dof_table_aux);

  return ctx;
}

/********************************************************************************/

void destroy_gaussian_measure(gaussian_measure_ctx* measure) {
  free(measure->gp_board);
  free(measure->dof_table);
}


/********************************************************************************/