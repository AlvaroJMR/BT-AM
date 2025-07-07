/**
 * @file Quadrature-Measure.hpp
 * @author Miguel Molinos, Pilar Ariza, Miguel Ortiz
 * ([migmolper](https://github.com/migmolper),[mpariza](https://github.com/mpariza),[mortizcaltech](https://github.com/mortizcaltech))
 * @brief Gaussian meassure
 * @version 0.1
 * @date 2023-01-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef Quadrature_Measure_HPP
#define Quadrature_Measure_HPP

// clang-format off
#include "Atoms/Atom.hpp"
#include "Atoms/Neighbors.hpp"
// clang-format on

#include <Kokkos_Core.hpp>

/**
 * @brief Arguments of the integral
 *
 */
typedef struct {

  /*! @param num_sites: Number of sites */
  int num_sites;

  /*! @param intergal_dim: Number of dimensions of the integral */
  int intergal_dim;

  /*! @param dof_table: Table with the active degree of freedom */
  int *dof_table;

  /*! @param gp_board: Integration order */
  int *gp_board;

  /*! @param mean_q_ij: Mean value of q at site i and j */
  double *mean_q_ij;

  /*! @param stddev_q_ij: Standard desviation of q at site i and j */
  double *stddev_q_ij;

  /*! @param xi_ij: Molar fraction of sites i and j */
  double *xi_ij;

  /*! @param AtomicSpecie: List of atomic species of each site */
  AtomicSpecie *spc;

} gaussian_measure_ctx;

/* Gaussian measure for Kokkos*/

typedef struct {

  int num_sites;

  int intergal_dim;

  Int_Vector_Default dof_table;

  Int_Vector_Default dof_table_aux;

  Int_Vector_Default gp_board;

  Int_Vector_Default active_dof;

  double *mean_q_ij;

  double *stddev_q_ij;

  double *xi_ij;

  AtomicSpecie *spc;

} gaussian_measure_ctx_kokkos;


typedef struct {

  int num_sites;

  int intergal_dim;

  int* dof_table;

  int* dof_table_aux;

  int* gp_board;

  int* active_dof;

  double *mean_q_ij;

  double *stddev_q_ij;

  double *xi_ij;

  AtomicSpecie *spc;

} gaussian_measure_ctx_kokkos_s;

/**
 * @brief Fill out integral context
 *
 * @param mean_q_ij Mean value of q at site i and j
 * @param stddev_q_ij Standard desviation of q at sites i and j
 * @param xi_ij Molar fraction of sites i and j
 * @param AtomicSpecie
 * @param dof_table Table with the relation of i, j and k
 * @param NumSites Number of sites to evaluate the integral
 */
gaussian_measure_ctx fill_out_gaussian_measure(double *mean_q_ij,   //!
                                               double *stddev_q_ij, //!
                                               double *xi_ij,       //!
                                               AtomicSpecie *spc,   //!
                                               int *dof_table,      //!
                                               unsigned int NumSites);

/**
 * @brief Free memory inside of the gaussian measure variable
 *
 * @param measure
 */
void destroy_gaussian_measure(gaussian_measure_ctx *measure);

/********************************************************************************/


KOKKOS_INLINE_FUNCTION void fill_out_gaussian_measure_Kokkos (double* mean_q_ij,
  double* stddev_q_ij,
  double* xi_ij, AtomicSpecie* spc,
  int* dof_table,
  unsigned int NumSites,
  gaussian_measure_ctx_kokkos* ctx) {

unsigned int dim = NumberDimensions;

ctx->num_sites = NumSites;
ctx->mean_q_ij = mean_q_ij;
ctx->stddev_q_ij = stddev_q_ij;
ctx->xi_ij = xi_ij;
ctx->spc = spc;

for (size_t i = 0; i < ctx->dof_table_aux.extent(0); ++i) {
ctx->dof_table_aux(i) = 0;
}

for (size_t i = 0; i < ctx->active_dof.extent(0); ++i) {
ctx->active_dof(i) = 0;
}

for (size_t i = 0; i < ctx->gp_board.extent(0); ++i) {
ctx->gp_board(i) = 0;
}


//! Copy dof table
for (unsigned int i = 0; i < NumSites; i++) {
for (unsigned int j = 0; j < NumSites; j++) {
ctx->dof_table(i * NumSites + j) = dof_table[i * NumSites + j];
ctx->dof_table_aux(i * NumSites + j) = dof_table[i * NumSites + j];
}
}

//! Remove redundant dofs in dof table
for (unsigned int i = 0; i < NumSites; i++) {

for (unsigned int j = 0; j < NumSites; j++) {

if (j != i) {
for (unsigned int k = 0; k < NumSites; k++) {
if (ctx->dof_table_aux(j * NumSites + k) ==
ctx->dof_table_aux(i * NumSites + k)) {
ctx->dof_table_aux(j * NumSites + k) = 0;
}
}
}
}
}

int counter = 0;
for (unsigned int i = 0; i < NumSites; i++) {
int counter_i = 0;
for (unsigned int j = 0; j < NumSites; j++) {
counter_i += ctx->dof_table_aux(i * NumSites + j);
}
if (counter_i > 0) {
ctx->active_dof(i) = 1;
counter++;
}
}

unsigned int NumRows = counter * dim;
unsigned int NumCols = NumSites * dim;

int i_new = 0;
for (unsigned int i = 0; i < NumSites; i++) {
if (ctx->active_dof(i)) {
for (unsigned int j = 0; j < NumSites; j++) {
if (ctx->dof_table_aux(i * NumSites + j)) {
for (unsigned int k = 0; k < dim; k++) {
ctx->gp_board(i_new * dim * NumCols + j * dim + k * NumCols + k) = 1;
}
}
}
i_new++;
}
}

ctx->intergal_dim = counter * dim;

}

/***************************************************************************************** */

KOKKOS_INLINE_FUNCTION void fill_out_gaussian_measure_Kokkos_s (double* mean_q_ij,
  double* stddev_q_ij,
  double* xi_ij, AtomicSpecie* spc,
  int* dof_table,
  unsigned int NumSites,
  gaussian_measure_ctx_kokkos_s* ctxs) {

unsigned int dim = NumberDimensions;

ctxs->num_sites = NumSites;
ctxs->mean_q_ij = mean_q_ij;
ctxs->stddev_q_ij = stddev_q_ij;
ctxs->xi_ij = xi_ij;
ctxs->spc = spc;


for (size_t i = 0; i < NumSites * NumSites; ++i) {
  ctxs->dof_table_aux[i] = 0;
}

for (size_t i = 0; i < NumSites; ++i) {
  ctxs->active_dof[i] = 0;
}

for (size_t i = 0; i < NumSites * NumSites * NumberDimensions * NumberDimensions; ++i) {
  ctxs->gp_board[i] = 0;
}


//! Copy dof table
for (unsigned int i = 0; i < NumSites; i++) {
for (unsigned int j = 0; j < NumSites; j++) {
  ctxs->dof_table[i * NumSites + j] = dof_table[i * NumSites + j];
  ctxs->dof_table_aux[i * NumSites + j] = dof_table[i * NumSites + j];
}
}

//! Remove redundant dofs in dof table
for (unsigned int i = 0; i < NumSites; i++) {

for (unsigned int j = 0; j < NumSites; j++) {

if (j != i) {
for (unsigned int k = 0; k < NumSites; k++) {
if (ctxs->dof_table_aux[j * NumSites + k] ==
  ctxs->dof_table_aux[i * NumSites + k]) {
  ctxs->dof_table_aux[j * NumSites + k] = 0;
}
}
}
}
}

int counter = 0;
for (unsigned int i = 0; i < NumSites; i++) {
int counter_i = 0;
for (unsigned int j = 0; j < NumSites; j++) {
counter_i += ctxs->dof_table_aux[i * NumSites + j];
}
if (counter_i > 0) {
  ctxs->active_dof[i] = 1;
counter++;
}
}

unsigned int NumRows = counter * dim;
unsigned int NumCols = NumSites * dim;

int i_new = 0;
for (unsigned int i = 0; i < NumSites; i++) {
if (ctxs->active_dof[i]) {
for (unsigned int j = 0; j < NumSites; j++) {
if (ctxs->dof_table_aux[i * NumSites + j]) {
for (unsigned int k = 0; k < dim; k++) {
  ctxs->gp_board[i_new * dim * NumCols + j * dim + k * NumCols + k] = 1;
}
}
}
i_new++;
}
}

ctxs->intergal_dim = counter * dim;

}

typedef Kokkos::View<gaussian_measure_ctx_kokkos*, DefaultLayout, HostMemorySpace> gaussian_measure_ctx_Host;
typedef Kokkos::View<gaussian_measure_ctx_kokkos*, DefaultLayout, DefaultMemorySpace> gaussian_measure_ctx_Default;                                               

#endif // Quadrature_Measure_HPP