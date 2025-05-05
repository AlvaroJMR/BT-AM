/**
 * @file MgHx-ADP.cpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief Angular Dependant Potential
 * @version 0.1
 * @date 2023-07-08
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <cmath>
#include <cstdlib>
#ifdef USE_MPI
#include <mpi.h>
#endif
#include "ADP/MgHx-ADP.hpp"
#include "Atoms/Atom.hpp"
#include "Macros.hpp"
#include "Numerical/Quadrature-Measure.hpp"
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <Kokkos_Core.hpp>
#include <Utils/KokkosUtils.hpp>



using namespace std;

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern adpPotential adp_MgMg;
extern adpPotential adp_HH;
extern adpPotential adp_MgH;


extern double element_mass[112];

/**
 * @brief Function to compute the energy density between sites i and j
 *
 * @param rho_ij Partial energy density between sites i and j
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void rho_ij(double* rho_ij, const double* n_ij, const double* q_ij,
                   const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the energy density between sites i and j
 *
 * @param rho_ij Partial energy density between sites i and j
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
KOKKOS_FUNCTION static void rho_ij_kokkos(double* rho_ij, const double* n, const double* q,
  const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default);

KOKKOS_FUNCTION static void V_pair_ij_kokkos(double* V_pair_ij, const double* n, const double* q,
  const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default);

KOKKOS_FUNCTION static void V_dipole_ij1j2_kokkos(double* V_dipole_ij1j2, const double* n,
  const double* q, const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default); 

KOKKOS_FUNCTION static void V_quadrupole_ij1j2_kokkos(double* V_quadrupole_ij1_ij2, const double* n,
  const double* q, const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default);  
/**
 * @brief Function to compute the gradient of the energy density between sites i
 * and j. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_rho_ij Gradient of the energy density between sites i and j
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void d_rho_ij_dq(int direction, double* Dr_rho_ij, const double* n_ij,
                        const double* q_ij, const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the gradient of the energy density between sites i
 * and j. (NUMERICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_rho_ij Gradient of the energy density between sites i and j
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void d_rho_ij_dq_FD(int direction, double* d_rho_ij_dq, const double* n,
                           const double* q, const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the hessian of the energy density between sites i
 * and j. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param DrDr_rho_ij Hessian of the energy density between sites i and j
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void d2_rho_ij_dq2(int direction, double* DrDr_rho_ij,
                          const double* n_ij, const double* q_ij,
                          const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the hessian of the energy density between sites i
 * and j. (NUMERICAL)
 *
 * @param direction Direction of the derivative
 * @param DrDr_rho_ij Hessian of the energy density between sites i and j
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void d2_rho_ij_dq2_FD(int direction, double* d2_rho_ij_dq,
                             const double* n, const double* q,
                             const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the derivative of the elcetron density with
 * respect the occupancy
 *
 * @param direction Direction of the derivative
 * @param d_rho_ij_dn Derivative of the elcetron density with respect the
 * occupancy
 * @param n Occupancy of sites i and j
 * @param q Coordinates of sites i and j
 */
static void d_rho_ij_dn(int direction, double* d_rho_ij_dn, const double* n,
                        const double* q, const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the pairing contribution of the site j to i
 *
 * @param V_pair_ij Partial pairing contribution of the site j to i
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void V_pair_ij(double* V_pair_ij, const double* n_ij, const double* q_ij,
                      const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the gradient of the pairing contribution of the
 * site j to i. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_pair_ij Gradient of the pairing contribution of the site j to i
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void dV_pair_ij_dq(int direction, double* Dr_pair_ij, const double* n_ij,
                          const double* q_ij, const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the gradient of the pairing contribution of the
 * site j to i. (NUMERICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_pair_ij Gradient of the pairing contribution of the site j to i
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void dV_pair_ij_dq_FD(int direction, double* dV_pair_ij_dq,
                             const double* n, const double* q,
                             const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the hessian of the pairing contribution of the
 * site j to i. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param DrDr_pair_ij Hessian of the pairing contribution of the site j to i
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void d2V_pair_ij_dq2(int direction, double* DrDr_pair_ij,
                            const double* n_ij, const double* q_ij,
                            const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the hessian of the pairing contribution of the
 * site j to i. (NUMERICAL)
 *
 * @param direction Direction of the derivative
 * @param DrDr_pair_ij Hessian of the pairing contribution of the site j to i
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void d2V_pair_ij_dq2_FD(int direction, double* d2V_pair_ij_dq,
                               const double* n, const double* q,
                               const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the derivative of the pairing contribution with
 * respect the occupancy
 *
 * @param direction Direction of the derivatives
 * @param dV_pair_ij_dn Derivative of the pairing contribution with respect the
 * occupancy
 * @param n_ij Occupancy of sites i and j
 * @param q_ij Coordinates of sites i and j
 */
static void dV_pair_ij_dn(int direction, double* dV_pair_ij_dn, const double* n,
                          const double* q, const AtomicSpecie* spc_ij);

/**
 * @brief Function to compute the dipole contribution of the sites j and k to i
 *
 * @param V_dipole_ij1_ij2 Partial dipole contribution of the sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void V_dipole_ij1j2(double* V_dipole_ij1_ij2, const double* n_ijk,
                           const double* q_ijk, const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the gradient of the dipole contribution of the
 * sites j and k to i. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_dipole_distortion_ijk Gradient of the dipole contribution of the
 * sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV_dipole_ij1j2_dq(int direction, double* Dr_dipole_distortion_ijk,
                               const double* n_ijk, const double* q_ijk,
                               const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the gradient of the dipole contribution of the
 * sites j and k to i. (NUMERICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_dipole_distortion_ijk Gradient of the dipole contribution of the
 * sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV_dipole_ij1j2_dq_FD(int direction, double* dV_dipole_ij1j2_dq,
                                  const double* n_ijk, const double* q_ijk,
                                  const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the hessian of the dipole contribution of the
 * sites j and k to i. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param DrDr_dipole_distortion_ijk Hessian of the dipole contribution of the
 * sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV2_dipole_ij1j2_dq2(int direction,
                                 double* DrDr_dipole_distortion_ijk,
                                 const double* n_ijk, const double* q_ijk,
                                 const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the hessian of the dipole contribution of the
 * sites j and k to i. (NUMERICAL)
 *
 * @param direction Direction of the derivative
 * @param DrDr_dipole_distortion_ijk Hessian of the dipole contribution of the
 * sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV2_dipole_ij1j2_dq2_FD(int direction, double* d2V_dipole_ij1j2_dq,
                                    const double* n_ijk, const double* q_ijk,
                                    const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the derivative of the dipole with respect the
 * occupancy
 *
 * @param direction Direction of the derivative
 * @param dV_dipole_ij1j2_dn Derivative of the dipole with respect the occupancy
 * @param n Occupancy of sites i, j and k
 * @param q Coordinates of sites i, j and k
 */
static void dV_dipole_ij1j2_dn(int direction, double* dV_dipole_ij1j2_dn,
                               const double* n_ijk, const double* q_ijk,
                               const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the quadrupole contribution of the sites j and k
 * to i
 *
 * @param V_quadrupole_ij1_ij2 Partial quadrupole contribution of the sites j
 * and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void V_quadrupole_ij1j2(double* V_quadrupole_ij1_ij2,
                               const double* n_ijk, const double* q_ijk,
                               const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the gradient of the quadrupole contribution of the
 * sites j and k to i. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_quadrupole_distortion_ijk Gradient of the quadrupole contribution
 * of the sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV_quadrupole_ij1j2_dq(int direction,
                                   double* Dr_quadrupole_distortion_ijk,
                                   const double* n_ijk, const double* q_ijk,
                                   const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the gradient of the quadrupole contribution of the
 * sites j and k to i. (NUMERICAL)
 *
 * @param direction Direction of the derivative
 * @param Dr_quadrupole_distortion_ijk Gradient of the quadrupole contribution
 * of the sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV_quadrupole_ij1j2_dq_FD(int direction,
                                      double* dV_quadrupole_ij1_ij2_dq,
                                      const double* n_ijk, const double* q_ijk,
                                      const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the hessian of the quadrupole contribution of the
 * sites j and k to i. (ANALYTICAL)
 *
 * @param direction Direction of the derivative
 * @param DrDr_quadrupole_distortion_ijk Hessian of the quadrupole contribution
 * of the sites j and k to i
 * @param n_ijk Occupancy of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV2_quadrupole_ij1j2_dq2(int direction,
                                     double* DrDr_quadrupole_distortion_ijk,
                                     const double* n_ijk, const double* q_ijk,
                                     const AtomicSpecie* spc_ijk);

/**
 * @brief Function to compute the hessian of the quadrupole contribution of the
 * sites j and k to i. (NUMERICAL)
 *
 * @param direction Direction ofd the derivative
 * @param DrDr_quadrupole_distortion_ijk Hessian of the quadrupole contribution
 * of the sites j and k to i
 * @param n_ijk Coordinates of sites i, j and k
 * @param q_ijk Coordinates of sites i, j and k
 */
static void dV2_quadrupole_ij1j2_dq2_FD(int direction,
                                        double* d2V_quadrupole_ij1j2_dq,
                                        const double* n_ijk,
                                        const double* q_ijk,
                                        const AtomicSpecie* spc_ijk);

/**
 * @brief Function ot compute the derivative of the quadrupole term with respect
 * the occupancy
 *
 * @param direction Direction ofd the derivative
 * @param dV_quadrupole_ij1_ij2_dn Derivative of the quadrupole term with
 * respect the occupancy
 * @param n Coordinates of sites i, j and k
 * @param q Coordinates of sites i, j and k
 */
static void dV_quadrupole_ij1j2_dn(int direction,
                                   double* dV_quadrupole_ij1_ij2_dn,
                                   const double* n_ijk, const double* q_ijk,
                                   const AtomicSpecie* spc_ijk);

/**
 * @brief Function devoted to compute the hessian of the distance norm
 *
 * @param hessian_r_ij
 * @param r_ij
 * @param norm_r_ij
 */
static void compute_hessian_norm_r(double* hessian_r_ij, const double* r_ij,
                                   const double norm_r_ij);

/********************************************************************************/

int init_adp_MgHx(adpPotential* adp, species_comb_MgH adp_material,
                  const char* PotentialsFolder) {

  std::cout.precision(14);
  double dr_rho, drho_embed, dr_pair, dr_u, dr_w;
  int i, error;
  bool alloy = false;
  char Name_file_t[10000];
  FILE* f_adp;
  AtomicSpecie scp_material;

  switch (adp_material) {
    case MgMg:

      scp_material = Mg;

      snprintf(Name_file_t, sizeof(Name_file_t), "%s/adpmg.dat",
               PotentialsFolder);
#ifdef DEBUG_MODE
      if (rank_MPI == 0)
        std::cout << "Reading ADP Mg-Mg from: " << Name_file_t << std::endl;
#endif
      f_adp = fopen(Name_file_t, "r");
      if (f_adp == NULL) {
        std::cout << "" RED "File corrupted" RESET "" << std::endl;
        return EXIT_FAILURE;
      }
      break;
    case HH:

      scp_material = H;

      snprintf(Name_file_t, sizeof(Name_file_t), "%s/adph.dat",
               PotentialsFolder);
#ifdef DEBUG_MODE
      if (rank_MPI == 0)
        std::cout << "Reading ADP H-H from: " << Name_file_t << std::endl;
#endif
      f_adp = fopen(Name_file_t, "r");
      if (f_adp == NULL) {
        std::cout << "" RED "File corrupted" RESET "" << std::endl;
        return EXIT_FAILURE;
      }
      break;
    case MgH:
      snprintf(Name_file_t, sizeof(Name_file_t), "%s/adpmgh.dat",
               PotentialsFolder);
#ifdef DEBUG_MODE
      if (rank_MPI == 0)
        std::cout << "Reading ADP Mg-H from: " << Name_file_t << std::endl;
#endif
      f_adp = fopen(Name_file_t, "r");
      if (f_adp == NULL) {
        std::cout << "" RED "File corrupted" RESET "" << std::endl;
        return EXIT_FAILURE;
      }
      alloy = true;
      break;
    default:
      if (rank_MPI == 0) printf("init_ADP: Unknown material\n");
      exit(0);
  }

  if (alloy == false) {

    error =
        fscanf(f_adp, "%lf %lf %lf \n", &(adp->mass), &(adp->factor),
               &(adp->r_cutoff));  // the r_cutoff is useless here, we read it
                                   // again when reading the position of atoms

    element_mass[scp_material] = adp->mass;

    error = fscanf(f_adp, "%d %lf \n", &(adp->n_rho), &dr_rho);

    init_spline(&adp->rho, adp->n_rho, dr_rho);

    #if defined(KOKKOS_ENABLE_CUDA)

    read_spline(f_adp, adp->n_rho, adp->rho );

    #else
    for (i = 0; i < adp->n_rho; i++) {
      error = fscanf(f_adp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
                     &adp->rho.a[i], &adp->rho.b[i], &adp->rho.c[i],
                     &adp->rho.d[i], &adp->rho.db[i], &adp->rho.dc[i],
                     &adp->rho.dd[i], &adp->rho.ddc[i], &adp->rho.ddd[i]);
    }
    #endif
    error = fscanf(f_adp, "%d %lf \n", &adp->n_embed, &drho_embed);
    init_spline(&adp->embed, adp->n_embed, drho_embed);

    #if defined(KOKKOS_ENABLE_CUDA)

    read_spline(f_adp, adp->n_embed, adp->embed );
    
    #else
    for (i = 0; i < adp->n_embed; i++) {
      error = fscanf(f_adp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
                     &adp->embed.a[i], &adp->embed.b[i], &adp->embed.c[i],
                     &adp->embed.d[i], &adp->embed.db[i], &adp->embed.dc[i],
                     &adp->embed.dd[i], &adp->embed.ddc[i], &adp->embed.ddd[i]);
    }
    #endif
  } else {
    error = fscanf(f_adp, "%lf \n",
                   &adp->r_cutoff);  // the r_cutoff is useless here, we read it
                                     // again when reading the position of atoms
    init_spline(&adp->rho, 1, 0);
    init_spline(&adp->embed, 1, 0);
  }

  //! @brief Read spline for the pairing term phi
  error = fscanf(f_adp, "%d %lf \n", &adp->n_pair, &dr_pair);
  init_spline(&adp->pair, adp->n_pair, dr_pair);

  #if defined(KOKKOS_ENABLE_CUDA)

  read_spline(f_adp, adp->n_pair, adp->pair );
  
  #else
  for (i = 0; i < adp->n_pair; i++) {
    error = fscanf(f_adp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
                   &adp->pair.a[i], &adp->pair.b[i], &adp->pair.c[i],
                   &adp->pair.d[i], &adp->pair.db[i], &adp->pair.dc[i],
                   &adp->pair.dd[i], &adp->pair.ddc[i], &adp->pair.ddd[i]);
  }
  #endif
  //! @brief Read spline for the dipole function u
  error = fscanf(f_adp, "%d %lf \n", &adp->n_u, &dr_u);
  init_spline(&adp->u, adp->n_u, dr_u);

  #if defined(KOKKOS_ENABLE_CUDA)

  read_spline(f_adp, adp->n_u, adp->u );
  
  #else
  for (i = 0; i < adp->n_u; i++) {
    error = fscanf(f_adp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
                   &adp->u.a[i], &adp->u.b[i], &adp->u.c[i], &adp->u.d[i],  //!
                   &adp->u.db[i], &adp->u.dc[i], &adp->u.dd[i],             //!
                   &adp->u.ddc[i], &adp->u.ddd[i]);                         //!
  }
  #endif
  //! @brief Read spline for the quadrupole function w
  error = fscanf(f_adp, "%d %lf \n", &adp->n_w, &dr_w);
  init_spline(&adp->w, adp->n_w, dr_w);

  #if defined(KOKKOS_ENABLE_CUDA)

  read_spline(f_adp, adp->n_w, adp->w );
  
  #else
  for (i = 0; i < adp->n_w; i++) {
    error = fscanf(f_adp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
                   &adp->w.a[i], &adp->w.b[i], &adp->w.c[i], &adp->w.d[i],  //!
                   &adp->w.db[i], &adp->w.dc[i], &adp->w.dd[i],             //!
                   &adp->w.ddc[i], &adp->w.ddd[i]);                         //!
  }
  #endif
  fclose(f_adp);

  return EXIT_SUCCESS;
}

/********************************************************************************/

potential_function rho_ij_adp_MgHx_constructor() {
  potential_function rho_ij_adp_MgHx;
  rho_ij_adp_MgHx.F = rho_ij;
  rho_ij_adp_MgHx.FK = rho_ij_kokkos;

  rho_ij_adp_MgHx.dF_dq = d_rho_ij_dq;
  rho_ij_adp_MgHx.d2F_dq2 = d2_rho_ij_dq2;

  rho_ij_adp_MgHx.dF_dq_FD = d_rho_ij_dq_FD;
  rho_ij_adp_MgHx.d2F_dq2_FD = d2_rho_ij_dq2_FD;

  rho_ij_adp_MgHx.dF_dn = d_rho_ij_dn;


  return rho_ij_adp_MgHx;
}

/********************************************************************************/

KOKKOS_FUNCTION static void rho_ij_kokkos(double* rho_ij, const double* n, const double* q,
  const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set to zero the density
  *rho_ij = 0.0;

  //! Set cubic spline to evaluate the energy density
  CubicSpline rho_j;
  if (spc[j] == Mg) {
    rho_j = adp_Device_Default(0).rho;
  } else if (spc[1] == H) {
    rho_j = adp_Device_Default(1).rho;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += dsqr(r_ij[alpha]);
  }

  double norm_r_ij = sqrt(r2_ij);
  double result = cubic_spline(&rho_j, norm_r_ij);
  *rho_ij = n[j] * result;
}

/********************************************************************************/
 static void rho_ij(double* rho_ij, const double* n, const double* q,
  const AtomicSpecie* spc) {

unsigned int dim = NumberDimensions;
unsigned int i = 0, j = 1;
double r2_ij = 0.0;
double r_ij[3];

//! Set to zero the density
*rho_ij = 0.0;

//! Set cubic spline to evaluate the energy density
CubicSpline rho_j;
if (spc[j] == Mg) {
rho_j = adp_MgMg.rho;
} else if (spc[j] == H) {
rho_j = adp_HH.rho;
}

//! Compute parameters
for (unsigned int alpha = 0; alpha < dim; alpha++) {
r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
r2_ij += DSQR(r_ij[alpha]);
}
double norm_r_ij = sqrt(r2_ij);

*rho_ij = n[j] * cubic_spline(&rho_j, norm_r_ij);
}

/********************************************************************************/

static void d_rho_ij_dq(int direction, double* d_rho_ij_dq, const double* n,
                        const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set d_rho_ij_dq to zero
  for (unsigned int dof = 0; dof < 3; dof++) {
    d_rho_ij_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the energy density
  CubicSpline rho_j;
  if (spc[j] == Mg) {
    rho_j = adp_MgMg.rho;
  } else if (spc[j] == H) {
    rho_j = adp_HH.rho;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += DSQR(r_ij[alpha]);
  }
  double norm_r_ij = sqrt(r2_ij);
  double norm_r_ij_m1 = 1.0 / norm_r_ij;

  double n_d_rho_ij = n[j] * d_cubic_spline(&rho_j, norm_r_ij);

  //! Direction i
  if (direction == 0) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      d_rho_ij_dq[alpha] = n_d_rho_ij * norm_r_ij_m1 * r_ij[alpha];
    }
  }

  //! Direction j
  if (direction == 1) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      d_rho_ij_dq[alpha] = -n_d_rho_ij * norm_r_ij_m1 * r_ij[alpha];
    }
  }
}

/********************************************************************************/

static void d_rho_ij_dq_FD(int direction, double* d_rho_ij_dq, const double* n,
                           const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 2;
  unsigned int d_idx;
  double dr = 0.0001;  // *rc;
  double f_mm, f_m, f_p, f_pp;

  //! Set to zero
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    d_rho_ij_dq[alpha] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  rho_ij(&f_0, n, q, spc);

  //! Allocate memory
  double* q_new = (double*)calloc(dim * num_sites, sizeof(double));
  for (unsigned int dof = 0; dof < dim * num_sites; dof++) {
    q_new[dof] = q[dof];
  }

  for (unsigned int alpha = 0; alpha < dim; alpha++) {

    d_idx = direction * dim + alpha;

    // f(x - 2*Dx,y)
    q_new[d_idx] = q[d_idx] - dr * 2;
    f_mm = 0.0;
    rho_ij(&f_mm, n, q_new, spc);

    // f(x - Dx,y)
    q_new[d_idx] = q[d_idx] - dr;
    f_m = 0.0;
    rho_ij(&f_m, n, q_new, spc);

    // f(x + Dx,y)
    q_new[d_idx] = q[d_idx] + dr;
    f_p = 0.0;
    rho_ij(&f_p, n, q_new, spc);

    // f(x + 2*Dx,y)
    q_new[d_idx] = q[d_idx] + dr * 2;
    f_pp = 0.0;
    rho_ij(&f_pp, n, q_new, spc);

    d_rho_ij_dq[alpha] =
        ((f_mm - f_0) - 8.0 * (f_m - f_0) + 8.0 * (f_p - f_0) - (f_pp - f_0)) /
        (12.0 * dr);
  }

  //! Free memory
  free(q_new);
}

/********************************************************************************/

static void d2_rho_ij_dq2(int direction, double* d2_rho_ij_dq, const double* n,
                          const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2_rho_ij_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the energy density
  CubicSpline rho_j;
  if (spc[j] == Mg) {
    rho_j = adp_MgMg.rho;
  } else if (spc[j] == H) {
    rho_j = adp_HH.rho;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += DSQR(r_ij[alpha]);
  }
  double r2_ij_m1 = 1.0 / r2_ij;
  double norm_r_ij = sqrt(r2_ij);

  double hessian_r_ij[NumberDimensions * NumberDimensions] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  compute_hessian_norm_r(hessian_r_ij, r_ij, norm_r_ij);

  double n_d_rho_ij = n[j] * d_cubic_spline(&rho_j, norm_r_ij);
  double n_dd_rho_ij = n[j] * d2_cubic_spline(&rho_j, norm_r_ij);

  // ii direction
  if (direction == 0) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2_rho_ij_dq[alpha * dim + beta] =
            n_dd_rho_ij * r2_ij_m1 * (r_ij[alpha] * r_ij[beta]) +
            n_d_rho_ij * hessian_r_ij[alpha * dim + beta];
      }
    }
  }

  // ij or ji direction
  if ((direction == 1) || (direction == 2)) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2_rho_ij_dq[alpha * dim + beta] =
            -n_dd_rho_ij * r2_ij_m1 * (r_ij[alpha] * r_ij[beta]) -
            n_d_rho_ij * hessian_r_ij[alpha * dim + beta];
      }
    }
  }

  // jj direction
  if (direction == 3) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2_rho_ij_dq[alpha * dim + beta] =
            n_dd_rho_ij * r2_ij_m1 * (r_ij[alpha] * r_ij[beta]) +
            n_d_rho_ij * hessian_r_ij[alpha * dim + beta];
      }
    }
  }
}

/********************************************************************************/

static void d2_rho_ij_dq2_FD(int direction, double* d2_rho_ij_dq,
                             const double* n, const double* q,
                             const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 2;
  double dr = 0.0001;  // *rc;

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2_rho_ij_dq[dof] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  rho_ij(&f_0, n, q, spc);

  //! Allocate memory
  double* q_pp = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_p = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_m = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_mm = (double*)calloc(dim * num_sites, sizeof(double));

  //
  for (unsigned int alpha = 0; alpha < dim; alpha++) {

    unsigned int d_dof_i = direction / (1 + num_sites) * dim + alpha;

    //! Compute the positions
    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int beta = 0; beta < dim; beta++) {

        unsigned int dof_idx_aux = site_idx * dim + beta;

        bool D_alpha = (dof_idx_aux == d_dof_i) ? true : false;

        q_pp[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr * 2;

        q_p[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr;

        q_m[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr;

        q_mm[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr * 2;
      }
    }

    //! Evaluate functions

    // f(x + 2*Dx,y)
    double f_pp = 0.0;
    rho_ij(&f_pp, n, q_pp, spc);

    // f(x + Dx,y)
    double f_p = 0.0;
    rho_ij(&f_p, n, q_p, spc);

    // f(x - Dx,y)
    double f_m = 0.0;
    rho_ij(&f_m, n, q_m, spc);

    // f(x - 2*Dx,y)
    double f_mm = 0.0;
    rho_ij(&f_mm, n, q_mm, spc);

    d2_rho_ij_dq[alpha * dim + alpha] =
        (-f_pp + 16.0 * f_p - 30.0 * f_0 + 16.0 * f_m - f_mm) /
        (12.0 * dr * dr);
  }

  //! Free memory
  free(q_pp);
  free(q_p);
  free(q_m);
  free(q_mm);
}

/********************************************************************************/

static void d_rho_ij_dn(int direction, double* d_rho_ij_dn, const double* n,
                        const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set to zero the density
  *d_rho_ij_dn = 0.0;

  //! Set cubic spline to evaluate the energy density
  CubicSpline rho_j;
  if (spc[j] == Mg) {
    rho_j = adp_MgMg.rho;
  } else if (spc[j] == H) {
    rho_j = adp_HH.rho;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += DSQR(r_ij[alpha]);
  }
  double norm_r_ij = sqrt(r2_ij);

  if (direction == j) {
    *d_rho_ij_dn = cubic_spline(&rho_j, norm_r_ij);
  }
}

/********************************************************************************/

potential_function V_pair_ij_adp_MgHx_constructor() {
  potential_function V_pair_ij_adp_MgHx;

  V_pair_ij_adp_MgHx.F = V_pair_ij;
  V_pair_ij_adp_MgHx.FK = V_pair_ij_kokkos;

  //! Analytical derivatives dV_pair_ij_dq and d2V_pair_ij_dq2
  V_pair_ij_adp_MgHx.dF_dq = dV_pair_ij_dq;
  V_pair_ij_adp_MgHx.d2F_dq2 = d2V_pair_ij_dq2;

  //! Numerical derivatives of dV_pair_ij_dq and d2V_pair_ij_dq2
  V_pair_ij_adp_MgHx.dF_dq_FD = dV_pair_ij_dq_FD;
  V_pair_ij_adp_MgHx.d2F_dq2_FD = d2V_pair_ij_dq2_FD;

  //! Analytical derivatives of dV_pair_ij_dn
  V_pair_ij_adp_MgHx.dF_dn = dV_pair_ij_dn;

  return V_pair_ij_adp_MgHx;
}

/********************************************************************************/

static void V_pair_ij(double* V_pair_ij, const double* n, const double* q,
                      const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set to zero the pair term contribution to the energy
  *V_pair_ij = 0.0;

  //! Set cubic spline to evaluate the pair interation curve
  CubicSpline pair_ij;
  if ((spc[i] == Mg) && (spc[j] == Mg)) {
    pair_ij = adp_MgMg.pair;
  } else if ((spc[i] == H) && (spc[j] == H)) {
    pair_ij = adp_HH.pair;
  } else {
    pair_ij = adp_MgH.pair;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += DSQR(r_ij[alpha]);
  }
  double norm_r_ij = sqrt(r2_ij);

  double nn_phi_ij1 = n[i] * n[j] * cubic_spline(&pair_ij, norm_r_ij);

  *V_pair_ij = 0.5 * nn_phi_ij1;
}

KOKKOS_FUNCTION static void V_pair_ij_kokkos(double* V_pair_ij, const double* n, const double* q,
  const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default) {

unsigned int dim = NumberDimensions;
unsigned int i = 0, j = 1;
double r2_ij = 0.0;
double r_ij[3];

//! Set to zero the pair term contribution to the energy
*V_pair_ij = 0.0;

//! Set cubic spline to evaluate the pair interation curve
CubicSpline pair_ij;
if ((spc[i] == Mg) && (spc[j] == Mg)) {
pair_ij = adp_Device_Default(0).pair;
} else if ((spc[i] == H) && (spc[j] == H)) {
pair_ij = adp_Device_Default(1).pair;
} else {
pair_ij = adp_Device_Default(2).pair;
}

//! Compute parameters
for (unsigned int alpha = 0; alpha < dim; alpha++) {
r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
r2_ij += dsqr(r_ij[alpha]);
}
double norm_r_ij = sqrt(r2_ij);

double nn_phi_ij1 = n[i] * n[j] * cubic_spline(&pair_ij, norm_r_ij);

*V_pair_ij = 0.5 * nn_phi_ij1;
}

/********************************************************************************/

static void dV_pair_ij_dq(int direction, double* dV_pair_ij_dq, const double* n,
                          const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set to zero
  for (unsigned int dof = 0; dof < 3; dof++) {
    dV_pair_ij_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the pair interation curve
  CubicSpline pair_ij;
  if ((spc[i] == Mg) && (spc[j] == Mg)) {
    pair_ij = adp_MgMg.pair;
  } else if ((spc[i] == H) && (spc[j] == H)) {
    pair_ij = adp_HH.pair;
  } else {
    pair_ij = adp_MgH.pair;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += DSQR(r_ij[alpha]);
  }
  double norm_r_ij = sqrt(r2_ij);
  double norm_r_ij_m1 = 1.0 / norm_r_ij;
  double nn_d_pair_ij = n[i] * n[j] * d_cubic_spline(&pair_ij, norm_r_ij);

  //! Direction i
  if (direction == i) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_pair_ij_dq[alpha] = 0.5 * nn_d_pair_ij * norm_r_ij_m1 * r_ij[alpha];
    }
  }

  //! Direction j
  if (direction == j) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_pair_ij_dq[alpha] = -0.5 * nn_d_pair_ij * norm_r_ij_m1 * r_ij[alpha];
    }
  }
}

/********************************************************************************/

static void dV_pair_ij_dq_FD(int direction, double* dV_pair_ij_dq,
                             const double* n, const double* q,
                             const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 2;
  unsigned int d_idx;
  double dr = 0.00001;  // *rc;
  double f_mm, f_m, f_p, f_pp;

  //! Set to zero
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    dV_pair_ij_dq[alpha] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  V_pair_ij(&f_0, n, q, spc);

  //! Allocate memory
  double* q_new = (double*)calloc(dim * num_sites, sizeof(double));
  for (unsigned int dof = 0; dof < dim * num_sites; dof++) {
    q_new[dof] = q[dof];
  }

  for (unsigned int alpha = 0; alpha < dim; alpha++) {

    d_idx = direction * dim + alpha;

    // f(x - 2*Dx,y)
    q_new[d_idx] = q[d_idx] - dr * 2;
    f_mm = 0.0;
    V_pair_ij(&f_mm, n, q_new, spc);

    // f(x - Dx,y)
    q_new[d_idx] = q[d_idx] - dr;
    f_m = 0.0;
    V_pair_ij(&f_m, n, q_new, spc);

    // f(x + Dx,y)
    q_new[d_idx] = q[d_idx] + dr;
    f_p = 0.0;
    V_pair_ij(&f_p, n, q_new, spc);

    // f(x + 2*Dx,y)
    q_new[d_idx] = q[d_idx] + dr * 2;
    f_pp = 0.0;
    V_pair_ij(&f_pp, n, q_new, spc);

    dV_pair_ij_dq[alpha] =
        ((f_mm - f_0) - 8.0 * (f_m - f_0) + 8.0 * (f_p - f_0) - (f_pp - f_0)) /
        (12.0 * dr);
  }

  //! Free memory
  free(q_new);
}

/********************************************************************************/

static void d2V_pair_ij_dq2(int direction, double* d2V_pair_ij_dq,
                            const double* n, const double* q,
                            const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2V_pair_ij_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the pair interation curve
  CubicSpline pair_ij;
  if ((spc[i] == Mg) && (spc[j] == Mg)) {
    pair_ij = adp_MgMg.pair;
  } else if ((spc[i] == H) && (spc[j] == H)) {
    pair_ij = adp_HH.pair;
  } else {
    pair_ij = adp_MgH.pair;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += DSQR(r_ij[alpha]);
  }
  double r2_ij_m1 = 1.0 / r2_ij;
  double norm_r_ij = sqrt(r2_ij);

  double hessian_r_ij[NumberDimensions * NumberDimensions] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  compute_hessian_norm_r(hessian_r_ij, r_ij, norm_r_ij);

  double nn_d_pair_ij = n[i] * n[j] * d_cubic_spline(&pair_ij, norm_r_ij);
  double nn_dd_pair_ij = n[i] * n[j] * d2_cubic_spline(&pair_ij, norm_r_ij);

  // ii direction
  if (direction == 0) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_pair_ij_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
            (nn_dd_pair_ij * r2_ij_m1 * (r_ij[alpha] * r_ij[beta]) +
             nn_d_pair_ij * hessian_r_ij[alpha * dim + beta]);
      }
    }
  }

  // ij and ji directions
  if ((direction == 1) || (direction == 2)) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_pair_ij_dq[alpha * dim + beta] =
            -(1.0 / 2.0) *
            (nn_dd_pair_ij * r2_ij_m1 * (r_ij[alpha] * r_ij[beta]) +
             nn_d_pair_ij * hessian_r_ij[alpha * dim + beta]);
      }
    }
  }

  // jj direction
  if (direction == 3) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_pair_ij_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
            (nn_dd_pair_ij * r2_ij_m1 * (r_ij[alpha] * r_ij[beta]) +
             nn_d_pair_ij * hessian_r_ij[alpha * dim + beta]);
      }
    }
  }
}

/********************************************************************************/

static void d2V_pair_ij_dq2_FD(int direction, double* d2V_pair_ij_dq,
                               const double* n, const double* q,
                               const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 2;
  double dr = 0.0001;  // *rc;

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2V_pair_ij_dq[dof] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  V_pair_ij(&f_0, n, q, spc);

  //! Allocate memory
  double* q_pp = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_p = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_m = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_mm = (double*)calloc(dim * num_sites, sizeof(double));

  //
  for (unsigned int alpha = 0; alpha < dim; alpha++) {

    unsigned int d_dof_i = direction / (1 + num_sites) * dim + alpha;

    //! Compute the positions
    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int beta = 0; beta < dim; beta++) {

        unsigned int dof_idx_aux = site_idx * dim + beta;

        bool D_alpha = (dof_idx_aux == d_dof_i) ? true : false;

        q_pp[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr * 2;

        q_p[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr;

        q_m[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr;

        q_mm[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr * 2;
      }
    }

    //! Evaluate functions

    // f(x + 2*Dx,y)
    double f_pp = 0.0;
    V_pair_ij(&f_pp, n, q_pp, spc);

    // f(x + Dx,y)
    double f_p = 0.0;
    V_pair_ij(&f_p, n, q_p, spc);

    // f(x - Dx,y)
    double f_m = 0.0;
    V_pair_ij(&f_m, n, q_m, spc);

    // f(x - 2*Dx,y)
    double f_mm = 0.0;
    V_pair_ij(&f_mm, n, q_mm, spc);

    d2V_pair_ij_dq[alpha * dim + alpha] =
        (-f_pp + 16.0 * f_p - 30.0 * f_0 + 16.0 * f_m - f_mm) /
        (12.0 * dr * dr);
  }

  //! Free memory
  free(q_pp);
  free(q_p);
  free(q_m);
  free(q_mm);
}

/********************************************************************************/

static void dV_pair_ij_dn(int direction, double* dV_pair_ij_dn, const double* n,
                          const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j = 1;
  double r2_ij = 0.0;
  double r_ij[3];

  //! Set to zero the pair term contribution to the energy
  *dV_pair_ij_dn = 0.0;

  //! Set cubic spline to evaluate the pair interation curve
  CubicSpline pair_ij;
  if ((spc[i] == Mg) && (spc[j] == Mg)) {
    pair_ij = adp_MgMg.pair;
  } else if ((spc[i] == H) && (spc[j] == H)) {
    pair_ij = adp_HH.pair;
  } else {
    pair_ij = adp_MgH.pair;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij[alpha] = q[dim * i + alpha] - q[dim * j + alpha];
    r2_ij += DSQR(r_ij[alpha]);
  }
  double norm_r_ij = sqrt(r2_ij);

  //! Direction i
  if (direction == i) {
    double d_nn_phi_ij1 = n[j] * cubic_spline(&pair_ij, norm_r_ij);
    *dV_pair_ij_dn = 0.5 * d_nn_phi_ij1;
  }

  //! Direction j
  if (direction == j) {
    double d_nn_phi_ij1 = n[i] * cubic_spline(&pair_ij, norm_r_ij);
    *dV_pair_ij_dn = 0.5 * d_nn_phi_ij1;
  }
}

/********************************************************************************/

potential_function V_dipole_ij1j2_adp_MgHx_constructor() {

  potential_function V_dipole_ij1j2_adp_MgHx;

  V_dipole_ij1j2_adp_MgHx.F = V_dipole_ij1j2;
  V_dipole_ij1j2_adp_MgHx.FK = V_dipole_ij1j2_kokkos;

  V_dipole_ij1j2_adp_MgHx.dF_dq = dV_dipole_ij1j2_dq;
  V_dipole_ij1j2_adp_MgHx.d2F_dq2 = dV2_dipole_ij1j2_dq2;

  V_dipole_ij1j2_adp_MgHx.dF_dq_FD = dV_dipole_ij1j2_dq_FD;
  V_dipole_ij1j2_adp_MgHx.d2F_dq2_FD = dV2_dipole_ij1j2_dq2_FD;

  //! Analytical derivatives of dV_dipole_ij1j2_dn
  V_dipole_ij1j2_adp_MgHx.dF_dn = dV_dipole_ij1j2_dn;

  return V_dipole_ij1j2_adp_MgHx;
}

/********************************************************************************/

static void V_dipole_ij1j2(double* V_dipole_ij1j2, const double* n,
                           const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Contribution of the dipole to the energy
  *V_dipole_ij1j2 = 0.0;

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline u_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    u_ij1 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    u_ij1 = adp_HH.u;
  } else {
    u_ij1 = adp_MgH.u;
  }

  CubicSpline u_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    u_ij2 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    u_ij2 = adp_HH.u;
  } else {
    u_ij2 = adp_MgH.u;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij2 = sqrt(r2_ij2);

  double nn_u_ij1 = n[i] * n[j1] * cubic_spline(&u_ij1, norm_r_ij1);
  double nn_u_ij2 = n[i] * n[j2] * cubic_spline(&u_ij2, norm_r_ij2);

  *V_dipole_ij1j2 = (1.0 / 2.0) * nn_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2;
}

/********************************************************************************/

KOKKOS_FUNCTION static void V_dipole_ij1j2_kokkos(double* V_dipole_ij1j2, const double* n,
  const double* q, const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default) {

unsigned int dim = NumberDimensions;
unsigned int i = 0, j1 = 1, j2 = 2;
double r2_ij1 = 0.0;
double r2_ij2 = 0.0;
double r_ij1__dot__r_ij2 = 0.0;
double r_ij1[3];
double r_ij2[3];

//! Contribution of the dipole to the energy
*V_dipole_ij1j2 = 0.0;

//! Set cubic spline to evaluate the dipole interation curve
CubicSpline u_ij1;
if ((spc[i] == Mg) && (spc[j1] == Mg)) {
u_ij1 = adp_Device_Default(0).u;
} else if ((spc[i] == H) && (spc[j1] == H)) {
u_ij1 = adp_Device_Default(1).u;
} else {
u_ij1 = adp_Device_Default(2).u;
}

CubicSpline u_ij2;
if ((spc[i] == Mg) && (spc[j2] == Mg)) {
u_ij2 = adp_Device_Default(0).u;
} else if ((spc[i] == H) && (spc[j2] == H)) {
u_ij2 = adp_Device_Default(1).u;
} else {
u_ij2 = adp_Device_Default(2).u;
}

//! Compute parameters
for (unsigned int alpha = 0; alpha < dim; alpha++) {
r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
r2_ij1 += dsqr(r_ij1[alpha]);
r2_ij2 += dsqr(r_ij2[alpha]);
r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
}
double norm_r_ij1 = sqrt(r2_ij1);
double norm_r_ij2 = sqrt(r2_ij2);

double nn_u_ij1 = n[i] * n[j1] * cubic_spline(&u_ij1, norm_r_ij1);
double nn_u_ij2 = n[i] * n[j2] * cubic_spline(&u_ij2, norm_r_ij2);

*V_dipole_ij1j2 = (1.0 / 2.0) * nn_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2;
}
/********************************************************************************/

static void dV_dipole_ij1j2_dq(int direction, double* dV_dipole_ij1j2_dq,
                               const double* n, const double* q,
                               const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Set to zero
  for (unsigned int dof = 0; dof < 3; dof++) {
    dV_dipole_ij1j2_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline u_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    u_ij1 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    u_ij1 = adp_HH.u;
  } else {
    u_ij1 = adp_MgH.u;
  }

  CubicSpline u_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    u_ij2 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    u_ij2 = adp_HH.u;
  } else {
    u_ij2 = adp_MgH.u;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij1_m1 = 1.0 / norm_r_ij1;
  double norm_r_ij2 = sqrt(r2_ij2);
  double norm_r_ij2_m1 = 1.0 / norm_r_ij2;

  double nn_u_ij1 = n[i] * n[j1] * cubic_spline(&u_ij1, norm_r_ij1);
  double nn_u_ij2 = n[i] * n[j2] * cubic_spline(&u_ij2, norm_r_ij2);

  double nn_d_u_ij1 = n[i] * n[j1] * d_cubic_spline(&u_ij1, norm_r_ij1);
  double nn_d_u_ij2 = n[i] * n[j2] * d_cubic_spline(&u_ij2, norm_r_ij2);

  //! Direction i
  if (direction == 0) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_dipole_ij1j2_dq[alpha] =
          0.5 * (nn_d_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     r_ij1[alpha] +
                 nn_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     r_ij2[alpha] +
                 nn_u_ij1 * nn_u_ij2 * (r_ij1[alpha] + r_ij2[alpha]));
    }
  }

  //! Direction j1
  if (direction == 1) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_dipole_ij1j2_dq[alpha] =
          -0.5 * (nn_d_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                      r_ij1[alpha] +
                  nn_u_ij1 * nn_u_ij2 * r_ij2[alpha]);
    }
  }

  //! Direction j2
  if (direction == 2) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_dipole_ij1j2_dq[alpha] =
          -0.5 * (nn_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                      r_ij2[alpha] +
                  nn_u_ij1 * nn_u_ij2 * r_ij1[alpha]);
    }
  }
}

/********************************************************************************/

static void dV_dipole_ij1j2_dq_FD(int direction, double* dV_dipole_ij1j2_dq,
                                  const double* n, const double* q,
                                  const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 3;
  unsigned int d_idx;
  double dr = 0.0001;  // *rc;
  double f_mm, f_m, f_p, f_pp;

  //! Set to zero
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    dV_dipole_ij1j2_dq[alpha] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  V_dipole_ij1j2(&f_0, n, q, spc);

  //! Allocate memory
  double* q_new = (double*)calloc(dim * num_sites, sizeof(double));
  for (unsigned int dof = 0; dof < dim * num_sites; dof++) {
    q_new[dof] = q[dof];
  }

  for (unsigned int alpha = 0; alpha < dim; alpha++) {

    d_idx = direction * dim + alpha;

    // f(x - 2*Dx,y)
    q_new[d_idx] = q[d_idx] - dr * 2;
    f_mm = 0.0;
    V_dipole_ij1j2(&f_mm, n, q_new, spc);

    // f(x - Dx,y)
    q_new[d_idx] = q[d_idx] - dr;
    f_m = 0.0;
    V_dipole_ij1j2(&f_m, n, q_new, spc);

    // f(x + Dx,y)
    q_new[d_idx] = q[d_idx] + dr;
    f_p = 0.0;
    V_dipole_ij1j2(&f_p, n, q_new, spc);

    // f(x + 2*Dx,y)
    q_new[d_idx] = q[d_idx] + dr * 2;
    f_pp = 0.0;
    V_dipole_ij1j2(&f_pp, n, q_new, spc);

    dV_dipole_ij1j2_dq[alpha] =
        ((f_mm - f_0) - 8.0 * (f_m - f_0) + 8.0 * (f_p - f_0) - (f_pp - f_0)) /
        (12.0 * dr);
  }

  //! Free memory
  free(q_new);
}

/********************************************************************************/

static void dV2_dipole_ij1j2_dq2(int direction, double* d2V_dipole_ij1j2_dq,
                                 const double* n, const double* q,
                                 const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2V_dipole_ij1j2_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline u_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    u_ij1 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    u_ij1 = adp_HH.u;
  } else {
    u_ij1 = adp_MgH.u;
  }

  CubicSpline u_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    u_ij2 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    u_ij2 = adp_HH.u;
  } else {
    u_ij2 = adp_MgH.u;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double r2_ij1_m1 = 1.0 / r2_ij1;
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij1_m1 = 1.0 / norm_r_ij1;
  double r2_ij2_m1 = 1.0 / r2_ij2;
  double norm_r_ij2 = sqrt(r2_ij2);
  double norm_r_ij2_m1 = 1.0 / norm_r_ij2;

  double hessian_r_ij1[NumberDimensions * NumberDimensions] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  compute_hessian_norm_r(hessian_r_ij1, r_ij1, norm_r_ij1);
  double hessian_r_ij2[NumberDimensions * NumberDimensions] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  compute_hessian_norm_r(hessian_r_ij2, r_ij2, norm_r_ij2);

  double Identity[NumberDimensions * NumberDimensions] = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  double nn_u_ij1 = n[i] * n[j1] * cubic_spline(&u_ij1, norm_r_ij1);
  double nn_u_ij2 = n[i] * n[j2] * cubic_spline(&u_ij2, norm_r_ij2);

  double nn_d_u_ij1 = n[i] * n[j1] * d_cubic_spline(&u_ij1, norm_r_ij1);
  double nn_d_u_ij2 = n[i] * n[j2] * d_cubic_spline(&u_ij2, norm_r_ij2);

  double nn_dd_u_ij1 = n[i] * n[j1] * d2_cubic_spline(&u_ij1, norm_r_ij1);
  double nn_dd_u_ij2 = n[i] * n[j2] * d2_cubic_spline(&u_ij2, norm_r_ij2);

  // ii direction
  if (direction == 0) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_dipole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
            (nn_dd_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2 * r2_ij1_m1 *
                 (r_ij1[alpha] * r_ij1[beta]) +
             nn_d_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                 norm_r_ij2_m1 * (r_ij1[alpha] * r_ij2[beta]) +
             nn_d_u_ij1 * nn_u_ij2 * norm_r_ij1_m1 *
                 (r_ij1[alpha] * r_ij2[beta] + r_ij1[alpha] * r_ij1[beta]) +
             nn_d_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2 *
                 hessian_r_ij1[alpha * dim + beta] +
             nn_d_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                 norm_r_ij1_m1 * (r_ij2[alpha] * r_ij1[beta]) +
             nn_u_ij1 * nn_dd_u_ij2 * r_ij1__dot__r_ij2 * r2_ij2_m1 *
                 (r_ij2[alpha] * r_ij2[beta]) +
             nn_u_ij1 * nn_d_u_ij2 * norm_r_ij2_m1 *
                 (r_ij2[alpha] * r_ij1[beta] + r_ij2[alpha] * r_ij2[beta]) +
             nn_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 *
                 hessian_r_ij2[alpha * dim + beta] +
             nn_d_u_ij1 * nn_u_ij2 * norm_r_ij1_m1 *
                 (r_ij1[alpha] * r_ij1[beta] + r_ij2[alpha] * r_ij1[beta]) +
             nn_u_ij1 * nn_d_u_ij2 * norm_r_ij2_m1 *
                 (r_ij1[alpha] * r_ij2[beta] + r_ij2[alpha] * r_ij2[beta]) +
             nn_u_ij1 * nn_u_ij2 * 2 * Identity[alpha * dim + beta]);
      }
    }
  }

  // j1j1 direction
  if (direction == 4) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_dipole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
            (nn_dd_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2 * r2_ij1_m1 *
                 (r_ij1[alpha] * r_ij1[beta]) +
             nn_d_u_ij1 * nn_u_ij2 * norm_r_ij1_m1 *
                 (r_ij1[alpha] * r_ij2[beta] + r_ij2[alpha] * r_ij1[beta]) +
             nn_d_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2 *
                 hessian_r_ij1[alpha * dim + beta]);
      }
    }
  }

  // j1j2 direction
  if (direction == 5) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_dipole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
            (nn_d_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                 norm_r_ij2_m1 * (r_ij1[alpha] * r_ij2[beta]) +
             nn_d_u_ij1 * nn_u_ij2 * norm_r_ij1_m1 *
                 (r_ij1[alpha] * r_ij1[beta]) +
             nn_u_ij1 * nn_d_u_ij2 * norm_r_ij2_m1 *
                 (r_ij2[alpha] * r_ij2[beta]) +
             nn_u_ij1 * nn_u_ij2 * Identity[alpha * dim + beta]);
      }
    }
  }

  // j2j1 direction
  if (direction == 7) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_dipole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
            (nn_d_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                 norm_r_ij1_m1 * (r_ij2[alpha] * r_ij1[beta]) +
             nn_u_ij1 * nn_d_u_ij2 * norm_r_ij2_m1 *
                 (r_ij2[alpha] * r_ij2[beta]) +
             nn_d_u_ij1 * nn_u_ij2 * norm_r_ij1_m1 *
                 (r_ij1[alpha] * r_ij1[beta]) +
             nn_u_ij1 * nn_u_ij2 * Identity[alpha * dim + beta]);
      }
    }
  }

  // j2j2 direction
  if (direction == 8) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_dipole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
            (nn_u_ij1 * nn_dd_u_ij2 * r_ij1__dot__r_ij2 * r2_ij2_m1 *
                 (r_ij2[alpha] * r_ij2[beta]) +
             nn_u_ij1 * nn_d_u_ij2 * norm_r_ij2_m1 *
                 (r_ij2[alpha] * r_ij1[beta] + r_ij1[alpha] * r_ij2[beta]) +
             nn_u_ij1 * nn_d_u_ij2 * r_ij1__dot__r_ij2 *
                 hessian_r_ij2[alpha * dim + beta]);
      }
    }
  }
}

/********************************************************************************/

static void dV2_dipole_ij1j2_dq2_FD(int direction, double* d2V_dipole_ij1j2_dq,
                                    const double* n, const double* q,
                                    const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 3;
  double dr = 1.0e-4;  // *rc;

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2V_dipole_ij1j2_dq[dof] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  V_dipole_ij1j2(&f_0, n, q, spc);

  //! Allocate memory
  double* q_pp = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_p = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_m = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_mm = (double*)calloc(dim * num_sites, sizeof(double));

  //
  for (unsigned int alpha = 0; alpha < dim; alpha++) {

    unsigned int d_dof_i = direction / (1 + num_sites) * dim + alpha;

    //! Compute the positions
    for (unsigned int site_idx = 0; site_idx < num_sites; site_idx++) {

      for (unsigned int gamma = 0; gamma < dim; gamma++) {

        unsigned int dof_idx_aux = site_idx * dim + gamma;
        bool D_alpha = (dof_idx_aux == d_dof_i) ? true : false;

        q_pp[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr * 2;

        q_p[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr;

        q_m[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr;

        q_mm[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr * 2;
      }
    }

    //! Evaluate functions

    // f(x + 2*Dx,y)
    double f_pp = 0.0;
    V_dipole_ij1j2(&f_pp, n, q_pp, spc);

    // f(x + Dx,y)
    double f_p = 0.0;
    V_dipole_ij1j2(&f_p, n, q_p, spc);

    // f(x - Dx,y)
    double f_m = 0.0;
    V_dipole_ij1j2(&f_m, n, q_m, spc);

    // f(x - 2*Dx,y)
    double f_mm = 0.0;
    V_dipole_ij1j2(&f_mm, n, q_mm, spc);

    d2V_dipole_ij1j2_dq[alpha * dim + alpha] =
        (-f_pp + 16.0 * f_p - 30.0 * f_0 + 16.0 * f_m - f_mm) /
        (12.0 * dr * dr);
  }

  //! Free memory
  free(q_pp);
  free(q_p);
  free(q_m);
  free(q_mm);
}

/********************************************************************************/

static void dV_dipole_ij1j2_dn(int direction, double* dV_dipole_ij1j2_dn,
                               const double* n, const double* q,
                               const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Contribution of the dipole to the energy
  *dV_dipole_ij1j2_dn = 0.0;

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline u_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    u_ij1 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    u_ij1 = adp_HH.u;
  } else {
    u_ij1 = adp_MgH.u;
  }

  CubicSpline u_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    u_ij2 = adp_MgMg.u;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    u_ij2 = adp_HH.u;
  } else {
    u_ij2 = adp_MgH.u;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij2 = sqrt(r2_ij2);

  //! Direction i
  if (direction == i) {
    double nn_u_ij1 = n[i] * n[j1] * cubic_spline(&u_ij1, norm_r_ij1);
    double nn_u_ij2 = n[i] * n[j2] * cubic_spline(&u_ij2, norm_r_ij2);

    double d_nn_u_ij1 = n[j1] * cubic_spline(&u_ij1, norm_r_ij1);
    double d_nn_u_ij2 = n[j2] * cubic_spline(&u_ij2, norm_r_ij2);

    *dV_dipole_ij1j2_dn =
        (1.0 / 2.0) * d_nn_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2 +
        (1.0 / 2.0) * nn_u_ij1 * d_nn_u_ij2 * r_ij1__dot__r_ij2;
  }

  //! Direction j1
  if (direction == j1) {
    double d_nn_u_ij1 = n[i] * cubic_spline(&u_ij1, norm_r_ij1);
    double nn_u_ij2 = n[i] * n[j2] * cubic_spline(&u_ij2, norm_r_ij2);

    *dV_dipole_ij1j2_dn =
        (1.0 / 2.0) * d_nn_u_ij1 * nn_u_ij2 * r_ij1__dot__r_ij2;
  }

  //! Direction j2
  if (direction == j2) {
    double nn_u_ij1 = n[i] * n[j1] * cubic_spline(&u_ij1, norm_r_ij1);
    double d_nn_u_ij2 = n[i] * cubic_spline(&u_ij2, norm_r_ij2);

    *dV_dipole_ij1j2_dn =
        (1.0 / 2.0) * nn_u_ij1 * d_nn_u_ij2 * r_ij1__dot__r_ij2;
  }
}

/********************************************************************************/

potential_function V_quadrupole_ij1j2_adp_MgHx_constructor() {

  potential_function V_quadrupole_ij1j2_adp_MgHx;

  V_quadrupole_ij1j2_adp_MgHx.F = V_quadrupole_ij1j2;
  V_quadrupole_ij1j2_adp_MgHx.FK = V_quadrupole_ij1j2_kokkos;


  V_quadrupole_ij1j2_adp_MgHx.dF_dq = dV_quadrupole_ij1j2_dq;
  V_quadrupole_ij1j2_adp_MgHx.d2F_dq2 = dV2_quadrupole_ij1j2_dq2;

  V_quadrupole_ij1j2_adp_MgHx.dF_dq_FD = dV_quadrupole_ij1j2_dq_FD;
  V_quadrupole_ij1j2_adp_MgHx.d2F_dq2_FD = dV2_quadrupole_ij1j2_dq2_FD;

  V_quadrupole_ij1j2_adp_MgHx.dF_dn = dV_quadrupole_ij1j2_dn;

  return V_quadrupole_ij1j2_adp_MgHx;
}

/********************************************************************************/

static void V_quadrupole_ij1j2(double* V_quadrupole_ij1_ij2, const double* n,
                               const double* q, const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Set to zero the Partial quadrupole contribution of the sites j and k to i
  *V_quadrupole_ij1_ij2 = 0.0;

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline w_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    w_ij1 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    w_ij1 = adp_HH.w;
  } else {
    w_ij1 = adp_MgH.w;
  }

  CubicSpline w_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    w_ij2 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    w_ij2 = adp_HH.w;
  } else {
    w_ij2 = adp_MgH.w;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij2 = sqrt(r2_ij2);

  double nn_w_ij1 = n[i] * n[j1] * cubic_spline(&w_ij1, norm_r_ij1);
  double nn_w_ij2 = n[i] * n[j2] * cubic_spline(&w_ij2, norm_r_ij2);

  *V_quadrupole_ij1_ij2 =
      (1.0 / 2.0) * nn_w_ij1 * nn_w_ij2 * DSQR(r_ij1__dot__r_ij2) -
      (1.0 / 6.0) * nn_w_ij1 * nn_w_ij2 * r2_ij1 * r2_ij2;
}

/*********************************************************************************/

KOKKOS_FUNCTION static void V_quadrupole_ij1j2_kokkos(double* V_quadrupole_ij1_ij2, const double* n,
  const double* q, const AtomicSpecie* spc, AdpPotencial_Device adp_Device_Default) {

unsigned int dim = NumberDimensions;
unsigned int i = 0, j1 = 1, j2 = 2;
double r2_ij1 = 0.0;
double r2_ij2 = 0.0;
double r_ij1__dot__r_ij2 = 0.0;
double r_ij1[3];
double r_ij2[3];

//! Set to zero the Partial quadrupole contribution of the sites j and k to i
*V_quadrupole_ij1_ij2 = 0.0;

//! Set cubic spline to evaluate the dipole interation curve
CubicSpline w_ij1;
if ((spc[i] == Mg) && (spc[j1] == Mg)) {
w_ij1 = adp_Device_Default(0).w;
} else if ((spc[i] == H) && (spc[j1] == H)) {
w_ij1 = adp_Device_Default(1).w;
} else {
w_ij1 = adp_Device_Default(2).w;
}

CubicSpline w_ij2;
if ((spc[i] == Mg) && (spc[j2] == Mg)) {
w_ij2 = adp_Device_Default(0).w;
} else if ((spc[i] == H) && (spc[j2] == H)) {
w_ij2 = adp_Device_Default(1).w;
} else {
w_ij2 = adp_Device_Default(2).w;
}

//! Compute parameters
for (unsigned int alpha = 0; alpha < dim; alpha++) {
r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
r2_ij1 += dsqr(r_ij1[alpha]);
r2_ij2 += dsqr(r_ij2[alpha]);
r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
}
double norm_r_ij1 = sqrt(r2_ij1);
double norm_r_ij2 = sqrt(r2_ij2);

double nn_w_ij1 = n[i] * n[j1] * cubic_spline(&w_ij1, norm_r_ij1);
double nn_w_ij2 = n[i] * n[j2] * cubic_spline(&w_ij2, norm_r_ij2);

*V_quadrupole_ij1_ij2 =
(1.0 / 2.0) * nn_w_ij1 * nn_w_ij2 * dsqr(r_ij1__dot__r_ij2) -
(1.0 / 6.0) * nn_w_ij1 * nn_w_ij2 * r2_ij1 * r2_ij2;
}

/********************************************************************************/

static void dV_quadrupole_ij1j2_dq(int direction,
                                   double* dV_quadrupole_ij1_ij2_dq,
                                   const double* n, const double* q,
                                   const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Set to zero
  for (unsigned int dof = 0; dof < 3; dof++) {
    dV_quadrupole_ij1_ij2_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline w_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    w_ij1 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    w_ij1 = adp_HH.w;
  } else {
    w_ij1 = adp_MgH.w;
  }

  CubicSpline w_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    w_ij2 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    w_ij2 = adp_HH.w;
  } else {
    w_ij2 = adp_MgH.w;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij1_m1 = 1.0 / norm_r_ij1;
  double norm_r_ij2 = sqrt(r2_ij2);
  double norm_r_ij2_m1 = 1.0 / norm_r_ij2;
  double dsqr_r_ij1__dot__r_ij2 = DSQR(r_ij1__dot__r_ij2);

  double nn_w_ij1 = n[i] * n[j1] * cubic_spline(&w_ij1, norm_r_ij1);
  double nn_w_ij2 = n[i] * n[j2] * cubic_spline(&w_ij2, norm_r_ij2);

  double nn_d_w_ij1 = n[i] * n[j1] * d_cubic_spline(&w_ij1, norm_r_ij1);
  double nn_d_w_ij2 = n[i] * n[j2] * d_cubic_spline(&w_ij2, norm_r_ij2);

  //! Direction i
  if (direction == 0) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_quadrupole_ij1_ij2_dq[alpha] =
          (1.0 / 2.0) * (nn_d_w_ij1 * nn_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                             norm_r_ij1_m1 * r_ij1[alpha] +
                         nn_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                             norm_r_ij2_m1 * r_ij2[alpha] +
                         nn_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 *
                             (r_ij1[alpha] + r_ij2[alpha])) -
          (1.0 / 6.0) *
              (nn_d_w_ij1 * nn_w_ij2 * norm_r_ij1 * r2_ij2 * r_ij1[alpha] +
               nn_w_ij1 * nn_d_w_ij2 * r2_ij1 * norm_r_ij2 * r_ij2[alpha] +
               nn_w_ij1 * nn_w_ij2 * 2 * r2_ij2 * r_ij1[alpha] +
               nn_w_ij1 * nn_w_ij2 * 2 * r2_ij1 * r_ij2[alpha]);
    }
  }

  //! Direction j1
  if (direction == 1) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_quadrupole_ij1_ij2_dq[alpha] =
          -0.5 * (nn_d_w_ij1 * nn_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                      norm_r_ij1_m1 * r_ij1[alpha] +
                  nn_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * r_ij2[alpha]) +
          (1.0 / 6.0) *
              (nn_d_w_ij1 * nn_w_ij2 * norm_r_ij1 * r2_ij2 * r_ij1[alpha] +
               nn_w_ij1 * nn_w_ij2 * 2 * r2_ij2 * r_ij1[alpha]);
    }
  }

  //! Direction j2
  if (direction == 2) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      dV_quadrupole_ij1_ij2_dq[alpha] =
          -0.5 * (nn_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                      norm_r_ij2_m1 * r_ij2[alpha] +
                  nn_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * r_ij1[alpha]) +
          (1.0 / 6.0) *
              (nn_w_ij1 * nn_d_w_ij2 * r2_ij1 * norm_r_ij2 * r_ij2[alpha] +
               nn_w_ij1 * nn_w_ij2 * 2 * r2_ij1 * r_ij2[alpha]);
    }
  }
}

/********************************************************************************/

static void dV_quadrupole_ij1j2_dq_FD(int direction,
                                      double* dV_quadrupole_ij1_ij2_dq,
                                      const double* n, const double* q,
                                      const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 3;
  unsigned int d_idx;
  double dr = 0.0001;  // *rc;
  double f_mm, f_m, f_p, f_pp;

  //! Set to zero
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    dV_quadrupole_ij1_ij2_dq[alpha] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  V_quadrupole_ij1j2(&f_0, n, q, spc);

  //! Allocate memory
  double* q_new = (double*)calloc(dim * num_sites, sizeof(double));
  for (unsigned int dof = 0; dof < dim * num_sites; dof++) {
    q_new[dof] = q[dof];
  }

  for (unsigned int alpha = 0; alpha < dim; alpha++) {

    d_idx = direction * dim + alpha;

    // f(x - 2*Dx,y)
    q_new[d_idx] = q[d_idx] - dr * 2;
    f_mm = 0.0;
    V_quadrupole_ij1j2(&f_mm, n, q_new, spc);

    // f(x - Dx,y)
    q_new[d_idx] = q[d_idx] - dr;
    f_m = 0.0;
    V_quadrupole_ij1j2(&f_m, n, q_new, spc);

    // f(x + Dx,y)
    q_new[d_idx] = q[d_idx] + dr;
    f_p = 0.0;
    V_quadrupole_ij1j2(&f_p, n, q_new, spc);

    // f(x + 2*Dx,y)
    q_new[d_idx] = q[d_idx] + dr * 2;
    f_pp = 0.0;
    V_quadrupole_ij1j2(&f_pp, n, q_new, spc);

    dV_quadrupole_ij1_ij2_dq[alpha] =
        ((f_mm - f_0) - 8.0 * (f_m - f_0) + 8.0 * (f_p - f_0) - (f_pp - f_0)) /
        (12.0 * dr);
  }

  //! Free memory
  free(q_new);
}

/********************************************************************************/

static void dV2_quadrupole_ij1j2_dq2(int direction,
                                     double* d2V_quadrupole_ij1j2_dq,
                                     const double* n, const double* q,
                                     const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2V_quadrupole_ij1j2_dq[dof] = 0.0;
  }

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline w_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    w_ij1 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    w_ij1 = adp_HH.w;
  } else {
    w_ij1 = adp_MgH.w;
  }

  CubicSpline w_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    w_ij2 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    w_ij2 = adp_HH.w;
  } else {
    w_ij2 = adp_MgH.w;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double r2_ij1_m1 = 1.0 / r2_ij1;
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij1_m1 = 1.0 / norm_r_ij1;
  double r2_ij2_m1 = 1.0 / r2_ij2;
  double norm_r_ij2 = sqrt(r2_ij2);
  double norm_r_ij2_m1 = 1.0 / norm_r_ij2;
  double dsqr_r_ij1__dot__r_ij2 = DSQR(r_ij1__dot__r_ij2);

  double hessian_r_ij1[NumberDimensions * NumberDimensions] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  compute_hessian_norm_r(hessian_r_ij1, r_ij1, norm_r_ij1);
  double hessian_r_ij2[NumberDimensions * NumberDimensions] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  compute_hessian_norm_r(hessian_r_ij2, r_ij2, norm_r_ij2);

  double Identity[NumberDimensions * NumberDimensions] = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  double nn_w_ij1 = n[i] * n[j1] * cubic_spline(&w_ij1, norm_r_ij1);
  double nn_w_ij2 = n[i] * n[j2] * cubic_spline(&w_ij2, norm_r_ij2);

  double nn_d_w_ij1 = n[i] * n[j1] * d_cubic_spline(&w_ij1, norm_r_ij1);
  double nn_d_w_ij2 = n[i] * n[j2] * d_cubic_spline(&w_ij2, norm_r_ij2);

  double nn_dd_w_ij1 = n[i] * n[j1] * d2_cubic_spline(&w_ij1, norm_r_ij1);
  double nn_dd_w_ij2 = n[i] * n[j2] * d2_cubic_spline(&w_ij2, norm_r_ij2);

  // ii direction
  if (direction == 0) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_quadrupole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
                (nn_dd_w_ij1 * nn_w_ij2 * dsqr_r_ij1__dot__r_ij2 * r2_ij1_m1 *
                     (r_ij1[alpha] * r_ij1[beta]) +
                 nn_d_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     norm_r_ij1_m1 * norm_r_ij2_m1 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij1[alpha] * r_ij1[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     hessian_r_ij1[alpha * dim + beta] +
                 nn_d_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     norm_r_ij2_m1 * norm_r_ij1_m1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_dd_w_ij2 * dsqr_r_ij1__dot__r_ij2 * r2_ij2_m1 *
                     (r_ij2[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij2[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     hessian_r_ij2[alpha * dim + beta] +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij1[alpha] * r_ij1[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij2[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij1[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij2[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 4 * r_ij1__dot__r_ij2 *
                     Identity[alpha * dim + beta]) -
            (1.0 / 6.0) *
                (nn_dd_w_ij1 * nn_w_ij2 * r2_ij2 *
                     (r_ij1[alpha] * r_ij1[beta]) +
                 nn_d_w_ij1 * nn_d_w_ij2 * norm_r_ij1 * norm_r_ij2 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r2_ij2 *
                     ((r_ij1[alpha] * r_ij1[beta]) / norm_r_ij1) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r2_ij1 *
                     ((r_ij1[alpha] * r_ij2[beta]) / norm_r_ij1) +
                 nn_d_w_ij1 * nn_w_ij2 * r2_ij1 * r2_ij2 *
                     hessian_r_ij1[alpha * dim + beta] +
                 nn_d_w_ij1 * nn_d_w_ij2 * norm_r_ij2 * norm_r_ij1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_dd_w_ij2 * r2_ij1 *
                     (r_ij2[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r2_ij2 *
                     ((r_ij2[alpha] * r_ij1[beta]) / norm_r_ij2) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r2_ij1 *
                     ((r_ij2[alpha] * r_ij2[beta]) / norm_r_ij2) +
                 nn_w_ij1 * nn_d_w_ij2 * r2_ij2 * r2_ij1 *
                     hessian_r_ij2[alpha * dim + beta] +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r2_ij2 *
                     ((r_ij1[alpha] * r_ij1[beta]) / norm_r_ij1) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * norm_r_ij1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * norm_r_ij2 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r2_ij1 *
                     ((r_ij2[alpha] * r_ij2[beta]) / norm_r_ij2) +
                 nn_w_ij1 * nn_w_ij2 * 4 * (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * r2_ij2 *
                     Identity[alpha * dim + beta] +
                 nn_w_ij1 * nn_w_ij2 * 4 * (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * r2_ij1 *
                     Identity[alpha * dim + beta]);
      }
    }
  }

  // j1j1 direction
  if (direction == 4) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_quadrupole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
                (nn_dd_w_ij1 * nn_w_ij2 * dsqr_r_ij1__dot__r_ij2 * r2_ij1_m1 *
                     (r_ij1[alpha] * r_ij1[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     hessian_r_ij1[alpha * dim + beta] +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij2[alpha] * r_ij2[beta])) -
            (1.0 / 6.0) * (nn_dd_w_ij1 * nn_w_ij2 * r2_ij2 *
                               (r_ij1[alpha] * r_ij1[beta]) +
                           nn_d_w_ij1 * nn_w_ij2 * 2 * r2_ij2 *
                               ((r_ij1[alpha] * r_ij1[beta]) / norm_r_ij1) +
                           nn_d_w_ij1 * nn_w_ij2 * r2_ij1 * r2_ij2 *
                               hessian_r_ij1[alpha * dim + beta] +
                           nn_d_w_ij1 * nn_w_ij2 * 2 * r2_ij2 *
                               ((r_ij1[alpha] * r_ij1[beta]) / norm_r_ij1) +
                           nn_w_ij1 * nn_w_ij2 * 2 * r2_ij2 *
                               Identity[alpha * dim + beta]);
      }
    }
  }

  // j1j2 direction
  if (direction == 5) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_quadrupole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
                (nn_d_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     norm_r_ij1_m1 * norm_r_ij2_m1 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij1[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij2[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 *
                     Identity[alpha * dim + beta]) -
            (1.0 / 6.0) *
                (nn_d_w_ij1 * nn_d_w_ij2 * norm_r_ij1 * norm_r_ij2 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * norm_r_ij1 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * norm_r_ij2 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 4 * (r_ij1[alpha] * r_ij2[beta]));
      }
    }
  }

  // j2j1 direction
  if (direction == 7) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_quadrupole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
                (nn_d_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     norm_r_ij2_m1 * norm_r_ij1_m1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij2[alpha] * r_ij2[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij1_m1 *
                     (r_ij1[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * r_ij1__dot__r_ij2 *
                     Identity[alpha * dim + beta]) -
            (1.0 / 6.0) *
                (nn_d_w_ij1 * nn_d_w_ij2 * norm_r_ij2 * norm_r_ij1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * norm_r_ij2 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_d_w_ij1 * nn_w_ij2 * 2 * norm_r_ij1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 4 * (r_ij2[alpha] * r_ij1[beta]));
      }
    }
  }

  // j2j2 direction
  if (direction == 8) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {
      for (unsigned int beta = 0; beta < dim; beta++) {
        d2V_quadrupole_ij1j2_dq[alpha * dim + beta] =
            (1.0 / 2.0) *
                (nn_w_ij1 * nn_dd_w_ij2 * dsqr_r_ij1__dot__r_ij2 * r2_ij2_m1 *
                     (r_ij2[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij2[alpha] * r_ij1[beta]) +
                 nn_w_ij1 * nn_d_w_ij2 * dsqr_r_ij1__dot__r_ij2 *
                     hessian_r_ij2[alpha * dim + beta] +
                 nn_w_ij1 * nn_d_w_ij2 * 2 * r_ij1__dot__r_ij2 * norm_r_ij2_m1 *
                     (r_ij1[alpha] * r_ij2[beta]) +
                 nn_w_ij1 * nn_w_ij2 * 2 * (r_ij1[alpha] * r_ij1[beta])) -
            (1.0 / 6.0) * (nn_w_ij1 * nn_dd_w_ij2 * r2_ij1 *
                               (r_ij2[alpha] * r_ij2[beta]) +
                           nn_w_ij1 * nn_d_w_ij2 * 2 * r2_ij1 *
                               ((r_ij2[alpha] * r_ij2[beta]) / norm_r_ij2) +
                           nn_w_ij1 * nn_d_w_ij2 * r2_ij2 * r2_ij1 *
                               hessian_r_ij2[alpha * dim + beta] +
                           nn_w_ij1 * nn_d_w_ij2 * 2 * r2_ij1 *
                               ((r_ij2[alpha] * r_ij2[beta]) / norm_r_ij2) +
                           nn_w_ij1 * nn_w_ij2 * 2 * r2_ij1 *
                               Identity[alpha * dim + beta]);
      }
    }
  }
}

/********************************************************************************/

static void dV2_quadrupole_ij1j2_dq2_FD(int direction,
                                        double* d2V_quadrupole_ij1j2_dq,
                                        const double* n, const double* q,
                                        const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int num_sites = 3;
  double dr = 1.0e-4;  // *rc;

  //! Set to zero
  for (unsigned int dof = 0; dof < 9; dof++) {
    d2V_quadrupole_ij1j2_dq[dof] = 0.0;
  }

  //! Evaluate function in the central value
  double f_0 = 0.0;
  V_quadrupole_ij1j2(&f_0, n, q, spc);

  //! Allocate memory
  double* q_pp = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_p = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_m = (double*)calloc(dim * num_sites, sizeof(double));
  double* q_mm = (double*)calloc(dim * num_sites, sizeof(double));

  // Diagonal terms
  if ((direction == 0) || (direction == 4) || (direction == 8)) {
    for (unsigned int alpha = 0; alpha < dim; alpha++) {

      unsigned int d_dof_i = direction / (1 + num_sites) * dim + alpha;

      //! Compute the positions
      for (unsigned int aux_site_idx = 0; aux_site_idx < num_sites;
           aux_site_idx++) {

        for (unsigned int gamma = 0; gamma < dim; gamma++) {

          unsigned int dof_idx_aux = aux_site_idx * dim + gamma;
          bool D_alpha = (dof_idx_aux == d_dof_i) ? true : false;

          q_pp[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr * 2;

          q_p[dof_idx_aux] = q[dof_idx_aux] + D_alpha * dr;

          q_m[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr;

          q_mm[dof_idx_aux] = q[dof_idx_aux] - D_alpha * dr * 2;
        }
      }

      //! Evaluate functions

      // f(x + 2*Dx,y)
      double f_pp = 0.0;
      V_quadrupole_ij1j2(&f_pp, n, q_pp, spc);

      // f(x + Dx,y)
      double f_p = 0.0;
      V_quadrupole_ij1j2(&f_p, n, q_p, spc);

      // f(x - Dx,y)
      double f_m = 0.0;
      V_quadrupole_ij1j2(&f_m, n, q_m, spc);

      // f(x - 2*Dx,y)
      double f_mm = 0.0;
      V_quadrupole_ij1j2(&f_mm, n, q_mm, spc);

      d2V_quadrupole_ij1j2_dq[alpha * dim + alpha] =
          (-f_pp + 16.0 * f_p - 30.0 * f_0 + 16.0 * f_m - f_mm) /
          (12.0 * dr * dr);
    }
  }

  //! Free memory
  free(q_pp);
  free(q_p);
  free(q_m);
  free(q_mm);
}

/********************************************************************************/

static void dV_quadrupole_ij1j2_dn(int direction,
                                   double* dV_quadrupole_ij1_ij2_dn,
                                   const double* n, const double* q,
                                   const AtomicSpecie* spc) {

  unsigned int dim = NumberDimensions;
  unsigned int i = 0, j1 = 1, j2 = 2;
  double r2_ij1 = 0.0;
  double r2_ij2 = 0.0;
  double r_ij1__dot__r_ij2 = 0.0;
  double r_ij1[3];
  double r_ij2[3];

  //! Set to zero the Partial quadrupole contribution of the sites j and k to i
  *dV_quadrupole_ij1_ij2_dn = 0.0;

  //! Set cubic spline to evaluate the dipole interation curve
  CubicSpline w_ij1;
  if ((spc[i] == Mg) && (spc[j1] == Mg)) {
    w_ij1 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j1] == H)) {
    w_ij1 = adp_HH.w;
  } else {
    w_ij1 = adp_MgH.w;
  }

  CubicSpline w_ij2;
  if ((spc[i] == Mg) && (spc[j2] == Mg)) {
    w_ij2 = adp_MgMg.w;
  } else if ((spc[i] == H) && (spc[j2] == H)) {
    w_ij2 = adp_HH.w;
  } else {
    w_ij2 = adp_MgH.w;
  }

  //! Compute parameters
  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    r_ij1[alpha] = q[dim * i + alpha] - q[dim * j1 + alpha];
    r_ij2[alpha] = q[dim * i + alpha] - q[dim * j2 + alpha];
    r2_ij1 += DSQR(r_ij1[alpha]);
    r2_ij2 += DSQR(r_ij2[alpha]);
    r_ij1__dot__r_ij2 += r_ij1[alpha] * r_ij2[alpha];
  }
  double norm_r_ij1 = sqrt(r2_ij1);
  double norm_r_ij2 = sqrt(r2_ij2);

  //! Direction i
  if (direction == i) {
    double nn_w_ij1 = n[i] * n[j1] * cubic_spline(&w_ij1, norm_r_ij1);
    double nn_w_ij2 = n[i] * n[j2] * cubic_spline(&w_ij2, norm_r_ij2);

    double d_nn_w_ij1 = n[j1] * cubic_spline(&w_ij1, norm_r_ij1);
    double d_nn_w_ij2 = n[j2] * cubic_spline(&w_ij2, norm_r_ij2);

    double DSQR_r_ij1__dot__r_ij2 = DSQR(r_ij1__dot__r_ij2);

    *dV_quadrupole_ij1_ij2_dn =
        (1.0 / 2.0) * d_nn_w_ij1 * nn_w_ij2 * DSQR_r_ij1__dot__r_ij2 +
        (1.0 / 2.0) * nn_w_ij1 * d_nn_w_ij2 * DSQR_r_ij1__dot__r_ij2 -
        (1.0 / 6.0) * d_nn_w_ij1 * nn_w_ij2 * r2_ij1 * r2_ij2 -
        (1.0 / 6.0) * nn_w_ij1 * d_nn_w_ij2 * r2_ij1 * r2_ij2;
  }

  //! Direction j1
  if (direction == j1) {
    double d_nn_w_ij1 = n[i] * cubic_spline(&w_ij1, norm_r_ij1);
    double nn_w_ij2 = n[i] * n[j2] * cubic_spline(&w_ij2, norm_r_ij2);

    *dV_quadrupole_ij1_ij2_dn =
        (1.0 / 2.0) * d_nn_w_ij1 * nn_w_ij2 * DSQR(r_ij1__dot__r_ij2) -
        (1.0 / 6.0) * d_nn_w_ij1 * nn_w_ij2 * r2_ij1 * r2_ij2;
  }

  //! Direction j2
  if (direction == j2) {
    double nn_w_ij1 = n[i] * n[j1] * cubic_spline(&w_ij1, norm_r_ij1);
    double d_nn_w_ij2 = n[i] * cubic_spline(&w_ij2, norm_r_ij2);

    *dV_quadrupole_ij1_ij2_dn =
        (1.0 / 2.0) * nn_w_ij1 * d_nn_w_ij2 * DSQR(r_ij1__dot__r_ij2) -
        (1.0 / 6.0) * nn_w_ij1 * d_nn_w_ij2 * r2_ij1 * r2_ij2;
  }
}

/********************************************************************************/

static void compute_hessian_norm_r(double* hessian_r_ij, const double* r_ij,
                                   const double norm_r_ij) {

  unsigned int dim = NumberDimensions;
  double norm_r_ij_m1 = 1.0 / norm_r_ij;
  double t_ij[NumberDimensions] = {
      norm_r_ij_m1 * r_ij[0], norm_r_ij_m1 * r_ij[1], norm_r_ij_m1 * r_ij[2]};

  for (unsigned int alpha = 0; alpha < dim; alpha++) {
    for (unsigned int beta = 0; beta < dim; beta++) {
      hessian_r_ij[alpha * dim + beta] =
          (alpha == beta) * norm_r_ij_m1 -
          (t_ij[alpha] * t_ij[beta]) * norm_r_ij_m1;
    }
  }
}

/********************************************************************************/