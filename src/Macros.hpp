/**
 * @file Macros.hpp
 * @author Miguel Molinos ([migmolper](https://github.com/migmolper))
 * @brief
 * @version 0.1
 * @date 2022-07-19
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef _MACROS_H_
#define _MACROS_H_

#include "petscis.h"
#include <Eigen/Dense>
#include <iostream>
#include <petsc/private/dmimpl.h>
#include <petscao.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscdmswarm.h>
#include <petscerror.h>
#include <petscsf.h>
#include <petscvec.h>
#include <Kokkos_Core.hpp>

#ifndef PETSC_SUCCESS
#define PETSC_SUCCESS 0
#endif

#ifndef PETSC_ERR_NOT_CONVERGED
#define PETSC_ERR_NOT_CONVERGED 91 // solver did not converge
#endif

#ifndef PETSC_ERR_RETURN
#define PETSC_ERR_RETURN 99
#endif

#define sqrt_2 1.4142135623730951
#define sqrt_3 1.7320508075688772
#define sqrt_6 2.449489742783178

//! Auxiliar Eigen function to map
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> VectorType;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixType;
typedef Eigen::Matrix<int, 1, Eigen::Dynamic> List1D;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    List2D;

typedef Kokkos::DefaultExecutionSpace              DefaultExecSpace;
typedef DefaultExecSpace::memory_space             DefaultMemorySpace;
typedef Kokkos::HostSpace                          HostMemorySpace;
typedef Kokkos::IndexType<PetscInt>                IndexType;
typedef DefaultExecSpace::array_layout             DefaultLayout;

/**
 * @brief Create dictionary for the species
 *
 */
enum AtomicSpecie {
  XX,
  //! @param H: Hydrogen, Z: 1.0079
  H,
  //! @param He: Helium, Z: 4.0026
  He,
  Li,
  Be,
  B,
  C,
  N,
  O,
  F,
  Ne,
  Na,
  //! @param Mg: Magnesium, Z: 24.305
  Mg,
  //! @param Al: Aluminium, Z: 26.981
  Al,
  Si,
  P,
  S,
  Cl,
  Ar,
  K,
  Ca,
  Sc,
  Ti,
  V,
  Cr,
  Mn,
  Fe,
  Co,
  Ni,
  //! @param Cu: Copper, Z: 63,546
  Cu,
  Zn,
  Ga,
  Ge,
  As,
  Se,
  Br,
  Kr,
  Rb,
  Sr,
  Y,
  Zr,
  Nb,
  Mo,
  Tc,
  Ru,
  Rh,
  Pd,
  Ag,
  Cd,
  In,
  Sn,
  Sb,
  Te,
  I,
  Xe,
  Cs,
  Ba,
  La,
  Ce,
  Pr,
  Nd,
  Pm,
  Sm,
  Eu,
  Gd,
  Tb,
  Dy,
  Ho,
  Er,
  Tm,
  Yb,
  Lu,
  Hf,
  Ta,
  //! @param W: Wolfram (Tungsten), Z: 183.84
  W,
  Re,
  Os,
  Ir,
  Pt,
  Au,
  Hg,
  Tl,
  Pb,
  Bi,
  Po,
  At,
  Rn,
  Fr,
  Ra,
  Ac,
  Th,
  Pa,
  U,
  Np,
  Pu,
  Am,
  Cm,
  Bk,
  Cf,
  Es,
  Fm,
  Md,
  No,
  Lr,
  Rf,
  Db,
  Sg,
  Bh,
  Hs,
  Mt,
  Ds,
  Rg
};

/*******************************************************/

enum Miller_Index {
  //! @param idx_0_0_0: direction (0, 0, 0)
  idx_0_0_0,
  //! @param idx_m1_m1_m1: direction (-1, -1, -1)
  idx_m1_m1_m1,
  //! @param idx_m1_m1_0: direction (-1, -1, 0)
  idx_m1_m1_0,
  //! @param idx_m1_m1_0: direction (-1, -1, 1)
  idx_m1_m1_p1,
  //! @param idx_m1_0_m1: direction (-1, 0, -1)
  idx_m1_0_m1,
  //! @param idx_m1_0_0: direction (-1, 0, 0)
  idx_m1_0_0,
  //! @param idx_m1_0_p1: direction (-1, 0, 1)
  idx_m1_0_p1,
  //! @param idx_m1_p1_m1: direction (-1, 1, -1)
  idx_m1_p1_m1,
  //! @param idx_m1_p1_0: direction (-1, 1, 0)
  idx_m1_p1_0,
  //! @param idx_m1_p1_p1: direction (-1, 1, 1)
  idx_m1_p1_p1,
  //! @param idx_0_m1_m1: direction (0, -1, -1)
  idx_0_m1_m1,
  //! @param idx_0_m1_0: direction (0, -1, 0)
  idx_0_m1_0,
  //! @param idx_0_m1_p1: direction (0, -1, 1)
  idx_0_m1_p1,
  //! @param idx_0_0_m1: direction (0, 0, -1)
  idx_0_0_m1,
  //! @param idx_0_0_p1: direction (0, 0, 1)
  idx_0_0_p1,
  //! @param idx_0_p1_m1: direction (0, 1, -1)
  idx_0_p1_m1,
  //! @param idx_0_p1_0: direction (0, 1, 0)
  idx_0_p1_0,
  //! @param idx_0_p1_p1: direction (0, 1, 1)
  idx_0_p1_p1,
  //! @param idx_p1_m1_m1: direction (1, -1, -1)
  idx_p1_m1_m1,
  //! @param idx_p1_m1_0: direction (1, -1, 0)
  idx_p1_m1_0,
  //! @param idx_p1_m1_p1: direction (1, -1, 1)
  idx_p1_m1_p1,
  //! @param idx_p1_0_m1: direction (1, 0, -1)
  idx_p1_0_m1,
  //! @param idx_p1_0_0: direction (1, 0, 0)
  idx_p1_0_0,
  //! @param idx_p1_0_p1: direction (1, 0, 1)
  idx_p1_0_p1,
  //! @param idx_p1_p1_m1: direction (1, 1, -1)
  idx_p1_p1_m1,
  //! @param idx_p1_p1_0: direction (1, 1, 0)
  idx_p1_p1_0,
  //! @param idx_p1_p1_p1: direction (1, 1, 1)
  idx_p1_p1_p1,
};

/*******************************************************/

typedef struct {

  int numneigh;

  const PetscInt *mech_neighs_ptr;

} AtomTopology;


/*******************************************************/
typedef Kokkos::View<double*, Kokkos::LayoutRight, HostMemorySpace> View_Double_Vector_Host;
typedef Kokkos::View<double*, DefaultLayout, DefaultMemorySpace> View_Double_Vector_Device;

/**
 * @brief Nonuniform cubic splines with n intervals
 *
 */
typedef struct CubicSpline {

  //! @param  dx: increment of the independent variable
  double dx;

  //! @param  n: number of segments of the cubic spline
  int n;

  #if USE_KOKKOS == 0

  //! @param x: independent variable
  Eigen::VectorXd x;

  //! @param a: coefficient of grade 0 of the cubic spline function
  Eigen::VectorXd a;

  //! @param b: coefficient of grade 1 of the cubic spline function
  Eigen::VectorXd b;

  //! @param c: coefficient of grade 2 of the cubic spline function
  Eigen::VectorXd c;

  //! @param d: coefficient of grade 3 of the cubic spline function
  Eigen::VectorXd d;

  //! @param db: coefficient of grade 0 of the first derivative of the cubic
  //! spline function
  Eigen::VectorXd db;

  //! @param dc: coefficient of grade 1 of the first derivative of the cubic
  //! spline function
  Eigen::VectorXd dc;

  //! @param dd: coefficient of grade 2 of the first derivative of the cubic
  //! spline function
  Eigen::VectorXd dd;

  //! @param ddc: coefficient of grade 0 of the second derivative of the cubic
  //! spline function
  Eigen::VectorXd ddc;

  //! @param ddd: coefficient of grade 1 of the second derivative of the cubic
  //! spline function
  Eigen::VectorXd ddd;

  #endif

  #if USE_KOKKOS == 1

  //! @param x: independent variable
  View_Double_Vector_Device x;

  //! @param a: coefficient of grade 0 of the cubic spline function
  View_Double_Vector_Device a;

  //! @param b: coefficient of grade 1 of the cubic spline function
  View_Double_Vector_Device b;

  //! @param c: coefficient of grade 2 of the cubic spline function
  View_Double_Vector_Device c;

  //! @param d: coefficient of grade 3 of the cubic spline function
  View_Double_Vector_Device d;

  //! @param db: coefficient of grade 0 of the first derivative of the cubic
  //! spline function
  View_Double_Vector_Device db;

  //! @param dc: coefficient of grade 1 of the first derivative of the cubic
  //! spline function
  View_Double_Vector_Device dc;

  //! @param dd: coefficient of grade 2 of the first derivative of the cubic
  //! spline function
  View_Double_Vector_Device dd;

  //! @param ddc: coefficient of grade 0 of the second derivative of the cubic
  //! spline function
  View_Double_Vector_Device ddc;

  //! @param ddd: coefficient of grade 1 of the second derivative of the cubic
  //! spline function
  View_Double_Vector_Device ddd;

  #endif


} CubicSpline;

/*******************************************************/

/**
 * @brief This structures defines a function and its derivatives
 *
 */
typedef struct {

  /*! @param F: Integrand */
  void (*F)(double *F, const double *xi, const double *q,
            const AtomicSpecie *spc);

  void (*FK)(double *F, const double *xi, const double *q,
            const AtomicSpecie *spc, CubicSpline rho_j);          

  /*! @param dF_dq: Gradient of the function (analytical) */
  void (*dF_dq)(int direction, double *dF_dq, const double *xi, const double *q,
                const AtomicSpecie *spc);

  /*! @param d2F_dq2: Hessian of the function (analytical) */
  void (*d2F_dq2)(int direction, double *d2F_dq2, const double *xi,
                  const double *q, const AtomicSpecie *spc);

  /*! @param dF_dq_FD: Gradient of the function (numerical) */
  void (*dF_dq_FD)(int direction, double *dF_dq, const double *xi,
                   const double *q, const AtomicSpecie *spc);

  /*! @param d2F_dq2_FD: Hessian of the function (numerical) */
  void (*d2F_dq2_FD)(int direction, double *d2F_dq2, const double *xi,
                     const double *q, const AtomicSpecie *spc);

  /*! @param dF_dn: Gradient of the function with respect the occupancy */
  void (*dF_dn)(int direction, double *dF_dn, const double *xi, const double *q,
                const AtomicSpecie *spc);

} potential_function;

/*******************************************************/

typedef struct dump_file {

  /** @param n_atoms Number of atoms */
  int n_atoms;

  /** @param specie: Integer which defines the atomic specie
   */
  AtomicSpecie *specie;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Boundary conditions
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  DMBoundaryType bx;
  DMBoundaryType by;
  DMBoundaryType bz;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    List of diffusive atoms
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  int *diffusive_idx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    System Lagrange multipliers
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /** @param beta: Thermodynamic Lagrange multiplier */
  double *beta;

  /** @param beta_bcc: Index for the sites with beta boundary condition */
  int *beta_bcc;

  /** @param gamma: Chemical Lagrange multiplier */
  double *gamma;

  /** @param gamma_bcc: Index for the sites with gamma boundary condition */
  int *gamma_bcc;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Position-related variables
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /** @param mean_q: Mean value of each atomic position */
  double *mean_q;

  /** @param stdv_q: Standard desviation of each atomic position */
  double *stdv_q;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Chemical variables
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /** @param xi: Molar fraction (mean occupancy) */
  double *xi;

} dump_file;

/*******************************************************/

/**
 * @brief Global variable to define a DMD simulation
 *
 */
typedef struct DMD {

  /*! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   @brief System information
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !*/
  //! @param Variable: with the atomistic data
  DM atomistic_data;

  //! @param n_sites_global: Number of atoms in the global domain
  PetscInt n_sites_global;

  //! @param n_sites_local: Number of atoms in the local domain (without ghost)
  PetscInt n_sites_local;

  //! @param n_ghost: Number of ghost atoms in the local domain
  PetscInt n_ghost;

  /*! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  @brief Enviroment thermo-chemo-mechanical variables
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !*/
  //! @param Pressure_env: Enviromental value of the pressure
  double Pressure_env;

  //! @param Temperature_env: Enviromental value of the temperature
  double Temperature_env;

  //! @param ChemicalPotential_env: Enviromental value of the chemical potential
  //! ({mu}). {mu} = {gamma}/{beta}
  double ChemicalPotential_env;

  //! @param F: Deformation gradient
  Eigen::Matrix3d F;

  /*! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   @brief Topological variables of each site
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !*/
  /** @param neigh: Local indices of I atoms (we only need for the cell) */
  IS *mechanical_neighs_idx;

  /** @param diffusive_neighs_idx: Table with the list of neighbors-idx of
   * each diffusive site */
  IS *diffusive_neighs_idx;

  /*! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @brief Topological variables for the themo-mechanical equations
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !*/
  /** @param n_mechanical_sites_local: Number of mechanical sites (local) */
  PetscInt n_mechanical_sites_local;

  /** @param n_mechanical_sites_ghost: Number of mechanical ghost sites */
  PetscInt n_mechanical_sites_ghost;

  /** @param active_mech_sites: List with the active mechanical sites */
  IS active_mech_sites;

  /*! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @brief Topological variables for the chemical equation
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !*/
  /** @param n_diffusive_sites: Number of diffusive sites (global) */
  PetscInt n_diffusive_sites;

  /** @param n_diffusive_sites_local: Number of diffusive sites (local) */
  PetscInt n_diffusive_sites_local;

  /** @param n_diffusive_sites_ghost: Number of diffusive ghost sites */
  PetscInt n_diffusive_sites_ghost;

  /** @param active_diff_sites: List with the active diffusive sites using local
   * numbering */
  IS active_diff_sites;

  /*! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   @brief Topological variables for the domain decompisition
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !*/
  //! @param background_mesh: with the background mesh
  DM background_mesh;

  //! @param bounding_cell: with the bounding cell
  DM bounding_cell;

  /*! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @brief Miscelaneous variables
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !*/
  /** @param dump2petsc_mapping: Allow the user to track the index of each atom
   * from the .dump prdering and petsc ordering */
  AO dump2petsc_mapping;

} DMD;

/**
 * @brief Structure which contains the necessary information to evaluate the
 * interatomic potential
 *
 */
typedef struct adpPotential {

  /**
   * @brief
   *
   */
  int n_embed;

  int n_rho;
  int n_pair;
  int n_u;
  int n_w;

  /**
   * @brief Evaluation of the embedded energy
   *
   */
  CubicSpline embed;

  /**
   * @brief Evaluation of the density function
   *
   */
  CubicSpline rho;

  /**
   * @brief
   *
   */
  CubicSpline pair;
  CubicSpline u;
  CubicSpline w;

  /**
   * @brief Mass of the specie
   *
   */
  double mass;
  double radius;
  double factor;

  /**
   * @brief Cut-off radious for the specie
   *
   */
  double r_cutoff;

} adpPotential;

/**
 * @brief Structure which contains user define equations to evaluate any sort of
 * equilibrium equation
 *
 */
typedef struct dmd_equations {

  /**
   * @brief
   *
   * @param site_i
   * @param mean_q:Mean value of q
   * @param xi Molar fraction
   * @param specie Atomic specie
   * @param atom_topology_i List of neighs
   * @return double
   */
  double (*evaluate_rho_i)(unsigned int site_i,           //!
                           const Eigen::MatrixXd &mean_q, //!
                           const Eigen::VectorXd &xi,     //!
                           const AtomicSpecie *specie,    //!
                           const AtomTopology atom_topology_i);
  /**
   * @brief Compute \f$\frac{\partial V}{\partial \mathbf{q}_{i^*}}\f$:
   * \f[
   *  \frac{\partial V}{\partial
   * \mathbf{q}_{i^*}}\ =\ \frac{\partial}{\partial \mathbf{q}_{i^*}} \sum_i
   * V_i \f] Derivative of the total potential with respect the site
   * position
   * @param mean_q Position (\f$\mathbf{q}\f$) of the atomic
   * positions in the updated configuration
   * @param adp Information of the angular-dependant potential
   * @param atoms Atomistic information of the bulk box
   * @return Eigen::Vector3d
   */
  Eigen::Vector3d (*evaluate_DV_i_Dq_j)(unsigned int site_i_star,            //!
                                        unsigned int site_i,                 //!
                                        const Eigen::MatrixXd &mean_q,       //!
                                        const Eigen::VectorXd &xi,           //!
                                        const Eigen::VectorXd &rho,          //!
                                        const AtomicSpecie *specie,          //!
                                        const AtomTopology atom_topology_i); //!

  /**
   * @brief
   *
   */
  Eigen::Matrix3d (*evaluate_D2V_i_Dq2_j)(
      unsigned int site_i_star,            //!
      unsigned int site_i,                 //!
      const Eigen::MatrixXd &mean_q,       //!
      const Eigen::VectorXd &xi,           //!
      const Eigen::VectorXd &rho,          //!
      const AtomicSpecie *specie,          //!
      const AtomTopology atom_topology_i); //!

  /**
   * @brief Compute the derivative of the potential functional with respect
   * the right stretch tensor:
   \f[
   \frac{\partial V}{\partial \mathbf{U}}
   \f]
   * @param q Position (\f$\mathbf{q}\f$) of the atomic
   * positions in the updated configuration
   * @param q0 Position (\f$\mathbf{q}_0\f$) of the atomic
   * positions in the reference configuration
   * @param adp Information of the angular-dependant potential
   * @param atoms Atomistic information of the bulk box
   * @return Eigen::Matrix3d
   */
  Eigen::Matrix3d (*evaluate_DV_i_DF)(unsigned int site_i,                 //!
                                      const Eigen::MatrixXd &mean_q,       //!
                                      const Eigen::MatrixXd &mean_q0,      //!
                                      const Eigen::VectorXd &xi,           //!
                                      const Eigen::VectorXd &rho,          //!
                                      const AtomicSpecie *specie,          //!
                                      const AtomTopology atom_topology_i); //!

  /**
   * @brief
   *
   * @param site_i
   * @param mean_q
   * @param stdv_q
   * @param xi
   * @param specie
   * @param atom_topology_i
   */
  double (*evaluate_mf_rho_i)(unsigned int site_i,           //!
                              const Eigen::MatrixXd &mean_q, //!
                              const Eigen::VectorXd &stdv_q, //!
                              const Eigen::VectorXd &xi,     //!
                              const AtomicSpecie *specie,    //!
                              const AtomTopology atom_topology_i);

  /**
   * @brief Total thermalized potential of the system
   * @param site_i Site to evaluate the potential
   * @param mean_q Mean position (\f$\bar{\mathbf{q}}\f$) of the atomic
   * positions in the updated configuration
   * @param stdv_q Standard desviation (\f$\sigma\f$) of the position
   * @param adp Information of the angular-dependant potential
   * @param atoms Atomistic information of the bulk box
   * @return V0
   */
  double (*evaluate_V0_i)(unsigned int site_i,
                          const Eigen::MatrixXd &mean_q,       //!
                          const Eigen::VectorXd &stdv_q,       //!
                          const Eigen::VectorXd &xi,           //!
                          const Eigen::VectorXd &mf_rho,       //!
                          const AtomicSpecie *specie,          //!
                          const AtomTopology atom_topology_i); //!

  /**
   * @brief Compute meanfield entropy at site i
   *
   * @param site_i: Site to evaluate the potential
   * @param mean_q: Mean value of q
   * @param stdv_q: Standard desviation of q
   * @param xi: Molar fraction
   * @param mf_rho: Meanfield energy density
   * @param beta: Lagrange multiplier (thermal)
   * @param gamma: Lagrange multiplier (chemical)
   * @param specie: Atomic specie
   * @param atom_topology_i: Atom topology
   * @return double
   */
  double (*evaluate_S0_i)(unsigned int site_i,           //!
                          const Eigen::MatrixXd &mean_q, //!
                          const Eigen::VectorXd &stdv_q, //!
                          const Eigen::VectorXd &xi,     //!
                          const Eigen::VectorXd &mf_rho, //!
                          const Eigen::VectorXd &beta,   //!
                          const Eigen::VectorXd &gamma,  //!
                          const AtomicSpecie *specie,    //!
                          const AtomTopology atom_topology_i);

  /**
   * @brief Compute the derivatives of the thermalised functional with respect
   the mean position of each atomis site i:
   *
   \f[
    \frac{\partial V_0}{\partial \bar{\mathbf{q}}_i}
   \f]
   *
   */
  Eigen::Vector3d (*evaluate_DV0_i_Dmeanq_j)(
      unsigned int site_i_star,            //!
      unsigned int site_i,                 //!
      const Eigen::MatrixXd &mean_q,       //! Mean value of q
      const Eigen::VectorXd &stdv_q,       //! Standard desviation of q
      const Eigen::VectorXd &xi,           //! Molar fraction
      const Eigen::VectorXd &mf_rho,       //!
      const AtomicSpecie *specie,          //! Atom
      const AtomTopology atom_topology_i); //!

  /**
   * @brief CCompute the hessian of the potential with respect the mean position
   of
   * the site positions.
   \f[
    \frac{\partial^2 V_0}{\partial \bar{\mathbf{q}}_i^2}
   \f]
   * @param mean_q Mean position (\f$\bar{\mathbf{q}}\f$) of the atomic
   positions in the updated
   * configuration
   * @param stdv_q Standard desviation (\f$\sigma\f$) of the position
   * @param adp Information of the angular-dependant potential
   * @param atoms Atomistic information of the bulk box
   * @return Eigen::MatrixXd
   */
  Eigen::Matrix3d (*evaluate_D2V0_i_Dmeanq2_j)(
      unsigned int site_i_star,            //!
      unsigned int site_i,                 //!
      const Eigen::MatrixXd &mean_q,       //!
      const Eigen::VectorXd &stdv_q,       //!
      const Eigen::VectorXd &xi,           //!
      const Eigen::VectorXd &mf_rho,       //!
      const AtomicSpecie *specie,          //!
      const AtomTopology atom_topology_i); //!

  /**
   * @brief Compute the derivatives of the free-entropy functional with respect
   the standard desviation of each atomis site i:
   *
   \f[
    \frac{\partial V_0}{\partial \sigma_i}
   \f]
   *
   */
  double (*evaluate_DV0_i_Dstdvq_j)(
      unsigned int site_i_star,            //!
      unsigned int site_i,                 //!
      const Eigen::MatrixXd &mean_q,       //! Mean value of q
      const Eigen::VectorXd &stdv_q,       //! Standard desviation of q
      const Eigen::VectorXd &xi,           //! Molar fraction
      const Eigen::VectorXd &mf_rho,       //!
      const AtomicSpecie *specie,          //! Atom
      const AtomTopology atom_topology_i); //!

  /**
   * @brief Compute the derivatives of the free-entropy functional with respect
   the meand and standard desviation of the position at the atomic site i:
   *
   \f[
    \frac{\partial V_0}{\partial \bar{\mathbf{q}}_i} \\
    \frac{\partial V_0}{\partial \sigma_i}
   \f]
   *
   */
  Eigen::Vector4d (*evaluate_DV0_i_Dq_j)(
      unsigned int site_i_star,            //!
      unsigned int site_i,                 //!
      const Eigen::MatrixXd &mean_q,       //! Mean value of q
      const Eigen::VectorXd &stdv_q,       //! Standard desviation of q
      const Eigen::VectorXd &xi,           //! Molar fraction
      const Eigen::VectorXd &mf_rho,       //!
      const AtomicSpecie *specie,          //! Atom
      const AtomTopology atom_topology_i); //!

  /**
   * @brief Compute the derivatives of the mean-fielf potential with respect
   the molar fraction of each atomis site i:
   *
   \f[
    \frac{\partial V_0}{\partial \xi_i}
   \f]
   * @param mean_q Mean position (\f$\bar{\mathbf{q}}\f$) of the atomic
   positions in the updated
   * configuration
   * @param stdv_q Standard desviation (\f$\sigma\f$) of the position
   * @param adp Information of the angular-dependant potential
   * @param atoms Atomistic information of the bulk box
   * @return double
   */
  double (*evaluate_DV0_i_Dxi_j)(
      unsigned int site_i_star,            //!
      unsigned int site_i,                 //!
      const Eigen::MatrixXd &mean_q,       //! Mean value of q
      const Eigen::VectorXd &stdv_q,       //! Standard desviation of q
      const Eigen::VectorXd &xi,           //! Molar fraction
      const Eigen::VectorXd &mf_rho,       //!
      const AtomicSpecie *specie,          //! Atom
      const AtomTopology atom_topology_i); //!

  /**
   * @brief Compute the desrivative of the free-entropy functional with respect
   * the deformation gradient tensor:
   \f[
   \frac{\partial \mathcal{L}_0}{\partial \mathbf{F}}
   \f]
   *
   */
  Eigen::Matrix3d (*evaluate_DV0_i_DF)(
      unsigned int site_i,                 //! Site to evalute the potential
      const Eigen::MatrixXd &mean_q,       //! Mean value of q
      const Eigen::MatrixXd &mean_q0,      //! Mean value of q
      const Eigen::VectorXd &stdv_q,       //! Standard desviation of q
      const Eigen::VectorXd &xi,           //! Molar fraction
      const Eigen::VectorXd &mf_rho,       //!
      const AtomicSpecie *specie,          //! Atom
      const AtomTopology atom_topology_i); //!

} dmd_equations;

/*
  Color text
*/
#define RESET "\033[0m"
#define BLACK "\033[30m"              /* Black */
#define RED "\033[31m"                /* Red */
#define GREEN "\033[32m"              /* Green */
#define YELLOW "\033[33m"             /* Yellow */
#define BLUE "\033[34m"               /* Blue */
#define MAGENTA "\033[35m"            /* Magenta */
#define CYAN "\033[36m"               /* Cyan */
#define WHITE "\033[37m"              /* White */
#define BOLDBLACK "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED "\033[1m\033[31m"     /* Bold Red */
#define BOLDGREEN "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW "\033[1m\033[33m"  /* Bold Yellow */
#define BOLDBLUE "\033[1m\033[34m"    /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m" /* Bold Magenta */
#define BOLDCYAN "\033[1m\033[36m"    /* Bold Cyan */
#define BOLDWHITE "\033[1m\033[37m"   /* Bold White */

/*
  Constant macros
*/
#define MAXW 100
#define MAXC 1000
#define NumberDimensions 3
#define TOL_NR 10E-6
#define TOL_zero 10E-23
#define PI 3.14159265358979323846

/**
 * @brief Define HPC constants
 *
 */
//! The length of the buffer used to efficient dynamic re-sizing
#define BufferLenght 4

/* OpenMP macros */
#define NUM_OPENMP_THREADS 8

/**
 * @brief Define physical constants
 *
 */

// Reduced Planck constant -> 6.5821192815×10−4 [eV·ps]
#define h_planck 6.5821192815E-4
// Boltzmann constant [ev/K]
#define k_B 8.617332478E-5
// 1 u.m.a = 103.4993333E-6 [ev·A^{-2}·ps^{2}]
#define unit_change_uma 103.4993333E-6
// the factor 98.22694969 is to change the
// units: 98.22694969 1/ps = 1 ((eV/A^2)/u.m.a.)^(1/2)
#define unit_change_w 98.227002603

//! Angular dependant potential cutoff
#define r_cutoff_ADP_MgHx 6.2934034034
#define r_cutoff_ADP_AlCu 6.2872100000

#define r_cutoff_Eb 2.6

#define maxneigh 250
#define max_chemical_neighs 50
#define maxneigh10 1000

//! This values has been computed in a MgHx hcp cell and they
//! are the values which warraty the stability of the system. If the occupancy
//! is below this value do not compute the equilibrium Zero value for the
//! occupancy
#define max_occupancy 0.99
#define min_occupancy 5e-4
#define min_stdv_q 0.001
#define max_stdv_q 3.0

/*
  Math macros
*/
static float sqr_arg;
#define SQR(a) ((sqr_arg = (a)) == 0.0 ? 0.0 : sqr_arg * sqr_arg)
static double dsqr_arg;
#define DSQR(a) ((dsqr_arg = (a)) == 0.0 ? 0.0 : dsqr_arg * dsqr_arg)
static double dmax_arg1, dmax_arg2;
#define DMAX(a, b)                                                             \
  (dmax_arg1 = (a), dmax_arg2 = (b),                                           \
   (dmax_arg1) > (dmax_arg2) ? (dmax_arg1) : (dmax_arg2))
static double dmin_arg1, dmin_arg2;
#define DMIN(a, b)                                                             \
  (dmin_arg1 = (a), dmin_arg2 = (b),                                           \
   (dmin_arg1) < (dmin_arg2) ? (dmin_arg1) : (dmin_arg2))
static float max_arg1, max_arg2;
#define FMAX(a, b)                                                             \
  (max_arg1 = (a), max_arg2 = (b),                                             \
   (max_arg1) > (max_arg2) ? (max_arg1) : (max_arg2))
static float min_arg1, min_arg2;
#define FMIN(a, b)                                                             \
  (min_arg1 = (a), min_arg2 = (b),                                             \
   (min_arg1) < (min_arg2) ? (min_arg1) : (min_arg2))
static long lmax_arg1, lmax_arg2;
#define LMAX(a, b)                                                             \
  (lmax_arg1 = (a), lmax_arg2 = (b),                                           \
   (lmax_arg1) > (lmax_arg2) ? (lmax_arg1) : (lmax_arg2))
static long lmin_arg1, lmin_arg2;
#define LMIN(a, b)                                                             \
  (lmin_arg1 = (a), lmin_arg2 = (b),                                           \
   (lmin_arg1) < (lmin_arg2) ? (lmin_arg1) : (lmin_arg2))
static int imax_arg1, imax_arg2;
#define IMAX(a, b)                                                             \
  (imax_arg1 = (a), imax_arg2 = (b),                                           \
   (imax_arg1) > (imax_arg2) ? (imax_arg1) : (imax_arg2))
static int imin_arg1, imin_arg2;
#define IMIN(a, b)                                                             \
  (imin_arg1 = (a), imin_arg2 = (b),                                           \
   (imin_arg1) < (imin_arg2) ? (imin_arg1) : (imin_arg2))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a)) s

#endif


/*
 Kokkos 
*/

typedef Kokkos::View<double**, DefaultLayout, DefaultMemorySpace> PetscScalar_Matrix_Default;
typedef Kokkos::View<PetscScalar*,  DefaultLayout, DefaultMemorySpace> PetscScalar_Vector_Default;

typedef Kokkos::View<double**, Kokkos::LayoutRight, HostMemorySpace, Kokkos::MemoryUnmanaged> PetscScalar_Matrix_Host;
typedef Kokkos::View<PetscScalar*,  Kokkos::LayoutRight, HostMemorySpace, Kokkos::MemoryUnmanaged> PetscScalar_Vector_Host;

typedef Kokkos::View<PetscInt*, Kokkos::LayoutRight, HostMemorySpace, Kokkos::MemoryUnmanaged> PetscInt_Vector_Host;
typedef Kokkos::View<PetscInt*, DefaultLayout, DefaultMemorySpace> PetscInt_Vector_Default;

typedef Kokkos::View<AtomTopology*, Kokkos::LayoutRight, HostMemorySpace, Kokkos::MemoryUnmanaged> AtomTopology_Host;
typedef Kokkos::View<AtomTopology*, DefaultLayout, DefaultMemorySpace> AtomTopology_Default;

typedef Kokkos::View<AtomicSpecie*, Kokkos::LayoutRight, HostMemorySpace, Kokkos::MemoryUnmanaged> AtomSpecie_Host;
typedef Kokkos::View<AtomicSpecie*, DefaultLayout, DefaultMemorySpace> AtomSpecie_Default;

typedef Kokkos::View<adpPotential*, Kokkos::LayoutRight, HostMemorySpace> AdpPotencial_Host;
typedef Kokkos::View<adpPotential*, DefaultLayout, DefaultMemorySpace> AdpPotencial_Device;



