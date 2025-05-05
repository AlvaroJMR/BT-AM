#include <iostream>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "ADP/density.hpp"
#include "Atoms/Atom.hpp"
#include "Atoms/Ghosts.hpp"
#include "Atoms/Neighbors.hpp"
#include "Atoms/Topology.hpp"
#include "IO/dump-input.hpp"
#include "Macros.hpp"
#include "Numerical/cubic-spline.hpp"
#include "Periodic-Boundary/boundary_conditions.hpp"
#include "Variables.hpp"
#include <petscksp.h>
#include <Kokkos_Core.hpp>
#include <Utils/KokkosUtils.hpp>
#ifdef USE_SLEPC
#include <slepcmfn.h>
#endif

extern PetscMPIInt size_MPI;
extern PetscMPIInt rank_MPI;

extern PetscInt ndiv_mesh_X;
extern PetscInt ndiv_mesh_Y;
extern PetscInt ndiv_mesh_Z;

extern adpPotential adp_MgMg;
extern adpPotential adp_HH;
extern adpPotential adp_MgH;

extern double element_mass[112];

extern char OutputFolder[MAXC];
static char help[] = "Bachelor's thesis: Álvaro Montaño Rosa \n";

int main(int argc, char **argv) {

  snprintf(OutputFolder, sizeof(OutputFolder), "%s", "./");

  try {

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_MPI);
    MPI_Comm_size(MPI_COMM_WORLD, &size_MPI);
#endif


    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
    // Initialize PETSc
    PetscFunctionBeginUser;
    PetscInitialize(&argc, &argv, 0, help);

    ndiv_mesh_X = 3;
    ndiv_mesh_Y = 3;
    ndiv_mesh_Z = 3;

    const char Inputs[10000] = "inputs";
    const char SimulationFile[10000] =
        "inputs/Mg-hcp-cube-x20-x15-x15-periodic.dump";

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Command line options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscOptionsSetValue(NULL, "-minV_dF_snes_atol", "1.e-12");
    PetscOptionsSetValue(NULL, "-minV_dF_snes_type", "ngmres");
    PetscOptionsSetValue(NULL, "-minV_dF_snes_ngmres_m", "3");
    PetscOptionsSetValue(NULL, "-minV_dF_snes_linesearch_type", "cp");

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Read information from dump file
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    dump_file Simulation_dump_data = read_dump_information(SimulationFile);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Initialize atomistic simulation
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    DMD Simulation;
    PetscCall(init_DMD_simulation(&Simulation, Simulation_dump_data));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free dump data
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    free_dump_information(&Simulation_dump_data);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Output data
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(DMView(Simulation.atomistic_data, PETSC_VIEWER_STDOUT_WORLD));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Create ghost atoms
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(DMSwarmSetMigrateType(Simulation.atomistic_data,
                                    DMSWARM_MIGRATE_BASIC));
    PetscCall(DMSwarmCreateGhostAtoms(&Simulation, r_cutoff_ADP_MgHx));
    PetscCall(DMView(Simulation.atomistic_data, PETSC_VIEWER_STDOUT_WORLD));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Compute neighs
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(list_of_active_mechanical_sites_MgHx(&Simulation));

    PetscCall(neighbors(&Simulation));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Initialize MgHx potential and equations
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    init_adp_MgHx(&adp_MgMg, MgMg, Inputs);
    init_adp_MgHx(&adp_HH, HH, Inputs);
    init_adp_MgHx(&adp_MgH, MgH, Inputs);

    //    dmd_equations system_equations = DMD_MgHx_constructor();

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Output data
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(
        DMSwarmViewXDMF(Simulation.atomistic_data,
                        "outputs/Mg-hcp-cube-x20-x15-x15-periodic-0.xmf"));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Relax the system solving the equation DPsi_Du = 0 to get the lattice
      parameter
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    //    PetscCall(mechanical_relaxation_bulk(&Simulation, system_equations));

    //    Vec xi;
    //    Kokkos::View<PetscScalar *, Kokkos::Cuda> k_xi;

    //    PetscCall(DMSwarmCreateLocalVectorFromField(Simulation.atomistic_data,
    //                                                "molar-fraction", &xi));
    //    PetscCall(VecGetKokkosView(xi, &k_xi));

    // double
    // evaluate_rho_i_adp_MgHx_kokkos(unsigned int site_i,           //!
    //                                const Eigen::MatrixXd &mean_q, //! Mean q
    //                                const Eigen::VectorXd &xi,     //! Molar
    //                                fraction const AtomicSpecie *specie, //!
    //                                Atom const AtomTopology atom_topology_i)

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Compute energy density
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    //! Get local number of sites in the simulation (without ghost)
    PetscInt n_sites_local = Simulation.n_sites_local;

    //! Get local number of sites in the simulation (with ghost)
    PetscInt n_sites_local_ghosted;
    PetscCall(
        DMSwarmGetLocalSize(Simulation.atomistic_data, &n_sites_local_ghosted));

    //! Get number of ghost particles
    PetscInt n_sites_ghost = n_sites_local_ghosted - n_sites_local;

    //!
    PetscInt n_mechanical_sites_local = Simulation.n_mechanical_sites_local;

    //!
    AtomTopology *atom_topology =
        (AtomTopology *)malloc(n_sites_local_ghosted * sizeof(AtomTopology));

    for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++) {

      PetscCall(read_atom_topology(&atom_topology[site_u],
                                   Simulation.mechanical_neighs_idx[site_u]));
    }

    AtomTopologyKokkos* atomTopologyKokkos = convert_atomTopology_array_to_Kokkos(atom_topology,n_sites_local_ghosted);

    //!
    IS active_mech_sites = Simulation.active_mech_sites;
    PetscInt *active_mech_sites_ptr;
    PetscCall(ISGetIndices(active_mech_sites,
                           (const PetscInt **)&active_mech_sites_ptr));

    PetscScalar *mean_q_ptr;
    PetscCall(DMSwarmGetField(Simulation.atomistic_data, DMSwarmPICField_coor,
                              NULL, NULL, (void **)&mean_q_ptr));
    Eigen::Map<MatrixType> mean_q(mean_q_ptr, n_sites_local_ghosted, 3);

    PetscScalar *mf_rho_ptr;
    PetscCall(DMSwarmGetField(Simulation.atomistic_data, "mf-rho", NULL, NULL,
                              (void **)&mf_rho_ptr));
    Eigen::Map<VectorType> mf_rho(mf_rho_ptr, n_sites_local_ghosted);

    PetscScalar *xi_ptr;
    PetscCall(DMSwarmGetField(Simulation.atomistic_data, "molar-fraction", NULL,
                              NULL, (void **)&xi_ptr));
    Eigen::Map<VectorType> xi(xi_ptr, n_sites_local_ghosted);

    AtomicSpecie *specie_ptr;
    PetscCall(DMSwarmGetField(Simulation.atomistic_data, "specie", NULL, NULL,
                              (void **)&specie_ptr));

    PetscInt *idx_q_ptr;
    PetscCall(DMSwarmGetField(Simulation.atomistic_data, "idx", NULL, NULL,
                              (void **)&idx_q_ptr));
          
                          
     Kokkos::Timer timer2;
    #pragma omp parallel for schedule(runtime)
    for (PetscInt mech_site_u = 0; mech_site_u < n_mechanical_sites_local;
         mech_site_u++) {

      //! Get index of the site u
      PetscInt site_u = active_mech_sites_ptr[mech_site_u];

      //! @brief Evaluate energy density at site u
      mf_rho(site_u) = evaluate_rho_i_adp_MgHx_kokkos(
          site_u, mean_q, xi, specie_ptr, atom_topology[site_u]);
    }
    double time2 = timer2.seconds();

    PetscScalar_Vector_Host mf_rho_Host(mf_rho_ptr, static_cast<size_t>(n_sites_local_ghosted));
    PetscScalar_Vector_Default mf_rho_Default("mf_rho_Default",static_cast<size_t>(n_sites_local_ghosted));
    Kokkos::deep_copy(mf_rho_Default,mf_rho_Host);

    PetscScalar_Matrix_Host mean_q_Kokkos_Host(mean_q_ptr,n_sites_local_ghosted,3);
    PetscScalar_Matrix_Default mean_q_Kokkos_Default("mean_q_Kokkos_Default",static_cast<size_t>(n_sites_local_ghosted),3);
    Kokkos::deep_copy(mean_q_Kokkos_Default,mean_q_Kokkos_Host);

    PetscScalar_Vector_Host xi_Kokkos_Host(xi_ptr, static_cast<size_t>(n_sites_local_ghosted));
    PetscScalar_Vector_Default xi_Kokkos_Default("xi_Kokkos_Default",static_cast<size_t>(n_sites_local_ghosted));
    Kokkos::deep_copy(xi_Kokkos_Default,xi_Kokkos_Host);
    
    PetscInt active_mech_sites_index;
    PetscCall(ISGetLocalSize(active_mech_sites, &active_mech_sites_index));

    PetscInt_Vector_Host active_mech_sites_Kokkos_Host(active_mech_sites_ptr, static_cast<size_t>(active_mech_sites_index));
    PetscInt_Vector_Default active_mech_sites_Kokkos_Default("Device_mechanical_sites", static_cast<size_t>(active_mech_sites_index));
    Kokkos::deep_copy(active_mech_sites_Kokkos_Default, active_mech_sites_Kokkos_Host);

    AtomTopology_Host atomTopology_Kokkos_Host(atom_topology, n_sites_local_ghosted);
    AtomTopology_Default atomTopology_Kokkos_Default("atomTopology_Kokkos_Default",n_sites_local_ghosted);
    Kokkos::deep_copy(atomTopology_Kokkos_Default, atomTopology_Kokkos_Host);

    AtomTopologyKokkos_Host atomTopologyKokkos_Kokkos_Host(atomTopologyKokkos, n_sites_local_ghosted);
    AtomTopologyKokkos_Default atomTopologyKokkos_Kokkos_Default("atomTopology_Kokkos_Default",n_sites_local_ghosted);
    Kokkos::deep_copy(atomTopologyKokkos_Kokkos_Default, atomTopologyKokkos_Kokkos_Host);

    PetscInt atomSpecie_Index;
    PetscCall(DMSwarmGetLocalSize(Simulation.atomistic_data, &atomSpecie_Index));

    AtomSpecie_Host atomSpecie_Kokkos_Host(specie_ptr, atomSpecie_Index);
    AtomSpecie_Default atomSpecie_Kokkos_Default("atomSpecie_Kokkos_Default", atomSpecie_Index);
    Kokkos::deep_copy(atomSpecie_Kokkos_Default, atomSpecie_Kokkos_Host);

    //Kokkos::printf("atomSpecie_Index = %d\n", atomSpecie_Index);
    //Kokkos::printf("specie_ptr[0] = %d\n", static_cast<int>(specie_ptr[0]));
    //Kokkos::printf("atomTopology_Kokkos_Default[0] = %f\n", atomSpecie_Kokkos_Host(0));


    AdpPotencial_Device adp_Device_Default;//("adp_Device", 2);
    // AdpPotencial_Host adp_Host("adp_Host", 2);

    // adp_Host(0) = adp_MgMg;
    // adp_Host(1) = adp_HH;

    // Kokkos::deep_copy(adp_Device_Default, adp_Host);

    copy_adpPotential_to_device(adp_Device_Default, adp_MgMg, 0 );

    copy_adpPotential_to_device(adp_Device_Default, adp_HH, 1 );

    copy_adpPotential_to_device(adp_Device_Default, adp_MgH, 2 );



    /* adp_HH_Kokkos_Host = AdpPotencial_Host("adp_HH_Kokkos_Host", 1);
    adp_HH_Kokkos_Host(0) = adp_HH;
    adp_HH_Kokkos_Default = AdpPotencial_Device("adp_HH_Kokkos_Default", 1);
    Kokkos::deep_copy(adp_HH_Kokkos_Default, adp_HH_Kokkos_Host);
    
        adp_MgMg_Kokkos_Host = AdpPotencial_Host("adp_MgMg_Kokkos_Host", 1);
    adp_MgMg_Kokkos_Host(0) = adp_MgMg;
    adp_MgMg_Kokkos_Default = AdpPotencial_Device("adp_MgMg_Kokkos_Default", 1);
    Kokkos::deep_copy(adp_MgMg_Kokkos_Default, adp_MgMg_Kokkos_Host);
    
    */

    View_Double_Matrix_Device mean_q_ij1_all("mean_q_ij1_all", n_mechanical_sites_local, 6);

    Kokkos::Timer timer;
    Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_mechanical_sites_local), KOKKOS_LAMBDA(PetscInt mech_site_u) {
        PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);
    
        auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all, mech_site_u, Kokkos::ALL());
    
        mf_rho_Default(site_u) = evaluate_rho_i_adp_MgHx_kokkos_Device(
            site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, atomSpecie_Kokkos_Default, atomTopologyKokkos_Kokkos_Default(site_u), mean_q_ij1, adp_Device_Default);
    });

  Kokkos::fence();
  double time = timer.seconds();
  std::cout << "Tiempo que ha tardado en las operaciones con Kokkos: " << time << " seconds" << std::endl;
  std::cout << "Tiempo que ha tardado en las operaciones sin Kokkos: " << time2 << " seconds" << std::endl;

double eigen_mean = mf_rho.mean();

double eigen_variance = ((mf_rho.array() - eigen_mean).square().sum()) / static_cast<double>(n_sites_local_ghosted-1);

std::cout << "Eigen: Media = " << eigen_mean
          << ", Varianza = " << eigen_variance << std::endl;

auto mf_rho_Mirrow = Kokkos::create_mirror_view(mf_rho_Default);
Kokkos::deep_copy(mf_rho_Mirrow, mf_rho_Default);

Eigen::Map<VectorType> mf_rho_test(mf_rho_Mirrow.data(), n_sites_local_ghosted);

double Kokkos_mean_test = mf_rho_test.mean();
double Kokkos_variance_test = ((mf_rho_test.array() - Kokkos_mean_test).square().sum())
                              / static_cast<double>(n_sites_local_ghosted - 1);

std::cout << "Kokkos: Media = " << Kokkos_mean_test
          << ", Varianza = " << Kokkos_variance_test << std::endl;
     

  //! Migrate ghost field (energy density) sin Kokkos

  PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
    &idx_q_ptr[n_sites_local], mf_rho_ptr));
  
  // PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "idx", NULL, NULL, (void**)&idx_q_ptr));
  

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Evaluate free entropy
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
double V_local = 0.0;

#pragma omp parallel for reduction(+ : V_local) schedule(runtime)
for (PetscInt site_u = 0; site_u < n_sites_local; site_u++) {

//! @brief Evaluate the potential energy site u
double V_u = evaluate_V_i_adp_MgHx(
site_u, mean_q, xi, mf_rho, specie_ptr, atom_topology[site_u]);

//! @brief Update local contribution of the residual equation
V_local += V_u;
}

std::cout << "Acabe potencial sin Kokkos: " << V_local << std::endl;

// Copy back to host to migrateGhostfield
auto mf_rho_Host_copy = Kokkos::create_mirror_view(mf_rho_Default);
Kokkos::deep_copy(mf_rho_Host_copy, mf_rho_Default);
PetscScalar* mf_rho_ptr_Kokkos = mf_rho_Host_copy.data();


PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
                                   &idx_q_ptr[n_sites_local], mf_rho_ptr_Kokkos));

Kokkos::deep_copy(mf_rho_Default, mf_rho_Host_copy);

// Repeat in Kokkos
double V_local_Kokkos = 0.0;
View_Double_Matrix_Device mean_q_ij1_all_n_local("mean_q_ij1_all", n_sites_local, 6);

Kokkos::parallel_reduce(
    "EvaluatePotentialEnergy", 
    Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
    KOKKOS_LAMBDA(const PetscInt site_u, double& V_u) {
      auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, site_u, Kokkos::ALL());
        V_u += evaluate_V_i_adp_MgHx_Kokkos(
            site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, mf_rho_Default, 
            atomSpecie_Kokkos_Default, atomTopologyKokkos_Kokkos_Default(site_u), adp_Device_Default, mean_q_ij1);
    },
    V_local_Kokkos);

  std::cout << "Acabe potencial en Kokkos: " << V_local_Kokkos << std::endl;

  PetscScalar* stdv_q_ptr;
  PetscCall(DMSwarmGetField(Simulation.atomistic_data, "stdv-q", NULL, NULL,
                            (void**)&stdv_q_ptr));
  Eigen::Map<VectorType> stdv_q(stdv_q_ptr, n_sites_local_ghosted);
  
  Kokkos::Timer timer3;
  #pragma omp parallel for schedule(runtime)
  for (PetscInt site_u = 0; site_u < n_sites_local; site_u++) {

    //! @brief Evaluate energy density at site u
    mf_rho(site_u) = evaluate_mf_rho_i_adp_MgHx(
        site_u, mean_q, stdv_q, xi, specie_ptr, atom_topology[site_u]);
  }

  std::cout << "Tiempo que ha tardado en las operaciones sin Kokkos: " << timer3.seconds() << " seconds" << std::endl;


  PetscScalar_Vector_Host stdv_q_ptr_Kokkos_Host(stdv_q_ptr, static_cast<size_t>(n_sites_local_ghosted));
  PetscScalar_Vector_Default stdv_q_ptr_Kokkos_Default("Device_mechanical_sites", static_cast<size_t>(n_sites_local_ghosted));
  Kokkos::deep_copy(stdv_q_ptr_Kokkos_Default, stdv_q_ptr_Kokkos_Host);
  
  Kokkos::Timer timer4;
  Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_sites_local), KOKKOS_LAMBDA(PetscInt n_sites_local_u) {

    auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, n_sites_local_u, Kokkos::ALL());

    mf_rho_Default(n_sites_local_u) = evaluate_mf_rho_i_adp_MgHx_Kokkos(
      n_sites_local_u, mean_q_Kokkos_Default, stdv_q_ptr_Kokkos_Default, xi_Kokkos_Default, atomSpecie_Kokkos_Default, atomTopologyKokkos_Kokkos_Default(n_sites_local_u), mean_q_ij1, adp_Device_Default);
});

std::cout << "Tiempo que ha tardado en las operaciones con Kokkos: " << timer4.seconds() << " seconds" << std::endl;


/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Get the thermal Lagrange Multiplier (beta) vector
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

PetscScalar* beta_ptr;
PetscCall(DMSwarmGetField(Simulation.atomistic_data, "beta", NULL, NULL,
                          (void**)&beta_ptr));
Eigen::Map<VectorType> beta(beta_ptr, n_sites_local_ghosted);

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Get the chemical Lagrange Multiplier (gamma) vector
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
PetscScalar* gamma_ptr;
PetscCall(DMSwarmGetField(Simulation.atomistic_data, "gamma", NULL, NULL,
                          (void**)&gamma_ptr));
Eigen::Map<VectorType> gamma(gamma_ptr, n_sites_local_ghosted);

  double L0_local = 0.0;

#pragma omp parallel for reduction(+ : L0_local) schedule(runtime)
  for (PetscInt site_u = 0; site_u < n_sites_local; site_u++) {

    //! @brief Evaluate the free entropy at site u
    double S0_u = evaluate_S0_i_adp_MgHx(
        site_u, mean_q, stdv_q, xi, mf_rho, beta, gamma, specie_ptr,
        atom_topology[site_u]);

    //! @brief Update local contribution of the residual equation
    L0_local += k_B * S0_u;
  }

  PetscScalar_Vector_Host beta_ptr_Kokkos_Host(stdv_q_ptr, static_cast<size_t>(n_sites_local_ghosted));
  PetscScalar_Vector_Default beta_ptr_Kokkos_Default("beta_ptr_Device", static_cast<size_t>(n_sites_local_ghosted));
  Kokkos::deep_copy(beta_ptr_Kokkos_Default, beta_ptr_Kokkos_Host);

  PetscScalar_Vector_Host gamma_ptr_Kokkos_Host(stdv_q_ptr, static_cast<size_t>(n_sites_local_ghosted));
  PetscScalar_Vector_Default gamma_ptr_Kokkos_Default("gamma_ptr_Device", static_cast<size_t>(n_sites_local_ghosted));
  Kokkos::deep_copy(gamma_ptr_Kokkos_Default, gamma_ptr_Kokkos_Host);

  View_Double_Vector_Host element_mass_Host("element_mass_Host", 112);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> temp_element_mass(element_mass, 112);
  View_Double_Vector_Device element_mass_Device("element_mass_Device", 112);
  Kokkos::deep_copy(element_mass_Host, temp_element_mass);
  Kokkos::deep_copy(element_mass_Device, element_mass_Host);

  double L0_Local_Kokkos = 0.0;

  Kokkos::parallel_reduce(
      "EvaluateFreeEntropy", 
      Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
      KOKKOS_LAMBDA(const PetscInt site_u, double& local_entropy) {

        auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, site_u, Kokkos::ALL());

          double S0_u = evaluate_S0_i_adp_MgHx_Kokkos(
              site_u, mean_q_Kokkos_Default, stdv_q_ptr_Kokkos_Default, xi_Kokkos_Default, 
              mf_rho_Default, beta_ptr_Kokkos_Default, gamma_ptr_Kokkos_Default, atomSpecie_Kokkos_Default, 
              atomTopologyKokkos_Kokkos_Default(site_u), mean_q_ij1, adp_Device_Default, element_mass_Device);
  
          //! @brief Update local contribution of the residual equation
          local_entropy += k_B * S0_u;
      },
      L0_Local_Kokkos);  

    //! Migrate ghost field (energy density)
    //    PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
    //                                       &idx_q_ptr[n_sites_local],
    //                                       mf_rho_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "beta", NULL, NULL,
      (void**)&beta_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "gamma", NULL, NULL,
      (void**)&gamma_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "stdv-q", NULL,
                                  NULL, (void **)&stdv_q_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data,
                                  DMSwarmPICField_coor, NULL, NULL,
                                  (void **)&mean_q_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "mf-rho", NULL,
                                  NULL, (void **)&mf_rho_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "molar-fraction",
                                  NULL, NULL, (void **)&xi_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "specie", NULL,
                                  NULL, (void **)&specie_ptr));

    PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "idx", NULL, NULL,
                                  (void **)&idx_q_ptr));

    PetscCall(ISRestoreIndices(active_mech_sites,
                               (const PetscInt **)&active_mech_sites_ptr));

    //!  Restore atom topology
    for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++) {

      PetscCall(restore_atom_topology(
          &atom_topology[site_u], Simulation.mechanical_neighs_idx[site_u]));
    }

    free(atom_topology);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Output data
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(
        DMSwarmViewXDMF(Simulation.atomistic_data,
                        "outputs/Mg-hcp-cube-x20-x15-x15-periodic-1.xmf"));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Delete the list of active mechanical sites
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    n_mechanical_sites_local = 0;
    PetscCall(ISGetLocalSize(Simulation.active_mech_sites,
                             &n_mechanical_sites_local));

    if (n_mechanical_sites_local >= 1) {
      PetscCall(ISDestroy(&Simulation.active_mech_sites));
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Destroy list of neighbors and other important information
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(destroy_mechanical_topology(&Simulation));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Destroy ghost atoms
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(DMSwarmDestroyGhostAtoms(&Simulation));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Output data
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(DMView(Simulation.atomistic_data, PETSC_VIEWER_STDOUT_WORLD));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Free work space.
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    //! @brief Destroy atomistic context
    PetscCall(destroy_DMD_simulation(&Simulation));

    //! @brief Destroy ADP context

    // Finalize PETSc
    PetscFinalize();
    // Finalize MPI
#ifdef USE_MPI
    MPI_Finalize();
#endif

    // Finalize Kokkos
    Kokkos::finalize();
  }

    return 0;
  } catch (std::exception &exception) {
    if (rank_MPI == 0) {
      std::cerr << "Test: " << exception.what() << std::endl;
    }
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  }
}
