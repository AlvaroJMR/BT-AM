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
    std::cout << "Tiempo que ha tardado en las operaciones sin Kokkos: " << time2 << " seconds" << std::endl;

    Kokkos::Timer timer;
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

    PetscInt atomSpecie_Index;
    PetscCall(DMSwarmGetLocalSize(Simulation.atomistic_data, &atomSpecie_Index));

    AtomSpecie_Host atomSpecie_Kokkos_Host(specie_ptr, atomSpecie_Index);
    AtomSpecie_Default atomSpecie_Kokkos_Default("atomSpecie_Kokkos_Default", atomSpecie_Index);
    Kokkos::deep_copy(atomSpecie_Kokkos_Default, atomSpecie_Kokkos_Host);

    //Kokkos::printf("atomSpecie_Index = %d\n", atomSpecie_Index);
    //Kokkos::printf("specie_ptr[0] = %d\n", static_cast<int>(specie_ptr[0]));
    //Kokkos::printf("atomTopology_Kokkos_Default[0] = %f\n", atomSpecie_Kokkos_Host(0));

    AdpPotencial_Device adp_Device_Default;

    copy_adpPotential_to_device(adp_Device_Default, adp_MgMg, 0 );

    copy_adpPotential_to_device(adp_Device_Default, adp_HH, 1 );


    /* adp_HH_Kokkos_Host = AdpPotencial_Host("adp_HH_Kokkos_Host", 1);
    adp_HH_Kokkos_Host(0) = adp_HH;
    adp_HH_Kokkos_Default = AdpPotencial_Device("adp_HH_Kokkos_Default", 1);
    Kokkos::deep_copy(adp_HH_Kokkos_Default, adp_HH_Kokkos_Host);
    
        adp_MgMg_Kokkos_Host = AdpPotencial_Host("adp_MgMg_Kokkos_Host", 1);
    adp_MgMg_Kokkos_Host(0) = adp_MgMg;
    adp_MgMg_Kokkos_Default = AdpPotencial_Device("adp_MgMg_Kokkos_Default", 1);
    Kokkos::deep_copy(adp_MgMg_Kokkos_Default, adp_MgMg_Kokkos_Host);
    
    */

    View_Double_Vector_Device mean_q_ij1("mean_q_ij1",6);

  Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType> (0, n_mechanical_sites_local), KOKKOS_LAMBDA(PetscInt mech_site_u) {

    PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);

    mf_rho_Host(site_u) = evaluate_rho_i_adp_MgHx_kokkos_Device(
      site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, atomSpecie_Kokkos_Default, atomTopology_Kokkos_Default(site_u), mean_q_ij1, adp_Device_Default);
  });
  Kokkos::fence();
  double time = timer.seconds();
  std::cout << "Tiempo que ha tardado en las operaciones con Kokkos: " << time << " seconds" << std::endl;
  //


    //! Migrate ghost field (energy density)
    //    PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
    //                                       &idx_q_ptr[n_sites_local],
    //                                       mf_rho_ptr));

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
