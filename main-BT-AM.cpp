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

extern PetscInt size_MPI_X;
extern PetscInt size_MPI_Y;
extern PetscInt size_MPI_Z;

extern adpPotential adp_MgMg;
extern adpPotential adp_HH;
extern adpPotential adp_MgH;

extern double element_mass[112];

extern char OutputFolder[MAXC];
static char help[] = "Bachelor's thesis: Álvaro Montaño Rosa \n";

int main(int argc, char **argv)
{

  snprintf(OutputFolder, sizeof(OutputFolder), "%s", "./");

  try
  {

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_MPI);
    MPI_Comm_size(MPI_COMM_WORLD, &size_MPI);
#endif

    // Initialize Kokkos
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024);
    Kokkos::initialize(argc, argv);
    {
      // Initialize PETSc
      PetscFunctionBeginUser;
      PetscInitialize(&argc, &argv, 0, help);

      ndiv_mesh_X = 400;
      ndiv_mesh_Y = 400;
      ndiv_mesh_Z = 400;

      size_MPI_X = 1;
      size_MPI_Y = 1;
      size_MPI_Z = 1;

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

      for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++)
      {

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
          
                            
    Kokkos::Timer timer_rho_i_adp;
    #pragma omp parallel for schedule(runtime)
    for (PetscInt mech_site_u = 0; mech_site_u < n_mechanical_sites_local;
      mech_site_u++) {

        //! Get index of the site u
        PetscInt site_u = active_mech_sites_ptr[mech_site_u];

        //! @brief Evaluate energy density at site u
        mf_rho(site_u) = evaluate_rho_i_adp_MgHx_kokkos(
            site_u, mean_q, xi, specie_ptr, atom_topology[site_u]);
      }
#pragma omp barrier
      double time = timer_rho_i_adp.seconds();

      // AtomTopologyKokkos* atomTopologyKokkos = convert_atomTopology_array_to_Kokkos(atom_topology,n_sites_local_ghosted);
      //  Initialize atom topology in Kokkos
      std::vector<PetscInt> mech_neighs_ptr_Kokkos;
      std::vector<int> atom_topology_offsets(n_sites_local_ghosted + 1);
      std::vector<int> numneigh_Kokkos(n_sites_local_ghosted);

      atom_topology_offsets[0] = 0;
      for (int i = 0; i < n_sites_local_ghosted; ++i)
      {
        int m = atom_topology[i].numneigh;
        numneigh_Kokkos[i] = m;
        mech_neighs_ptr_Kokkos.insert(
            mech_neighs_ptr_Kokkos.end(),
            atom_topology[i].mech_neighs_ptr,
            atom_topology[i].mech_neighs_ptr + m);
        atom_topology_offsets[i + 1] = atom_topology_offsets[i] + m;
      }

      Int_Vector_Default mech_neighs_ptr_Kokkos_view(
          "mech_neighs_ptr_Kokkos",
          mech_neighs_ptr_Kokkos.size());

      Int_Vector_Default atom_topology_offsets_view(
          "atom_topology_offsets",
          atom_topology_offsets.size());

      Int_Vector_Default numneigh_Kokkos_view(
          "numneigh_Kokkos",
          numneigh_Kokkos.size());

      Kokkos::fence();

      {
        auto mech_neighs_ptr_Kokkos_view_host = PetscInt_Vector_Host(
            mech_neighs_ptr_Kokkos.data(),
            mech_neighs_ptr_Kokkos.size());
        Kokkos::deep_copy(mech_neighs_ptr_Kokkos_view, mech_neighs_ptr_Kokkos_view_host);

        auto atom_topology_offsets_view_host = Int_Vector_Host(
            atom_topology_offsets.data(),
            atom_topology_offsets.size());
        Kokkos::deep_copy(atom_topology_offsets_view, atom_topology_offsets_view_host);

        auto numneigh_Kokkos_view_host = Int_Vector_Host(
            numneigh_Kokkos.data(),
            numneigh_Kokkos.size());
        Kokkos::deep_copy(numneigh_Kokkos_view, numneigh_Kokkos_view_host);
      }
      ///////////////////////////////////////////////////////////

      PetscScalar_Vector_Default mf_rho_Default("mf_rho_Default", n_sites_local_ghosted);
      PetscScalar_Matrix_Default mean_q_Kokkos_Default("mean_q_Kokkos_Default", n_sites_local_ghosted, 3);
      PetscScalar_Vector_Default xi_Kokkos_Default("xi_Kokkos_Default", n_sites_local_ghosted);

      {

        PetscScalar_Vector_Host mf_rho_Host(mf_rho_ptr, n_sites_local_ghosted);
        Kokkos::deep_copy(mf_rho_Default, mf_rho_Host);

        PetscScalar_Matrix_Host mean_q_Kokkos_Host(mean_q_ptr, n_sites_local_ghosted, 3);
        Kokkos::deep_copy(mean_q_Kokkos_Default, mean_q_Kokkos_Host);

        PetscScalar_Vector_Host xi_Kokkos_Host(xi_ptr, static_cast<size_t>(n_sites_local_ghosted));
        Kokkos::deep_copy(xi_Kokkos_Default, xi_Kokkos_Host);
      }

      PetscInt active_mech_sites_index;
      PetscCall(ISGetLocalSize(active_mech_sites, &active_mech_sites_index));

      PetscInt_Vector_Host active_mech_sites_Kokkos_Host(active_mech_sites_ptr, active_mech_sites_index);
      PetscInt_Vector_Default active_mech_sites_Kokkos_Default("Device_mechanical_sites", active_mech_sites_index);
      Kokkos::deep_copy(active_mech_sites_Kokkos_Default, active_mech_sites_Kokkos_Host);

      AtomTopology_Host atomTopology_Kokkos_Host(atom_topology, n_sites_local_ghosted);
      AtomTopology_Default atomTopology_Kokkos_Default("atomTopology_Kokkos_Default", n_sites_local_ghosted);
      Kokkos::deep_copy(atomTopology_Kokkos_Default, atomTopology_Kokkos_Host);

      AtomSpecie_Host atomSpecie_Kokkos_Host(specie_ptr, n_sites_local_ghosted);
      AtomSpecie_Default atomSpecie_Kokkos_Default("atomSpecie_Kokkos_Default", n_sites_local_ghosted);
      Kokkos::deep_copy(atomSpecie_Kokkos_Default, atomSpecie_Kokkos_Host);

      SoA_ADP soa_ADP;
      initSoA_ADP(adp_MgMg, adp_HH, adp_MgH, soa_ADP);

      Kokkos::fence();

      SoADevice snap;
      copyHostToDevice(soa_ADP, snap);

      Kokkos::fence();

      DevSnap devSnap("devSnap");

      {
        auto host_mv = Kokkos::create_mirror_view(devSnap);
        host_mv() = snap;
        Kokkos::deep_copy(devSnap, host_mv);
      }

      Kokkos::fence();
      DevSnapUnmanaged devSnapUM(devSnap.data());
      Kokkos::fence();
      View_Double_Matrix_Device mean_q_ij1_all("mean_q_ij1_all", n_mechanical_sites_local, 6);

      Kokkos::Timer timer_rho_i_adp_Kokkos;

      Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_mechanical_sites_local), KOKKOS_LAMBDA(PetscInt mech_site_u) {
      PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);
  
      AtomTopology topology;
      topology.numneigh       = numneigh_Kokkos_view(site_u);
      topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                              atom_topology_offsets_view(site_u);

       mf_rho_Default(site_u) = evaluate_rho_i_adp_MgHx_kokkos_Device(
          site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, atomSpecie_Kokkos_Default, topology, devSnapUM); });

  Kokkos::fence();

  double time2 = timer_rho_i_adp_Kokkos.seconds();

  Kokkos::Timer timer_rho_i_adp_Kokkos_SIMD;

    Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_mechanical_sites_local), KOKKOS_LAMBDA(PetscInt mech_site_u) {
      PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);
  
      AtomTopology topology;
      topology.numneigh       = numneigh_Kokkos_view(site_u);
      topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                              atom_topology_offsets_view(site_u);

       mf_rho_Default(site_u) = evaluate_rho_i_adp_MgHx_kokkos_Device_SIMD(
          site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, atomSpecie_Kokkos_Default, topology, devSnapUM);    
  });

  Kokkos::fence();

  double time3 = timer_rho_i_adp_Kokkos_SIMD.seconds();

      Kokkos::Timer timer_rho_i_adp_Kokkos2;

      TeamPolicy policy(n_mechanical_sites_local, Kokkos::AUTO, Kokkos::AUTO);

      const std::size_t bytes_per_thread = 6 * sizeof(double);
      policy.set_scratch_size(
          0,
          Kokkos::PerTeam(0),
          Kokkos::PerThread(bytes_per_thread));

      Kokkos::parallel_for("Active_mech_sites", policy, KOKKOS_LAMBDA(const Member &team) {

    const int mech_site_u = team.league_rank();  
    const PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);
    double* mean_q_ij1 = static_cast<double*>(
      team.thread_scratch(0).get_shmem(bytes_per_thread)
    );

    const unsigned int numneigh_site_i = numneigh_Kokkos_view(site_u);
    const PetscInt* mech_neighs_i = mech_neighs_ptr_Kokkos_view.data() +
    atom_topology_offsets_view(site_u);
  
    const AtomicSpecie spc_i = atomSpecie_Kokkos_Default(site_u);
    const double xi_i = xi(site_u);
    auto mean_q_i = extractRowBlock(mean_q_Kokkos_Default, site_u, 0, 3);
    
    if (xi_i < min_occupancy) {
      mf_rho_Default(site_u) = 0.0;
      return;
    }

    double rho_i = 0.0;
    //Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, 0};
    //Kokkos::Array<unsigned int, 2>  sites_ij1 = Kokkos::Array<unsigned int, 2> {site_u, 0};
    //AtomicSpecie spc_ij1[2] = {spc_i, {}};
      Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(team, numneigh_site_i),
        [&](int idx, double& acc){
          unsigned int site_j1 = mech_neighs_i[idx];
          AtomicSpecie spc_j1 = atomSpecie_Kokkos_Default(site_j1);
          double xi_j1 = xi(site_j1);
          auto mean_q_j1 = extractRowBlock(mean_q_Kokkos_Default, site_j1, 0, 3);
      
          if (xi_j1 < min_occupancy) {
            return;
          }
      
          int dof_table_ij[4] = {1, 0, 0, 1};
      
          concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
          // xi_ij1[1] = xi_j1;
          // sites_ij1[1] = site_j1;
          // spc_ij1[1] = spc_j1; 
          Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
          Kokkos::Array<unsigned int, 2>  sites_ij1 = Kokkos::Array<unsigned int, 2> {site_u, site_j1};
          AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};

          double rho_ij = 0.0;
      
          rho_ij_adp_MgHx_dispatcher functions_rho_ij { Functions_Enum::FK, &rho_ij, xi_ij1.data(), mean_q_ij1, spc_ij1, devSnapUM.data() };
          
          functions_rho_ij();
          acc += rho_ij;
        },
        rho_i
      );
  
      mf_rho_Default(site_u) = rho_i; });

      Kokkos::fence();
      double time2K = timer_rho_i_adp_Kokkos2.seconds();
      PetscBarrier((PetscObject)NULL);
      /*PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
        &idx_q_ptr[n_sites_local], mf_rho_ptr));


      // Copy back to host to migrateGhostfield
      //! Migrate ghost field (energy density) con Kokkos
      auto mf_rho_Host_copy = Kokkos::create_mirror_view(mf_rho_Default);
      Kokkos::deep_copy(mf_rho_Host_copy, mf_rho_Default);
      PetscScalar* mf_rho_ptr_Kokkos = mf_rho_Host_copy.data();


      PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
                                         &idx_q_ptr[n_sites_local], mf_rho_ptr_Kokkos));

      Kokkos::deep_copy(mf_rho_Default, mf_rho_Host_copy);    */

      auto mf_rho_Mirrow = Kokkos::create_mirror_view(mf_rho_Default);
      Kokkos::deep_copy(mf_rho_Mirrow, mf_rho_Default);

      Eigen::Map<VectorType> mf_rho_test(mf_rho_Mirrow.data(), n_sites_local_ghosted);

      double eigen_mean = mf_rho.mean();

      double eigen_variance = ((mf_rho.array() - eigen_mean).square().sum()) / static_cast<double>(n_sites_local_ghosted - 1);

      double Kokkos_mean_test = mf_rho_test.mean();
      double Kokkos_variance_test = ((mf_rho_test.array() - Kokkos_mean_test).square().sum()) / static_cast<double>(n_sites_local_ghosted - 1);

std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos: evaluate_rho_i_adp_MgHx " << time << " segundos" << " Resultados: " << eigen_mean << std::endl;
std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos: evaluate_rho_i_adp_MgHx_kokkos_Device " << time2 << " segundos" << " Resultados: " << Kokkos_mean_test << std::endl;
std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos y ThreadVectorRange: evaluate_rho_i_adp_MgHx_kokkos_Device " << time2K << " segundos" << " Resultados: " << Kokkos_mean_test << std::endl;          
std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos y reorganización SIMD : evaluate_rho_i_adp_MgHx_kokkos_Device_SIMD " << time3 << " segundos" << " Resultados: " << Kokkos_mean_test << std::endl;          

  
  // PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "idx", NULL, NULL, (void**)&idx_q_ptr));
  

      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Evaluate free entropy
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      double V_local = 0.0;

      Kokkos::Timer timer_V_i_adp;
#pragma omp parallel for reduction(+ : V_local) schedule(runtime)
      for (PetscInt site_u = 0; site_u < n_sites_local; site_u++)
      {

        //! @brief Evaluate the potential energy site u
        double V_u = evaluate_V_i_adp_MgHx(
            site_u, mean_q, xi, mf_rho, specie_ptr, atom_topology[site_u]);

        //! @brief Update local contribution of the residual equation
        V_local += V_u;
      }
#pragma omp barrier

      double time_V_i_adp = timer_V_i_adp.seconds();
      double V_local_Kokkos = 0.0;
      View_Double_Matrix_Device mean_q_ij1_all_n_local("mean_q_ij1_all", n_sites_local, 6);

      Kokkos::Timer timer_V_i_adp_Kokkos;
      Kokkos::parallel_reduce(
          "EvaluatePotentialEnergy",
          Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
          KOKKOS_LAMBDA(const PetscInt site_u, double &V_u) {
            AtomTopology topology;
            topology.numneigh = numneigh_Kokkos_view(site_u);
            topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                       atom_topology_offsets_view(site_u);
            V_u += evaluate_V_i_adp_MgHx_Kokkos(
                site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, mf_rho_Default,
                atomSpecie_Kokkos_Default, topology, devSnapUM);
          },
          V_local_Kokkos);

      Kokkos::fence();
      double time_V_i_adp_Kokkos = timer_V_i_adp_Kokkos.seconds();

      PetscScalar_Vector_Default retrieve_V_u_results_Default("retrieve_V_u_results_Default", n_sites_local);

      Kokkos::Timer timer_V_i_adp_Kokkos2;

      TeamPolicy policy4(n_mechanical_sites_local, Kokkos::AUTO, Kokkos::AUTO);

    const std::size_t bytes_per_thread4 = 9 * sizeof(double);
    policy4.set_scratch_size(
      0,
      Kokkos::PerTeam(0),
      Kokkos::PerThread(bytes_per_thread4)
    );
    
    Kokkos::parallel_for(
      "EvaluatePotentialEnergy",
      policy4,
      KOKKOS_LAMBDA(const Member& team){

        unsigned int dim = NumberDimensions;
        const int site_i  = team.league_rank();
        AtomTopology topology;
        topology.numneigh       = numneigh_Kokkos_view(site_i);
        topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                atom_topology_offsets_view(site_i);
    
        double mf_rho_i    = mf_rho_Default(site_i);
        double xi_i        = xi_Kokkos_Default(site_i);
        AtomicSpecie spc_i = atomSpecie_Kokkos_Default(site_i);
        auto mean_q_i = extractRowBlock(mean_q_Kokkos_Default, site_i, 0, 3);
    
    
        double V_embed_i = 0.0; 
        double V_pair_i = 0.0;   
        double V_dip_i = 0.0;    
        double V_quad_i = 0.0;   
        double V_i = 0.0;        

        double* mean_q_ij1 = static_cast<double*>(
          team.thread_scratch(0).get_shmem(bytes_per_thread4)
        );
        
        unsigned int numneigh_site_i = topology.numneigh;
        const PetscInt* mech_neighs_i = topology.mech_neighs_ptr;
        
        
        if (xi_i < min_occupancy) {
        return;
        }
    
        Trio localT{0.0, 0.0, 0.0};
        SumTrio trioReducer(localT);
    
        Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange (team, numneigh_site_i),
          [&](unsigned idx_j1, Trio& accThread){
        unsigned int site_j1 = mech_neighs_i[idx_j1];
        AtomicSpecie spc_j1 = atomSpecie_Kokkos_Default(site_j1);
        double xi_j1 = xi_Kokkos_Default(site_j1);
        auto mean_q_j1 = extractRowBlock(mean_q_Kokkos_Default, site_j1, 0, 3);
        
        if (xi_j1 < min_occupancy) {
          return;
          }
        
        concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
        Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
        AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
        
        double V_pair_ij = 0.0;

        {         
        V_pair_ij_adp_MgHx_dispatcher function_V_pair_ij { Functions_Enum::FK, &V_pair_ij, xi_ij1.data(), mean_q_ij1, spc_ij1, devSnapUM.data() };
        function_V_pair_ij();
        }
        accThread.c += V_pair_ij;
        Pair pairT{0.0, 0.0};
        SumPair pairReducer(pairT);
    
        Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(team, idx_j1, numneigh_site_i),
          KOKKOS_LAMBDA(int idx_j2, Pair& accThreadVector) {
      
          unsigned int site_j2 = mech_neighs_i[idx_j2];
          AtomicSpecie spc_j2 = atomSpecie_Kokkos_Default(site_j2);
          double xi_j2 = xi_Kokkos_Default(site_j2);
          auto mean_q_j2 = extractRowBlock(mean_q_Kokkos_Default, site_j2, 0, 3);
        
          //! If the site is empty, skip from the evaluation
          if (xi_j2 < min_occupancy) {
          return;
          }
        
          auto mean_q_ij1j2 = concatenateToArray<Kokkos::View<double*>, 9>(mean_q_i, mean_q_j1, mean_q_j2);
          Kokkos::Array<double, 3>  xi_ij1j2 = Kokkos::Array<double, 3> {xi_i, xi_j1, xi_j2};
          AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};
        
          double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;
        
          double V_dip_ij1j2 = 0.0;
          { 
            V_dipole_ij1j2_dispatcher function_V_dipole_ij1j2 { Functions_Enum::FK, &V_dip_ij1j2, xi_ij1j2.data(), mean_q_ij1j2.data(), spc_ij1j2, devSnapUM.data() };
            function_V_dipole_ij1j2();
          }
          accThreadVector.a += factor_j1j2 * V_dip_ij1j2;
        
          double V_quad_ij1j2 = 0.0;
          {
            V_quadrupole_ij1j2_dispatcher function_V_quadrupole_ij1j2 { Functions_Enum::FK, &V_quad_ij1j2, xi_ij1j2.data(), mean_q_ij1j2.data(), spc_ij1j2, devSnapUM.data() };
            function_V_quadrupole_ij1j2();  
          }
          accThreadVector.b += factor_j1j2 * V_quad_ij1j2;
          }, pairReducer
        );
        accThread.a += pairT.a;
        accThread.b += pairT.b;
        }, trioReducer
      );
        CubicSpline embed_ii;
        if (spc_i == Mg) {
        embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, devSnapUM.data());
        } else if (spc_i == H) {
        embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, devSnapUM.data());
        }
    
        double mf_F_i = cubic_spline_Kokkos(&embed_ii, mf_rho_i);
        V_embed_i = xi_i * mf_F_i;
        
        V_i = V_embed_i + localT.a + localT.b + localT.c;
    
        retrieve_V_u_results_Default(site_i) = V_i;
      
      }
    );

    Kokkos::fence();
    double time_V_i_adp_Kokkos2 = timer_V_i_adp_Kokkos2.seconds();
    double V_local_Kokkos_ThreadVectorRange = 0.0;
    Kokkos::parallel_reduce("SumS0_i",
      Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
      KOKKOS_LAMBDA(const int i, double& sum){
        sum += retrieve_V_u_results_Default(i);
      },
      V_local_Kokkos_ThreadVectorRange
    );

    Kokkos::Timer timer_V_i_adp_Kokkos_SIMD;
    double V_local_Kokkos_SIMD = 0.0;
    Kokkos::parallel_reduce(
    "EvaluatePotentialEnergy", 
    Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
    KOKKOS_LAMBDA(const PetscInt site_u, double& V_u) {
      AtomTopology topology;
      topology.numneigh       = numneigh_Kokkos_view(site_u);
      topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                              atom_topology_offsets_view(site_u);   
        V_u += evaluate_V_i_adp_MgHx_Kokkos_SIMD(
            site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, mf_rho_Default, 
            atomSpecie_Kokkos_Default, topology, devSnapUM);
    },
    V_local_Kokkos_SIMD);

    Kokkos::fence();
    double time_V_i_adp_Kokkos_SIMD = timer_V_i_adp_Kokkos_SIMD.seconds();    

std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos: evaluate_V_i_adp_MgHx " << time_V_i_adp << " segundos: " << " Resultados: "<< V_local << std::endl;
std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos: evaluate_V_i_adp_MgHx_Kokkos " << time_V_i_adp_Kokkos << " segundos: " << " Resultados: "<< V_local_Kokkos << std::endl;
std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos y ThreadVectorRange: evaluate_V_i_adp_MgHx_Kokkos " << time_V_i_adp_Kokkos2 << " segundos: " << " Resultados: "<< V_local_Kokkos_ThreadVectorRange << std::endl;
std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos y reorganización SIMD: evaluate_V_i_adp_MgHx_Kokkos " << time_V_i_adp_Kokkos_SIMD << " segundos: " << " Resultados: "<< V_local_Kokkos_SIMD << std::endl;

      PetscScalar *stdv_q_ptr;
      PetscCall(DMSwarmGetField(Simulation.atomistic_data, "stdv-q", NULL, NULL,
                                (void **)&stdv_q_ptr));
      Eigen::Map<VectorType> stdv_q(stdv_q_ptr, n_sites_local_ghosted);

      Kokkos::Timer timer_mf_rho_i_adp;

#pragma omp parallel for schedule(runtime)
      for (PetscInt site_u = 0; site_u < n_sites_local; site_u++)
      {

        //! @brief Evaluate energy density at site u
        mf_rho(site_u) = evaluate_mf_rho_i_adp_MgHx(
            site_u, mean_q, stdv_q, xi, specie_ptr, atom_topology[site_u]);
      }
#pragma omp barrier
      double time_mf_rho_i_adp = timer_mf_rho_i_adp.seconds();

      PetscScalar_Vector_Host stdv_q_ptr_Kokkos_Host(stdv_q_ptr, static_cast<size_t>(n_sites_local_ghosted));
      PetscScalar_Vector_Default stdv_q_ptr_Kokkos_Default("Device_mechanical_sites", static_cast<size_t>(n_sites_local_ghosted));
      Kokkos::deep_copy(stdv_q_ptr_Kokkos_Default, stdv_q_ptr_Kokkos_Host);

      /* Inicialización de gaussianMeasures*/

      int MaxNumSites = 3;
      Int_Matrix_Default dof_table_view("dof_table", n_sites_local, MaxNumSites * MaxNumSites);
      Int_Matrix_Default gp_board_view("gp_board", n_sites_local, MaxNumSites * MaxNumSites * NumberDimensions * NumberDimensions);
      Int_Matrix_Default dof_table_aux("dof_table", n_sites_local, MaxNumSites * MaxNumSites);
      Int_Matrix_Default active_dof("active_dof", n_sites_local, MaxNumSites);

      gaussian_measure_ctx_Default ctx("ctx", n_sites_local);

      initGaussianContexts(ctx,
                           dof_table_view,
                           gp_board_view,
                           dof_table_aux,
                           active_dof);

      Kokkos::Timer timer_mf_rho_i_adp_Kokkos;

      // Pasarlo a view para CUDA
      bool multipole_integral = false;
#if defined(MULTIPOLE_INTEGRAL)
      multipole_integral = true;
#elif defined(GH3TH_INTEGRAL)
      multipole_integral = false;
#else
#error "Define MULTIPOLE_INTEGRAL or GH3TH_INTEGRAL"
#endif

      Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_sites_local), KOKKOS_LAMBDA(PetscInt n_sites_local_u) {
        double result = 0.0;
        AtomTopology topology;
        topology.numneigh = numneigh_Kokkos_view(n_sites_local_u);
        topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                   atom_topology_offsets_view(n_sites_local_u);

        mf_rho_Default(n_sites_local_u) = evaluate_mf_rho_i_adp_MgHx_Kokkos(
            n_sites_local_u, mean_q_Kokkos_Default, stdv_q_ptr_Kokkos_Default,
            xi_Kokkos_Default, atomSpecie_Kokkos_Default,
            topology, devSnapUM,
            ctx(n_sites_local_u), multipole_integral);
      });

      Kokkos::fence();

      TeamPolicy policy2(n_sites_local, Kokkos::AUTO, Kokkos::AUTO);

      const std::size_t bytes_per_thread2 = MaxNumSites * (1 + MaxNumSites * (2 + NumberDimensions * NumberDimensions)) * sizeof(int) + sizeof(gaussian_measure_ctx_kokkos_s);
      policy2.set_scratch_size(
          0,
          Kokkos::PerTeam(0),
          Kokkos::PerThread(bytes_per_thread2));

      double time_mf_rho_i_adp_Kokkos = timer_mf_rho_i_adp_Kokkos.seconds();

      Kokkos::Timer timer_mf_rho_i_adp_Kokkos2;

      Kokkos::parallel_for("Active_mech_sites", policy2, KOKKOS_LAMBDA(const Member &team) {

    const int mech_site_u = team.league_rank();  
    PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);
    auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all, mech_site_u, Kokkos::ALL());
    AtomTopology topology;
    topology.numneigh       = numneigh_Kokkos_view(site_u);
    topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                              atom_topology_offsets_view(site_u);
                              
    char* mem = static_cast<char*>(
      team.team_scratch(0).get_shmem(bytes_per_thread2)
    );
    gaussian_measure_ctx_kokkos_s* ctxs = reinterpret_cast<gaussian_measure_ctx_kokkos_s*>(mem);    
    int* dof_table = reinterpret_cast<int*>(mem + sizeof(gaussian_measure_ctx_kokkos_s));
    int* gp_board = reinterpret_cast<int*>(mem + sizeof(gaussian_measure_ctx_kokkos_s) + MaxNumSites * MaxNumSites *sizeof(int) );
    int* dof_table_aux = reinterpret_cast<int*>(mem + sizeof(gaussian_measure_ctx_kokkos_s) + (MaxNumSites * MaxNumSites * NumberDimensions * NumberDimensions + MaxNumSites * MaxNumSites) * sizeof(int));
    int* active_dof = reinterpret_cast<int*>(mem + sizeof(gaussian_measure_ctx_kokkos_s) + (MaxNumSites * MaxNumSites * NumberDimensions * NumberDimensions + 2 * (MaxNumSites * MaxNumSites)) * sizeof(int));

    ctxs->dof_table = dof_table;
    ctxs->gp_board = gp_board;
    ctxs->dof_table_aux = dof_table_aux;
    ctxs->active_dof = active_dof;
  
    unsigned int numneigh_site_i = topology.numneigh;
    const PetscInt* mech_neighs_i = topology.mech_neighs_ptr;
  
    AtomicSpecie spc_i = atomSpecie_Kokkos_Default(site_u);
    double xi_i = xi(site_u);
    double stdv_q_i = stdv_q(site_u);
    auto mean_q_i = extractRowBlock(mean_q_Kokkos_Default, site_u, 0, 3);
    
    if (xi_i < min_occupancy) {
      mf_rho_Default(site_u) = 0.0;
      return;
    }                   
  
    double mf_rho_i = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(team, numneigh_site_i),
        [&](int idx, double& acc){
          unsigned int site_j1 = mech_neighs_i[idx];
          AtomicSpecie spc_j1 = atomSpecie_Kokkos_Default(site_j1);
          double xi_j1 = xi(site_j1);
          double stdv_q_j1 = stdv_q(site_j1);
          auto mean_q_j1 = extractRowBlock(mean_q_Kokkos_Default, site_j1, 0, 3);
      
          if (xi_j1 < min_occupancy) {
            return;
          }
      
          int dof_table_ij[4] = {1, 0, 0, 1};
      
          concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
          Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
          Kokkos::Array<unsigned int, 2>  sites_ij1 = Kokkos::Array<unsigned int, 2> {site_u, site_j1};
          AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
          double stdv_q_ij1[2] = {stdv_q_i, stdv_q_j1};

          double mf_rho_ij = 0.0;
          fill_out_gaussian_measure_Kokkos_s(mean_q_ij1.data(), stdv_q_ij1,
            xi_ij1.data(), spc_ij1, dof_table_ij, 2, ctxs);
  
          {
            rho_ij_adp_MgHx_dispatcher function;  
            if (multipole_integral) {
              meanfield_integral_mp_Kokkos<rho_ij_adp_MgHx_dispatcher>(&mf_rho_ij, ctxs, devSnapUM.data(), function);
            } else {
              meanfield_integral_gh3th_Kokkos_s<rho_ij_adp_MgHx_dispatcher>(&mf_rho_ij, ctxs, devSnapUM.data(), function);
            }
          }  
          acc += mf_rho_ij;

        },
        mf_rho_i
      );

      mf_rho_Default(site_u) = mf_rho_i; });

      Kokkos::fence();
      double time_mf_rho_i_adp_Kokkos2 = timer_mf_rho_i_adp_Kokkos2.seconds();
      double eigen_mean1 = mf_rho.mean();
      double eigen_variance1 = ((mf_rho.array() - eigen_mean1).square().sum()) / static_cast<double>(n_sites_local_ghosted - 1);

      auto mf_rho_Mirrow1 = Kokkos::create_mirror_view(mf_rho_Default);
      Kokkos::deep_copy(mf_rho_Mirrow1, mf_rho_Default);

      Eigen::Map<VectorType> mf_rho_test1(mf_rho_Mirrow1.data(), n_sites_local_ghosted);

      double Kokkos_mean_test1 = mf_rho_test1.mean();
      double Kokkos_variance_test1 = ((mf_rho_test1.array() - Kokkos_mean_test1).square().sum()) / static_cast<double>(n_sites_local_ghosted - 1);

      std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos: evaluate_mf_rho_i_adp_MgHx " << time_mf_rho_i_adp << " seconds" << " Resultados: " << eigen_mean1 << std::endl;
      std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos: evaluate_mf_rho_i_adp_MgHx_Kokkos " << time_mf_rho_i_adp_Kokkos << " seconds" << " Resultados: " << Kokkos_mean_test1 << std::endl;
      std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos y ThreadVectorRange: evaluate_mf_rho_i_adp_MgHx_Kokkos " << time_mf_rho_i_adp_Kokkos2 << " seconds" << " Resultados: " << Kokkos_mean_test1 << std::endl;

      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
          Get the thermal Lagrange Multiplier (beta) vector
          - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

      PetscScalar *beta_ptr;
      PetscCall(DMSwarmGetField(Simulation.atomistic_data, "beta", NULL, NULL,
                                (void **)&beta_ptr));
      Eigen::Map<VectorType> beta(beta_ptr, n_sites_local_ghosted);

      /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Get the chemical Lagrange Multiplier (gamma) vector
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
      PetscScalar *gamma_ptr;
      PetscCall(DMSwarmGetField(Simulation.atomistic_data, "gamma", NULL, NULL,
                                (void **)&gamma_ptr));
      Eigen::Map<VectorType> gamma(gamma_ptr, n_sites_local_ghosted);

      double L0_local = 0.0;

      Kokkos::Timer timer_S0_i_adp;

#pragma omp parallel for reduction(+ : L0_local) schedule(runtime)
      for (PetscInt site_u = 0; site_u < n_sites_local; site_u++)
      {

        //! @brief Evaluate the free entropy at site u
        double S0_u = evaluate_S0_i_adp_MgHx(
            site_u, mean_q, stdv_q, xi, mf_rho, beta, gamma, specie_ptr,
            atom_topology[site_u]);

        //! @brief Update local contribution of the residual equation
        L0_local += k_B * S0_u;
      }
#pragma omp barrier
      double time_S0_i_adp = timer_S0_i_adp.seconds();

      PetscScalar_Vector_Host beta_ptr_Kokkos_Host(beta_ptr, static_cast<size_t>(n_sites_local_ghosted));
      PetscScalar_Vector_Default beta_ptr_Kokkos_Default("beta_ptr_Device", static_cast<size_t>(n_sites_local_ghosted));
      Kokkos::deep_copy(beta_ptr_Kokkos_Default, beta_ptr_Kokkos_Host);

      PetscScalar_Vector_Host gamma_ptr_Kokkos_Host(stdv_q_ptr, static_cast<size_t>(n_sites_local_ghosted));
      PetscScalar_Vector_Default gamma_ptr_Kokkos_Default("gamma_ptr_Device", static_cast<size_t>(n_sites_local_ghosted));
      Kokkos::deep_copy(gamma_ptr_Kokkos_Default, gamma_ptr_Kokkos_Host);

      View_Double_Vector_Host element_mass_Host("element_mass_Host", 112);
      Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> temp_element_mass(element_mass, 112);
      View_Double_Vector_Device element_mass_Device("element_mass_Device", 112);
      Kokkos::deep_copy(element_mass_Host, temp_element_mass);
      Kokkos::deep_copy(element_mass_Device, element_mass_Host);

      double *mean_q_ij1_all_n_local_data = mean_q_ij1_all_n_local.data();

      double L0_Local_Kokkos = 0.0;
      Kokkos::Timer timer_S0_i_adp_Kokkos;
      Kokkos::parallel_reduce(
          "EvaluateFreeEntropy",
          Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
          KOKKOS_LAMBDA(const PetscInt site_u, double &local_entropy) {
            AtomTopology topology;
            topology.numneigh = numneigh_Kokkos_view(site_u);
            topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                       atom_topology_offsets_view(site_u);
            double S0_u = evaluate_S0_i_adp_MgHx_Kokkos(
                site_u, mean_q_Kokkos_Default, stdv_q_ptr_Kokkos_Default, xi_Kokkos_Default,
                mf_rho_Default, beta_ptr_Kokkos_Default, gamma_ptr_Kokkos_Default, atomSpecie_Kokkos_Default,
                topology, devSnapUM, element_mass_Device, ctx(site_u), multipole_integral);

            //! @brief Update local contribution of the residual equation
            local_entropy += k_B * S0_u;
          },
          L0_Local_Kokkos);

  Kokkos::fence();



  double time_S0_i_adp_Kokkos = timer_S0_i_adp_Kokkos.seconds();

      Kokkos::Timer timer_S0_i_adp_Kokkos2;
      PetscScalar_Vector_Default retrieve_S0_results_Default("retrieve_S0_results_Default", n_sites_local);

      Kokkos::parallel_for(
          "EvaluateFreeEntropy",
          policy,
          KOKKOS_LAMBDA(const Member &team) {
            unsigned int dim = NumberDimensions;
            const int site_i = team.league_rank();
            auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, site_i, Kokkos::ALL());
            AtomTopology topology;
            topology.numneigh = numneigh_Kokkos_view(site_i);
            topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                       atom_topology_offsets_view(site_i);

            double mf_rho_i = mf_rho_Default(site_i);
            double stdv_q_i = stdv_q_ptr_Kokkos_Default(site_i);
            double xi_i = xi_Kokkos_Default(site_i);
            double beta_i = beta_ptr_Kokkos_Default(site_i);
            double gamma_i = gamma_ptr_Kokkos_Default(site_i);
            AtomicSpecie spc_i = atomSpecie_Kokkos_Default(site_i);
            double m_i = unit_change_uma * xi_i * element_mass_Device(spc_i);
            auto mean_q_i = extractRowBlock(mean_q_Kokkos_Default, site_i, 0, 3);

            double V0_embed_i = 0.0; //! Meanfield Embedded forces term
            double V0_pair_i = 0.0;  //! Meanfield Pairing forces term
            double V0_dip_i = 0.0;   //! Meanfield Dipole distortion term
            double V0_quad_i = 0.0;  //! Meanfield Quadrupole distortion term
            double V0_i = 0.0;       //! Total potential

            unsigned int numneigh_site_i = topology.numneigh;
            const PetscInt *mech_neighs_i = topology.mech_neighs_ptr;

            if (xi_i < min_occupancy)
            {
              return;
            }

            Trio localT{0.0, 0.0, 0.0};
            SumTrio trioReducer(localT);

    Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange (team, numneigh_site_i),
      [&](unsigned idx_j1, Trio& accThread){
    unsigned int site_j1 = mech_neighs_i[idx_j1];
    AtomicSpecie spc_j1 = atomSpecie_Kokkos_Default(site_j1);
    double xi_j1 = xi_Kokkos_Default(site_j1);
    double stdv_q_j1 = stdv_q_ptr_Kokkos_Default(site_j1);
    auto mean_q_j1 = extractRowBlock(mean_q_Kokkos_Default, site_j1, 0, 3);
    
    if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
    return;
    }
    
    concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
    Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
    double stdv_q_ij1[2] = {stdv_q_i, stdv_q_j1};
    AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
    
    
    int dof_table_ij[4] = {1, 0, 0, 1};
    
    
    fill_out_gaussian_measure_Kokkos(mean_q_ij1.data(), stdv_q_ij1,
      xi_ij1.data(), spc_ij1, dof_table_ij, 2, &ctx(site_i));      
          
    double V0_pair_ij = 0.0;
    { 
    V_pair_ij_adp_MgHx_dispatcher function;
    
    if (multipole_integral) {
      meanfield_integral_mp_Kokkos<V_pair_ij_adp_MgHx_dispatcher>(&V0_pair_ij, &ctx(site_i), devSnapUM.data(), function);
    } else {
      meanfield_integral_gh3th_Kokkos<V_pair_ij_adp_MgHx_dispatcher>(&V0_pair_ij, &ctx(site_i), devSnapUM.data(), function);
    } 
    }
    accThread.c += V0_pair_ij;
    Pair pairT{0.0, 0.0};
    SumPair pairReducer(pairT);

                  Kokkos::parallel_reduce(
                      Kokkos::ThreadVectorRange(team, idx_j1, numneigh_site_i),
                      KOKKOS_LAMBDA(int idx_j2, Pair &accThreadVector) {
                        unsigned int site_j2 = mech_neighs_i[idx_j2];
                        AtomicSpecie spc_j2 = atomSpecie_Kokkos_Default(site_j2);
                        double xi_j2 = xi_Kokkos_Default(site_j2);
                        double stdv_q_j2 = stdv_q_ptr_Kokkos_Default(site_j2);
                        auto mean_q_j2 = extractRowBlock(mean_q_Kokkos_Default, site_j2, 0, 3);

                        //! If the site is empty, skip from the evaluation
                        if (xi_j2 < min_occupancy)
                        {
                          return;
                        }

                        int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

                        if (site_j1 == site_j2)
                        {
                          dof_table_ij1j2[5] = 1;
                          dof_table_ij1j2[7] = 1;
                        }
                        auto mean_q_ij1j2 = concatenateToArray<Kokkos::View<double *>, 9>(mean_q_i, mean_q_j1, mean_q_j2);
                        Kokkos::Array<double, 3> xi_ij1j2 = Kokkos::Array<double, 3>{xi_i, xi_j1, xi_j2};
                        AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};
                        double stdv_q_ij1j2[3] = {stdv_q_i, stdv_q_j1, stdv_q_j2};

                        double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;

                        fill_out_gaussian_measure_Kokkos(mean_q_ij1j2.data(), stdv_q_ij1j2,
                                                         xi_ij1j2.data(), spc_ij1j2, dof_table_ij1j2, 3, &ctx(site_i));

                        double V0_dip_ij1j2 = 0.0;
                        {
                          V_dipole_ij1j2_dispatcher function_V_dipole_ij1j2;
                          if (multipole_integral)
                          {
                            meanfield_integral_mp_Kokkos<V_dipole_ij1j2_dispatcher>(&V0_dip_ij1j2, &ctx(site_i), devSnapUM.data(), function_V_dipole_ij1j2);
                          }
                          else
                          {
                            meanfield_integral_gh3th_Kokkos<V_dipole_ij1j2_dispatcher>(&V0_dip_ij1j2, &ctx(site_i), devSnapUM.data(), function_V_dipole_ij1j2);
                          }
                        }
                        accThreadVector.a += factor_j1j2 * V0_dip_ij1j2;

                        double V0_quad_ij1j2 = 0.0;
                        {
                          V_quadrupole_ij1j2_dispatcher function_V_quadrupole_ij1j2;
                          if (multipole_integral)
                          {
                            meanfield_integral_mp_Kokkos<V_quadrupole_ij1j2_dispatcher>(&V0_quad_ij1j2, &ctx(site_i), devSnapUM.data(), function_V_quadrupole_ij1j2);
                          }
                          else
                          {
                            meanfield_integral_gh3th_Kokkos<V_quadrupole_ij1j2_dispatcher>(&V0_quad_ij1j2, &ctx(site_i), devSnapUM.data(), function_V_quadrupole_ij1j2);
                          }
                        }
                        accThreadVector.b += factor_j1j2 * V0_quad_ij1j2;
                      },
                      pairReducer);
                  accThread.a += pairT.a;
                  accThread.b += pairT.b;
                },
                trioReducer);
            CubicSpline embed_ii;
            if (spc_i == Mg)
            {
              embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, devSnapUM.data());
            }
            else if (spc_i == H)
            {
              embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, devSnapUM.data());
            }

            double mf_F_i = cubic_spline_Kokkos(&embed_ii, mf_rho_i);
            V0_embed_i = xi_i * mf_F_i;

            V0_i = V0_embed_i + localT.a + localT.b + localT.c;

            double H0_i = 1.0 / (2.0 * beta_i) + V0_i;

            double log_Z0_i = 3.0 * log((stdv_q_i * sqrt(m_i / beta_i)) / h_planck);
            if (xi_i < 1.0)
            {
              log_Z0_i += log(1.0 / (1.0 - xi_i));
            }
            double gamma_xi_i = 0.0;
            if (spc_i == H)
            {
              gamma_xi_i = gamma_i * (xi_i > 0.9999 ? 0.9999 : xi_i);
            }
            double S0_i = -log_Z0_i + beta_i * H0_i - gamma_xi_i;

            retrieve_S0_results_Default(site_i) += k_B * S0_i;
          });

      Kokkos::fence();
      double time_S0_i_adp_Kokkos2 = timer_S0_i_adp_Kokkos2.seconds();

      Kokkos::parallel_reduce("SumS0_i", Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local), KOKKOS_LAMBDA(const int i, double &sum) { sum += retrieve_S0_results_Default(i); }, L0_Local_Kokkos);

  std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos: evaluate_S0_i_adp_MgHx " << time_S0_i_adp << " segundos" << " Resultados: " << L0_local << std::endl;
  std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos: evaluate_S0_i_adp_MgHx_Kokkos " << time_S0_i_adp_Kokkos << " segundos" << " Resultados: " << L0_Local_Kokkos << std::endl;
  std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos y ThreadVectorRange: evaluate_S0_i_adp_MgHx_Kokkos " << time_S0_i_adp_Kokkos2 << " segundos" << " Resultados: " << L0_Local_Kokkos << std::endl;

  
  unsigned int dim = NumberDimensions;

      Vec X_dist, X_loc;
      Vec Y_dist, Y_loc;
      PetscScalar *X_loc_ptr;
      PetscScalar *Y_loc_ptr;

      PetscInt n_dof_local = dim * n_sites_local;
      PetscInt n_dof_ghost = dim * n_sites_ghost;
      PetscInt *idx_dof_ghost = (PetscInt *)malloc(n_dof_ghost * sizeof(PetscInt));
      for (int i = 0; i < n_sites_ghost; i++)
      {
        for (int j = 0; j < dim; j++)
        {
          idx_dof_ghost[i * dim + j] = idx_q_ptr[n_sites_local + i] * dim + j;
        }
      }

      PetscCall(VecCreateGhost(PETSC_COMM_WORLD, n_dof_local, PETSC_DETERMINE,
                               n_dof_ghost, idx_dof_ghost, &X_dist));
      PetscCall(VecSetUp(X_dist));

      PetscCall(VecGhostGetLocalForm(X_dist, &X_loc));
      PetscCall(VecGetArray(X_loc, &X_loc_ptr));

#pragma omp parallel for schedule(runtime)
      for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++)
      {
        for (PetscInt alpha = 0; alpha < dim; alpha++)
        {
          X_loc_ptr[site_u * dim + alpha] = mean_q_ptr[site_u * dim + alpha];
        }
      }
      PetscCall(VecRestoreArray(X_loc, &X_loc_ptr));
      PetscCall(VecGhostRestoreLocalForm(X_dist, &X_loc));

      PetscCall(VecDuplicate(X_dist, &Y_dist));
      PetscCall(VecSetOptionsPrefix(Y_dist, "minV_dq_RHS_"));

      PetscCall(VecGhostUpdateBegin(X_dist, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecGhostUpdateEnd(X_dist, INSERT_VALUES, SCATTER_FORWARD));

      PetscCall(VecGhostGetLocalForm(X_dist, &X_loc));
      PetscCall(VecGetArray(X_loc, &X_loc_ptr));

#pragma omp parallel for schedule(runtime)
      for (PetscInt mech_site_u = 0; mech_site_u < n_mechanical_sites_local;
           mech_site_u++)
      {

        //! Get index of the site u
        PetscInt site_u = active_mech_sites_ptr[mech_site_u];

        //! @brief Evaluate energy density at site u
        mf_rho(site_u) = evaluate_rho_i_adp_MgHx_kokkos(
            site_u, mean_q, xi, specie_ptr, atom_topology[site_u]);
      }

      PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
                                         &idx_q_ptr[n_sites_local], mf_rho_ptr));

      PetscCall(VecSet(Y_dist, 0.0));

      PetscCall(VecGhostGetLocalForm(Y_dist, &Y_loc));
      PetscCall(VecGetArray(Y_loc, &Y_loc_ptr));

      Kokkos::Timer timer_DV_i_Dq_u_adp;
#pragma omp parallel for schedule(runtime)
      for (PetscInt mech_site_u = 0; mech_site_u < n_mechanical_sites_local;
           mech_site_u++)
      {

        //! Get index of the site u
        PetscInt site_u = active_mech_sites_ptr[mech_site_u];

        //! @brief Evaluate the local gradient of V0 at site u
        Eigen::Vector3d dV_dq_u = Eigen::Vector3d::Zero();

        //! @brief Evaluate gradient potential at site u
        {
          dV_dq_u += evaluate_DV_i_Dq_u_adp_MgHx(site_u, site_u, mean_q, xi,
                                                 mf_rho, specie_ptr,
                                                 atom_topology[site_u]);
        }

        //! @brief Evaluate the local gradient of V0 at the neighs of site u
        // atom_topology[site_u].numneigh
        for (PetscInt idx_i = 0; idx_i < atom_topology[site_u].numneigh; idx_i++)
        {

          //! Get index of the site i
          PetscInt site_i = atom_topology[site_u].mech_neighs_ptr[idx_i];

          //! @brief Evaluate gradient potential at site i
          dV_dq_u += evaluate_DV_i_Dq_u_adp_MgHx(site_u, site_i, mean_q, xi,
                                                 mf_rho, specie_ptr,
                                                 atom_topology[site_i]);
        }

        //! Fill residual vector
        for (PetscInt alpha = 0; alpha < dim; alpha++)
        {
          Y_loc_ptr[site_u * dim + alpha] = dV_dq_u(alpha);
        }
      }

#pragma omp barrier
      double time_DV_i_Dq_u_adp = timer_DV_i_Dq_u_adp.seconds();
      PetscInt Y_loc_size;

      PetscCall(VecGetLocalSize(Y_loc, &Y_loc_size));

      PetscScalar_Vector_Host Y_loc_view(Y_loc_ptr, static_cast<size_t>(Y_loc_size));
      PetscScalar_Vector_Default Y_loc_view_device("Y_loc_view_device", static_cast<size_t>(Y_loc_size));
      Kokkos::deep_copy(Y_loc_view_device, Y_loc_view);

      ThreeD_Double_View aux_view("aux_view", 3, 11, n_mechanical_sites_local);
      Kokkos::deep_copy(aux_view, 0.0);

      PetscScalar_Vector_Host mf_rho_Host(mf_rho_ptr, static_cast<size_t>(n_sites_local_ghosted));
      PetscScalar_Vector_Default mf_rho_Default1("mf_rho_Default1", static_cast<size_t>(n_sites_local_ghosted));
      Kokkos::deep_copy(mf_rho_Default1, mf_rho_Host);

      Kokkos::Timer timer_DV_i_Dq_u_adp_Kokkos;
      Kokkos::parallel_for(
          "EvaluateGradientPotential",
          Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_mechanical_sites_local),
          KOKKOS_LAMBDA(const PetscInt mech_site_u) {
            //! Get index of the site u
            PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);

            //! @brief Evaluate the local gradient of V0 at site u
            // Eigen::Vector3d dV_dq_u = Eigen::Vector3d::Zero();

            auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, site_u, Kokkos::ALL());
            double dV_dq_u[3] = {0.0, 0.0, 0.0};

            AtomTopology topologySiteU;
            topologySiteU.numneigh = numneigh_Kokkos_view(site_u);
            topologySiteU.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                            atom_topology_offsets_view(site_u);

            //! @brief Evaluate gradient potential at site u
            {
              auto temp = evaluate_DV_i_Dq_u_adp_MgHx_Kokkos(
                  site_u, site_u, mean_q_Kokkos_Default, xi_Kokkos_Default,
                  mf_rho_Default1, atomSpecie_Kokkos_Default,
                  topologySiteU, mean_q_ij1, aux_view, devSnapUM);

              dV_dq_u[0] += temp(0);
              dV_dq_u[1] += temp(1);
              dV_dq_u[2] += temp(2);
            }

            //! @brief Evaluate the local gradient of V0 at the neighbors of site u
            for (PetscInt idx_i = 0; idx_i < numneigh_Kokkos_view(site_u); idx_i++)
            {
              //! Get index of the site i
              PetscInt site_i = topologySiteU.mech_neighs_ptr[idx_i];
              AtomTopology topology;
              topology.numneigh = numneigh_Kokkos_view(site_i);
              topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                         atom_topology_offsets_view(site_i);
              //! @brief Evaluate gradient potential at site i
              auto temp = evaluate_DV_i_Dq_u_adp_MgHx_Kokkos(
                  site_u, site_i, mean_q_Kokkos_Default, xi_Kokkos_Default,
                  mf_rho_Default1, atomSpecie_Kokkos_Default,
                  topology, mean_q_ij1, aux_view, devSnapUM);
              dV_dq_u[0] += temp(0);
              dV_dq_u[1] += temp(1);
              dV_dq_u[2] += temp(2);
            }

            //! Fill residual vector
            for (PetscInt alpha = 0; alpha < dim; alpha++)
            {
              Y_loc_view_device(site_u * dim + alpha) = dV_dq_u[alpha];
            }
          });

      Kokkos::fence();

      double time_DV_i_Dq_u_adp_Kokkos = timer_DV_i_Dq_u_adp_Kokkos.seconds();

  // With SIMD

  double sum_device = 0.0;
Kokkos::parallel_reduce("SumY_loc", Kokkos::RangePolicy<DefaultExecSpace>(0, n_mechanical_sites_local),
  KOKKOS_LAMBDA(const size_t i, double &localSum) {
    localSum += Y_loc_view_device(i);
}, sum_device);

  double mean_device = sum_device / n_mechanical_sites_local;

  Kokkos::Timer timer_DV_i_Dq_u_adp_Kokkos_SIMD;
  Kokkos::parallel_for(
    "EvaluateGradientPotential", 
    Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_mechanical_sites_local),
    KOKKOS_LAMBDA(const PetscInt mech_site_u) {
        //! Get index of the site u
        PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);
    
        AtomTopology topologySiteU;
        topologySiteU.numneigh       = numneigh_Kokkos_view(site_u);
        topologySiteU.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                atom_topology_offsets_view(site_u); 
        double dV_dq_u[3] = {0.0, 0.0, 0.0};


        //! @brief Evaluate gradient potential at site u
        {
          auto temp = evaluate_DV_i_Dq_u_adp_MgHx_Kokkos_SIMD(
                site_u, site_u, mean_q_Kokkos_Default, xi_Kokkos_Default,
                mf_rho_Default1, atomSpecie_Kokkos_Default,
                topologySiteU, devSnapUM);

                dV_dq_u[0] += temp[0];
                dV_dq_u[1] += temp[1];
                dV_dq_u[2] += temp[2];
            }
    
        //! @brief Evaluate the local gradient of V0 at the neighbors of site u
        for (PetscInt idx_i = 0; idx_i < numneigh_Kokkos_view(site_u); idx_i++) {
            //! Get index of the site i
            PetscInt site_i = topologySiteU.mech_neighs_ptr[idx_i];            AtomTopology topology;
            topology.numneigh       = numneigh_Kokkos_view(site_i);
            topology.mech_neighs_ptr = mech_neighs_ptr_Kokkos_view.data() +
                                  atom_topology_offsets_view(site_i);     
            //! @brief Evaluate gradient potential at site i
            auto temp = evaluate_DV_i_Dq_u_adp_MgHx_Kokkos_SIMD(
                site_u, site_i, mean_q_Kokkos_Default, xi_Kokkos_Default,
                mf_rho_Default1, atomSpecie_Kokkos_Default,
                topology, devSnapUM);
                dV_dq_u[0] += temp[0];
                dV_dq_u[1] += temp[1];
                dV_dq_u[2] += temp[2];
            }
    
            //! Fill residual vector
        for (PetscInt alpha = 0; alpha < dim; alpha++) {
            Y_loc_view_device(site_u * dim + alpha) = dV_dq_u[alpha];
        }
    });

  Kokkos::fence();

  double time_DV_i_Dq_u_adp_Kokkos_SIMD = timer_DV_i_Dq_u_adp_Kokkos_SIMD.seconds();

  ////////

// n_mechanical_sites_local
  double sum_host = 0.0;
for (size_t i = 0; i < n_mechanical_sites_local; i++) {
  sum_host += Y_loc_ptr[i];
}
double mean_host = sum_host / n_mechanical_sites_local;

double sum_device_SIMD = 0.0;
Kokkos::parallel_reduce("SumY_loc", Kokkos::RangePolicy<DefaultExecSpace>(0, n_mechanical_sites_local),
  KOKKOS_LAMBDA(const size_t i, double &localSum) {
    localSum += Y_loc_view_device(i);
}, sum_device_SIMD);

  double mean_device_SIMD = sum_device_SIMD / n_mechanical_sites_local;
  Kokkos::fence();

  std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos: " << time_DV_i_Dq_u_adp << " segundos:" << " Resultados: " << mean_host << std::endl;
  std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos: " << time_DV_i_Dq_u_adp_Kokkos << " segundos:" << " Resultados: " <<  mean_device << std::endl;
  std::cout << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos y reorganización SIMD: " << time_DV_i_Dq_u_adp_Kokkos_SIMD << " segundos:" << " Resultados: " <<  mean_device_SIMD << std::endl;

  // Finalize Kokkos
  destroy_adp_MgHx(&adp_MgMg);
  destroy_adp_MgHx(&adp_HH);
  destroy_adp_MgHx(&adp_MgH);
  ctx = gaussian_measure_ctx_Default();
  devSnap = DevSnap();
  clearSoA_ADP(soa_ADP);
  
    //! Migrate ghost field (energy density)
    //    PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
    //                                       &idx_q_ptr[n_sites_local],
    //                                       mf_rho_ptr));

      PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "beta", NULL, NULL,
                                    (void **)&beta_ptr));

      PetscCall(DMSwarmRestoreField(Simulation.atomistic_data, "gamma", NULL, NULL,
                                    (void **)&gamma_ptr));

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
      for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++)
      {

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

      if (n_mechanical_sites_local >= 1)
      {
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
    }
    Kokkos::finalize();
    return 0;
  }
  catch (std::exception &exception)
  {
    if (rank_MPI == 0)
    {
      std::cerr << "Test: " << exception.what() << std::endl;
    }
#ifdef USE_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  }
}
