#include <iostream>
#include <unistd.h>
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
    // cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024);
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

    PetscScalar* stdv_q_ptr;
    PetscCall(DMSwarmGetField(Simulation.atomistic_data, "stdv-q", NULL, NULL,
                              (void**)&stdv_q_ptr));
    Eigen::Map<VectorType> stdv_q(stdv_q_ptr, n_sites_local_ghosted);
    
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

    MPI_File outputFile;
    MPI_File_open(MPI_COMM_WORLD, "Resultados.txt",
              MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND,
              MPI_INFO_NULL, &outputFile);

    std::ostringstream out;


    bool useKokkos = false;
    int opt;
    while ((opt = getopt(argc, argv, "k")) != -1) {
      switch (opt) {
        case 'k': 
          useKokkos = true;
          break;
      }
    }    
    {
    if (!useKokkos) { 
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

    if (rank_MPI == 0){
      std::cout << "Tiempo que ha tardado en las operaciones sin Kokkos: " << time2 << " seconds" << std::endl;
    }

    //! Migrate ghost field (energy density) sin Kokkos

    PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
      &idx_q_ptr[n_sites_local], mf_rho_ptr));

    double eigen_mean_loc = mf_rho.mean();
    double eigen_mean_sum = 0.0;
    double eigen_mean_all = 0.0;
      
    MPI_Allreduce(&eigen_mean_loc, &eigen_mean_sum, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD);
    eigen_mean_all = eigen_mean_sum / static_cast<double>(size_MPI);
      
    if (rank_MPI == 0) {
      std::cout << "Media global de la densidad de energia = " << eigen_mean_all << std::endl;
    }
    
    double eigen_mean = mf_rho.mean();
      
    double eigen_variance = ((mf_rho.array() - eigen_mean).square().sum()) / static_cast<double>(n_sites_local_ghosted-1);

    if (rank_MPI == 0){      
      std::cout << "Eigen: Media = " << eigen_mean
                << ", Varianza = " << eigen_variance << std::endl;
      
      }

    out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos evaluate_rho_i_adp_MgHx_kokkos: = " << time2 << " Resultados: "<< eigen_mean <<"\n";
    std::string str = out.str();
    MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    out.str("");     

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Evaluate free entropy
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    double V_local = 0.0;

    Kokkos::Timer timer;
    #pragma omp parallel for reduction(+ : V_local) schedule(runtime)
    for (PetscInt site_u = 0; site_u < n_sites_local; site_u++) {

    //! @brief Evaluate the potential energy site u
    double V_u = evaluate_V_i_adp_MgHx(
      site_u, mean_q, xi, mf_rho, specie_ptr, atom_topology[site_u]);

      //! @brief Update local contribution of the residual equation
      V_local += V_u;
    }
    
    double time = timer.seconds();
    if (rank_MPI == 0) {
      std::cout << "Acabe potencial sin Kokkos: " << V_local << std::endl;
    }  

    out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos evaluate_V_i_adp_MgHx: = " << time << " Resultados: "<< V_local <<"\n";
    str = out.str();
    MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    out.str("");

    Kokkos::Timer timer3;
    #pragma omp parallel for schedule(runtime)
    for (PetscInt site_u = 0; site_u < n_sites_local; site_u++) {
  
      //! @brief Evaluate energy density at site u
      mf_rho(site_u) = evaluate_mf_rho_i_adp_MgHx(
          site_u, mean_q, stdv_q, xi, specie_ptr, atom_topology[site_u]);
    }
   
    double eigen_mean1 = mf_rho.mean();
    double eigen_variance1 = ((mf_rho.array() - eigen_mean1).square().sum()) / static_cast<double>(n_sites_local_ghosted - 1);

    if (rank_MPI == 0){
      std::cout << "Tiempo que ha tardado en las operaciones sin Kokkos: " << timer3.seconds() << " seconds" << std::endl;
      std::cout << "Eigen: Media = " << eigen_mean1 << ", Varianza = " << eigen_variance1 << std::endl;  
    }

    out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos evaluate_mf_rho_i_adp_MgHx: = " << timer3.seconds() << " Resultados: "<< eigen_mean1 <<"\n";
    str = out.str();
    MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    out.str("");

    double L0_local = 0.0;
    
    Kokkos::Timer timer5;
    #pragma omp parallel for reduction(+ : L0_local) schedule(runtime)
      for (PetscInt site_u = 0; site_u < n_sites_local; site_u++) {
    
        //! @brief Evaluate the free entropy at site u
        double S0_u = evaluate_S0_i_adp_MgHx(
            site_u, mean_q, stdv_q, xi, mf_rho, beta, gamma, specie_ptr,
            atom_topology[site_u]);
    
        //! @brief Update local contribution of the residual equation
        L0_local += k_B * S0_u;
      }

    double time5 = timer5.seconds();

    if (rank_MPI == 0){
      std::cout << "Tiempo sin Kokkos (timer5): " << time5 << " seconds" << std::endl;
      std::cout << "L0_local sin Kokkos: " << L0_local << std::endl;
    }

    out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos evaluate_S0_i_adp_MgHx: = " << time5 << " Resultados: "<< L0_local <<"\n";
    str = out.str();
    MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    out.str("");

    Vec X_dist, X_loc;
    Vec Y_dist, Y_loc;  
    PetscScalar* X_loc_ptr;
    PetscScalar* Y_loc_ptr;
  
    unsigned int dim = NumberDimensions;

    PetscInt n_dof_local = dim * n_sites_local;
    PetscInt n_dof_ghost = dim * n_sites_ghost;
    PetscInt* idx_dof_ghost = (PetscInt*)malloc(n_dof_ghost * sizeof(PetscInt));
    for (int i = 0; i < n_sites_ghost; i++) {
      for (int j = 0; j < dim; j++) {
        idx_dof_ghost[i * dim + j] = idx_q_ptr[n_sites_local + i] * dim + j;
      }
    }
  
    PetscCall(VecCreateGhost(PETSC_COMM_WORLD, n_dof_local, PETSC_DETERMINE,
                             n_dof_ghost, idx_dof_ghost, &X_dist));
    PetscCall(VecSetUp(X_dist));
  
    PetscCall(VecGhostGetLocalForm(X_dist, &X_loc));
    PetscCall(VecGetArray(X_loc, &X_loc_ptr));
    
  #pragma omp parallel for schedule(runtime)
    for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++) {
      for (PetscInt alpha = 0; alpha < dim; alpha++) {
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
         mech_site_u++) {
  
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
          
    Kokkos::Timer timer7;
    #pragma omp parallel for schedule(runtime)
      for (PetscInt mech_site_u = 0; mech_site_u < n_mechanical_sites_local;
            mech_site_u++) {
      
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
      for (PetscInt idx_i = 0; idx_i < atom_topology[site_u].numneigh; idx_i++) {
      
          //! Get index of the site i
          PetscInt site_i = atom_topology[site_u].mech_neighs_ptr[idx_i];
      
          //! @brief Evaluate gradient potential at site i
          dV_dq_u += evaluate_DV_i_Dq_u_adp_MgHx(site_u, site_i, mean_q, xi,
                                                        mf_rho, specie_ptr,
                                                        atom_topology[site_i]);
          }
      
          //! Fill residual vector
      for (PetscInt alpha = 0; alpha < dim; alpha++) {
          Y_loc_ptr[site_u * dim + alpha] = dV_dq_u(alpha);
      }
    }
  
    double time7 = timer7.seconds();
    if (rank_MPI == 0){
    std::cout << "Tiempo sin Kokkos (timer7): " << time7 << " seconds" << std::endl;
    }
    
    double sum_host = 0.0;
    for (size_t i = 0; i < (size_t)n_mechanical_sites_local; i++) {
      sum_host += Y_loc_ptr[i];
    }
    double mean_host = sum_host / n_mechanical_sites_local;
    if (rank_MPI == 0) {
    std::cout << "Media (host, Y_loc_ptr): " << mean_host << std::endl;
    }
    
    out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones sin Kokkos evaluate_DV_i_Dq_u_adp_MgHx: = " << time7 << " Resultados: "<< mean_host <<"\n";
    str = out.str();
    MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
    out.str("");

  } else {

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

    SoA_ADP H;
    initSoA_ADP(adp_MgMg, adp_HH, adp_MgH, H);

    Kokkos::fence();

    SoADevice snap;
    copyHostToDevice(H, snap);

    Kokkos::fence();

    DevSnap devSnap("devSnap");

    {
      auto host_mv = Kokkos::create_mirror_view(devSnap);
      host_mv()   = snap;
      Kokkos::deep_copy(devSnap, host_mv);
    }

    Kokkos::fence();
    DevSnapUnmanaged devSnapUM(devSnap.data());
    Kokkos::fence();



    View_Double_Matrix_Device mean_q_ij1_all("mean_q_ij1_all", n_mechanical_sites_local, 6);

    Kokkos::Timer timer;
     Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_mechanical_sites_local), KOKKOS_LAMBDA(PetscInt mech_site_u) {
        PetscInt site_u = active_mech_sites_Kokkos_Default(mech_site_u);
    
        auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all, mech_site_u, Kokkos::ALL());

         mf_rho_Default(site_u) = evaluate_rho_i_adp_MgHx_kokkos_Device(
            site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, atomSpecie_Kokkos_Default, atomTopologyKokkos_Kokkos_Default(site_u), mean_q_ij1, devSnapUM);    
    });

  Kokkos::fence();
  PetscBarrier((PetscObject)NULL);
  double time = timer.seconds();
  if (rank_MPI == 0){
  std::cout << "Kokkos backend: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
  std::cout << "Tiempo que ha tardado en las operaciones con Kokkos: " << time << " seconds" << std::endl;
  }

  

// Copy back to host to migrateGhostfield
//! Migrate ghost field (energy density) con Kokkos
auto mf_rho_Host_copy = Kokkos::create_mirror_view(mf_rho_Default);
Kokkos::deep_copy(mf_rho_Host_copy, mf_rho_Default);
PetscScalar* mf_rho_ptr_Kokkos = mf_rho_Host_copy.data();


PetscCall(DMSwarmMigrateGhostField(n_sites_local, n_sites_ghost, 1,
                                   &idx_q_ptr[n_sites_local], mf_rho_ptr_Kokkos));

Kokkos::deep_copy(mf_rho_Default, mf_rho_Host_copy);    

auto mf_rho_Mirrow = Kokkos::create_mirror_view(mf_rho_Default);
Kokkos::deep_copy(mf_rho_Mirrow, mf_rho_Default);

Eigen::Map<VectorType> mf_rho_test(mf_rho_Mirrow.data(), n_sites_local_ghosted);

double Kokkos_mean_test = mf_rho_test.mean();
double Kokkos_variance_test = ((mf_rho_test.array() - Kokkos_mean_test).square().sum())
                              / static_cast<double>(n_sites_local_ghosted - 1);
if (rank_MPI == 0){

std::cout << "Kokkos: Media = " << Kokkos_mean_test
          << ", Varianza = " << Kokkos_variance_test << std::endl;
}

out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos evaluate_rho_i_adp_MgHx_kokkos: = " << time << " Resultados: "<< Kokkos_mean_test <<"\n";
std::string str = out.str();
MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
out.str("");   
  
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Evaluate free entropy
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
double V_local_Kokkos = 0.0;
View_Double_Matrix_Device mean_q_ij1_all_n_local("mean_q_ij1_all", n_sites_local, 6);

Kokkos::Timer timer2;
Kokkos::parallel_reduce(
    "EvaluatePotentialEnergy", 
    Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
    KOKKOS_LAMBDA(const PetscInt site_u, double& V_u) {
      auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, site_u, Kokkos::ALL());
        V_u += evaluate_V_i_adp_MgHx_Kokkos(
            site_u, mean_q_Kokkos_Default, xi_Kokkos_Default, mf_rho_Default, 
            atomSpecie_Kokkos_Default, atomTopologyKokkos_Kokkos_Default(site_u), devSnapUM, mean_q_ij1);
    },
    V_local_Kokkos);

  if (rank_MPI == 0){  
    std::cout << "Acabe potencial en Kokkos: " << V_local_Kokkos << std::endl;
  }

  out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos evaluate_V_i_adp_MgHx: = " << timer2.seconds() << " Resultados: "<< V_local_Kokkos <<"\n";
  str = out.str();
  MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
  out.str("");

  PetscScalar_Vector_Host stdv_q_ptr_Kokkos_Host(stdv_q_ptr, static_cast<size_t>(n_sites_local_ghosted));
  PetscScalar_Vector_Default stdv_q_ptr_Kokkos_Default("Device_mechanical_sites", static_cast<size_t>(n_sites_local_ghosted));
  Kokkos::deep_copy(stdv_q_ptr_Kokkos_Default, stdv_q_ptr_Kokkos_Host);
  

  /* Inicialización de gaussianMeasures*/

    int MaxNumSites = 3;
    Int_Matrix_Default dof_table_view  ("dof_table",  n_sites_local, MaxNumSites * MaxNumSites);
    Int_Matrix_Default gp_board_view   ("gp_board",   n_sites_local, MaxNumSites * MaxNumSites * NumberDimensions * NumberDimensions);
    Int_Matrix_Default dof_table_aux  ("dof_table",  n_sites_local, MaxNumSites * MaxNumSites);
    Int_Matrix_Default active_dof   ("gp_board",   n_sites_local, MaxNumSites);

    gaussian_measure_ctx_Default ctx("ctx", n_sites_local);

    initGaussianContexts(ctx,
                        dof_table_view,
                        gp_board_view,
                        dof_table_aux,
                        active_dof
                        );

  Kokkos::Timer timer4;
  Kokkos::parallel_for("Active_mech_sites", Kokkos::RangePolicy<DefaultExecSpace, IndexType>(0, n_sites_local), KOKKOS_LAMBDA(PetscInt n_sites_local_u) {

    auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, n_sites_local_u, Kokkos::ALL());
    double result = 0.0;
    result = evaluate_mf_rho_i_adp_MgHx_Kokkos(
      n_sites_local_u, mean_q_Kokkos_Default, stdv_q_ptr_Kokkos_Default,
       xi_Kokkos_Default, atomSpecie_Kokkos_Default, 
       atomTopologyKokkos_Kokkos_Default(n_sites_local_u), 
       mean_q_ij1, devSnapUM,
      ctx(n_sites_local_u));

    mf_rho_Default(n_sites_local_u) =  result; 
}); 

double Kokkos_mean_test1; 
double Kokkos_variance_test1;
if (rank_MPI == 0){
std::cout << "Tiempo que ha tardado en las operaciones con Kokkos: " << timer4.seconds() << " seconds" << std::endl;

auto mf_rho_Mirrow1 = Kokkos::create_mirror_view(mf_rho_Default);
Kokkos::deep_copy(mf_rho_Mirrow1, mf_rho_Default);

Eigen::Map<VectorType> mf_rho_test1(mf_rho_Mirrow1.data(), n_sites_local_ghosted);

 Kokkos_mean_test1 = mf_rho_test1.mean();
 Kokkos_variance_test1 = ((mf_rho_test1.array() - Kokkos_mean_test1).square().sum())
                               / static_cast<double>(n_sites_local_ghosted - 1);

std::cout << "Kokkos: Media = " << Kokkos_mean_test1
          << ", Varianza = " << Kokkos_variance_test1 << std::endl;
}

  out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos evaluate_mf_rho_i_adp_MgHx_Kokkos: = " << timer4.seconds() << " Resultados: "<< Kokkos_mean_test1 <<"\n";
  str = out.str();
  MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
  out.str("");

  PetscScalar_Vector_Host beta_ptr_Kokkos_Host(beta_ptr, static_cast<size_t>(n_sites_local_ghosted));
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
  //PetscLogEvent  myWorkEvent;
  //PetscLogEventRegister("EvaluateFreeEntropy", PETSC_VIEWER_CLASSID, &myWorkEvent);
  //PetscLogEventBegin(myWorkEvent, 0,0,0,0);
  Kokkos::Timer timer6;
  Kokkos::parallel_reduce(
      "EvaluateFreeEntropy", 
      Kokkos::RangePolicy<DefaultExecSpace>(0, n_sites_local),
      KOKKOS_LAMBDA(const PetscInt site_u, double& local_entropy) {

        auto mean_q_ij1 = Kokkos::subview(mean_q_ij1_all_n_local, site_u, Kokkos::ALL());

          double S0_u = evaluate_S0_i_adp_MgHx_Kokkos(
              site_u, mean_q_Kokkos_Default, stdv_q_ptr_Kokkos_Default, xi_Kokkos_Default, 
              mf_rho_Default, beta_ptr_Kokkos_Default, gamma_ptr_Kokkos_Default, atomSpecie_Kokkos_Default, 
              atomTopologyKokkos_Kokkos_Default(site_u), mean_q_ij1, devSnapUM, element_mass_Device, ctx(site_u));
  
          //! @brief Update local contribution of the residual equation
          local_entropy += k_B * S0_u;
      },
      L0_Local_Kokkos);
  //PetscLogEventEnd(myWorkEvent,   0,0,0,0);
  double time6 = timer6.seconds();
  if (rank_MPI == 0){
  std::cout << "Tiempo con Kokkos (timer6): " << time6 << " seconds" << std::endl;
  std::cout << "L0_Local_Kokkos: " << L0_Local_Kokkos << std::endl;  
  }

  out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos evaluate_S0_i_adp_MgHx_Kokkos: = " << time6 << " Resultados: "<< L0_Local_Kokkos <<"\n";
  str = out.str();
  MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
  out.str("");


  unsigned int dim = NumberDimensions;

  Vec X_dist, X_loc;
  Vec Y_dist, Y_loc;  
  PetscScalar* X_loc_ptr;
  PetscScalar* Y_loc_ptr;


  PetscInt n_dof_local = dim * n_sites_local;
  PetscInt n_dof_ghost = dim * n_sites_ghost;
  PetscInt* idx_dof_ghost = (PetscInt*)malloc(n_dof_ghost * sizeof(PetscInt));
  for (int i = 0; i < n_sites_ghost; i++) {
    for (int j = 0; j < dim; j++) {
      idx_dof_ghost[i * dim + j] = idx_q_ptr[n_sites_local + i] * dim + j;
    }
  }

  PetscCall(VecCreateGhost(PETSC_COMM_WORLD, n_dof_local, PETSC_DETERMINE,
                           n_dof_ghost, idx_dof_ghost, &X_dist));
  PetscCall(VecSetUp(X_dist));

  PetscCall(VecGhostGetLocalForm(X_dist, &X_loc));
  PetscCall(VecGetArray(X_loc, &X_loc_ptr));
  
#pragma omp parallel for schedule(runtime)
  for (PetscInt site_u = 0; site_u < n_sites_local_ghosted; site_u++) {
    for (PetscInt alpha = 0; alpha < dim; alpha++) {
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
       mech_site_u++) {

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
        
  PetscInt Y_loc_size; 
  PetscCall(VecGetLocalSize(Y_loc, &Y_loc_size));

  PetscScalar_Vector_Host Y_loc_view(Y_loc_ptr, static_cast<size_t>(Y_loc_size));
  PetscScalar_Vector_Default Y_loc_view_device("Y_loc_view_device", static_cast<size_t>(Y_loc_size));
  Kokkos::deep_copy(Y_loc_view_device, Y_loc_view);


  ThreeD_Double_View aux_view("aux_view", 3, 11, n_mechanical_sites_local);
  Kokkos::deep_copy(aux_view, 0.0);
      
  Kokkos::Timer timer8;
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

        //! @brief Evaluate gradient potential at site u
        {
          auto temp = evaluate_DV_i_Dq_u_adp_MgHx_Kokkos(
                site_u, site_u, mean_q_Kokkos_Default, xi_Kokkos_Default,
                mf_rho_Default, atomSpecie_Kokkos_Default,
                atomTopologyKokkos_Kokkos_Default(site_u), mean_q_ij1, aux_view, devSnapUM);

                dV_dq_u[0] += temp(0);
                dV_dq_u[1] += temp(1);
                dV_dq_u[2] += temp(2);
            }
    
        //! @brief Evaluate the local gradient of V0 at the neighbors of site u
        for (PetscInt idx_i = 0; idx_i < atomTopologyKokkos_Kokkos_Default(site_u).numneigh; idx_i++) {
            //! Get index of the site i
            PetscInt site_i = atomTopologyKokkos_Kokkos_Default(site_u).mech_neighs_ptr(idx_i);
    
            //! @brief Evaluate gradient potential at site i
            auto temp = evaluate_DV_i_Dq_u_adp_MgHx_Kokkos(
                site_u, site_i, mean_q_Kokkos_Default, xi_Kokkos_Default,
                mf_rho_Default, atomSpecie_Kokkos_Default,
                atomTopologyKokkos_Kokkos_Default(site_i), mean_q_ij1, aux_view, devSnapUM);
                dV_dq_u[0] += temp(0);
                dV_dq_u[1] += temp(1);
                dV_dq_u[2] += temp(2);
            }
    
            //! Fill residual vector
        for (PetscInt alpha = 0; alpha < dim; alpha++) {
            Y_loc_view_device(site_u * dim + alpha) = dV_dq_u[alpha];
        }
    });

  Kokkos::fence();

double time8 = timer8.seconds();
double sum_device = 0.0;
Kokkos::parallel_reduce("SumY_loc", Kokkos::RangePolicy<DefaultExecSpace>(0, n_mechanical_sites_local),
  KOKKOS_LAMBDA(const size_t i, double &localSum) {
    localSum += Y_loc_view_device(i);
}, sum_device);

  double mean_device = sum_device / n_mechanical_sites_local;
  Kokkos::fence();

if (rank_MPI == 0){
  Kokkos::printf("Tiempo con Kokkos (timer8): %f seconds \n" , time8);
  Kokkos::printf("Media (device, Y_loc_view_device): %f\n", mean_device);
}

  out << "Rank " << rank_MPI << ": Tiempo que ha tardado en las operaciones con Kokkos evaluate_DV_i_Dq_u_adp_MgHx_Kokkos: = " << time8 << " Resultados: "<< mean_device <<"\n";
  str = out.str();
  MPI_File_write_ordered(outputFile, str.c_str(), str.size(), MPI_CHAR, MPI_STATUS_IGNORE);
  out.str("");
}
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
    limpiarADPPotencial(adp_MgMg);
    limpiarADPPotencial(adp_HH);
    limpiarADPPotencial(adp_MgH);
    // Finalize PETSc
    PetscFinalize();
    // Finalize MPI
#ifdef USE_MPI
    MPI_Finalize();
#endif
  }
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
