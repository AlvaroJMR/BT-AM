#include <iostream>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include "ADP/MgHx-mf-V-bulk.hpp"
#include "Atoms/Atom.hpp"
#include "Atoms/Ghosts.hpp"
#include "Atoms/Neighbors.hpp"
#include "Atoms/Topology.hpp"
#include "IO/dump-input.hpp"
#include "Macros.hpp"
#include "Mechanical-eqs/Mechanical-Relaxation-bulk.hpp"
#include "Numerical/cubic-spline.hpp"
#include "Periodic-Boundary/boundary_conditions.hpp"
#include "Variables.hpp"
#include <petscksp.h>
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

    dmd_equations system_equations = DMD_MgHx_constructor();

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

    Vec xi;
    Kokkos::View<PetscScalar *, Kokkos::Cuda> k_xi;

    PetscCall(DMSwarmCreateLocalVectorFromField(Simulation.atomistic_data,
                                                "molar-fraction", &xi));
    PetscCall(VecGetKokkosView(xi, &k_xi));



    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Output data
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscCall(
        DMSwarmViewXDMF(Simulation.atomistic_data,
                        "outputs/Mg-hcp-cube-x20-x15-x15-periodic-1.xmf"));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Delete the list of active mechanical sites
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscInt n_mechanical_sites_local = 0;
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
    destroy_adp_MgHx(&adp_MgMg);
    destroy_adp_MgHx(&adp_HH);
    destroy_adp_MgHx(&adp_MgH);

    // Finalize PETSc
    PetscFinalize();

    // Finalize MPI
#ifdef USE_MPI
    MPI_Finalize();
#endif

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
