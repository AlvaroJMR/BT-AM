/**
 * @file MgHx-mf-V-bulk.hpp
 * @author Miguel Molinos (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2023-05-19
 *
 * @copyright Copyright (c) 2023
 *
 */

 #ifndef mf_ADP_MgHx_kokkos_HPP
 #define mf_ADP_MgHx_kokkos_HPP
 
 #include "ADP/MgHx-ADP.hpp"
 #include "Macros.hpp"
 #include <Eigen/Dense>
 #include <Kokkos_Core.hpp>
 #include <Kokkos_SIMD.hpp>
 #include <Utils/KokkosUtils.hpp>
 #include "Numerical/Quadrature-Multipole.hpp"
 #include "Numerical/Quadrature-Hermitian-3th.hpp"
 
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


    KOKKOS_INLINE_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos_Device_SIMD(unsigned int site_i,
    const PetscScalar_Matrix_Default  &mean_q,
    const PetscScalar_Vector_Default &xi,
    const AtomSpecie_Default &specie,
    const AtomTopology atom_topology_i,
    const DevSnapUnmanaged &soADevice) {
    
    //! @brief Auxiliar variables
    using simd_t = Kokkos::Experimental::native_simd<double>;
    using simd_int_t = Kokkos::Experimental::native_simd<int>;
    double rho_i = 0.0;
    constexpr int W   = simd_t::size();

    //! @brief Get topologic information of site i
    unsigned int numneigh_site_i = atom_topology_i.numneigh;
    const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;
    CubicSpline rho_j_Mg = getSpline(AdpType::MgMg, SplineType::rho, CubicSpline(), soADevice.data());
    CubicSpline rho_j_H  = getSpline(AdpType::HH,  SplineType::rho, CubicSpline(), soADevice.data());
  
    //! @brief Get atomistic information of site i
    AtomicSpecie spc_i = specie(site_i);
    double xi_i = xi(site_i);
    double q_i0 = mean_q(site_i,0);
    double q_i1 = mean_q(site_i,1);
    double q_i2 = mean_q(site_i,2);
    simd_t qi0(q_i0);
    simd_t qi1(q_i1);
    simd_t qi2(q_i2);

    //! If the site is empty, skip from the evaluation
    if (xi_i < min_occupancy) {
      return 0.0;
    }
  
    //! @brief Compute the gradient of the potential with respect the
    //! mean value of the position at site i
    unsigned int idx_j1 = 0;
    for (; idx_j1 + W < 1; idx_j1+= W) {

      simd_int_t sites_j1;
      sites_j1.copy_from(mech_neighs_i + idx_j1, Kokkos::Experimental::simd_flag_default);
      AtomicSpecie spc_arr[W];
      simd_t qj0, qj1, qj2, n;
      for (int lane = 0; lane < W; ++lane) {
        qj0[lane] = mean_q(sites_j1[lane],0);
        qj1[lane] = mean_q(sites_j1[lane],1);
        qj2[lane] = mean_q(sites_j1[lane],2);
        n[lane]   = xi(sites_j1[lane]);
        spc_arr[lane] = specie(sites_j1[lane]);
      }
  
      simd_t dr0 = qi0 - qj0;
      simd_t dr1 = qi1 - qj1;
      simd_t dr2 = qi2 - qj2;

      simd_t r2  = Kokkos::fma(dr0, dr0,
                   Kokkos::fma(dr1, dr1,
                                             dr2*dr2));
      simd_t norm = Kokkos::sqrt(r2);

      CubicSpline rho_j[W];
      for (int lane = 0; lane < W; ++lane) {
        rho_j[lane] = (spc_arr[lane] == Mg) ? rho_j_Mg : rho_j_H;
      }
      simd_t val = n * cubic_spline_Kokkos_SIMD<W>(rho_j, norm);
        
    }

    for (; idx_j1 < numneigh_site_i; ++idx_j1) {
      unsigned site_j = mech_neighs_i[idx_j1];
      double d0 = q_i0 - mean_q(site_j,0);
      double d1 = q_i1 - mean_q(site_j,1);
      double d2 = q_i2 - mean_q(site_j,2);
      double norm = std::sqrt(d0*d0 + d1*d1 + d2*d2);
      CubicSpline spline = (specie(site_j) == Mg
                          ? rho_j_Mg
                          : rho_j_H);
      double val = cubic_spline_Kokkos(&spline, norm);
      rho_i += xi(site_j) * val;
    }
    
    return rho_i;
  }
 
  KOKKOS_INLINE_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos_Device(unsigned int site_i,
    const PetscScalar_Matrix_Default  &mean_q,
    const PetscScalar_Vector_Default &xi,
    const AtomSpecie_Default &specie,
    const AtomTopology atom_topology_i,
    const DevSnapUnmanaged &soADevice) {
    
    //! @brief Auxiliar variables
    double rho_i = 0.0;
  
    //! @brief Get topologic information of site i
    unsigned int numneigh_site_i = atom_topology_i.numneigh;
    const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;
  
    //! @brief Get atomistic information of site i
    AtomicSpecie spc_i = specie(site_i);
    double xi_i = xi(site_i);
    auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
    
    //! If the site is empty, skip from the evaluation
    if (xi_i < min_occupancy) {
      return 0.0;
    }
  
    //! @brief Compute the gradient of the potential with respect the
    //! mean value of the position at site i
    for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {
      //! @brief Get atomistic information of site j
  
      unsigned int site_j1 = mech_neighs_i[idx_j1];
      AtomicSpecie spc_j1 = specie(site_j1);
      double xi_j1 = xi(site_j1);
      auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);
  
      //! If the site is empty, skip from the evaluation
      if (xi_j1 < min_occupancy) {
        continue;
      }
  
      //! @brief Create dof table
      int dof_table_ij[4] = {1, 0, 0, 1};

      double mean_q_ij1[6]; 
  
      //! @brief Fill data for the measure (i,j1)
      concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
      Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
      Kokkos::Array<unsigned int, 2>  sites_ij1 = Kokkos::Array<unsigned int, 2> {site_i, site_j1};
      AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
  
       //! @brief Compute energy density
      double rho_ij = 0.0;
  
      //! Create functions
      rho_ij_adp_MgHx_dispatcher functions_rho_ij { Functions_Enum::FK, &rho_ij, xi_ij1.data(), mean_q_ij1, spc_ij1, soADevice.data() };
      
      functions_rho_ij();
      rho_i += rho_ij;
    } 
    
    return rho_i;
  }
 
 KOKKOS_FUNCTION double evaluate_rho_i_adp_MgHx_kokkos(unsigned int site_i,           //!
     const Eigen::MatrixXd &mean_q, //!
     const Eigen::VectorXd &xi,     //!
     const AtomicSpecie *specie,    //!
     const AtomTopology atom_topology_i);
 
 double evaluate_V_i_adp_MgHx(unsigned int site_i,                 //!
     const Eigen::MatrixXd& mean_q,       //!
     const Eigen::VectorXd& xi,           //!
     const Eigen::VectorXd& rho,          //!
     const AtomicSpecie* specie,          //!
     const AtomTopology atom_topology_i);
 
 KOKKOS_FUNCTION double evaluate_V_i_adp_MgHx_Kokkos(unsigned int site_i,                 //!
     const PetscScalar_Matrix_Default  &mean_q,
     const PetscScalar_Vector_Default &xi,
     const PetscScalar_Vector_Default& rho,
     const AtomSpecie_Default &specie,
     const AtomTopology atom_topology_i,
     const DevSnapUnmanaged &soADevice);
       
 
 double evaluate_mf_rho_i_adp_MgHx(unsigned int site_i,            //!
     const Eigen::MatrixXd& mean_q,  //!
     const Eigen::VectorXd& stdv_q,  //!
     const Eigen::VectorXd& xi,      //!
     const AtomicSpecie* specie,     //!
     const AtomTopology atom_topology_i);
 
 KOKKOS_FUNCTION double evaluate_mf_rho_i_adp_MgHx_Kokkos(unsigned int site_i,                 //!
   const PetscScalar_Matrix_Default  &mean_q,
   const PetscScalar_Vector_Default &stdv_q,
   const PetscScalar_Vector_Default &xi,
   const AtomSpecie_Default &specie,
   const AtomTopology atom_topology_i,
   const DevSnapUnmanaged &soADevice,
   const gaussian_measure_ctx_kokkos &ctx,
   const bool multipole_integral  //! If the integral is multipole or GH3TH  
   );
     
 double evaluate_S0_i_adp_MgHx(unsigned int site_i,                 //!
     const Eigen::MatrixXd& mean_q,       //!
     const Eigen::VectorXd& stdv_q,       //!
     const Eigen::VectorXd& xi,           //!
     const Eigen::VectorXd& mf_rho,       //!
     const Eigen::VectorXd& beta,         //!
     const Eigen::VectorXd& gamma,        //!
     const AtomicSpecie* specie,          //!
     const AtomTopology atom_topology_i); 
      
 Eigen::Vector3d evaluate_DV_i_Dq_u_adp_MgHx(
     unsigned int site_i_star,            //!
     unsigned int site_i,                 //!
     const Eigen::MatrixXd& mean_q,       //! Mean value of q
     const Eigen::VectorXd& xi,           //! Molar fraction
     const Eigen::VectorXd& rho,          //! Energy density
     const AtomicSpecie* specie,          //! Atom
     const AtomTopology atom_topology_i);
         
 KOKKOS_FUNCTION aux_Vector evaluate_DV_i_Dq_u_adp_MgHx_Kokkos(
     unsigned int site_i_star,            //!
     unsigned int site_i,                 //!
     const PetscScalar_Matrix_Default  &mean_q,       //! Mean value of q
     const PetscScalar_Vector_Default &xi,           //! Molar fraction
     const PetscScalar_Vector_Default &mf_rho,          //! Energy density
     const AtomSpecie_Default &specie,          //! Atom
     const AtomTopology atom_topology_i,
     const aux_Vector mean_q_ij1, 
     const ThreeD_Double_View aux_view,
     const DevSnapUnmanaged &soADevice); 

 KOKKOS_FUNCTION Kokkos::Array<double, 3> evaluate_DV_i_Dq_u_adp_MgHx_Kokkos_SIMD(
   unsigned int site_i_star,            //!
   unsigned int site_i,                 //!
   const PetscScalar_Matrix_Default  &mean_q,       //! Mean value of q
   const PetscScalar_Vector_Default &xi,           //! Molar fraction
   const PetscScalar_Vector_Default &mf_rho,          //! Energy density
   const AtomSpecie_Default &specie,          //! Atom
   const AtomTopology atom_topology_i,
   const DevSnapUnmanaged &soADevice);     

 KOKKOS_FUNCTION double evaluate_V_i_adp_MgHx_Kokkos_SIMD(unsigned int site_i,                 //!
   const PetscScalar_Matrix_Default  &mean_q,
   const PetscScalar_Vector_Default &xi,
   const PetscScalar_Vector_Default& rho,
   const AtomSpecie_Default &specie,
   const AtomTopology atom_topology_i,
   const DevSnapUnmanaged &soADevice);  
     
KOKKOS_INLINE_FUNCTION double evaluate_S0_i_adp_MgHx_Kokkos(unsigned int site_i, 
        const PetscScalar_Matrix_Default  &mean_q,
        const PetscScalar_Vector_Default &stdv_q,
        const PetscScalar_Vector_Default &xi,
        const PetscScalar_Vector_Default &mf_rho,       //!
        const PetscScalar_Vector_Default &beta,         //!
        const PetscScalar_Vector_Default &gamma,        //!
        const AtomSpecie_Default &specie,
        const AtomTopology atom_topology_i,
        const DevSnapUnmanaged &soADevice,
        const View_Double_Vector_Device element_mass,
        const gaussian_measure_ctx_kokkos &ctx,
        const bool multipole_integral  //! If the integral is multipole or GH3TH )  //! 
        )
                                                                                        
      {
      
      unsigned int dim = NumberDimensions;
      
      //! Define Integration rule
    
      //! Local variables
      double mf_rho_i = mf_rho(site_i);
      double V0_embed_i = 0.0;  //! Meanfield Embedded forces term
      double V0_pair_i = 0.0;   //! Meanfield Pairing forces term
      double V0_dip_i = 0.0;    //! Meanfield Dipole distortion term
      double V0_quad_i = 0.0;   //! Meanfield Quadrupole distortion term
      double V0_i = 0.0;        //! Total potential
      
      //! @brief Get topologic information of site i
      unsigned int numneigh_site_i = atom_topology_i.numneigh;
      const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;
      
      //! @brief Get atomistic information of site i
      auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
      double stdv_q_i = stdv_q(site_i);
      double xi_i = xi(site_i);
      double beta_i = beta(site_i);
      double gamma_i = gamma(site_i);
      AtomicSpecie spc_i = specie(site_i);
      double m_i = unit_change_uma * xi_i * element_mass(spc_i);

      double mean_q_ij1 [9];
      
      //! If the site is empty, skip from the evaluation
      if (xi_i < min_occupancy) {
      return 0.0;
      }
      
      for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {
      
      //! @brief Get atomistic information of site j
      unsigned int site_j1 = mech_neighs_i[idx_j1];
      AtomicSpecie spc_j1 = specie(site_j1);
      double xi_j1 = xi(site_j1);
      double stdv_q_j1 = stdv_q(site_j1);
      auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);
      
      //! If the site is empty, skip from the evaluation
      if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
      }
      
      //! @brief Fill data for the measure
      concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
      Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
      double stdv_q_ij1[2] = {stdv_q_i, stdv_q_j1};
      AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
      
      
      //! @brief Create dof table for i-j par
      int dof_table_ij[4] = {1, 0, 0, 1};
      
      //! @brief Create measure/functions
      
      fill_out_gaussian_measure_Kokkos(mean_q_ij1, stdv_q_ij1,
        xi_ij1.data(), spc_ij1, dof_table_ij, 2, &ctx);      
            
      //! @brief Compute meanfield pairing term
      double V0_pair_ij = 0.0;
      { 
      V_pair_ij_adp_MgHx_dispatcher function;
      
      if (multipole_integral) {
        meanfield_integral_mp_Kokkos<V_pair_ij_adp_MgHx_dispatcher>(&V0_pair_ij, &ctx, soADevice.data(), function);
      } else {
        meanfield_integral_gh3th_Kokkos<V_pair_ij_adp_MgHx_dispatcher>(&V0_pair_ij, &ctx, soADevice.data(), function);
      } 
      }
      V0_pair_i += V0_pair_ij;
      
      //! @brief Remove measure
      
      //! @brief Loop in the neighborhood considering the simmetry of the
      //! opration a_ij1 * a_ij2 = a_ij2 * a_ij1
      for (unsigned int idx_j2 = idx_j1; idx_j2 < numneigh_site_i; idx_j2++) {
      
      //! @brief Compute atomistic information of site j2
      unsigned int site_j2 = mech_neighs_i[idx_j2];
      AtomicSpecie spc_j2 = specie(site_j2);
      double xi_j2 = xi(site_j2);
      double stdv_q_j2 = stdv_q(site_j2);
      auto mean_q_j2 = extractRowBlock(mean_q, site_j2, 0, 3);
      
      //! If the site is empty, skip from the evaluation
      if (xi_j2 < min_occupancy) {
      continue;
      }
      
      //! @brief Create dof table for i-j1-j2
      int dof_table_ij1j2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      
      //! If the j1 match j2, avoid integration duplic
      if (site_j1 == site_j2) {
      dof_table_ij1j2[5] = 1;
      dof_table_ij1j2[7] = 1;
      }
      
      //! @brief Fill data for the measure
      auto mean_q_ij1j2 = concatenateToArray<Kokkos::View<double*>, 9>(mean_q_i, mean_q_j1, mean_q_j2);
      Kokkos::Array<double, 3>  xi_ij1j2 = Kokkos::Array<double, 3> {xi_i, xi_j1, xi_j2};
      AtomicSpecie spc_ij1j2[3] = {spc_i, spc_j1, spc_j2};
      double stdv_q_ij1j2[3] = {stdv_q_i, stdv_q_j1, stdv_q_j2};
      
      //! @brief Factor to consider the simmetry of the opration a_ij*a_ik =
      //! a_ik*a_ij
      double factor_j1j2 = (idx_j2 == idx_j1) ? 1.0 : 2.0;
      
      fill_out_gaussian_measure_Kokkos(mean_q_ij1j2.data(), stdv_q_ij1j2,
        xi_ij1j2.data(), spc_ij1j2, dof_table_ij1j2, 3, &ctx);
      
      //! Compute meanfield dipole angular term
      double V0_dip_ij1j2 = 0.0;
      { 
      V_dipole_ij1j2_dispatcher function_V_dipole_ij1j2;
      if (multipole_integral) {
        meanfield_integral_mp_Kokkos<V_dipole_ij1j2_dispatcher>(&V0_dip_ij1j2, &ctx, soADevice.data(), function_V_dipole_ij1j2);
      } else {
        meanfield_integral_gh3th_Kokkos<V_dipole_ij1j2_dispatcher>(&V0_dip_ij1j2, &ctx, soADevice.data(), function_V_dipole_ij1j2);
      }
      }
      V0_dip_i += factor_j1j2 * V0_dip_ij1j2;
      
      //! Compute meanfield quadrupole angular term
      double V0_quad_ij1j2 = 0.0;
      {
      V_quadrupole_ij1j2_dispatcher function_V_quadrupole_ij1j2;
      if (multipole_integral) {
        meanfield_integral_mp_Kokkos<V_quadrupole_ij1j2_dispatcher>(&V0_quad_ij1j2, &ctx, soADevice.data(), function_V_quadrupole_ij1j2);
      } else {
        meanfield_integral_gh3th_Kokkos<V_quadrupole_ij1j2_dispatcher>(&V0_quad_ij1j2, &ctx, soADevice.data(), function_V_quadrupole_ij1j2);
      }
      }
      V0_quad_i += factor_j1j2 * V0_quad_ij1j2;
      
      }
      }
    //  Kokkos::printf("Time for loop in the neighborhood si Kokkos= %f\n", timer11.seconds());

      //! @brief Add the contribution of the embedded forces
      CubicSpline embed_ii;
      if (spc_i == Mg) {
      embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, soADevice.data());
      } else if (spc_i == H) {
      embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, soADevice.data());
      }
      double mf_F_i = cubic_spline_Kokkos(&embed_ii, mf_rho_i);
      V0_embed_i = xi_i * mf_F_i;
      
      //! @brief Add up each contribution to the meanfield potential
      V0_i = V0_embed_i + V0_pair_i + V0_dip_i + V0_quad_i;
      
      //! @brief Compute mean meanfield Hamiltonian
      double H0_i = 1.0 / (2.0 * beta_i) + V0_i;
      
      
      //! @brief Compute log of the grand-cannonical partition function
      double log_Z0_i = 3.0 * log((stdv_q_i * sqrt(m_i / beta_i)) / h_planck);
      if (xi_i < 1.0) {
      log_Z0_i += log(1.0 / (1.0 - xi_i));
      }
      
      //! @brief Chemical multiplier
      double gamma_xi_i = 0.0;
      if (spc_i == H) {
      gamma_xi_i = gamma_i * (xi_i > 0.9999 ? 0.9999 : xi_i);
      }
      
      //! @brief Compute the meanfield Entropy
      double S0_i = -log_Z0_i + beta_i * H0_i - gamma_xi_i;
      
      return S0_i;
      }


KOKKOS_INLINE_FUNCTION double evaluate_S0_i_adp_MgHx_Kokkos_SIMD(unsigned int site_i, 
        const PetscScalar_Matrix_Default  &mean_q,
        const PetscScalar_Vector_Default &stdv_q,
        const PetscScalar_Vector_Default &xi,
        const PetscScalar_Vector_Default &mf_rho,       //!
        const PetscScalar_Vector_Default &beta,         //!
        const PetscScalar_Vector_Default &gamma,        //!
        const AtomSpecie_Default &specie,
        const AtomTopology atom_topology_i,
        const DevSnapUnmanaged &soADevice,
        const View_Double_Vector_Device element_mass,
        gaussian_measure_ctx_kokkos_s ctxs[],
        const bool multipole_integral  //! If the integral is multipole or GH3TH )  //! 
        )
                                                                                        
      {

      using simd_t = Kokkos::Experimental::native_simd<double>;
      constexpr int W   = simd_t::size();        
      
      unsigned int dim = NumberDimensions;
      
      //! Define Integration rule
    
      //! Local variables
      double mf_rho_i = mf_rho(site_i);
      double V0_embed_i = 0.0;  //! Meanfield Embedded forces term
      double V0_pair_i = 0.0;   //! Meanfield Pairing forces term
      double V0_dip_i = 0.0;    //! Meanfield Dipole distortion term
      double V0_quad_i = 0.0;   //! Meanfield Quadrupole distortion term
      double V0_i = 0.0;        //! Total potential
      
      //! @brief Get topologic information of site i
      unsigned int numneigh_site_i = atom_topology_i.numneigh;
      const PetscInt* mech_neighs_i = atom_topology_i.mech_neighs_ptr;
      
      //! @brief Get atomistic information of site i
      auto mean_q_i = extractRowBlock(mean_q, site_i, 0, 3);
      double stdv_q_i = stdv_q(site_i);
      double xi_i = xi(site_i);
      double beta_i = beta(site_i);
      double gamma_i = gamma(site_i);
      AtomicSpecie spc_i = specie(site_i);
      double m_i = unit_change_uma * xi_i * element_mass(spc_i);

      double mean_q_ij1 [9];
      
      //! If the site is empty, skip from the evaluation
      if (xi_i < min_occupancy) {
      return 0.0;
      }
      
      for (unsigned int idx_j1 = 0; idx_j1 < numneigh_site_i; idx_j1++) {
      
      //! @brief Get atomistic information of site j
      unsigned int site_j1 = mech_neighs_i[idx_j1];
      AtomicSpecie spc_j1 = specie(site_j1);
      double xi_j1 = xi(site_j1);
      double stdv_q_j1 = stdv_q(site_j1);
      auto mean_q_j1 = extractRowBlock(mean_q, site_j1, 0, 3);
      
      //! If the site is empty, skip from the evaluation
      if ((spc_j1 == H) && (xi_j1 < min_occupancy)) {
      continue;
      }
      
      //! @brief Fill data for the measure
      concatenateVectors(mean_q_i, mean_q_j1, mean_q_ij1);
      Kokkos::Array<double, 2>  xi_ij1 = Kokkos::Array<double, 2> {xi_i, xi_j1};
      double stdv_q_ij1[2] = {stdv_q_i, stdv_q_j1};
      AtomicSpecie spc_ij1[2] = {spc_i, spc_j1};
      
      
      //! @brief Create dof table for i-j par
      int dof_table_ij[4] = {1, 0, 0, 1};
      
      //! @brief Create measure/functions
      
      fill_out_gaussian_measure_Kokkos_s(mean_q_ij1, stdv_q_ij1,
        xi_ij1.data(), spc_ij1, dof_table_ij, 2, &ctxs[0]);      

      //! @brief Compute meanfield pairing term
      double V0_pair_ij = 0.0;
      { 
      V_pair_ij_adp_MgHx_dispatcher function;
      
      if (multipole_integral) {
        meanfield_integral_mp_Kokkos<V_pair_ij_adp_MgHx_dispatcher>(&V0_pair_ij, &ctxs[0], soADevice.data(), function);
      } else {
        meanfield_integral_gh3th_Kokkos_s<V_pair_ij_adp_MgHx_dispatcher>(&V0_pair_ij, &ctxs[0], soADevice.data(), function);
      }
      }
      V0_pair_i += V0_pair_ij;
      
      for (unsigned int idx_j2 = idx_j1; idx_j2 + W < numneigh_site_i; idx_j2+= W) {
      
      unsigned   sites_j2[W];
      AtomicSpecie spc_j2_arr[W];
      double     xi_j2_arr[W], stdv_q_j2_arr[W];
      double     mean_q_j2_arr[W*3];
      for(int lane=0; lane<W; ++lane) {
        unsigned site = mech_neighs_i[idx_j2+lane];
        sites_j2[lane]      = site;
        spc_j2_arr[lane]    = specie(site);
        xi_j2_arr[lane]     = xi(site);
        stdv_q_j2_arr[lane] = stdv_q(site);
        auto row = extractRowBlock(mean_q, site, 0,3);
        mean_q_j2_arr[3*lane+0] = row[0];
        mean_q_j2_arr[3*lane+1] = row[1];
        mean_q_j2_arr[3*lane+2] = row[2];
      };
      
      int dof_table_arr[W][9];
      for(int lane=0; lane<W; ++lane) {
        for(int k=0; k<9; ++k)
          dof_table_arr[lane][k] = (k==0||k==4||k==8) ? 1 : 0;
        if (sites_j2[lane] == site_j1) {
          dof_table_arr[lane][5] = 1;
          dof_table_arr[lane][7] = 1;
        }
      }
      
    
    for(int lane=0; lane<W; ++lane) {

      double mean_q_ij1j2[9];
      for(int d=0; d<3; ++d) {
        mean_q_ij1j2[3*d+0] = mean_q_i[3*0 + d];
        mean_q_ij1j2[3*d+1] = mean_q_j1[3*0 + d];
        mean_q_ij1j2[3*d+2] = mean_q_j2_arr[3*lane + d];
      }

      Kokkos::Array<double,3> xi_ij1j2 = { xi_i, xi_j1, xi_j2_arr[lane] };
      AtomicSpecie spc_ij1j2[3]      = { spc_i, spc_j1, spc_j2_arr[lane] };
      double stdv_q_ij1j2[3]         = { stdv_q_i, stdv_q_j1, stdv_q_j2_arr[lane] };

      // fill out gaussian_measure
      fill_out_gaussian_measure_Kokkos_s(
        mean_q_ij1j2, stdv_q_ij1j2,
        xi_ij1j2.data(), spc_ij1j2,
        dof_table_arr[lane], 3, &ctxs[lane]
      );
    }

      double V0_dip_ij1j2[W] = {0.0};
      {
      V_dipole_ij1j2_SIMD_dispatcher function_dipole_ij1j2_SIMD;
      V_dipole_ij1j2_dispatcher function_V_dipole_ij1j2;
      if (multipole_integral) {
        meanfield_integral_mp_Kokkos<V_dipole_ij1j2_dispatcher>(&V0_dip_ij1j2[0], &ctxs[0], soADevice.data(), function_V_dipole_ij1j2);
      } else {
        meanfield_integral_gh3th_Kokkos_SIMD<V_dipole_ij1j2_SIMD_dispatcher, V_dipole_ij1j2_dispatcher>(V0_dip_ij1j2, ctxs, soADevice.data(), function_dipole_ij1j2_SIMD, function_V_dipole_ij1j2);
      }
      }
      for (int lane = 0; lane < W; ++lane) {
        double factor_j1j2_lane = ((idx_j2 + lane) == idx_j1) ? 1.0 : 2.0;
        V0_dip_i += factor_j1j2_lane * V0_dip_ij1j2[lane];
      }
      double V0_quad_ij1j2[W] = {0.0};
      {
      V_quadrupole_ij1j2_SIMD_dispatcher function_quadrupole_ij1j2_SIMD;
      V_quadrupole_ij1j2_dispatcher function_V_quadrupole_ij1j2;
      if (multipole_integral) {
        meanfield_integral_mp_Kokkos<V_quadrupole_ij1j2_dispatcher>(&V0_quad_ij1j2[0], &ctxs[0], soADevice.data(), function_V_quadrupole_ij1j2);
      } else {
        meanfield_integral_gh3th_Kokkos_SIMD<V_quadrupole_ij1j2_SIMD_dispatcher, V_quadrupole_ij1j2_dispatcher>(V0_quad_ij1j2, ctxs, soADevice.data(), function_quadrupole_ij1j2_SIMD, function_V_quadrupole_ij1j2);
      }
      }
      for (int lane = 0; lane < W; ++lane) {
      double factor_j1j2_lane = ((idx_j2 + lane) == idx_j1) ? 1.0 : 2.0;
      V0_quad_i += factor_j1j2_lane * V0_quad_ij1j2[lane];
      }
      }
      }
      CubicSpline embed_ii;
      if (spc_i == Mg) {
      embed_ii = getSpline(AdpType::MgMg, SplineType::embed, embed_ii, soADevice.data());
      } else if (spc_i == H) {
      embed_ii = getSpline(AdpType::HH, SplineType::embed, embed_ii, soADevice.data());
      }
      double mf_F_i = cubic_spline_Kokkos(&embed_ii, mf_rho_i);
      V0_embed_i = xi_i * mf_F_i;
      
      V0_i = V0_embed_i + V0_pair_i + V0_dip_i + V0_quad_i;
      
      double H0_i = 1.0 / (2.0 * beta_i) + V0_i;
      
      
      double log_Z0_i = 3.0 * log((stdv_q_i * sqrt(m_i / beta_i)) / h_planck);
      if (xi_i < 1.0) {
      log_Z0_i += log(1.0 / (1.0 - xi_i));
      }
      
      double gamma_xi_i = 0.0;
      if (spc_i == H) {
      gamma_xi_i = gamma_i * (xi_i > 0.9999 ? 0.9999 : xi_i);
      }
      
      double S0_i = -log_Z0_i + beta_i * H0_i - gamma_xi_i;
      
      return S0_i;
      }      
     
 
 #endif /* mf_ADP_MgHx_kokkos_HPP */