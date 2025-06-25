#ifndef KOKKOS_UTILS_HPP
#define KOKKOS_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <array>

template <typename View>
KOKKOS_INLINE_FUNCTION
auto extractRowBlock(const View &matrix, int row, int startCol, int numCols) {
    return Kokkos::subview(matrix, row, std::make_pair(startCol, startCol + numCols));
}

template <typename Input_View>
KOKKOS_INLINE_FUNCTION
void concatenateVectors(const Input_View &v1,
                        const Input_View &v2,
                        double *out)
{
    auto n1 = v1.extent(0);
    auto n2 = v2.extent(0);
    for (size_t i = 0; i < n1; i++) {
        out[i] = v1(i);
    }
    for (size_t i = 0; i < n2; i++) {
        out[n1 + i] = v2(i);
    }
}

template <typename View, size_t size>
KOKKOS_INLINE_FUNCTION
std::array<typename View::value_type, size> concatenateToArray(
    const View& v1, const View& v2, const View& v3) {
    std::array<typename View::value_type, size> out;

    size_t index = 0;

    for (size_t i = 0; i < v1.extent(0); i++) {
        out[index++] = v1(i);
    }

    for (size_t i = 0; i < v2.extent(0); i++) {
        out[index++] = v2(i);
    }

    for (size_t i = 0; i < v3.extent(0); i++) {
        out[index++] = v3(i);
    }

    return out;
}

template<typename DestView, typename Scalar, typename... SrcViews>
KOKKOS_INLINE_FUNCTION
void multi_sum_scaled (const DestView dest,
                      Scalar alpha,
                      const SrcViews&... srcs) {
  const std::size_t n = dest.extent(0);
  for (std::size_t k = 0; k < n; ++k) {
    dest(k) += alpha * (srcs(k) + ...);
  }
}


/* KOKKOS_INLINE_FUNCTION void deep_copy_cubic_spline(CubicSpline &dest, const CubicSpline &src) {
    dest.dx = src.dx;
    dest.n  = src.n;
    dest.x   = View_Double_Vector_Device("dest.x", src.n + 1);
    dest.a   = View_Double_Vector_Device("dest.a", src.n + 1);
    dest.b   = View_Double_Vector_Device("dest.b", src.n + 1);
    dest.c   = View_Double_Vector_Device("dest.c", src.n + 1);
    dest.d   = View_Double_Vector_Device("dest.d", src.n + 1);
    dest.db  = View_Double_Vector_Device("dest.db", src.n + 1);
    dest.dc  = View_Double_Vector_Device("dest.dc", src.n + 1);
    dest.dd  = View_Double_Vector_Device("dest.dd", src.n + 1);
    dest.ddc = View_Double_Vector_Device("dest.ddc", src.n + 1);
    dest.ddd = View_Double_Vector_Device("dest.ddd", src.n + 1);


    Kokkos::deep_copy(dest.x, src.x);
    Kokkos::deep_copy(dest.a, src.a);
    Kokkos::deep_copy(dest.b, src.b);
    Kokkos::deep_copy(dest.c, src.c);
    Kokkos::deep_copy(dest.d, src.d);
    Kokkos::deep_copy(dest.db, src.db);
    Kokkos::deep_copy(dest.dc, src.dc);
    Kokkos::deep_copy(dest.dd, src.dd);
    Kokkos::deep_copy(dest.ddc, src.ddc);
    Kokkos::deep_copy(dest.ddd, src.ddd);

}
  
KOKKOS_INLINE_FUNCTION void deep_copy_adpPotential(adpPotential &dest, const adpPotential &src) {


  dest.n_embed = src.n_embed;
  dest.n_rho = src.n_rho;
  dest.n_pair = src.n_pair;
  dest.n_u = src.n_u;
  dest.n_w = src.n_w;
  dest.mass = src.mass;
  dest.radius = src.radius;
  dest.factor = src.factor;
  dest.r_cutoff = src.r_cutoff;


  deep_copy_cubic_spline(dest.embed, src.embed);
  deep_copy_cubic_spline(dest.rho, src.rho);
  deep_copy_cubic_spline(dest.pair, src.pair);
  deep_copy_cubic_spline(dest.u, src.u);
  deep_copy_cubic_spline(dest.w, src.w);


}
  

KOKKOS_INLINE_FUNCTION void copy_adpPotential_to_device(
  AdpPotencial_Device &adpDevice,
  const adpPotential   &hostAdp,
  int                    pos)
{
  static auto mirror = Kokkos::create_mirror_view(adpDevice);

  Kokkos::deep_copy(mirror, adpDevice);

  if (pos >= 0 && pos < static_cast<int>(mirror.extent(0))) {
    deep_copy_adpPotential(mirror(pos), hostAdp);
  }
  Kokkos::deep_copy(adpDevice, mirror);


} */


/* inline void copy_spline_scalars_device(
  CubicSpline       &dst,
  const CubicSpline &src,
  const char        *label)
{
  Kokkos::printf("Probando escalares\n");

  auto dx_val = src.dx;
  auto n_val  = src.n;

  CubicSpline *p_dst = &dst;

  Kokkos::printf("Debe de fallar aqui\n");

  Kokkos::parallel_for(
    label,
    Kokkos::RangePolicy< Kokkos::DefaultExecutionSpace >(0,1),
    [=] KOKKOS_FUNCTION (int)
    {
      p_dst->dx = dx_val;
      p_dst->n  = n_val;
    }
  );
}

inline void copy_cubic_spline_device(
  CubicSpline       &dst,
  const CubicSpline &src,
  const char        *label = "copy_spline")
{
  Kokkos::printf("Probando\n");

  std::size_t N = std::size_t(src.n) + 1;

  dst.x   = View_Double_Vector_Device("dst.x",   N);
  Kokkos::printf("Probando más\n");
  dst.a   = View_Double_Vector_Device("dst.a",   N);
  dst.b   = View_Double_Vector_Device("dst.b",   N);
  dst.c   = View_Double_Vector_Device("dst.c",   N);
  dst.d   = View_Double_Vector_Device("dst.d",   N);
  dst.db  = View_Double_Vector_Device("dst.db",  N);
  dst.dc  = View_Double_Vector_Device("dst.dc",  N);
  dst.dd  = View_Double_Vector_Device("dst.dd",  N);
  dst.ddc = View_Double_Vector_Device("dst.ddc", N);
  dst.ddd = View_Double_Vector_Device("dst.ddd", N);

  Kokkos::printf("Copiar\n");

  Kokkos::deep_copy(Kokkos::subview(dst.x,   Kokkos::ALL()), src.x);
  Kokkos::deep_copy(Kokkos::subview(dst.a,   Kokkos::ALL()), src.a);
  Kokkos::deep_copy(Kokkos::subview(dst.b,   Kokkos::ALL()), src.b);
  Kokkos::deep_copy(Kokkos::subview(dst.c,   Kokkos::ALL()), src.c);
  Kokkos::deep_copy(Kokkos::subview(dst.d,   Kokkos::ALL()), src.d);
  Kokkos::deep_copy(Kokkos::subview(dst.db,  Kokkos::ALL()), src.db);
  Kokkos::deep_copy(Kokkos::subview(dst.dc,  Kokkos::ALL()), src.dc);
  Kokkos::deep_copy(Kokkos::subview(dst.dd,  Kokkos::ALL()), src.dd);
  Kokkos::deep_copy(Kokkos::subview(dst.ddc, Kokkos::ALL()), src.ddc);
  Kokkos::deep_copy(Kokkos::subview(dst.ddd, Kokkos::ALL()), src.ddd);

  Kokkos::printf("Terminando de probar\n");


  copy_spline_scalars_device(dst, src, label);
}

inline void copy_adpPotential_to_device(
  AdpPotencial_Device &adpDevice,
  const adpPotential  &hostAdp,
  int                   pos)
{
  int M = int(adpDevice.extent(0));
  if (pos<0 || pos>=M) throw std::out_of_range("pos fuera de rango");

  Kokkos::printf("Antes de copiar escalares\n");

  auto n_embed = hostAdp.n_embed;
  auto n_rho   = hostAdp.n_rho;
  auto n_pair  = hostAdp.n_pair;
  auto n_u     = hostAdp.n_u;
  auto n_w     = hostAdp.n_w;
  auto mass    = hostAdp.mass;
  auto radius  = hostAdp.radius;
  auto factor  = hostAdp.factor;
  auto r_cut   = hostAdp.r_cutoff;

  Kokkos::printf("Antes de pillar el tipo\n");

  using DPS = typename AdpPotencial_Device::value_type;

  Kokkos::printf("Despues de pillar el tipo\n");

  DPS *p_dst = adpDevice.data() + pos;

  Kokkos::printf("Voy a copiar escalares\n");

  Kokkos::parallel_for(
    "copy_adp_scalars",
    Kokkos::RangePolicy< Kokkos::DefaultExecutionSpace >(0,1),
    [=] KOKKOS_FUNCTION(int)
    {
      p_dst->n_embed  = n_embed;
      p_dst->n_rho    = n_rho;
      p_dst->n_pair   = n_pair;
      p_dst->n_u      = n_u;
      p_dst->n_w      = n_w;
      p_dst->mass     = mass;
      p_dst->radius   = radius;
      p_dst->factor   = factor;
      p_dst->r_cutoff = r_cut;
    }
  );

  copy_cubic_spline_device(p_dst->embed, hostAdp.embed, "copy_embed");
  copy_cubic_spline_device(p_dst->rho,   hostAdp.rho,   "copy_rho");
  copy_cubic_spline_device(p_dst->pair,  hostAdp.pair,  "copy_pair");
  copy_cubic_spline_device(p_dst->u,     hostAdp.u,     "copy_u");
  copy_cubic_spline_device(p_dst->w,     hostAdp.w,     "copy_w");

} */

/* inline void copy_adpPotential_to_device(
  AdpPotencial_Device &adpDevice,
  const adpPotential  &hostAdp,
  int                   pos)
{
  int N = adpDevice.extent(0);
  if (pos < 0 || pos >= N) throw std::out_of_range("pos fuera de rango");

  auto ne = hostAdp.n_embed;
  auto nr = hostAdp.n_rho;
  auto np = hostAdp.n_pair;
  auto nu = hostAdp.n_u;
  auto nw = hostAdp.n_w;
  auto m  = hostAdp.mass;
  auto r  = hostAdp.radius;
  auto f  = hostAdp.factor;
  auto rc = hostAdp.r_cutoff;

  {
    auto p = adpDevice.data() + pos;
    Kokkos::parallel_for(
      "copy_adp_scalars",
      Kokkos::RangePolicy<>(0,1),
      [=] KOKKOS_FUNCTION(int)
      {
        p->n_embed  = ne;
        p->n_rho    = nr;
        p->n_pair   = np;
        p->n_u      = nu;
        p->n_w      = nw;
        p->mass     = m;
        p->radius   = r;
        p->factor   = f;
        p->r_cutoff = rc;
      }
    );
  }

  auto syncSpline = [&](auto &dvHost, auto &dvDev){
    dvDev.modify_host();
    Kokkos::deep_copy(dvDev.view_host(), dvHost.view_host());
    dvDev.sync_device();
  };

  auto &dst = adpDevice(pos);

  syncSpline(hostAdp.embed.x,   dst.embed.x);
  syncSpline(hostAdp.embed.a,   dst.embed.a);
  syncSpline(hostAdp.embed.b,   dst.embed.b);
  syncSpline(hostAdp.embed.c,   dst.embed.c);
  syncSpline(hostAdp.embed.d,   dst.embed.d);
  syncSpline(hostAdp.embed.db,  dst.embed.db);
  syncSpline(hostAdp.embed.dc,  dst.embed.dc);
  syncSpline(hostAdp.embed.dd,  dst.embed.dd);
  syncSpline(hostAdp.embed.ddc, dst.embed.ddc);
  syncSpline(hostAdp.embed.ddd, dst.embed.ddd);

  syncSpline(hostAdp.rho.x,   dst.rho.x);
  syncSpline(hostAdp.rho.a,   dst.rho.a);
  syncSpline(hostAdp.rho.b,   dst.rho.b);
  syncSpline(hostAdp.rho.c,   dst.rho.c);
  syncSpline(hostAdp.rho.d,   dst.rho.d);
  syncSpline(hostAdp.rho.db,  dst.rho.db);
  syncSpline(hostAdp.rho.dc,  dst.rho.dc);
  syncSpline(hostAdp.rho.dd,  dst.rho.dd);
  syncSpline(hostAdp.rho.ddc, dst.rho.ddc);
  syncSpline(hostAdp.rho.ddd, dst.rho.ddd);

  syncSpline(hostAdp.pair.x,   dst.pair.x);
  syncSpline(hostAdp.pair.a,   dst.pair.a);
  syncSpline(hostAdp.pair.b,   dst.pair.b);
  syncSpline(hostAdp.pair.c,   dst.pair.c);
  syncSpline(hostAdp.pair.d,   dst.pair.d);
  syncSpline(hostAdp.pair.db,  dst.pair.db);
  syncSpline(hostAdp.pair.dc,  dst.pair.dc);
  syncSpline(hostAdp.pair.dd,  dst.pair.dd);
  syncSpline(hostAdp.pair.ddc, dst.pair.ddc);
  syncSpline(hostAdp.pair.ddd, dst.pair.ddd);

  syncSpline(hostAdp.u.x,   dst.u.x);
  syncSpline(hostAdp.u.a,   dst.u.a);
  syncSpline(hostAdp.u.b,   dst.u.b);
  syncSpline(hostAdp.u.c,   dst.u.c);
  syncSpline(hostAdp.u.d,   dst.u.d);
  syncSpline(hostAdp.u.db,  dst.u.db);
  syncSpline(hostAdp.u.dc,  dst.u.dc);
  syncSpline(hostAdp.u.dd,  dst.u.dd);
  syncSpline(hostAdp.u.ddc, dst.u.ddc);
  syncSpline(hostAdp.u.ddd, dst.u.ddd);

  syncSpline(hostAdp.w.x,   dst.w.x);
  syncSpline(hostAdp.w.a,   dst.w.a);
  syncSpline(hostAdp.w.b,   dst.w.b);
  syncSpline(hostAdp.w.c,   dst.w.c);
  syncSpline(hostAdp.w.d,   dst.w.d);
  syncSpline(hostAdp.w.db,  dst.w.db);
  syncSpline(hostAdp.w.dc,  dst.w.dc);
  syncSpline(hostAdp.w.dd,  dst.w.dd);
  syncSpline(hostAdp.w.ddc, dst.w.ddc);
  syncSpline(hostAdp.w.ddd, dst.w.ddd);
} */


/* inline void copy_adpPotential_to_device(
  AdpPot_DualView    &adpDual,
  const adpPotential &hostAdp,
  int                  pos)
{
  int N = adpDual.extent(0);
  if (pos < 0 || pos >= N) {
    throw std::out_of_range("pos fuera de rango");
  }

  adpDual.modify_host();

  adpPotential &dst = adpDual.view_host()(pos);

  dst.n_embed  = hostAdp.n_embed;
  dst.n_rho    = hostAdp.n_rho;
  dst.n_pair   = hostAdp.n_pair;
  dst.n_u      = hostAdp.n_u;
  dst.n_w      = hostAdp.n_w;
  dst.mass     = hostAdp.mass;
  dst.radius   = hostAdp.radius;
  dst.factor   = hostAdp.factor;
  dst.r_cutoff = hostAdp.r_cutoff;

  Kokkos::printf("Despues de copiar escalares\n");


  auto syncSpline = [&](DualView_Vector_Double &dvDst, const DualView_Vector_Double &dvSrc) {
    dvDst.modify_host();
    Kokkos::deep_copy(dvDst.view_host(), dvSrc.view_host());
    dvDst.sync_device();
  }; 

  syncSpline(dst.embed.x,   hostAdp.embed.x);
  syncSpline(dst.embed.a,   hostAdp.embed.a);
  syncSpline(dst.embed.b,   hostAdp.embed.b);
  syncSpline(dst.embed.c,   hostAdp.embed.c);
  syncSpline(dst.embed.d,   hostAdp.embed.d);
  syncSpline(dst.embed.db,  hostAdp.embed.db);
  syncSpline(dst.embed.dc,  hostAdp.embed.dc);
  syncSpline(dst.embed.dd,  hostAdp.embed.dd);
  syncSpline(dst.embed.ddc, hostAdp.embed.ddc);
  syncSpline(dst.embed.ddd, hostAdp.embed.ddd);

  Kokkos::printf("Despues de copiar embed\n");

  syncSpline(dst.rho.x,   hostAdp.rho.x);
  syncSpline(dst.rho.a,   hostAdp.rho.a);
  syncSpline(dst.rho.b,   hostAdp.rho.b);
  syncSpline(dst.rho.c,   hostAdp.rho.c);
  syncSpline(dst.rho.d,   hostAdp.rho.d);
  syncSpline(dst.rho.db,  hostAdp.rho.db);
  syncSpline(dst.rho.dc,  hostAdp.rho.dc);
  syncSpline(dst.rho.dd,  hostAdp.rho.dd);
  syncSpline(dst.rho.ddc, hostAdp.rho.ddc);
  syncSpline(dst.rho.ddd, hostAdp.rho.ddd);

  syncSpline(dst.pair.x,   hostAdp.pair.x);
  syncSpline(dst.pair.a,   hostAdp.pair.a);
  syncSpline(dst.pair.b,   hostAdp.pair.b);
  syncSpline(dst.pair.c,   hostAdp.pair.c);
  syncSpline(dst.pair.d,   hostAdp.pair.d);
  syncSpline(dst.pair.db,  hostAdp.pair.db);
  syncSpline(dst.pair.dc,  hostAdp.pair.dc);
  syncSpline(dst.pair.dd,  hostAdp.pair.dd);
  syncSpline(dst.pair.ddc, hostAdp.pair.ddc);
  syncSpline(dst.pair.ddd, hostAdp.pair.ddd);

  syncSpline(dst.u.x,   hostAdp.u.x);
  syncSpline(dst.u.a,   hostAdp.u.a);
  syncSpline(dst.u.b,   hostAdp.u.b);
  syncSpline(dst.u.c,   hostAdp.u.c);
  syncSpline(dst.u.d,   hostAdp.u.d);
  syncSpline(dst.u.db,  hostAdp.u.db);
  syncSpline(dst.u.dc,  hostAdp.u.dc);
  syncSpline(dst.u.dd,  hostAdp.u.dd);
  syncSpline(dst.u.ddc, hostAdp.u.ddc);
  syncSpline(dst.u.ddd, hostAdp.u.ddd);

  syncSpline(dst.w.x,   hostAdp.w.x);
  syncSpline(dst.w.a,   hostAdp.w.a);
  syncSpline(dst.w.b,   hostAdp.w.b);
  syncSpline(dst.w.c,   hostAdp.w.c);
  syncSpline(dst.w.d,   hostAdp.w.d);
  syncSpline(dst.w.db,  hostAdp.w.db);
  syncSpline(dst.w.dc,  hostAdp.w.dc);
  syncSpline(dst.w.dd,  hostAdp.w.dd);
  syncSpline(dst.w.ddc, hostAdp.w.ddc);
  syncSpline(dst.w.ddd, hostAdp.w.ddd);


  Kokkos::printf("Despues de copiar todo\n");

  // adpDual.sync_device();
} */

template<typename T>
KOKKOS_INLINE_FUNCTION
double dsqr(T a) {
  double da = static_cast<double>(a);
  return (da == 0.0) ? 0.0 : da * da;
}

/* KOKKOS_INLINE_FUNCTION void read_spline(FILE * f_adp, int index, CubicSpline dest ) {
  int error;
  View_Double_Vector_Host x = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host a = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host b = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host c = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host d = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host db = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host dc = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host dd = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host ddc = View_Double_Vector_Host("HostSpline->x", index + 1);
  View_Double_Vector_Host ddd = View_Double_Vector_Host("HostSpline->x", index + 1);

  for (int i = 0; i < index; i++) {
      error = fscanf(f_adp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf \n",
                     &a[i], &b[i], &c[i],
                     &d[i], &db[i], &dc[i],
                     &dd[i], &ddc[i], &ddd[i]);
    }


  Kokkos::deep_copy(dest.a,a);   
  Kokkos::deep_copy(dest.b,b);   
  Kokkos::deep_copy(dest.c,c);   
  Kokkos::deep_copy(dest.d,d);   
  Kokkos::deep_copy(dest.db,db);   
  Kokkos::deep_copy(dest.dc,dc);   
  Kokkos::deep_copy(dest.dd,dd);   
  Kokkos::deep_copy(dest.ddc,ddc);   
  Kokkos::deep_copy(dest.ddd,ddd);   

} */


inline AtomTopologyKokkos convert_single_atomTopology_to_Kokkos(const AtomTopology &atom_top) {
    AtomTopologyKokkos result;
    result.numneigh = atom_top.numneigh;
    size_t n = static_cast<size_t>(atom_top.numneigh);
    
    Kokkos::View<const PetscInt*, DefaultLayout, Kokkos::HostSpace> host_view(atom_top.mech_neighs_ptr, n);
    
    result.mech_neighs_ptr = PetscInt_Vector_Default("Atom_Topology_mech_neighs_ptr", n);
    
    Kokkos::deep_copy(result.mech_neighs_ptr, host_view);
    
    return result;
  }
  
  inline AtomTopologyKokkos* convert_atomTopology_array_to_Kokkos(const AtomTopology* atomArray, size_t count) {

    AtomTopologyKokkos* result = new AtomTopologyKokkos[count];
    for (size_t i = 0; i < count; i++) {
      result[i] = convert_single_atomTopology_to_Kokkos(atomArray[i]);
    }
    return result;
  }


inline void initSoA_ADP(
  const adpPotential &adpMgMg,
  const adpPotential &adpHH,
  const adpPotential &adpMgH,
  SoA_ADP            &H)
  {
    constexpr int M = 3;
  
    int ne[3] = { adpMgMg.n_embed, adpHH.n_embed, adpMgH.n_embed };
    int nr[3] = { adpMgMg.n_rho,   adpHH.n_rho,   adpMgH.n_rho   };
    int np[3] = { adpMgMg.n_pair,  adpHH.n_pair,  adpMgH.n_pair  };
    int nu[3] = { adpMgMg.n_u,     adpHH.n_u,     adpMgH.n_u     };
    int nw[3] = { adpMgMg.n_w,     adpHH.n_w,     adpMgH.n_w     };
  
    size_t total_embed = (ne[0]+1)+(ne[1]+1)+(ne[2]+1);
    size_t total_rho   = (nr[0]+1)+(nr[1]+1)+(nr[2]+1);
    size_t total_pair  = (np[0]+1)+(np[1]+1)+(np[2]+1);
    size_t total_u     = (nu[0]+1)+(nu[1]+1)+(nu[2]+1);
    size_t total_w     = (nw[0]+1)+(nw[1]+1)+(nw[2]+1);
  
    H.n_embed = { "n_embed", M };
    H.n_rho   = { "n_rho",   M };
    H.n_pair  = { "n_pair",  M };
    H.n_u     = { "n_u",     M };
    H.n_w     = { "n_w",     M };
  
    H.mass     = { "mass",     M };
    H.radius   = { "radius",   M };
    H.factor   = { "factor",   M };
    H.r_cutoff = { "r_cutoff", M };

    H.dx_embed = { "dx_embed", M };
    H.dx_rho   = { "dx_rho",   M };
    H.dx_pair  = { "dx_pair",  M };
    H.dx_u     = { "dx_u",     M };
    H.dx_w     = { "dx_w",     M };
  
    H.embed_x   = { "embed_x",   total_embed };
    H.embed_a   = { "embed_a",   total_embed };
    H.embed_b   = { "embed_b",   total_embed };
    H.embed_c   = { "embed_c",   total_embed };
    H.embed_d   = { "embed_d",   total_embed };
    H.embed_db  = { "embed_db",  total_embed };
    H.embed_dc  = { "embed_dc",  total_embed };
    H.embed_dd  = { "embed_dd",  total_embed };
    H.embed_ddc = { "embed_ddc", total_embed };
    H.embed_ddd = { "embed_ddd", total_embed };
  
    H.rho_x   = { "rho_x",   total_rho };
    H.rho_a   = { "rho_a",   total_rho };
    H.rho_b   = { "rho_b",   total_rho };
    H.rho_c   = { "rho_c",   total_rho };
    H.rho_d   = { "rho_d",   total_rho };
    H.rho_db  = { "rho_db",  total_rho };
    H.rho_dc  = { "rho_dc",  total_rho };
    H.rho_dd  = { "rho_dd",  total_rho };
    H.rho_ddc = { "rho_ddc", total_rho };
    H.rho_ddd = { "rho_ddd", total_rho };
  
    H.pair_x   = { "pair_x",   total_pair };
    H.pair_a   = { "pair_a",   total_pair };
    H.pair_b   = { "pair_b",   total_pair };
    H.pair_c   = { "pair_c",   total_pair };
    H.pair_d   = { "pair_d",   total_pair };
    H.pair_db  = { "pair_db",  total_pair };
    H.pair_dc  = { "pair_dc",  total_pair };
    H.pair_dd  = { "pair_dd",  total_pair };
    H.pair_ddc = { "pair_ddc", total_pair };
    H.pair_ddd = { "pair_ddd", total_pair };
  
    H.u_x   = { "u_x",   total_u };
    H.u_a   = { "u_a",   total_u };
    H.u_b   = { "u_b",   total_u };
    H.u_c   = { "u_c",   total_u };
    H.u_d   = { "u_d",   total_u };
    H.u_db  = { "u_db",  total_u };
    H.u_dc  = { "u_dc",  total_u };
    H.u_dd  = { "u_dd",  total_u };
    H.u_ddc = { "u_ddc", total_u };
    H.u_ddd = { "u_ddd", total_u };
  
    H.w_x   = { "w_x",   total_w };
    H.w_a   = { "w_a",   total_w };
    H.w_b   = { "w_b",   total_w };
    H.w_c   = { "w_c",   total_w };
    H.w_d   = { "w_d",   total_w };
    H.w_db  = { "w_db",  total_w };
    H.w_dc  = { "w_dc",  total_w };
    H.w_dd  = { "w_dd",  total_w };
    H.w_ddc = { "w_ddc", total_w };
    H.w_ddd = { "w_ddd", total_w };
  
    {
      auto ne_h  = H.n_embed.view_host();
      auto nr_h  = H.n_rho  .view_host();
      auto np_h  = H.n_pair .view_host();
      auto nu_h  = H.n_u    .view_host();
      auto nw_h  = H.n_w    .view_host();
      
      auto m_h   = H.mass   .view_host();
      auto r_h   = H.radius .view_host();
      auto f_h   = H.factor .view_host();
      auto rc_h  = H.r_cutoff.view_host();

      auto de_h = H.dx_embed.view_host();
      auto dr_h = H.dx_rho  .view_host();
      auto dp_h = H.dx_pair .view_host();
      auto du_h = H.dx_u    .view_host();
      auto dw_h = H.dx_w    .view_host();
      
      auto ex_h   = H.embed_x .view_host();
      auto ea_h   = H.embed_a .view_host();
      auto eb_h   = H.embed_b .view_host();
      auto ec_h   = H.embed_c .view_host();
      auto ed_h   = H.embed_d .view_host();
      auto edb_h  = H.embed_db.view_host();
      auto edc_h  = H.embed_dc.view_host();
      auto edd_h  = H.embed_dd.view_host();
      auto eddc_h = H.embed_ddc.view_host();
      auto eddd_h = H.embed_ddd.view_host();
      
      auto rx_h   = H.rho_x   .view_host();
      auto ra2_h  = H.rho_a   .view_host();
      auto rb2_h  = H.rho_b   .view_host();
      auto rc2_h  = H.rho_c   .view_host();
      auto rd2_h  = H.rho_d   .view_host();
      auto rdb_h  = H.rho_db  .view_host();
      auto rdc_h  = H.rho_dc  .view_host();
      auto rdd_h  = H.rho_dd  .view_host();
      auto rddc_h = H.rho_ddc .view_host();
      auto rddd_h = H.rho_ddd .view_host();
      
      auto px_h   = H.pair_x  .view_host();
      auto pa_h   = H.pair_a  .view_host();
      auto pb_h   = H.pair_b  .view_host();
      auto pc_h   = H.pair_c  .view_host();
      auto pd_h   = H.pair_d  .view_host();
      auto pdb_h  = H.pair_db .view_host();
      auto pdc_h  = H.pair_dc .view_host();
      auto pdd_h  = H.pair_dd .view_host();
      auto pddc_h = H.pair_ddc.view_host();
      auto pddd_h = H.pair_ddd.view_host();
      
      auto ux_h   = H.u_x     .view_host();
      auto ua_h   = H.u_a     .view_host();
      auto ub_h   = H.u_b     .view_host();
      auto uc_h   = H.u_c     .view_host();
      auto ud_h   = H.u_d     .view_host();
      auto udb_h  = H.u_db    .view_host();
      auto udc_h  = H.u_dc    .view_host();
      auto udd_h  = H.u_dd    .view_host();
      auto uddc_h = H.u_ddc   .view_host();
      auto uddd_h = H.u_ddd   .view_host();
      
      auto wx_h   = H.w_x     .view_host();
      auto wa_h   = H.w_a     .view_host();
      auto wb_h   = H.w_b     .view_host();
      auto wc_h   = H.w_c     .view_host();
      auto wd_h   = H.w_d     .view_host();
      auto wdb_h  = H.w_db    .view_host();
      auto wdc_h  = H.w_dc    .view_host();
      auto wdd_h  = H.w_dd    .view_host();
      auto wddc_h = H.w_ddc   .view_host();
      auto wddd_h = H.w_ddd   .view_host();
  
      int off_e=0, off_r=0, off_p=0, off_u=0, off_w=0;
  
      adpPotential const *arr[3] = { &adpMgMg, &adpHH, &adpMgH };
      for(int i=0;i<3;++i){
        const auto &hp = *arr[i];

        ne_h(i) = hp.n_embed;  nr_h(i) = hp.n_rho;
        np_h(i) = hp.n_pair;   nu_h(i) = hp.n_u;
        nw_h(i) = hp.n_w;
        m_h(i)  = hp.mass;     r_h(i)  = hp.radius;
        f_h(i)  = hp.factor;   rc_h(i) = hp.r_cutoff;

        de_h(i) = hp.embed.dx;
        dr_h(i) = hp.rho.dx;
        dp_h(i) = hp.pair.dx;
        du_h(i) = hp.u.dx;
        dw_h(i) = hp.w.dx;
      
      
        for(int k = 0; k <= hp.n_embed; ++k) {
          ex_h [off_e + k] = hp.embed.x  [k];
          ea_h [off_e + k] = hp.embed.a  [k];
          eb_h [off_e + k] = hp.embed.b  [k];
          ec_h [off_e + k] = hp.embed.c  [k];
          ed_h [off_e + k] = hp.embed.d  [k];
          edb_h[off_e + k] = hp.embed.db [k];
          edc_h[off_e + k] = hp.embed.dc [k];
          edd_h[off_e + k] = hp.embed.dd [k];
          eddc_h[off_e + k]= hp.embed.ddc[k];
          eddd_h[off_e + k]= hp.embed.ddd[k];
        }
        off_e += hp.n_embed + 1;
      
        for(int k = 0; k <= hp.n_rho; ++k) {
          rx_h [off_r + k] = hp.rho.x  [k];
          ra2_h[off_r + k] = hp.rho.a  [k];
          rb2_h[off_r + k] = hp.rho.b  [k];
          rc2_h[off_r + k] = hp.rho.c  [k];
          rd2_h[off_r + k] = hp.rho.d  [k];
          rdb_h[off_r + k] = hp.rho.db [k];
          rdc_h[off_r + k] = hp.rho.dc [k];
          rdd_h[off_r + k] = hp.rho.dd [k];
          rddc_h[off_r + k]= hp.rho.ddc[k];
          rddd_h[off_r + k]= hp.rho.ddd[k];
        }
        off_r += hp.n_rho + 1;
      
        for(int k = 0; k <= hp.n_pair; ++k) {
          px_h [off_p + k] = hp.pair.x  [k];
          pa_h [off_p + k] = hp.pair.a  [k];
          pb_h [off_p + k] = hp.pair.b  [k];
          pc_h [off_p + k] = hp.pair.c  [k];
          pd_h [off_p + k] = hp.pair.d  [k];
          pdb_h[off_p + k] = hp.pair.db [k];
          pdc_h[off_p + k] = hp.pair.dc [k];
          pdd_h[off_p + k] = hp.pair.dd [k];
          pddc_h[off_p + k]= hp.pair.ddc[k];
          pddd_h[off_p + k]= hp.pair.ddd[k];
        }
        off_p += hp.n_pair + 1;
      
        for(int k = 0; k <= hp.n_u; ++k) {
          ux_h [off_u + k] = hp.u.x  [k];
          ua_h [off_u + k] = hp.u.a  [k];
          ub_h [off_u + k] = hp.u.b  [k];
          uc_h [off_u + k] = hp.u.c  [k];
          ud_h [off_u + k] = hp.u.d  [k];
          udb_h[off_u + k] = hp.u.db [k];
          udc_h[off_u + k] = hp.u.dc [k];
          udd_h[off_u + k] = hp.u.dd [k];
          uddc_h[off_u + k]= hp.u.ddc[k];
          uddd_h[off_u + k]= hp.u.ddd[k];
        }
        off_u += hp.n_u + 1;
      
        for(int k = 0; k <= hp.n_w; ++k) {
          wx_h [off_w + k] = hp.w.x  [k];
          wa_h [off_w + k] = hp.w.a  [k];
          wb_h [off_w + k] = hp.w.b  [k];
          wc_h [off_w + k] = hp.w.c  [k];
          wd_h [off_w + k] = hp.w.d  [k];
          wdb_h[off_w + k] = hp.w.db [k];
          wdc_h[off_w + k] = hp.w.dc [k];
          wdd_h[off_w + k] = hp.w.dd [k];
          wddc_h[off_w + k]= hp.w.ddc[k];
          wddd_h[off_w + k]= hp.w.ddd[k];
        }
        off_w += hp.n_w + 1;
      }
    }
  
    H.n_embed   .modify_host(); H.n_embed   .sync_device();
    H.n_rho     .modify_host(); H.n_rho     .sync_device();
    H.n_pair    .modify_host(); H.n_pair    .sync_device();
    H.n_u       .modify_host(); H.n_u       .sync_device();
    H.n_w       .modify_host(); H.n_w       .sync_device();
    
    H.mass      .modify_host(); H.mass      .sync_device();
    H.radius    .modify_host(); H.radius    .sync_device();
    H.factor    .modify_host(); H.factor    .sync_device();
    H.r_cutoff  .modify_host(); H.r_cutoff  .sync_device();

    H.dx_embed  .modify_host(); H.dx_embed  .sync_device();
    H.dx_rho    .modify_host(); H.dx_rho    .sync_device();
    H.dx_pair   .modify_host(); H.dx_pair   .sync_device();
    H.dx_u      .modify_host(); H.dx_u      .sync_device();
    H.dx_w      .modify_host(); H.dx_w      .sync_device();
    
    H.embed_x   .modify_host(); H.embed_x   .sync_device();
    H.embed_a   .modify_host(); H.embed_a   .sync_device();
    H.embed_b   .modify_host(); H.embed_b   .sync_device();
    H.embed_c   .modify_host(); H.embed_c   .sync_device();
    H.embed_d   .modify_host(); H.embed_d   .sync_device();
    H.embed_db  .modify_host(); H.embed_db  .sync_device();
    H.embed_dc  .modify_host(); H.embed_dc  .sync_device();
    H.embed_dd  .modify_host(); H.embed_dd  .sync_device();
    H.embed_ddc .modify_host(); H.embed_ddc .sync_device();
    H.embed_ddd .modify_host(); H.embed_ddd .sync_device();
    
    H.rho_x     .modify_host(); H.rho_x     .sync_device();
    H.rho_a     .modify_host(); H.rho_a     .sync_device();
    H.rho_b     .modify_host(); H.rho_b     .sync_device();
    H.rho_c     .modify_host(); H.rho_c     .sync_device();
    H.rho_d     .modify_host(); H.rho_d     .sync_device();
    H.rho_db    .modify_host(); H.rho_db    .sync_device();
    H.rho_dc    .modify_host(); H.rho_dc    .sync_device();
    H.rho_dd    .modify_host(); H.rho_dd    .sync_device();
    H.rho_ddc   .modify_host(); H.rho_ddc   .sync_device();
    H.rho_ddd   .modify_host(); H.rho_ddd   .sync_device();
    
    H.pair_x    .modify_host(); H.pair_x    .sync_device();
    H.pair_a    .modify_host(); H.pair_a    .sync_device();
    H.pair_b    .modify_host(); H.pair_b    .sync_device();
    H.pair_c    .modify_host(); H.pair_c    .sync_device();
    H.pair_d    .modify_host(); H.pair_d    .sync_device();
    H.pair_db   .modify_host(); H.pair_db   .sync_device();
    H.pair_dc   .modify_host(); H.pair_dc   .sync_device();
    H.pair_dd   .modify_host(); H.pair_dd   .sync_device();
    H.pair_ddc  .modify_host(); H.pair_ddc  .sync_device();
    H.pair_ddd  .modify_host(); H.pair_ddd  .sync_device();
    
    H.u_x       .modify_host(); H.u_x       .sync_device();
    H.u_a       .modify_host(); H.u_a       .sync_device();
    H.u_b       .modify_host(); H.u_b       .sync_device();
    H.u_c       .modify_host(); H.u_c       .sync_device();
    H.u_d       .modify_host(); H.u_d       .sync_device();
    H.u_db      .modify_host(); H.u_db      .sync_device();
    H.u_dc      .modify_host(); H.u_dc      .sync_device();
    H.u_dd      .modify_host(); H.u_dd      .sync_device();
    H.u_ddc     .modify_host(); H.u_ddc     .sync_device();
    H.u_ddd     .modify_host(); H.u_ddd     .sync_device();
    
    H.w_x       .modify_host(); H.w_x       .sync_device();
    H.w_a       .modify_host(); H.w_a       .sync_device();
    H.w_b       .modify_host(); H.w_b       .sync_device();
    H.w_c       .modify_host(); H.w_c       .sync_device();
    H.w_d       .modify_host(); H.w_d       .sync_device();
    H.w_db      .modify_host(); H.w_db      .sync_device();
    H.w_dc      .modify_host(); H.w_dc      .sync_device();
    H.w_dd      .modify_host(); H.w_dd      .sync_device();
    H.w_ddc     .modify_host(); H.w_ddc     .sync_device();
    H.w_ddd     .modify_host(); H.w_ddd     .sync_device();
  }

  inline void copyHostToDevice(const SoA_ADP &H, SoADevice &D) {
  
    D.n_embed   = H.n_embed .view_device().data();
    D.n_rho     = H.n_rho   .view_device().data();
    D.n_pair    = H.n_pair  .view_device().data();
    D.n_u       = H.n_u     .view_device().data();
    D.n_w       = H.n_w     .view_device().data();
  
    D.mass      = H.mass    .view_device().data();
    D.radius    = H.radius  .view_device().data();
    D.factor    = H.factor  .view_device().data();
    D.r_cutoff  = H.r_cutoff.view_device().data();

    D.dx_embed  = H.dx_embed.view_device().data();
    D.dx_rho    = H.dx_rho  .view_device().data();
    D.dx_pair   = H.dx_pair .view_device().data();
    D.dx_u      = H.dx_u    .view_device().data();
    D.dx_w      = H.dx_w    .view_device().data();
  
    D.embed_x   = H.embed_x .view_device().data();
    D.embed_a   = H.embed_a .view_device().data();
    D.embed_b   = H.embed_b .view_device().data();
    D.embed_c   = H.embed_c .view_device().data();
    D.embed_d   = H.embed_d .view_device().data();
    D.embed_db  = H.embed_db.view_device().data();
    D.embed_dc  = H.embed_dc.view_device().data();
    D.embed_dd  = H.embed_dd.view_device().data();
    D.embed_ddc = H.embed_ddc.view_device().data();
    D.embed_ddd = H.embed_ddd.view_device().data();
  
    D.rho_x     = H.rho_x   .view_device().data();
    D.rho_a     = H.rho_a   .view_device().data();
    D.rho_b     = H.rho_b   .view_device().data();
    D.rho_c     = H.rho_c   .view_device().data();
    D.rho_d     = H.rho_d   .view_device().data();
    D.rho_db    = H.rho_db  .view_device().data();
    D.rho_dc    = H.rho_dc  .view_device().data();
    D.rho_dd    = H.rho_dd  .view_device().data();
    D.rho_ddc   = H.rho_ddc .view_device().data();
    D.rho_ddd   = H.rho_ddd .view_device().data();
  
    D.pair_x    = H.pair_x  .view_device().data();
    D.pair_a    = H.pair_a  .view_device().data();
    D.pair_b    = H.pair_b  .view_device().data();
    D.pair_c    = H.pair_c  .view_device().data();
    D.pair_d    = H.pair_d  .view_device().data();
    D.pair_db   = H.pair_db .view_device().data();
    D.pair_dc   = H.pair_dc .view_device().data();
    D.pair_dd   = H.pair_dd .view_device().data();
    D.pair_ddc  = H.pair_ddc.view_device().data();
    D.pair_ddd  = H.pair_ddd.view_device().data();
  
    D.u_x       = H.u_x     .view_device().data();
    D.u_a       = H.u_a     .view_device().data();
    D.u_b       = H.u_b     .view_device().data();
    D.u_c       = H.u_c     .view_device().data();
    D.u_d       = H.u_d     .view_device().data();
    D.u_db      = H.u_db    .view_device().data();
    D.u_dc      = H.u_dc    .view_device().data();
    D.u_dd      = H.u_dd    .view_device().data();
    D.u_ddc     = H.u_ddc   .view_device().data();
    D.u_ddd     = H.u_ddd   .view_device().data();
  
    D.w_x       = H.w_x     .view_device().data();
    D.w_a       = H.w_a     .view_device().data();
    D.w_b       = H.w_b     .view_device().data();
    D.w_c       = H.w_c     .view_device().data();
    D.w_d       = H.w_d     .view_device().data();
    D.w_db      = H.w_db    .view_device().data();
    D.w_dc      = H.w_dc    .view_device().data();
    D.w_dd      = H.w_dd    .view_device().data();
    D.w_ddc     = H.w_ddc   .view_device().data();
    D.w_ddd     = H.w_ddd   .view_device().data();
  }


/* inline void limpiarADPPotencial(adpPotential &pot) {
  
      pot.embed.x   = View_Double_Vector_Host();
      pot.embed.a   = View_Double_Vector_Host();
      pot.embed.b   = View_Double_Vector_Host();
      pot.embed.c   = View_Double_Vector_Host();
      pot.embed.d   = View_Double_Vector_Host();
      pot.embed.db  = View_Double_Vector_Host();
      pot.embed.dc  = View_Double_Vector_Host();
      pot.embed.dd  = View_Double_Vector_Host();
      pot.embed.ddc = View_Double_Vector_Host();
      pot.embed.ddd = View_Double_Vector_Host();
  
      pot.rho.x     = View_Double_Vector_Host();
      pot.rho.a     = View_Double_Vector_Host();
      pot.rho.b     = View_Double_Vector_Host();
      pot.rho.c     = View_Double_Vector_Host();
      pot.rho.d     = View_Double_Vector_Host();
      pot.rho.db    = View_Double_Vector_Host();
      pot.rho.dc    = View_Double_Vector_Host();
      pot.rho.dd    = View_Double_Vector_Host();
      pot.rho.ddc   = View_Double_Vector_Host();
      pot.rho.ddd   = View_Double_Vector_Host();
  
      pot.pair.x    = View_Double_Vector_Host();
      pot.pair.a    = View_Double_Vector_Host();
      pot.pair.b    = View_Double_Vector_Host();
      pot.pair.c    = View_Double_Vector_Host();
      pot.pair.d    = View_Double_Vector_Host();
      pot.pair.db   = View_Double_Vector_Host();
      pot.pair.dc   = View_Double_Vector_Host();
      pot.pair.dd   = View_Double_Vector_Host();
      pot.pair.ddc  = View_Double_Vector_Host();
      pot.pair.ddd  = View_Double_Vector_Host();
  
      pot.u.x       = View_Double_Vector_Host();
      pot.u.a       = View_Double_Vector_Host();
      pot.u.b       = View_Double_Vector_Host();
      pot.u.c       = View_Double_Vector_Host();
      pot.u.d       = View_Double_Vector_Host();
      pot.u.db      = View_Double_Vector_Host();
      pot.u.dc      = View_Double_Vector_Host();
      pot.u.dd      = View_Double_Vector_Host();
      pot.u.ddc     = View_Double_Vector_Host();
      pot.u.ddd     = View_Double_Vector_Host();
  
      pot.w.x       = View_Double_Vector_Host();
      pot.w.a       = View_Double_Vector_Host();
      pot.w.b       = View_Double_Vector_Host();
      pot.w.c       = View_Double_Vector_Host();
      pot.w.d       = View_Double_Vector_Host();
      pot.w.db      = View_Double_Vector_Host();
      pot.w.dc      = View_Double_Vector_Host();
      pot.w.dd      = View_Double_Vector_Host();
      pot.w.ddc     = View_Double_Vector_Host();
      pot.w.ddd     = View_Double_Vector_Host();
    } */

  KOKKOS_INLINE_FUNCTION
  CubicSpline getSpline(
    AdpType pot,
    SplineType sp,
    CubicSpline s,
    const SoADevice* soADevice) 
  {
    int idx = int(pot);
    double *dat_x, *dat_a, *dat_b, *dat_c, *dat_d;
    double *dat_db, *dat_dc, *dat_dd, *dat_ddc, *dat_ddd;
    double *dx_arr;
    int   *len_arr;

    switch(sp) {
      case SplineType::embed:
        dat_x   = soADevice->embed_x;    dat_a   = soADevice->embed_a;
        dat_b   = soADevice->embed_b;    dat_c   = soADevice->embed_c;
        dat_d   = soADevice->embed_d;    dat_db  = soADevice->embed_db;
        dat_dc  = soADevice->embed_dc;   dat_dd  = soADevice->embed_dd;
        dat_ddc = soADevice->embed_ddc;  dat_ddd = soADevice->embed_ddd;
        dx_arr  = soADevice->dx_embed;   len_arr = soADevice->n_embed;
        break;

      case SplineType::rho:
        dat_x   = soADevice->rho_x;      dat_a   = soADevice->rho_a;
        dat_b   = soADevice->rho_b;      dat_c   = soADevice->rho_c;
        dat_d   = soADevice->rho_d;      dat_db  = soADevice->rho_db;
        dat_dc  = soADevice->rho_dc;     dat_dd  = soADevice->rho_dd;
        dat_ddc = soADevice->rho_ddc;    dat_ddd = soADevice->rho_ddd;
        dx_arr  = soADevice->dx_rho;     len_arr = soADevice->n_rho;
        break;

      case SplineType::pair:
        dat_x   = soADevice->pair_x;     dat_a   = soADevice->pair_a;
        dat_b   = soADevice->pair_b;     dat_c   = soADevice->pair_c;
        dat_d   = soADevice->pair_d;     dat_db  = soADevice->pair_db;
        dat_dc  = soADevice->pair_dc;    dat_dd  = soADevice->pair_dd;
        dat_ddc = soADevice->pair_ddc;   dat_ddd = soADevice->pair_ddd;
        dx_arr  = soADevice->dx_pair;    len_arr = soADevice->n_pair;
        break;

      case SplineType::u:
        dat_x   = soADevice->u_x;        dat_a   = soADevice->u_a;
        dat_b   = soADevice->u_b;        dat_c   = soADevice->u_c;
        dat_d   = soADevice->u_d;        dat_db  = soADevice->u_db;
        dat_dc  = soADevice->u_dc;       dat_dd  = soADevice->u_dd;
        dat_ddc = soADevice->u_ddc;      dat_ddd = soADevice->u_ddd;
        dx_arr  = soADevice->dx_u;       len_arr = soADevice->n_u;
        break;

      case SplineType::w:
        dat_x   = soADevice->w_x;        dat_a   = soADevice->w_a;
        dat_b   = soADevice->w_b;        dat_c   = soADevice->w_c;
        dat_d   = soADevice->w_d;        dat_db  = soADevice->w_db;
        dat_dc  = soADevice->w_dc;       dat_dd  = soADevice->w_dd;
        dat_ddc = soADevice->w_ddc;      dat_ddd = soADevice->w_ddd;
        dx_arr  = soADevice->dx_w;       len_arr = soADevice->n_w;
        break;
    }

    size_t start = 0;
    for(int p = 0; p < idx; ++p) {
      start += size_t(len_arr[p] + 1);
    }

    s.isKernel = true;

    s.n  = len_arr[idx];
    s.dx = dx_arr[idx];

    s.x_d   = dat_x   + start;
    s.a_d   = dat_a   + start;
    s.b_d   = dat_b   + start;
    s.c_d   = dat_c   + start;
    s.d_d   = dat_d   + start;
    s.db_d  = dat_db  + start;
    s.dc_d  = dat_dc  + start;
    s.dd_d  = dat_dd  + start;
    s.ddc_d = dat_ddc + start;
    s.ddd_d = dat_ddd + start;

  return s;
}

void inline initGaussianContexts(
  gaussian_measure_ctx_Default ctx,
  Int_Matrix_Default dof_table_view,
  Int_Matrix_Default gp_board_view,
  Int_Matrix_Default dof_table_aux,
  Int_Matrix_Default active_dof
  ) {
  const int Nctx = ctx.extent(0);
  Kokkos::parallel_for("init_ctx", Nctx, KOKKOS_LAMBDA(int i){
    auto c = &ctx(i);

    c->dof_table   = Kokkos::subview(dof_table_view,  i, Kokkos::ALL());
    c->gp_board    = Kokkos::subview(gp_board_view,   i, Kokkos::ALL());
    c->dof_table_aux = Kokkos::subview(dof_table_aux, i, Kokkos::ALL());
    c->active_dof    = Kokkos::subview(active_dof,      i, Kokkos::ALL());

  });
}

inline void clearSoA_ADP(SoA_ADP &soa) {
  soa.n_embed   = DualView_Vector_Int();
  soa.n_rho     = DualView_Vector_Int();
  soa.n_pair    = DualView_Vector_Int();
  soa.n_u       = DualView_Vector_Int();
  soa.n_w       = DualView_Vector_Int();

  soa.mass      = DualView_Vector_Double();
  soa.radius    = DualView_Vector_Double();
  soa.factor    = DualView_Vector_Double();
  soa.r_cutoff  = DualView_Vector_Double();

  soa.dx_embed  = DualView_Vector_Double();
  soa.dx_rho    = DualView_Vector_Double();
  soa.dx_pair   = DualView_Vector_Double();
  soa.dx_u      = DualView_Vector_Double();
  soa.dx_w      = DualView_Vector_Double();

  soa.embed_x   = DualView_Vector_Double();
  soa.embed_a   = DualView_Vector_Double();
  soa.embed_b   = DualView_Vector_Double();
  soa.embed_c   = DualView_Vector_Double();
  soa.embed_d   = DualView_Vector_Double();
  soa.embed_db  = DualView_Vector_Double();
  soa.embed_dc  = DualView_Vector_Double();
  soa.embed_dd  = DualView_Vector_Double();
  soa.embed_ddc = DualView_Vector_Double();
  soa.embed_ddd = DualView_Vector_Double();

  soa.rho_x     = DualView_Vector_Double();
  soa.rho_a     = DualView_Vector_Double();
  soa.rho_b     = DualView_Vector_Double();
  soa.rho_c     = DualView_Vector_Double();
  soa.rho_d     = DualView_Vector_Double();
  soa.rho_db    = DualView_Vector_Double();
  soa.rho_dc    = DualView_Vector_Double();
  soa.rho_dd    = DualView_Vector_Double();
  soa.rho_ddc   = DualView_Vector_Double();
  soa.rho_ddd   = DualView_Vector_Double();

  soa.pair_x    = DualView_Vector_Double();
  soa.pair_a    = DualView_Vector_Double();
  soa.pair_b    = DualView_Vector_Double();
  soa.pair_c    = DualView_Vector_Double();
  soa.pair_d    = DualView_Vector_Double();
  soa.pair_db   = DualView_Vector_Double();
  soa.pair_dc   = DualView_Vector_Double();
  soa.pair_dd   = DualView_Vector_Double();
  soa.pair_ddc  = DualView_Vector_Double();
  soa.pair_ddd  = DualView_Vector_Double();

  soa.u_x       = DualView_Vector_Double();
  soa.u_a       = DualView_Vector_Double();
  soa.u_b       = DualView_Vector_Double();
  soa.u_c       = DualView_Vector_Double();
  soa.u_d       = DualView_Vector_Double();
  soa.u_db      = DualView_Vector_Double();
  soa.u_dc      = DualView_Vector_Double();
  soa.u_dd      = DualView_Vector_Double();
  soa.u_ddc     = DualView_Vector_Double();
  soa.u_ddd     = DualView_Vector_Double();

  soa.w_x       = DualView_Vector_Double();
  soa.w_a       = DualView_Vector_Double();
  soa.w_b       = DualView_Vector_Double();
  soa.w_c       = DualView_Vector_Double();
  soa.w_d       = DualView_Vector_Double();
  soa.w_db      = DualView_Vector_Double();
  soa.w_dc      = DualView_Vector_Double();
  soa.w_dd      = DualView_Vector_Double();
  soa.w_ddc     = DualView_Vector_Double();
  soa.w_ddd     = DualView_Vector_Double();
}  

#endif 


