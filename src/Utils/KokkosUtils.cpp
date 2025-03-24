/* #include <Kokkos_Core.hpp>
#include "Macros.hpp"
#include "KokkosUtils.hpp"


KOKKOS_FUNCTION
void deep_copy_cubic_spline(CubicSpline &dest, const CubicSpline &src) {
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
  
KOKKOS_FUNCTION
void deep_copy_adpPotential(adpPotential &dest, const adpPotential &src) {
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
  
KOKKOS_INLINE_FUNCTION 
void copy_adpPotential_to_device(AdpPotencial_Device &adpDevice, const adpPotential &hostAdp) {
    adpDevice = AdpPotencial_Device("deviceAdp", 1);
    
    adpPotential temp;
    deep_copy_adpPotential(temp, hostAdp);
    
    Kokkos::deep_copy(adpDevice, temp);
} */