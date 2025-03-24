#ifndef KOKKOS_UTILS_HPP
#define KOKKOS_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <array>

template <typename View>
KOKKOS_INLINE_FUNCTION
auto extractRowBlock(const View &matrix, int row, int startCol, int numCols) {
    return Kokkos::subview(matrix, row, std::make_pair(startCol, startCol + numCols));
}

template <typename Input_View, typename Output_View>
KOKKOS_INLINE_FUNCTION
void concatenateVectors(const Input_View &v1,
                        const Input_View &v2,
                        Output_View &out)
{
    auto n1 = v1.extent(0);
    auto n2 = v2.extent(0);
    for (size_t i = 0; i < n1; i++) {
        out(i) = static_cast<typename Output_View::value_type>(v1(i));
    }
    for (size_t i = 0; i < n2; i++) {
        out(n1 + i) = static_cast<typename Output_View::value_type>(v2(i));
    }
}


template <typename T>
KOKKOS_INLINE_FUNCTION
std::array<T, 2> concatenate(const T &a, const T &b) {
    return std::array<T, 2>{a, b};
}

KOKKOS_INLINE_FUNCTION void deep_copy_cubic_spline(CubicSpline &dest, const CubicSpline &src) {
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
  

KOKKOS_INLINE_FUNCTION void copy_adpPotential_to_device(AdpPotencial_Device &adpDevice,
                                 const adpPotential &hostAdp,
                                 int pos)
{
    if (adpDevice.data() == nullptr) {
        adpDevice = AdpPotencial_Device("deviceAdp", 2);
    }
    
    if (pos < adpDevice.extent(0)) {
        deep_copy_adpPotential(adpDevice(pos), hostAdp);
    }
}

template<typename T>
KOKKOS_INLINE_FUNCTION
double dsqr(T a) {
  double da = static_cast<double>(a);
  return (da == 0.0) ? 0.0 : da * da;
}
#endif 


