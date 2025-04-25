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
adpDevice = AdpPotencial_Device("deviceAdp", 3);
}

auto mirror = Kokkos::create_mirror_view(adpDevice);

Kokkos::deep_copy(mirror, adpDevice);

if (pos < static_cast<int>(mirror.extent(0))) {
mirror(pos) = hostAdp;
// deep_copy_adpPotential(mirror(pos), hostAdp);
}

Kokkos::deep_copy(adpDevice, mirror);
}

template<typename T>
KOKKOS_INLINE_FUNCTION
double dsqr(T a) {
  double da = static_cast<double>(a);
  return (da == 0.0) ? 0.0 : da * da;
}

KOKKOS_INLINE_FUNCTION void read_spline(FILE * f_adp, int index, CubicSpline dest ) {
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

}


inline AtomTopologyKokkos convert_single_atomTopology_to_Kokkos(const AtomTopology &atom_top) {
    AtomTopologyKokkos result;
    result.numneigh = atom_top.numneigh;
    size_t n = static_cast<size_t>(atom_top.numneigh);
    
    Kokkos::View<const PetscInt*, Kokkos::LayoutRight, Kokkos::HostSpace> host_view(atom_top.mech_neighs_ptr, n);
    
    result.mech_neighs_ptr = PetscInt_Vector_Default("mech_neighs", n);
    
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

#endif 


