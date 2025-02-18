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
#endif 


