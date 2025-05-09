#include <algorithm>
#include <unordered_map>
#include <limits>
#include <iostream>
#include "base_implementation/headers/utils.h"
#include <random>
#include <numeric>
#include <vector>
#include <cstdint>
#include <limits>
#include <string>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr std::size_t MaxIterations = 100;

void find_communities(HypergraphNotSparse& H){
    sycl::queue q;
    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    sycl::buffer<uint32_t> vlabels_buf(H.vertex_labels.data(), sycl::range<1>(N));
    sycl::buffer<uint32_t> helabels_buf(H.hyperedge_labels.data(), sycl::range<1>(E));
    sycl::buffer<

    bool stop = false;
    bool stop_flag_host = true;

    for (std::size_t iter = 0; iter < MaxIterations && !stop; ++iter) {
        std::cout << "SYCL iter: " << iter << std::endl;
    }
}