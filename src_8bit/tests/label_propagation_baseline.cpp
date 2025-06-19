#include <algorithm>
#include <unordered_map>
#include <limits>
#include <iostream>
#include <random>
#include <numeric>
#include <vector>
#include <cstdint>
#include <limits>
#include <string>
#include "../base_implementation/headers/algorithms.h"
#include "../base_implementation/headers/utils.h"
#include <chrono>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_vertices> <num_hyperedges> <probability>" << std::endl;
        return 1;
    }

    std::size_t num_vertices = std::stoul(argv[1]);
    std::size_t num_hyperedges = std::stoul(argv[2]);
    double probability = std::stod(argv[3]);

    HypergraphNotSparse H = generate_hypergraph(num_vertices, num_hyperedges, probability);

    find_communities(H_clone);

    return 0;
}

