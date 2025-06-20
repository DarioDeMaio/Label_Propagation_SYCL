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

HypergraphNotSparse clone_hypergraph(const HypergraphNotSparse& original) {
    HypergraphNotSparse copy;
    copy.num_vertices = original.num_vertices;
    copy.num_hyperedges = original.num_hyperedges;

    copy.incidence_matrix.resize(original.incidence_matrix.size());
    for (size_t i = 0; i < original.incidence_matrix.size(); ++i)
        copy.incidence_matrix[i] = original.incidence_matrix[i];

    copy.vertex_labels = original.vertex_labels;
    copy.hyperedge_labels = original.hyperedge_labels;

    return copy;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_vertices> <num_hyperedges> <probability>" << std::endl;
        return 1;
    }

    std::size_t num_vertices = std::stoul(argv[1]);
    std::size_t num_hyperedges = std::stoul(argv[2]);
    double probability = std::stod(argv[3]);

    // std::cout << "Generating hypergraph..." << std::endl;
    HypergraphNotSparse H = generate_hypergraph(num_vertices, num_hyperedges, probability);
    HypergraphNotSparse H_clone = clone_hypergraph(H);
    // std::cout << "Done." << std::endl;

    std::cout << std::endl << "Optimized Label Propagation:" << std::endl;
    find_communities_transpose(H);
    std::cout << "Done." << std::endl;
    
    std::cout << std::endl << "Baseline Label Propagation:" << std::endl;
    find_communities(H_clone);
    std::cout << "Done." << std::endl;

    for(size_t i = 0; i < H_clone.vertex_labels.size(); ++i) {
        if (H_clone.vertex_labels[i] != H.vertex_labels[i]) {
            std::cout << "v" << i << ": " << static_cast<int>(H_clone.vertex_labels[i]) << " != " << static_cast<int>(H.vertex_labels[i]) << "\n";
        }
        break;
    }

    return 0;
}

