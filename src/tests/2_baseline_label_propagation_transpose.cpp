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
    copy.incidence_matrix = original.incidence_matrix;
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

    std::cout << "Generating hypergraph..." << std::endl;
    HypergraphNotSparse H = generate_hypergraph(num_vertices, num_hyperedges, probability);
    HypergraphNotSparse H_clone = clone_hypergraph(H);
    std::cout << "Done." << std::endl;

    // std::cout << "Number of vertices: " << H.num_vertices << std::endl;
    // std::cout << "Number of hyperedges: " << H.num_hyperedges << std::endl;

    // std::cout << "Initial vertex labels: ";
    // for (auto label : H_clone.vertex_labels) {
    //     std::cout << label << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Initial hyperedge labels: ";
    // for (auto label : H_clone.hyperedge_labels) {
    //     std::cout << label << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Incidence matrix:\n";
    // for (size_t i = 0; i < H_clone.num_vertices; ++i) {
    //     for (size_t j = 0; j < H_clone.num_hyperedges; ++j) {
    //         std::cout << static_cast<int>(H_clone.incidence_matrix[i][j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "----------------------------------------------" << std::endl;

    // std::cout << "Initial vertex labels:\n";
    // for (std::size_t i = 0; i < H.vertex_labels.size(); ++i) {
    //     auto label = H_clone.vertex_labels[i];
    //     std::cout << "v" << i << ": " 
    //               << (label == std::numeric_limits<uint8_t>::max() ? "?" : std::to_string(label)) 
    //               << "\n";
    // }

    std::cout << std::endl << "Baseline Label Propagation:" << std::endl;
    find_communities(H);
    std::cout << "Done." << std::endl;

    std::cout << std::endl << "Optimized Label Propagation:" << std::endl;
    find_communities_transpose(H_clone);
    std::cout << "Done." << std::endl;

    // std::cout << "\nSize vertex_labels: " << H_clone.vertex_labels.size() << "\n";
    // std::cout << "Size hyperedge_labels: " << H_clone.hyperedge_labels.size() << "\n";

//     std::cout << "\nFinal vertex labels:\n";
//     for (std::size_t i = 0; i < H_clone.vertex_labels.size(); ++i) {
//         std::cout << "v" << i << ": " << static_cast<int>(H_clone.vertex_labels[i]) << "\n";
//     }

//     std::cout << "\nFinal hyperedge labels:\n";
//     for (std::size_t i = 0; i < H_clone.hyperedge_labels.size(); ++i) {
//         std::cout << "e" << i << ": " << static_cast<int>(H_clone.hyperedge_labels[i]) << "\n";
//     }

//     std::cout << "\n*****************************************************************************************\n";

//     std::cout << "\nFinal vertex labels:\n";
//     for (std::size_t i = 0; i < H_clone.vertex_labels.size(); ++i) {
//         std::cout << "v" << i << ": " << static_cast<int>(H_clone.vertex_labels[i]) << "\n";
//     }

//     std::cout << "\nFinal hyperedge labels:\n";
//     for (std::size_t i = 0; i < H_clone.hyperedge_labels.size(); ++i) {
//         std::cout << "e" << i << ": " << static_cast<int>(H_clone.hyperedge_labels[i]) << "\n";
//     }

    return 0;
}

