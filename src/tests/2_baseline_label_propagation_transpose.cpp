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

int main() {
    std::size_t num_vertices = 10000;
    std::size_t num_hyperedges = 30000;
    double probability = 0.3;

    // std::cout << "Generating hypergraph..." << std::endl;
    HypergraphNotSparse H = generate_hypergraph(num_vertices, num_hyperedges, probability);
    // std::cout << "Done." << std::endl;

    // std::cout << "Number of vertices: " << H.num_vertices << std::endl;
    // std::cout << "Number of hyperedges: " << H.num_hyperedges << std::endl;

    // std::cout << "Initial vertex labels: ";
    // for (auto label : H.vertex_labels) {
    //     std::cout << label << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Initial hyperedge labels: ";
    // for (auto label : H.hyperedge_labels) {
    //     std::cout << label << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Incidence matrix:\n";
    // for (size_t i = 0; i < H.num_vertices; ++i) {
    //     for (size_t j = 0; j < H.num_hyperedges; ++j) {
    //         std::cout << H.incidence_matrix[i][j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "----------------------------------------------" << std::endl;

    // std::cout << "Initial vertex labels:\n";
    // for (std::size_t i = 0; i < H.vertex_labels.size(); ++i) {
    //     auto label = H.vertex_labels[i];
    //     std::cout << "v" << i << ": " 
    //               << (label == std::numeric_limits<uint32_t>::max() ? "?" : std::to_string(label)) 
    //               << "\n";
    // }

    // std::cout << "\n Starting find_communities:\n";
    find_communities_transpose(H);
    // std::cout << "Done." << std::endl;

    // std::cout << "\nSize vertex_labels: " << H.vertex_labels.size() << "\n";
    // std::cout << "Size hyperedge_labels: " << H.hyperedge_labels.size() << "\n";

    
    // std::cout << "\nFinal vertex labels:\n";
    // for (std::size_t i = 0; i < H.vertex_labels.size(); ++i) {
    //     std::cout << "v" << i << ": " << H.vertex_labels[i] << "\n";
    // }

    // std::cout << "\nFinal hyperedge labels:\n";
    // for (std::size_t i = 0; i < H.hyperedge_labels.size(); ++i) {
    //     std::cout << "e" << i << ": " << H.hyperedge_labels[i] << "\n";
    // }

    return 0;
}
