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
#include "../first_optimization/headers/algorithms.h"

int main() {
    std::size_t num_vertices = 10;
    std::size_t num_hyperedges = 5;
    double probability = 0.3;

    std::cout << "Generating hypergraph..." << std::endl;
    Hypergraph H = generate_hypergraph(num_vertices, num_hyperedges, probability);
    std::cout << "Done." << std::endl;

    // std::cout << "Number of vertices: " << H.num_vertices << std::endl;
    // std::cout << "Number of hyperedges: " << H.num_hyperedges << std::endl;

    // std::cout << "Vertex labels: ";
    // for (auto label : H.vertex_labels) {
    //     std::cout << label << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Hyperedge labels: ";
    // for (auto label : H.hyperedge_labels) {
    //     std::cout << label << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "he2v index: ";
    // for (auto index : H.he2v_indices) {
    //     std::cout << index << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "he2v offset: ";
    // for (auto offset : H.he2v_offsets) {
    //     std::cout << offset << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "----------------------------------------------" << std::endl;

    // std::cout << "Initial vertex labels:\n";
    // for (std::size_t i = 0; i < H.vertex_labels.size(); ++i) {
    //     auto label = H.vertex_labels[i];
    //     std::cout << "v" << i << ": " 
    //               << (label == std::numeric_limits<uint32_t>::max() ? "?" : std::to_string(label)) 
    //               << "\n";
    // }

    first_parallel_find_communities(H);

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
