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
#include <unordered_set>
#include <cstdint>

#include <fstream>

void save_incidence_matrix(const HypergraphNotSparse& H, const std::string& filename) {
    std::ofstream file(filename);
    for (const auto& row : H.incidence_matrix) {
        for (std::size_t j = 0; j < row.size(); ++j) {
            file << static_cast<int>(row[j]);
            if (j < row.size() - 1) file << ' ';
        }
        file << '\n';
    }
}

void save_labels(const std::vector<std::uint32_t>& labels, const std::string& filename) {
    std::ofstream file(filename);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        file << static_cast<int>(labels[i]);
        if (i < labels.size() - 1) file << ' ';
    }
    file << '\n';
}


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_vertices> <num_hyperedges> <probability>" << std::endl;
        return 1;
    }

    std::size_t N = std::stoul(argv[1]);
    std::size_t E = std::stoul(argv[2]);
    double p = std::stod(argv[3]);

    std::cout << "Generazione ipergrafo con distribuzione power-law..." << std::endl;
    HypergraphNotSparse H = generate_hypergraph(N, E, p);

    save_incidence_matrix(H, "incidence_matrix.txt");
    save_labels(H.vertex_labels, "nodes_label.txt");
    save_labels(H.hyperedge_labels, "edges_label.txt");

    return 0;
}