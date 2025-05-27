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

void save_labels(const std::vector<std::uint8_t>& labels, const std::string& filename) {
    std::ofstream file(filename);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        file << static_cast<int>(labels[i]);
        if (i < labels.size() - 1) file << ' ';
    }
    file << '\n';
}


int main() {
    std::size_t N = 15000;
    std::size_t E = 50000;
    double p_unused = 0.5; // ignorato nella versione power-law, serviva per la funzione precedente

    std::cout << "Generazione ipergrafo con distribuzione power-law..." << std::endl;
    HypergraphNotSparse H = generate_hypergraph(N, E, p_unused);

    // std::cout << "\nGrado dei nodi (quanti iperarchi tocca ogni nodo):\n";
    // for (std::size_t i = 0; i < N; ++i) {
    //     int degree = std::accumulate(H.incidence_matrix[i].begin(), H.incidence_matrix[i].end(), 0);
    //     std::cout << "v" << i << ": " << degree << "\n";
    // }

    // std::cout << "\nGrado degli iperarchi (quanti nodi contiene ogni iperarco):\n";
    // for (std::size_t j = 0; j < E; ++j) {
    //     int degree = 0;
    //     for (std::size_t i = 0; i < N; ++i) {
    //         degree += H.incidence_matrix[i][j];
    //     }
    //     std::cout << "e" << j << ": " << degree << "\n";
    // }

    save_incidence_matrix(H, "incidence_matrix.txt");
    save_labels(H.vertex_labels, "nodes_label.txt");
    save_labels(H.hyperedge_labels, "edges_label.txt");

    return 0;
}