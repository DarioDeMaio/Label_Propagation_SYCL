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

int main() {
    std::size_t N = 100;
    std::size_t E = 250;
    double p_unused = 0.5; // ignorato nella versione power-law, serviva per la funzione precedente

    std::cout << "Generazione ipergrafo con distribuzione power-law..." << std::endl;
    HypergraphNotSparse H = generate_hypergraph(N, E, p_unused);

    std::cout << "\nGrado dei nodi (quanti iperarchi tocca ogni nodo):\n";
    for (std::size_t i = 0; i < N; ++i) {
        int degree = std::accumulate(H.incidence_matrix[i].begin(), H.incidence_matrix[i].end(), 0);
        std::cout << "v" << i << ": " << degree << "\n";
    }

    std::cout << "\nGrado degli iperarchi (quanti nodi contiene ogni iperarco):\n";
    for (std::size_t j = 0; j < E; ++j) {
        int degree = 0;
        for (std::size_t i = 0; i < N; ++i) {
            degree += H.incidence_matrix[i][j];
        }
        std::cout << "e" << j << ": " << degree << "\n";
    }

    return 0;
}