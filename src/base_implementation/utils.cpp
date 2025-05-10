#include"../base_implementation/headers/utils.h"
#include<random>
#include<numeric>
#include<vector>
#include<limits>
#include<string>

HypergraphNotSparse generate_hypergraph(std::size_t N, std::size_t E, double p) {
    HypergraphNotSparse H;
    H.num_vertices = N;
    H.num_hyperedges = E;

    // Reproducibility
    std::mt19937 gen(42);
    std::bernoulli_distribution dist(p);

    H.incidence_matrix.resize(N);
    for (size_t i = 0; i < N; ++i) {
        H.incidence_matrix[i].resize(E, 0);
    }

    for (std::uint32_t e = 0; e < E; e++) {
        for (std::uint32_t v = 0; v < N; v++) {
            if (dist(gen)) {
                H.incidence_matrix[v][e] = 1;
            }
        }
    }

    H.vertex_labels.resize(N);
    std::uniform_int_distribution<int> label_dist(0, 5);
    std::bernoulli_distribution labeled(0.4);

    for (size_t i = 0; i < N; ++i) {
        if (labeled(gen)) {
            H.vertex_labels[i] = label_dist(gen);
        } else {
            H.vertex_labels[i] = std::numeric_limits<std::uint32_t>::max();
        }
    }

    // Initialize hyperedge labels
    H.hyperedge_labels.resize(E);
    std::fill(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), std::numeric_limits<std::uint32_t>::max());

    return H;
}
