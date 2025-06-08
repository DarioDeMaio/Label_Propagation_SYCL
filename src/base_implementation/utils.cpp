#include"../base_implementation/headers/utils.h"
#include<random>
#include<numeric>
#include<vector>
#include<limits>
#include<string>
#include<iostream>
#include <unordered_set>

HypergraphNotSparse generate_hypergraph(std::size_t N, std::size_t E, double p) {
    HypergraphNotSparse H;
    H.num_vertices = N;
    H.num_hyperedges = E;

    std::mt19937 gen(42);
    std::bernoulli_distribution dist(p);

    H.incidence_matrix.resize(N, std::vector<uint32_t>(E, 0));

    for (std::size_t e = 0; e < E; ++e) {
        std::unordered_set<std::size_t> nodes;
        while (nodes.size() < 2) {
            std::size_t v = gen() % N;
            nodes.insert(v);
        }
        for (auto v : nodes) {
            H.incidence_matrix[v][e] = 1;
        }
    }

    for (std::size_t v = 0; v < N; ++v) {
        std::vector<std::size_t> incident;
        for (std::size_t e = 0; e < E; ++e) {
            if (H.incidence_matrix[v][e] == 1) {
                incident.push_back(e);
            }
        }

        while (incident.size() < 2) {
            std::size_t e = gen() % E;
            if (H.incidence_matrix[v][e] == 0) {
                H.incidence_matrix[v][e] = 1;
                incident.push_back(e);
            }
        }
    }

    for (std::size_t e = 0; e < E; ++e) {
        for (std::size_t v = 0; v < N; ++v) {
            if (H.incidence_matrix[v][e] == 0 && dist(gen)) {
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

    H.hyperedge_labels.resize(E);
    std::fill(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), std::numeric_limits<std::uint32_t>::max());

    return H;
}
