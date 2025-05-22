#include "../base_implementation/headers/utils.h"
#include <random>
#include <numeric>
#include <vector>
#include <limits>
#include <string>
#include <iostream>
#include <cmath>
#include <unordered_set>
#include <cstdint>
#include <algorithm>

int sample_powerlaw(double alpha, int min_k, int max_k, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double y = dist(gen);
    double a = std::pow(min_k, 1 - alpha);
    double b = std::pow(max_k, 1 - alpha);
    double x = std::pow(a + (b - a) * y, 1.0 / (1 - alpha));
    return std::max(min_k, std::min(max_k, static_cast<int>(std::round(x))));
}

HypergraphNotSparse generate_hypergraph(std::size_t N, std::size_t E, double p_unused) {
    HypergraphNotSparse H;
    H.num_vertices = N;
    H.num_hyperedges = E;

    std::mt19937 gen(42);

    H.incidence_matrix.resize(N, std::vector<uint32_t>(E, 0));

    double alpha = 2.5;
    int min_deg = 1;
    int max_deg_node = std::min<int>(E / 2, E);
    int max_deg_edge = std::min<int>(N / 2, N);

    std::vector<int> node_degrees(N);
    for (size_t i = 0; i < N; ++i) {
        node_degrees[i] = sample_powerlaw(alpha, min_deg, max_deg_node, gen);
    }

    std::vector<int> edge_sizes(E);
    for (size_t e = 0; e < E; ++e) {
        edge_sizes[e] = sample_powerlaw(alpha, min_deg, max_deg_edge, gen);
    }

    int sum_node_deg = std::accumulate(node_degrees.begin(), node_degrees.end(), 0);
    int sum_edge_size = std::accumulate(edge_sizes.begin(), edge_sizes.end(), 0);
    double scale = static_cast<double>(sum_node_deg) / sum_edge_size;

    for (auto& s : edge_sizes) {
        s = std::max(min_deg, static_cast<int>(std::round(s * scale)));
    }

    sum_edge_size = std::accumulate(edge_sizes.begin(), edge_sizes.end(), 0);

    std::vector<int> nodes_pool;
    for (size_t v = 0; v < N; ++v) {
        nodes_pool.insert(nodes_pool.end(), node_degrees[v], v);
    }

    std::shuffle(nodes_pool.begin(), nodes_pool.end(), gen);
    std::vector<int> remaining_node_deg = node_degrees;

    int pool_idx = 0;
    for (int e = 0; e < (int)E; ++e) {
        std::unordered_set<int> selected;
        int target_size = edge_sizes[e];

        while ((int)selected.size() < target_size && pool_idx < (int)nodes_pool.size()) {
            int v = nodes_pool[pool_idx++];
            if (remaining_node_deg[v] > 0 && selected.insert(v).second) {
                H.incidence_matrix[v][e] = 1;
                remaining_node_deg[v]--;
            }
        }

        if (selected.empty()) {
            std::uniform_int_distribution<int> force_dist(0, N - 1);
            while (true) {
                int v = force_dist(gen);
                if (H.incidence_matrix[v][e] == 0) {
                    H.incidence_matrix[v][e] = 1;
                    break;
                }
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

    H.hyperedge_labels.resize(E, std::numeric_limits<std::uint32_t>::max());

    return H;
}
