#include "headers/utils.h"
#include <random>
#include <numeric>
#include <vector>
#include <cstdint>
#include <limits>

Hypergraph generate_hypergraph(std::size_t N, std::size_t E, double p) {
    Hypergraph H;
    H.num_vertices = N;
    H.num_hyperedges = E;

    std::vector<std::vector<uint32_t>> v2he_tmp(N);
    std::vector<uint32_t> he2v_indices;
    std::vector<uint32_t> he2v_offsets;

    he2v_offsets.push_back(0);

    std::mt19937 rng(std::random_device{}());
    std::bernoulli_distribution include(p);

    for (size_t e = 0; e < E; ++e) {
        std::vector<uint32_t> vertices_in_edge;
        for (size_t v = 0; v < N; ++v) {
            if (include(rng)) {
                vertices_in_edge.push_back(v);
                v2he_tmp[v].push_back(e);
            }
        }

        // ensure at least one vertex per hyperedge
        if (vertices_in_edge.empty()) {
            uint32_t random_vertex = rng() % N;
            vertices_in_edge.push_back(random_vertex);
            v2he_tmp[random_vertex].push_back(e);
        }

        he2v_indices.insert(he2v_indices.end(), vertices_in_edge.begin(), vertices_in_edge.end());
        he2v_offsets.push_back(he2v_indices.size());
    }

    std::vector<uint32_t> v2he_indices;
    std::vector<uint32_t> v2he_offsets = {0};
    for (const auto& edges : v2he_tmp) {
        v2he_indices.insert(v2he_indices.end(), edges.begin(), edges.end());
        v2he_offsets.push_back(v2he_indices.size());
    }

    H.he2v_indices = std::move(he2v_indices);
    H.he2v_offsets = std::move(he2v_offsets);
    H.v2he_indices = std::move(v2he_indices);
    H.v2he_offsets = std::move(v2he_offsets);

    H.vertex_labels.resize(N);
    std::uniform_int_distribution<int> label_dist(0, 2);
    std::bernoulli_distribution labeled(0.5);

    for (size_t i = 0; i < N; ++i) {
        if (labeled(rng)) {
            H.vertex_labels[i] = label_dist(rng);
        } else {
            H.vertex_labels[i] = std::numeric_limits<std::uint32_t>::max();
        }
    }

    H.hyperedge_labels.resize(E);
    std::fill(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), std::numeric_limits<std::uint32_t>::max());

    return H;
}
