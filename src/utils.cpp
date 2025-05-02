#include "headers/utils.h"
#include <random>
#include <numeric>
#include <vector>
#include <cstdint>

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

        // ensure at least one vertex
        if (vertices_in_edge.empty()) {
            vertices_in_edge.push_back(rng() % N);  
            v2he_tmp[vertices_in_edge[0]].push_back(e);
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
    H.hyperedge_labels.resize(E);

    std::iota(H.vertex_labels.begin(), H.vertex_labels.end(), 0);  // label = id
    std::fill(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), 0);

    return H;
}
