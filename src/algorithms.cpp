#include <algorithm>
#include <unordered_map>
#include <limits>
#include <iostream>
#include "headers/utils.h"
#include <random>
#include <numeric>
#include <vector>
#include <cstdint>
#include <limits>
#include <string>

constexpr std::size_t MaxIterations = 100;

void sequential_find_communities(Hypergraph& H) {
    std::mt19937 rng(std::random_device{}());

    std::vector<std::uint32_t>& vlabels = H.vertex_labels;
    std::vector<std::uint32_t>& helabels = H.hyperedge_labels;

    std::vector<std::size_t> edges(H.num_hyperedges);
    std::iota(edges.begin(), edges.end(), 0);

    std::vector<std::size_t> vertices(H.num_vertices);
    std::iota(vertices.begin(), vertices.end(), 0);

    bool stop = false;
    std::size_t iter = 0;

    while (!stop && iter < MaxIterations) {
        std::cout << "iter: " << iter << std::endl;
        stop = true;

        for (std::size_t e : edges) {
            std::unordered_map<uint32_t, std::size_t> label_counts;

            for (std::size_t i = H.he2v_offsets[e]; i < H.he2v_offsets[e + 1]; ++i) {
                std::uint32_t v = H.he2v_indices[i];
                std::uint32_t vlabel = vlabels[v];

                if (vlabel != std::numeric_limits<std::uint32_t>::max()) {
                    label_counts[vlabel]++;
                }
            }

            if (!label_counts.empty()) {
                std::uint32_t best_label = 0;
                std::size_t max_count = 0;
                for (const auto& [label, count] : label_counts) {
                    if (count > max_count || (count == max_count && label < best_label)) {
                        best_label = label;
                        max_count = count;
                    }
                }
                helabels[e] = best_label;
            }
        }

        for (std::size_t v : vertices) {
            std::unordered_map<uint32_t, std::size_t> label_counts;

            for (std::size_t i = H.v2he_offsets[v]; i < H.v2he_offsets[v + 1]; ++i) {
                std::uint32_t e = H.v2he_indices[i];
                std::uint32_t elabel = helabels[e];

                if (elabel != std::numeric_limits<std::uint32_t>::max()) {
                    label_counts[elabel]++;
                }
            }

            if (!label_counts.empty()) {
                std::uint32_t best_label = 0;
                std::size_t max_count = 0;
                for (const auto& [label, count] : label_counts) {
                    if (count > max_count || (count == max_count && label < best_label)) {
                        best_label = label;
                        max_count = count;
                    }
                }

                if (vlabels[v] != best_label) {
                    vlabels[v] = best_label;
                    stop = false;
                }
            }
        }

        iter++;
    }
}