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

void find_communities(Hypergraph& H) {
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
        stop = true;

        std::shuffle(edges.begin(), edges.end(), rng);
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

        std::shuffle(vertices.begin(), vertices.end(), rng);
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

int main() {
    std::size_t num_vertices = 10;
    std::size_t num_hyperedges = 6;
    double probability = 0.3;

    Hypergraph H = generate_hypergraph(num_vertices, num_hyperedges, probability);

    std::cout << "Number of vertices: " << H.num_vertices << std::endl;
    std::cout << "Number of hyperedges: " << H.num_hyperedges << std::endl;

    std::cout << "Vertex labels: ";
    for (auto label : H.vertex_labels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    std::cout << "Hyperedge labels: ";
    for (auto label : H.hyperedge_labels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    std::cout << "he2v index: ";
    for (auto index : H.he2v_indices) {
        std::cout << index << " ";
    }
    std::cout << std::endl;

    std::cout << "he2v offset: ";
    for (auto offset : H.he2v_offsets) {
        std::cout << offset << " ";
    }
    std::cout << std::endl;

    std::cout << "----------------------------------------------" << std::endl;

    std::cout << "Initial vertex labels:\n";
    for (std::size_t i = 0; i < H.vertex_labels.size(); ++i) {
        auto label = H.vertex_labels[i];
        std::cout << "v" << i << ": " 
                  << (label == std::numeric_limits<uint32_t>::max() ? "?" : std::to_string(label)) 
                  << "\n";
    }

    find_communities(H);

    std::cout << "\nFinal vertex labels:\n";
    for (std::size_t i = 0; i < H.vertex_labels.size(); ++i) {
        std::cout << "v" << i << ": " << H.vertex_labels[i] << "\n";
    }

    std::cout << "\nFinal hyperedge labels:\n";
    for (std::size_t i = 0; i < H.hyperedge_labels.size(); ++i) {
        std::cout << "e" << i << ": " << H.hyperedge_labels[i] << "\n";
    }

    return 0;
}
