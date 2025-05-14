#include <algorithm>
#include <unordered_map>
#include <limits>
#include <iostream>
#include "../first_optimization/headers/utils.h"
#include <random>
#include <numeric>
#include <vector>
#include <cstdint>
#include <limits>
#include <string>
#include <sycl/sycl.hpp>
using namespace sycl;


constexpr std::size_t MaxIterations = 100;

void first_parallel_find_communities(Hypergraph& H) {
    sycl::queue q;
    try {
        q = sycl::queue(sycl::device{sycl::gpu_selector_v}, sycl::property::queue::enable_profiling{});
    } catch (sycl::exception const& e) {
        std::cout << "GPU non disponibile, uso la CPU.\n";
        sycl::queue(sycl::device{ sycl::cpu_selector_v });
    }

    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    constexpr std::size_t MaxIterations = 100;

    uint32_t* vlabels_usm = sycl::malloc_shared<uint32_t>(N, q);
    std::copy(H.vertex_labels.begin(), H.vertex_labels.end(), vlabels_usm);
    uint32_t* helabels_usm = sycl::malloc_shared<uint32_t>(E, q);
    std::copy(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), helabels_usm);
    uint32_t* he2v_indices_usm = sycl::malloc_shared<uint32_t>(H.he2v_indices.size(), q);
    std::copy(H.he2v_indices.begin(), H.he2v_indices.end(), he2v_indices_usm);
    uint32_t* he2v_offsets_usm = sycl::malloc_shared<uint32_t>(H.he2v_offsets.size(), q);
    std::copy(H.he2v_offsets.begin(), H.he2v_offsets.end(), he2v_offsets_usm);
    uint32_t* v2he_indices_usm = sycl::malloc_shared<uint32_t>(H.v2he_indices.size(), q);
    std::copy(H.v2he_indices.begin(), H.v2he_indices.end(), v2he_indices_usm);
    uint32_t* v2he_offsets_usm = sycl::malloc_shared<uint32_t>(H.v2he_offsets.size(), q);
    std::copy(H.v2he_offsets.begin(), H.v2he_offsets.end(), v2he_offsets_usm);

    bool stop = false;
    bool stop_flag_host = true;

    for (std::size_t iter = 0; iter < MaxIterations && !stop; ++iter) {
        std::cout << "SYCL iter: " << iter << std::endl;

        q.submit([&](sycl::handler& h) {
            auto vlabels = vlabels_usm;
            auto helabels = helabels_usm;
            auto he2v_indices = he2v_indices_usm;
            auto he2v_offsets = he2v_offsets_usm;

            h.parallel_for(sycl::range<1>(E), [=](sycl::id<1> e_id) {
                uint32_t label_counts[256] = {0};
                size_t start = he2v_offsets[e_id];
                size_t end = he2v_offsets[e_id + 1];
                for (size_t i = start; i < end; ++i) {
                    uint32_t v = he2v_indices[i];
                    uint32_t lbl = vlabels[v];
                    if (lbl != std::numeric_limits<uint32_t>::max() && lbl < 256) {
                        label_counts[lbl]++;
                    }
                }

                uint32_t best_label = std::numeric_limits<uint32_t>::max();
                size_t best_count = 0;
                for (uint32_t i = 0; i < 256; ++i) {
                    if (label_counts[i] > best_count || (label_counts[i] == best_count && i < best_label)) {
                        best_label = i;
                        best_count = label_counts[i];
                    }
                }

                if (best_label != std::numeric_limits<uint32_t>::max()) {
                    helabels[e_id] = best_label;
                }
            });
        }).wait();

        stop_flag_host = true;
        bool* stop_flag_usm = sycl::malloc_shared<bool>(1, q);
        *stop_flag_usm = stop_flag_host;

        q.submit([&](sycl::handler& h) {
            auto vlabels = vlabels_usm;
            auto helabels = helabels_usm;
            auto v2he_indices = v2he_indices_usm;
            auto v2he_offsets = v2he_offsets_usm;
            auto stop_flag = stop_flag_usm;

            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> v_id) {
                uint32_t label_counts[256] = {0};
                size_t start = v2he_offsets[v_id];
                size_t end = v2he_offsets[v_id + 1];
                for (size_t i = start; i < end; ++i) {
                    uint32_t e = v2he_indices[i];
                    uint32_t lbl = helabels[e];
                    if (lbl != std::numeric_limits<uint32_t>::max() && lbl < 256) {
                        label_counts[lbl]++;
                    }
                }

                uint32_t best_label = std::numeric_limits<uint32_t>::max();
                size_t best_count = 0;
                for (uint32_t i = 0; i < 256; ++i) {
                    if (label_counts[i] > best_count || (label_counts[i] == best_count && i < best_label)) {
                        best_label = i;
                        best_count = label_counts[i];
                    }
                }

                if (best_label != std::numeric_limits<uint32_t>::max() && vlabels[v_id] != best_label) {
                    vlabels[v_id] = best_label;
                    stop_flag[0] = false;
                }
            });
        }).wait();

        stop = *stop_flag_usm;
        sycl::free(stop_flag_usm, q);
    }

}
