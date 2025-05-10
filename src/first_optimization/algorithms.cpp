#include <algorithm>
#include <unordered_map>
#include <limits>
#include <iostream>
#include "first_optimization/headers/utils.h"
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

    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    constexpr std::size_t MaxIterations = 100;

    sycl::buffer<uint32_t> vlabels_buf(H.vertex_labels.data(), sycl::range<1>(N));
    sycl::buffer<uint32_t> helabels_buf(H.hyperedge_labels.data(), sycl::range<1>(E));
    sycl::buffer<uint32_t> he2v_indices_buf(H.he2v_indices.data(), sycl::range<1>(H.he2v_indices.size()));
    sycl::buffer<uint32_t> he2v_offsets_buf(H.he2v_offsets.data(), sycl::range<1>(H.he2v_offsets.size()));
    sycl::buffer<uint32_t> v2he_indices_buf(H.v2he_indices.data(), sycl::range<1>(H.v2he_indices.size()));
    sycl::buffer<uint32_t> v2he_offsets_buf(H.v2he_offsets.data(), sycl::range<1>(H.v2he_offsets.size()));

    bool stop = false;
    bool stop_flag_host = true;

    for (std::size_t iter = 0; iter < MaxIterations && !stop; ++iter) {
        std::cout << "SYCL iter: " << iter << std::endl;

        q.submit([&](sycl::handler& h) {
            auto vlabels = vlabels_buf.get_access<sycl::access::mode::read>(h);
            auto helabels = helabels_buf.get_access<sycl::access::mode::write>(h);
            auto he2v_indices = he2v_indices_buf.get_access<sycl::access::mode::read>(h);
            auto he2v_offsets = he2v_offsets_buf.get_access<sycl::access::mode::read>(h);

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
        });

        stop_flag_host = true;
        sycl::buffer<bool, 1> stop_buf(&stop_flag_host, sycl::range<1>(1));

        q.submit([&](sycl::handler& h) {
            auto vlabels = vlabels_buf.get_access<sycl::access::mode::read_write>(h);
            auto helabels = helabels_buf.get_access<sycl::access::mode::read>(h);
            auto v2he_indices = v2he_indices_buf.get_access<sycl::access::mode::read>(h);
            auto v2he_offsets = v2he_offsets_buf.get_access<sycl::access::mode::read>(h);
            auto stop_flag = stop_buf.get_access<sycl::access::mode::write>(h);

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
        });

        q.wait();

        sycl::host_accessor stop_acc(stop_buf, sycl::read_only);
        stop = stop_acc[0];
    }

    q.submit([&](sycl::handler& h) {
        auto acc = vlabels_buf.get_access<sycl::access::mode::read>(h);
    }).wait();
}
