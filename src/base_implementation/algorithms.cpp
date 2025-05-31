#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <limits>
#include <chrono>
#include <sycl/sycl.hpp>
#include "headers/utils.h"
#include <iostream>

using namespace sycl;
constexpr std::size_t MaxIterations = 100;
constexpr size_t TILE_SIZE = 16;
constexpr std::size_t MaxLabels = 16;


void find_communities(HypergraphNotSparse& H) {
    sycl::queue q(sycl::gpu_selector_v);

    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;
    constexpr size_t WorkGroupSize = 128;

    uint8_t* incidence_matrix_dev = sycl::malloc_device<uint8_t>(N * E, q);
    uint8_t* vlabels_dev = sycl::malloc_device<uint8_t>(N, q);
    uint8_t* helabels_dev = sycl::malloc_device<uint8_t>(E, q);
    int* stop_flag_dev = sycl::malloc_device<int>(1, q);

    std::vector<uint8_t> flat_incidence(N * E, 0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < H.incidence_matrix[i].size(); ++j) {
            flat_incidence[i * E + j] = H.incidence_matrix[i][j];
        }
    }

    q.memcpy(incidence_matrix_dev, flat_incidence.data(), N * E).wait();
    q.memcpy(vlabels_dev, H.vertex_labels.data(), N).wait();
    q.memcpy(helabels_dev, H.hyperedge_labels.data(), E).wait();

    std::vector<int> stop_flag_host(1);
    size_t iter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (iter < MaxIterations) {
        stop_flag_host[0] = 0;
        q.memcpy(stop_flag_dev, stop_flag_host.data(), sizeof(int)).wait();

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint8_t, 2> label_counts_acc({WorkGroupSize, MaxLabels}, h);

            h.parallel_for(
                sycl::nd_range<1>(((E + WorkGroupSize - 1) / WorkGroupSize) * WorkGroupSize, WorkGroupSize),
                [=](sycl::nd_item<1> idx) {
                    size_t e = idx.get_global_id(0);
                    if (e >= E) return;

                    auto label_counts = label_counts_acc[idx.get_local_id(0)];

                    for (size_t i = 0; i < MaxLabels; ++i) label_counts[i] = 0;

                    for (size_t v = 0; v < N; ++v) {
                        if (incidence_matrix_dev[v * E + e] == 1) {
                            uint8_t lbl = vlabels_dev[v];
                            if (lbl < MaxLabels && lbl != std::numeric_limits<uint8_t>::max()) {
                                label_counts[lbl]++;
                            }
                        }
                    }

                    uint8_t max_count = 0, best_label = std::numeric_limits<uint8_t>::max();
                    for (size_t i = 0; i < MaxLabels; ++i) {
                        if (label_counts[i] > max_count) {
                            max_count = label_counts[i];
                            best_label = i;
                        }
                    }

                    if (best_label != std::numeric_limits<uint8_t>::max()) {
                        helabels_dev[e] = best_label;
                    }
                });
        }).wait();

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<uint8_t, 2> label_counts_acc({WorkGroupSize, MaxLabels}, h);

            h.parallel_for(
                sycl::nd_range<1>(((N + WorkGroupSize - 1) / WorkGroupSize) * WorkGroupSize, WorkGroupSize),
                [=](sycl::nd_item<1> idx) {
                    size_t v = idx.get_global_id(0);
                    if (v >= N) return;

                    auto label_counts = label_counts_acc[idx.get_local_id(0)];

                    for (size_t i = 0; i < MaxLabels; ++i) label_counts[i] = 0;

                    for (size_t e = 0; e < E; ++e) {
                        if (incidence_matrix_dev[v * E + e] == 1) {
                            uint8_t lbl = helabels_dev[e];
                            if (lbl < MaxLabels && lbl != std::numeric_limits<uint8_t>::max()) {
                                label_counts[lbl]++;
                            }
                        }
                    }

                    uint8_t max_count = 0;
                    uint8_t best_label = vlabels_dev[v];

                    for (size_t i = 0; i < MaxLabels; ++i) {
                        if (label_counts[i] > max_count) {
                            max_count = label_counts[i];
                            best_label = i;
                        }
                    }

                    if (vlabels_dev[v] != best_label && best_label != std::numeric_limits<uint8_t>::max()) {
                        vlabels_dev[v] = best_label;
                        sycl::atomic_ref<int,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::global_space>
                            af(stop_flag_dev[0]);
                        af.store(1);
                    }
                });
        }).wait();

        q.memcpy(stop_flag_host.data(), stop_flag_dev, sizeof(int)).wait();
        if (stop_flag_host[0] == 0) break;

        iter++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Tempo totale ottimizzato (ms): " << total_time_ms << std::endl;

    q.memcpy(H.vertex_labels.data(), vlabels_dev, N).wait();
    q.memcpy(H.hyperedge_labels.data(), helabels_dev, E).wait();

    sycl::free(incidence_matrix_dev, q);
    sycl::free(vlabels_dev, q);
    sycl::free(helabels_dev, q);
    sycl::free(stop_flag_dev, q);
}

void transpose_incidence_matrix(sycl::queue& q, const std::vector<std::vector<uint8_t>>& incidence_matrix, uint8_t* incidence_matrix_T, size_t N, size_t E) {
    uint8_t* incidence_flat = sycl::malloc_shared<uint8_t>(N * E, q);

    for (size_t v = 0; v < N; ++v) {
        for (size_t e = 0; e < E; ++e) {
            incidence_flat[v * E + e] = incidence_matrix[v][e];
        }
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> tile(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(
                           sycl::range<2>((E + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                                          (N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE),
                           sycl::range<2>(TILE_SIZE, TILE_SIZE)),
                       [=](sycl::nd_item<2> item) {
            size_t e = item.get_global_id(0);
            size_t v = item.get_global_id(1);
            size_t local_e = item.get_local_id(0);
            size_t local_v = item.get_local_id(1);

            if (v < N && e < E) {
                tile[local_v][local_e] = incidence_flat[v * E + e];
            }
            item.barrier(sycl::access::fence_space::local_space);

            size_t transposed_e = item.get_group(1) * TILE_SIZE + local_e;
            size_t transposed_v = item.get_group(0) * TILE_SIZE + local_v;

            if (transposed_v < N && transposed_e < E) {
                incidence_matrix_T[transposed_e * N + transposed_v] = tile[local_e][local_v];
            }
        });
    }).wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Tempo per trasporre la matrice (ms): " << duration_ms << std::endl;

    sycl::free(incidence_flat, q);
}

void find_communities_transpose(HypergraphNotSparse& H) {
    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
        std::cout << "Selected device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    } catch (sycl::exception const& e) {
        std::cerr << "Failed to create SYCL queue: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    uint8_t* incidence_matrix_T = sycl::malloc_shared<uint8_t>(E * N, q);
    uint8_t* vlabels_usm = sycl::malloc_shared<uint8_t>(N, q);
    uint8_t* helabels_usm = sycl::malloc_shared<uint8_t>(E, q);
    size_t* edge_indices_usm = sycl::malloc_shared<size_t>(E, q);
    size_t* vertex_indices_usm = sycl::malloc_shared<size_t>(N, q);
    int* stop_flag_device = sycl::malloc_shared<int>(1, q);

    transpose_incidence_matrix(q, H.incidence_matrix, incidence_matrix_T, N, E);

    std::copy(H.vertex_labels.begin(), H.vertex_labels.end(), vlabels_usm);
    std::copy(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), helabels_usm);

    bool stop_flag_host = true;
    size_t iter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (stop_flag_host && iter < MaxIterations) {
        stop_flag_host = false;

        std::iota(edge_indices_usm, edge_indices_usm + E, 0);

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(E), [=](sycl::id<1> idx) {
                size_t e = edge_indices_usm[idx];
                uint8_t label_counts[MaxLabels] = {0};

                for (size_t v = 0; v < N; ++v) {
                    if (incidence_matrix_T[e * N + v] == 1) {
                        uint8_t lbl = vlabels_usm[v];
                        if (lbl < MaxLabels && lbl != std::numeric_limits<uint8_t>::max()) {
                            label_counts[lbl]++;
                        }
                    }
                }

                uint8_t max_count = 0, best_label = std::numeric_limits<uint8_t>::max();
                for (size_t i = 0; i < MaxLabels; ++i) {
                    if (label_counts[i] > max_count) {
                        max_count = label_counts[i];
                        best_label = i;
                    }
                }

                if (best_label != std::numeric_limits<uint8_t>::max()) {
                    helabels_usm[e] = best_label;
                }
            });
        }).wait();

        std::iota(vertex_indices_usm, vertex_indices_usm + N, 0);
        *stop_flag_device = 0;

        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                size_t v = vertex_indices_usm[idx];
                uint8_t label_counts[MaxLabels] = {0};

                for (size_t e = 0; e < E; ++e) {
                    if (incidence_matrix_T[e * N + v] == 1) {
                        uint8_t lbl = helabels_usm[e];
                        if (lbl < MaxLabels && lbl != std::numeric_limits<uint8_t>::max()) {
                            label_counts[lbl]++;
                        }
                    }
                }

                uint8_t max_count = 0;
                uint8_t best_label = vlabels_usm[v];

                for (size_t i = 0; i < MaxLabels; ++i) {
                    if (label_counts[i] > max_count) {
                        max_count = label_counts[i];
                        best_label = i;
                    }
                }

                if (vlabels_usm[v] != best_label && best_label != std::numeric_limits<uint8_t>::max()) {
                    vlabels_usm[v] = best_label;

                    sycl::atomic_ref<int,
                                     sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>
                        atomic_flag(*stop_flag_device);
                    atomic_flag.store(1);
                }
            });
        }).wait();

        stop_flag_host = (*stop_flag_device != 0);
        iter++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Tempo totale (ms): " << total_time_ms << std::endl;

    std::copy(vlabels_usm, vlabels_usm + N, H.vertex_labels.begin());
    std::copy(helabels_usm, helabels_usm + E, H.hyperedge_labels.begin());

    sycl::free(incidence_matrix_T, q);
    sycl::free(vlabels_usm, q);
    sycl::free(helabels_usm, q);
    sycl::free(edge_indices_usm, q);
    sycl::free(vertex_indices_usm, q);
    sycl::free(stop_flag_device, q);
}
