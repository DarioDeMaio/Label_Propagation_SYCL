#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <limits>
#include <chrono>
#include <sycl/sycl.hpp>
#include "headers/utils.h"
#include <iostream>
#include <iomanip> 

using namespace sycl;
constexpr std::size_t MaxIterations = 100;
constexpr size_t TILE_SIZE = 16;
constexpr size_t WorkGroupSize = 128;
constexpr std::size_t MaxLabels = 16;

void find_communities(HypergraphNotSparse& H) {
    sycl::queue q(sycl::gpu_selector_v);

    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    uint8_t* incidence_matrix_dev = sycl::malloc_device<uint8_t>(N * E, q);
    uint8_t* vlabels_dev = sycl::malloc_device<uint8_t>(N, q);
    uint8_t* helabels_dev = sycl::malloc_device<uint8_t>(E, q);
    int* stop_flag_dev = sycl::malloc_device<int>(1, q);

    constexpr uint8_t INVALID_LABEL = std::numeric_limits<uint8_t>::max();

    std::vector<uint8_t> flat_incidence(N * E, 0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < E; ++j)
            flat_incidence[i * E + j] = H.incidence_matrix[i][j];

    q.memcpy(incidence_matrix_dev, flat_incidence.data(), N * E * sizeof(uint8_t)).wait();
    q.memcpy(vlabels_dev, H.vertex_labels.data(), N * sizeof(uint8_t)).wait();
    q.memcpy(helabels_dev, H.hyperedge_labels.data(), E * sizeof(uint8_t)).wait();

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
                            if (lbl < MaxLabels && lbl != INVALID_LABEL) {
                                label_counts[lbl]++;
                            }
                        }
                    }

                    uint8_t max_count = 0, best_label = INVALID_LABEL;
                    for (size_t i = 0; i < MaxLabels; ++i) {
                        if (label_counts[i] > max_count) {
                            max_count = label_counts[i];
                            best_label = i;
                        }
                    }

                    if (best_label != INVALID_LABEL) {
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
                            if (lbl < MaxLabels && lbl != INVALID_LABEL) {
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

                    if (vlabels_dev[v] != best_label && best_label != INVALID_LABEL) {
                        vlabels_dev[v] = best_label;
                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
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
    std::cout << "Total time baseline (ms): " << total_time_ms << std::endl;

    assert(H.vertex_labels.size() == N && "vertex_labels size mismatch");
    assert(H.hyperedge_labels.size() == E && "hyperedge_labels size mismatch");

    q.memcpy(H.vertex_labels.data(), vlabels_dev, N * sizeof(uint8_t)).wait();
    q.memcpy(H.hyperedge_labels.data(), helabels_dev, E * sizeof(uint8_t)).wait();

    sycl::free(incidence_matrix_dev, q);
    sycl::free(vlabels_dev, q);
    sycl::free(helabels_dev, q);
    sycl::free(stop_flag_dev, q);
}

bool checkTransposeCorrectness(uint8_t* originalDev,
                               uint8_t* transposedDev,
                               sycl::queue& q,
                               size_t N, size_t E) {
    std::vector<uint8_t> originalHost(N * E);
    std::vector<uint8_t> transposedHost(E * N);

    q.memcpy(originalHost.data(), originalDev, N * E * sizeof(uint8_t)).wait();
    q.memcpy(transposedHost.data(), transposedDev, E * N * sizeof(uint8_t)).wait();

    for (size_t v = 0; v < N; ++v) {
        for (size_t e = 0; e < E; ++e) {
            uint8_t orig = originalHost[v * E + e];
            uint8_t transp = transposedHost[e * N + v];
            if (orig != transp) {
                std::cerr << "Mismatch at original[" << v << "][" << e << "] = "
                          << (int)orig << " vs transpose[" << e << "][" << v << "] = "
                          << (int)transp << "\n";
                return false;
            }
        }
    }

    // std::cout << "Transpose verification: OK \n";
    return true;
}

void transpose_incidence_matrix(sycl::queue& q, const std::vector<std::vector<uint8_t>>& incidence_matrix, uint8_t* incidence_matrix_T, uint8_t* incidence_matrix_dev, size_t N, size_t E) {
    std::vector<uint8_t> incidence_flat(N * E);

    for (size_t v = 0; v < N; ++v) {
        for (size_t e = 0; e < E; ++e) {
            incidence_flat[v * E + e] = incidence_matrix[v][e];
        }
    }
    q.memcpy(incidence_matrix_dev, incidence_flat.data(), N * E * sizeof(uint8_t)).wait();

    auto start_time = std::chrono::high_resolution_clock::now();

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint8_t, 2> tile(sycl::range<2>(TILE_SIZE, TILE_SIZE), h);

        h.parallel_for(sycl::nd_range<2>(
            sycl::range<2>((E + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE,
                           (N + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE),
            sycl::range<2>(TILE_SIZE, TILE_SIZE)),
            [=](sycl::nd_item<2> item) {
                size_t global_e = item.get_global_id(0);
                size_t global_v = item.get_global_id(1);

                size_t local_e = item.get_local_id(0);
                size_t local_v = item.get_local_id(1);

                if (global_v < N && global_e < E) {
                    tile[local_v][local_e] = incidence_matrix_dev[global_v * E + global_e];
                } else {
                    tile[local_v][local_e] = 0;
                }

                item.barrier(sycl::access::fence_space::local_space);

                size_t transposed_v = item.get_group(1) * TILE_SIZE + local_v;
                size_t transposed_e = item.get_group(0) * TILE_SIZE + local_e;

                if (transposed_v < N && transposed_e < E) {
                    incidence_matrix_T[transposed_e * N + transposed_v] = tile[local_v][local_e];
                }
            });
    }).wait();

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    // std::cout << "Tempo per trasporre la matrice (ms): " << duration_ms << std::endl;
}

void find_communities_transpose(HypergraphNotSparse& H) {
    sycl::queue q(sycl::gpu_selector_v);

    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    uint8_t* incidence_matrix_dev = sycl::malloc_device<uint8_t>(N * E, q);
    uint8_t* incidence_matrix_T_dev = sycl::malloc_device<uint8_t>(E * N, q);
    uint8_t* vlabels_dev = sycl::malloc_device<uint8_t>(N, q);
    uint8_t* helabels_dev = sycl::malloc_device<uint8_t>(E, q);
    int* stop_flag_dev = sycl::malloc_device<int>(1, q);

    constexpr uint8_t INVALID_LABEL = std::numeric_limits<uint8_t>::max();

    std::vector<uint8_t> flat_incidence(N * E, 0);
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < E; ++j)
            flat_incidence[i * E + j] = H.incidence_matrix[i][j];

    q.memcpy(incidence_matrix_dev, flat_incidence.data(), N * E * sizeof(uint8_t)).wait();
    q.memcpy(vlabels_dev, H.vertex_labels.data(), N * sizeof(uint8_t)).wait();
    q.memcpy(helabels_dev, H.hyperedge_labels.data(), E * sizeof(uint8_t)).wait();

    std::vector<int> stop_flag_host(1);

    transpose_incidence_matrix(q, H.incidence_matrix, incidence_matrix_T_dev, incidence_matrix_dev, N, E);

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
                            if (lbl < MaxLabels && lbl != INVALID_LABEL) {
                                label_counts[lbl]++;
                            }
                        }
                    }

                    uint8_t max_count = 0, best_label = INVALID_LABEL;
                    for (size_t i = 0; i < MaxLabels; ++i) {
                        if (label_counts[i] > max_count) {
                            max_count = label_counts[i];
                            best_label = i;
                        }
                    }

                    if (best_label != INVALID_LABEL) {
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
                        if (incidence_matrix_T_dev[e * N + v] == 1) {
                            uint8_t lbl = helabels_dev[e];
                            if (lbl < MaxLabels && lbl != INVALID_LABEL) {
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

                    if (vlabels_dev[v] != best_label && best_label != INVALID_LABEL) {
                        vlabels_dev[v] = best_label;
                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
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
    std::cout << "Total time transpose (ms): " << total_time_ms << std::endl;

    assert(H.vertex_labels.size() == N && "vertex_labels size mismatch");
    assert(H.hyperedge_labels.size() == E && "hyperedge_labels size mismatch");

    q.memcpy(H.vertex_labels.data(), vlabels_dev, N * sizeof(uint8_t)).wait();
    q.memcpy(H.hyperedge_labels.data(), helabels_dev, E * sizeof(uint8_t)).wait();

    sycl::free(incidence_matrix_dev, q);
    sycl::free(vlabels_dev, q);
    sycl::free(helabels_dev, q);
    sycl::free(stop_flag_dev, q);
}