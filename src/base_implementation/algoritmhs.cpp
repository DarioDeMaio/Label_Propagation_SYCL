#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <limits>
#include <chrono>
#include <sycl/sycl.hpp>
#include "headers/utils.h"

using namespace sycl;
constexpr std::size_t MaxIterations = 100;

void find_communities(HypergraphNotSparse& H) {
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::enable_profiling{});
    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    uint32_t* incidence_matrix_usm = sycl::malloc_shared<uint32_t>(N * E, q);
    uint32_t* vlabels_usm = sycl::malloc_shared<uint32_t>(N, q);
    uint32_t* helabels_usm = sycl::malloc_shared<uint32_t>(E, q);
    size_t* edge_indices_usm = sycl::malloc_shared<size_t>(E, q);
    size_t* vertex_indices_usm = sycl::malloc_shared<size_t>(N, q);

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < E; ++j) {
            incidence_matrix_usm[i * E + j] = (j < H.incidence_matrix[i].size()) ? H.incidence_matrix[i][j] : 0;
        }
    }
    std::copy(H.vertex_labels.begin(), H.vertex_labels.end(), vlabels_usm);
    std::copy(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), helabels_usm);

    bool stop_flag_host = true;
    size_t iter = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (stop_flag_host && iter < MaxIterations) {
        stop_flag_host = true;

        std::vector<size_t> edge_indices(E);
        std::iota(edge_indices.begin(), edge_indices.end(), 0);
        std::copy(edge_indices.begin(), edge_indices.end(), edge_indices_usm);

        sycl::event edge_event = q.submit([&](sycl::handler& h) {
            h.parallel_for(range<1>(E), [=](id<1> idx) {
                size_t e = edge_indices_usm[idx];
                uint32_t label_counts[1024] = {0};

                for (size_t v = 0; v < N; ++v) {
                    if (incidence_matrix_usm[v * E + e] == 1) {
                        uint32_t lbl = vlabels_usm[v];
                        if (lbl < 1024 && lbl != std::numeric_limits<uint32_t>::max()) {
                            label_counts[lbl]++;
                        }
                    }
                }

                uint32_t max_count = 0, best_label = std::numeric_limits<uint32_t>::max();
                for (size_t i = 0; i < 1024; ++i) {
                    if (label_counts[i] > max_count) {
                        max_count = label_counts[i];
                        best_label = i;
                    }
                }

                if (best_label != std::numeric_limits<uint32_t>::max())
                    helabels_usm[e] = best_label;
            });
        });
        edge_event.wait();

        std::vector<size_t> vertex_indices(N);
        std::iota(vertex_indices.begin(), vertex_indices.end(), 0);
        std::copy(vertex_indices.begin(), vertex_indices.end(), vertex_indices_usm);

        bool* stop_flag_device = sycl::malloc_shared<bool>(1, q);
        *stop_flag_device = true;

        sycl::event vertex_event = q.submit([&](sycl::handler& h) {
            h.parallel_for(range<1>(N), [=](id<1> idx) {
                size_t v = vertex_indices_usm[idx];
                uint32_t label_counts[1024] = {0};

                for (size_t e = 0; e < E; ++e) {
                    if (incidence_matrix_usm[v * E + e] == 1) {
                        uint32_t lbl = helabels_usm[e];
                        if (lbl < 1024 && lbl != std::numeric_limits<uint32_t>::max()) {
                            label_counts[lbl]++;
                        }
                    }
                }

                uint32_t max_count = 0, best_label = vlabels_usm[v];
                for (size_t i = 0; i < 1024; ++i) {
                    if (label_counts[i] > max_count) {
                        max_count = label_counts[i];
                        best_label = i;
                    }
                }

                if (vlabels_usm[v] != best_label && best_label != std::numeric_limits<uint32_t>::max()) {
                    vlabels_usm[v] = best_label;
                    *stop_flag_device = false;
                }
            });
        });
        vertex_event.wait();

        stop_flag_host = !(*stop_flag_device);
        sycl::free(stop_flag_device, q);

        iter++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Tempo totale (ms): " << total_time_ms << std::endl;

    std::copy(vlabels_usm, vlabels_usm + N, H.vertex_labels.begin());
    std::copy(helabels_usm, helabels_usm + E, H.hyperedge_labels.begin());

    sycl::free(incidence_matrix_usm, q);
    sycl::free(vlabels_usm, q);
    sycl::free(helabels_usm, q);
    sycl::free(edge_indices_usm, q);
    sycl::free(vertex_indices_usm, q);
}

void find_communities_transpose(HypergraphNotSparse& H) {
    sycl::queue q;
    try {
        q = sycl::queue(sycl::device{sycl::gpu_selector_v}, sycl::property::queue::enable_profiling{});
    } catch (sycl::exception const& e) {
        std::cout << "GPU non disponibile, uso la CPU.\n";
        sycl::queue(sycl::device{ sycl::cpu_selector_v });
    }
    const size_t N = H.num_vertices;
    const size_t E = H.num_hyperedges;

    uint32_t* incidence_matrix_usm = sycl::malloc_shared<uint32_t>(E * N, q);
    uint32_t* vlabels_usm = sycl::malloc_shared<uint32_t>(N, q);
    uint32_t* helabels_usm = sycl::malloc_shared<uint32_t>(E, q);
    size_t* edge_indices_usm = sycl::malloc_shared<size_t>(E, q);
    size_t* vertex_indices_usm = sycl::malloc_shared<size_t>(N, q);

    std::cout << "Incidence matrix will be copied." << std::endl;
    for (size_t v = 0; v < N; ++v) {
        for (size_t e = 0; e < E; ++e) {
            incidence_matrix_usm[e * N + v] = (e < H.incidence_matrix[v].size()) ? H.incidence_matrix[v][e] : 0;
        }
    }
    std::cout << "Incidence matrix copied." << std::endl;

    std::copy(H.vertex_labels.begin(), H.vertex_labels.end(), vlabels_usm);
    std::copy(H.hyperedge_labels.begin(), H.hyperedge_labels.end(), helabels_usm);

    bool stop_flag_host = true;
    size_t iter = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (stop_flag_host && iter < MaxIterations) {
        std::cout << "CPU iter: " << iter << std::endl;
        stop_flag_host = true;

        std::vector<size_t> edge_indices(E);
        std::iota(edge_indices.begin(), edge_indices.end(), 0);
        std::copy(edge_indices.begin(), edge_indices.end(), edge_indices_usm);

        std::cout << "First kernel execution." << std::endl;
        q.submit([&](sycl::handler& h) {
            sycl::stream out(1024, 256, h);
            h.parallel_for(range<1>(E), [=](id<1> idx) {
                size_t e = edge_indices_usm[idx];
                out << e << "\n";
                uint32_t label_counts[10] = {0};

                for (size_t v = 0; v < N; ++v) {
                    if (incidence_matrix_usm[e * N + v] == 1) {
                        uint32_t lbl = vlabels_usm[v];
                        if (lbl < 10 && lbl != std::numeric_limits<uint32_t>::max())
                            label_counts[lbl]++;
                    }
                }

                uint32_t max_count = 0, best_label = std::numeric_limits<uint32_t>::max();
                for (size_t i = 0; i < 10; ++i) {
                    if (label_counts[i] > max_count) {
                        max_count = label_counts[i];
                        best_label = i;
                    }
                }

                if (best_label != std::numeric_limits<uint32_t>::max())
                    helabels_usm[e] = best_label;
            });
        }).wait();
        
        std::cout << "Starting the second phase" << std::endl;

        std::vector<size_t> vertex_indices(N);
        std::iota(vertex_indices.begin(), vertex_indices.end(), 0);
        std::copy(vertex_indices.begin(), vertex_indices.end(), vertex_indices_usm);

        bool* stop_flag_device = sycl::malloc_shared<bool>(1, q);
        *stop_flag_device = true;

        q.submit([&](sycl::handler& h) {
            h.parallel_for(range<1>(N), [=](id<1> idx) {
                size_t v = vertex_indices_usm[idx];
                uint32_t label_counts[10] = {0};

                for (size_t e = 0; e < E; ++e) {
                    if (incidence_matrix_usm[e * N + v] == 1) {
                        uint32_t lbl = helabels_usm[e];
                        if (lbl < 10 && lbl != std::numeric_limits<uint32_t>::max())
                            label_counts[lbl]++;
                    }
                }

                uint32_t max_count = 0, best_label = vlabels_usm[v];
                for (size_t i = 0; i < 10; ++i) {
                    if (label_counts[i] > max_count) {
                        max_count = label_counts[i];
                        best_label = i;
                    }
                }

                if (vlabels_usm[v] != best_label && best_label != std::numeric_limits<uint32_t>::max()) {
                    vlabels_usm[v] = best_label;
                    *stop_flag_device = false;
                }
            });
        }).wait();

        stop_flag_host = !(*stop_flag_device);
        sycl::free(stop_flag_device, q);
        iter++;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Tempo totale (ms): " << total_time_ms << std::endl;

    std::copy(vlabels_usm, vlabels_usm + N, H.vertex_labels.begin());
    std::copy(helabels_usm, helabels_usm + E, H.hyperedge_labels.begin());

    sycl::free(incidence_matrix_usm, q);
    sycl::free(vlabels_usm, q);
    sycl::free(helabels_usm, q);
    sycl::free(edge_indices_usm, q);
    sycl::free(vertex_indices_usm, q);
}

