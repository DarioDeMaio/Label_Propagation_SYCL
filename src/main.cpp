#include <iostream>
#include "headers/utils.h"

int main() {
    std::size_t N = 5;
    std::size_t E = 3;
    double p = 0.5;

    Hypergraph H = generate_hypergraph(N, E, p);

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

    return 0;
}

