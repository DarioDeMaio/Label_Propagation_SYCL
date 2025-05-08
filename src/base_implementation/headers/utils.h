#ifndef HYPERGRAPHNOTSPARSE_H
#define HYPERGRAPHNOTSPARSE_H

#include <vector>
#include <cstdint>

struct HypergraphNotSparse
{
    std::size_t num_vertices;
    std::size_t num_hyperedges;
    
    std::vector<std::vector<std::uint32_t>> incidence_matrix;

    std::vector<std::uint32_t> vertex_labels;
    std::vector<std::uint32_t> hyperedge_labels;
};

HypergraphNotSparse generate_hypergraph(std::size_t N, std::size_t E, double p);

#endif
