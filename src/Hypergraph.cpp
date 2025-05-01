#include <vector>
#include <cstdint>

struct Hypergraph
{
    //number of vertices and hyperedges
    std::size_t num_vertices;
    std::size_t num_hyperedges;

    // CSR structure for hyperedges → vertices (E → V)
    std::vector<std::uint32_t> he2v_offsets;
    std::vector<std::uint32_t> he2v_indices;

    // CSR structure for vertices → hyperedges (V → E)
    std::vector<std::uint32_t> v2he_offsets;
    std::vector<std::uint32_t> v2he_indices;

    // Labels
    std::vector<std::uint32_t> vertex_labels;
    std::vector<std::uint32_t> hyperedge_labels;
};
