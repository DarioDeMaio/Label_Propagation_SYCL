# Label Propagation on Hypergraphs using SYCL

This project implements the Label Propagation Algorithm (LPA) for hypergraphs using SYCL for parallel execution. It was developed as part of a High-Performance Computing (HPC) course, focusing on both algorithmic correctness and performance on heterogeneous hardware.

## Project Overview

The Label Propagation Algorithm is a community detection method for graphs and hypergraphs. In this version, both vertices and hyperedges participate in the labeling process:

1. Each hyperedge updates its label based on the most frequent label among its incident vertices.
2. Each vertex then updates its label based on the most frequent label among its incident hyperedges.
3. The process iterates until convergence or until a specified maximum number of iterations is reached.

## Hypergraph Representation

The hypergraph is stored using incidence matrices, where each hyperedge is represented by the set of nodes it connects. This representation is well-suited for implementing label propagation and allows straightforward parallelization of vertex and hyperedge updates.

## Optimizations

The project introduces several performance optimizations in a progressive manner:

### Baseline Implementation - Incidence Matrix

The algorithm is first implemented using the raw incidence matrix. While simple and functional, this layout can lead to inefficient memory access patterns, particularly on GPUs.

### Transpose Optimization

The first optimization step transposes the incidence structure to enable coalesced memory access, especially for GPU execution. Accessing memory in a contiguous fashion reduces latency and improves throughput by leveraging the GPU's memory hierarchy more effectively.

### Compiler Optimization Flags (-O2)

The second optimization leverages the compiler's built-in optimizations via the -O2 flag. This enables loop unrolling, instruction-level parallelism, and better register allocation, improving performance with no changes to the source code.

### Load Balancing via Hyperedge Grouping

The third optimization balances the workload across GPU workgroups by ensuring that each group processes hyperedges of similar cardinality. This avoids idle threads and improves overall efficiency by reducing the execution time variance across workgroups.

## Building and Running

To compile the code with SYCL and optimizations:
```
icpx -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda "2_baseline_label_propagation_transpose.cpp" "../base_implementation/algorithms.cpp" "../base_implementation/utils.cpp" -o "label_prop.exe"
./"label_prop.exe" num_nodes num_hyperedges density
```