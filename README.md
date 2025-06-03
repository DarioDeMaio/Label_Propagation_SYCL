# Label Propagation on Hypergraphs using SYCL

This project implements the Label Propagation Algorithm (LPA) for hypergraphs using SYCL for parallel execution. It was developed as part of a High-Performance Computing (HPC) course, focusing on both algorithmic correctness and performance on heterogeneous hardware.

## Project Overview

The Label Propagation Algorithm is a community detection method for graphs and hypergraphs. In this version, both vertices and hyperedges participate in the labeling process:

1. Each hyperedge updates its label based on the most frequent label among its incident vertices.
2. Each vertex then updates its label based on the most frequent label among its incident hyperedges.
3. The process iterates until convergence or until a specified maximum number of iterations is reached.

## Hypergraph Representation

The hypergraph is represented using incidence matrices, where each row corresponds to a vertex and each column to a hyperedge. A nonzero entry in the matrix indicates the participation of a vertex in a hyperedge. This structure enables efficient traversal from vertices to hyperedges and vice versa, which is essential for implementing the two-phase label propagation process. To optimize memory usage, the incidence matrix is stored using uint8_t entries, which are sufficient to encode binary participation. This compact representation also enhances cache performance and supports efficient parallel access in SYCL kernels.

## Optimizations

The project introduces several performance optimizations in a progressive manner:

## Baseline Implementation
The initial implementation uses the raw incidence matrix to represent the hypergraph. While this straightforward approach ensures correctness and ease of development, it suffers from suboptimal memory access patterns leading to poor utilization of memory bandwidth and slower execution.

### Transpose Optimization

To address the inefficiencies in memory access, the incidence matrix is transposed using a tiled transposition algorithm. Tiling improves cache locality and enables coalesced memory access, which is critical for achieving high throughput on GPUs. By dividing the matrix into smaller tiles that fit into faster shared or local memory, the implementation minimizes uncoalesced reads and writes, resulting in significantly faster execution for label updates.

### Compiler Optimization Flags (-O2)

The next optimization leverages compiler-level enhancements through the -O2 flag. This enables automatic loop unrolling, instruction reordering, and improved register allocation, without requiring any manual changes to the code. These low-level optimizations further reduce runtime and allow the compiler to exploit hardware-level parallelism more effectively.

## Compiling and Running

To compile the code with SYCL and optimizations:
```
icpx -O2 -fsycl -fsycl-targets=nvptx64-nvidia-cuda "2_baseline_label_propagation_transpose.cpp" "../base_implementation/algorithms.cpp" "../base_implementation/utils.cpp" -o "label_prop.exe"
./"label_prop.exe" num_nodes num_hyperedges density
```

## Profiling on Windows (PowerShell)
To perform performance profiling using NVIDIA Nsight Compute on Windows:

Open the Intel oneAPI command prompt as Administrator.
This ensures all necessary environment variables and permissions are set correctly.

Compile the program using the same **icpx** command shown above.

Run the profiler using **ncu** with the following command:
```
ncu --set full --kernel-name regex:.* --export myprofile.ncu-rep ./label_prop.exe num_nodes num_hyperedges density
```
This captures detailed profiling information for all SYCL kernels and exports the report to myprofile.ncu-rep.

You can then open this file using NVIDIA Nsight Compute GUI for analysis of kernel execution, memory usage, and bottlenecks.


