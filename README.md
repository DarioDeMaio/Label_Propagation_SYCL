# Label Propagation on Hypergraphs using SYCL

This project implements the **Label Propagation Algorithm (LPA)** for **hypergraphs** using **SYCL** for parallel execution. It is part of a High-Performance Computing (HPC) course and focuses on both algorithmic design and performance optimization on modern heterogeneous hardware.

---

## Project Overview

The Label Propagation Algorithm is a community detection method used in graphs and hypergraphs. Each node (and in this case, also each hyperedge) starts with a unique label, and in each iteration:
- Hyperedges update their labels based on the most frequent label among their incident vertices.
- Vertices then update their labels based on the most frequent label among their incident hyperedges.

The process is repeated until convergence or until a maximum number of iterations is reached.

---

## Hypergraph Representation

To enable efficient access and parallelism, the hypergraph is stored using a **CSR-like format**:

- `he2v_offsets` and `he2v_indices`: define the set of vertices connected to each hyperedge.
- `v2he_offsets` and `v2he_indices`: define the set of hyperedges connected to each vertex.
- `vertex_labels` and `hyperedge_labels`: current labels of vertices and hyperedges, updated iteratively.

This representation:
- Reduces memory overhead,
- Allows coalesced memory accesses,
- Is friendly to SYCL parallel kernels.

---

## Optimizations

This project is structured to allow the following HPC-oriented optimizations:

- **Memory layout**: Contiguous arrays for efficient memory access on GPUs/CPUs.
- **Parallelization**: Vertex and hyperedge updates can be parallelized independently.
- **Shuffling**: Label updates are randomized each iteration to improve convergence speed.
- **Early stopping**: The algorithm halts early if no label changes occur.
- **Atomic operations / reduction kernels** (planned): To allow truly parallel label aggregation with correctness guarantees.

---

## Running the Project with Docker

Make sure you have Docker installed. Then, run:

```bash
docker pull intel/oneapi-hpckit:latest

docker run -it --gpus all --rm -v "$(pwd):/workspace" -w /workspace intel/oneapi-hpckit:latest /bin/bash
```

# Compiling and Running the Code

```bash
icpx -fsycl 1_baseline_label_propagation.cpp ../base_implementation/algoritmhs.cpp ../base_implementation/utils.cpp -o label_prop
./main
```

Future Work
----------------

- **Implement timing and performance benchmarking**: Compare the performance of different kernel launches and data layouts.
- **Add support for weighted hyperedges**: Allow edges to have weights, enabling weighted label propagation.
- **Optimize label aggregation using reduction kernels**: Implement a parallel reduction kernel to improve performance.
- **Compare convergence with a serial baseline**: Compare the performance and convergence of the parallel SYCL version with a serial baseline.

This project was developed as part of a university-level High-Performance Computing course. It explores the application of SYCL to a non-trivial algorithm on generalized graph structures (hypergraphs).
