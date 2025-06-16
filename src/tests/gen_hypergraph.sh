#!/bin/bash

SOURCE="generate_hypergraph.cpp"
ALGO_SRC="../base_implementation/algorithms.cpp"
UTILS_SRC="../base_implementation/utils.cpp"
EXECUTABLE="label_prop.exe"

clang++ -O2 -fsycl $SOURCE $ALGO_SRC $UTILS_SRC -o $EXECUTABLE

mkdir -p generated_hypergraphs

DENSITY=0.5

declare -a NODES=(1000 2000 3000)
declare -a EDGES=(10000 20000 30000)

for i in ${!NODES[@]}; do
    N=${NODES[$i]}
    E=${EDGES[$i]}
    echo "Generating hypergraph with ${N} nodes and ${E} hyperedges (p=${DENSITY})"
    ./$EXECUTABLE $N $E $DENSITY

    mv incidence_matrix.txt generated_hypergraphs/incidence_matrix_${N}_${E}.txt
    mv nodes_label.txt generated_hypergraphs/nodes_label_${N}_${E}.txt
    mv edges_label.txt generated_hypergraphs/edges_label_${N}_${E}.txt
done

echo "All hypergraphs have been generated and saved in 'generated_hypergraphs'."

