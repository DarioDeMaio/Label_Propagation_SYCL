num_runs=10

# Baseline

# log_dir="../logs/base"

# mkdir -p "$log_dir"

# for i in $(seq 1 $num_runs)
# do
#   echo "Compilazione run #$i..."
#   icpx -fsycl 1_baseline_label_propagation.cpp ../base_implementation/algoritmhs.cpp ../base_implementation/utils.cpp -o label_prop

#   echo "Esecuzione run #$i..."
#   ./label_prop > "$log_dir/base_$i.log"
  
#   echo "Run #$i completata. Log salvato in base_$i.log"
# done

# First optimization

log_dir="../logs/first_optimization"

mkdir -p "$log_dir"

for i in $(seq 1 $num_runs)
do
  echo "Compilazione run #$i..."
  icpx -fsycl 2_baseline_label_propagation_transpose.cpp ../base_implementation/algoritmhs.cpp ../base_implementation/utils.cpp -o label_prop

  echo "Esecuzione run #$i..."
  ./label_prop > "$log_dir/base_$i.log"
  
  echo "Run #$i completata. Log salvato in base_$i.log"
done