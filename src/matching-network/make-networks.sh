#!/bin/bash
for planes in 10 20 100 200 1000; do
    for training in "True" "False"; do
      python matching_network_lsh_training.py $training $classes $planes
    done
  done
done

for classes in `seq 3 10 3`; do
    python matching_network_lsh_one_all_training.py $classes $period
  done
done
