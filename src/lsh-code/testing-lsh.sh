#!/bin/bash

random_filename="random_results.csv"
trained_filename="trained_results.csv"

echo "method,iteration,nPlanes,effectiveness,cosine acc,lsh acc" > $random_filename
echo "method,iteration,nTrained,effectiveness,cosine acc,lsh acc" > $trained_filename

amount_done=0
amount_to_add=0.44
for method in "random" "trained"; do
    for iter in 100 200 500 1000 2000 5000 10000; do
        if [ "$method" = "random" ]; then
            for nPlanes in 10 20 50 100 200 500 1000 2000 5000; do
                data=$(python lsh_implementation_testing.py $method $nPlanes $iter | tail -n 3)
                cos_acc=$(echo $data | grep Cos | cut -d " " -f 3)
                lsh_acc=$(echo $data | grep LSH | cut -d " " -f 6)
                effect=$(echo $data | grep Effect | cut -d " " -f 8)
                echo "$method,$iter,$nPlanes,$effect,$cos_acc,$lsh_acc" >> $random_filename
                amount_done=$(bc <<< "$amount_done+$amount_to_add")
                echo "$amount_done% done"
            done
        elif [ "$method" = "trained" ]; then
            for nPassed in 1000 2000 5000 10000 20000 50000 55000; do
                data=$(python lsh_implementation_testing.py $method $nPassed $iter | tail -n 3)
                cos_acc=$(echo $data | grep Cos | cut -d " " -f 3)
                lsh_acc=$(echo $data | grep LSH | cut -d " " -f 6)
                effect=$(echo $data | grep Effect | cut -d " " -f 8)
                echo "$method,$iter,$nPassed,$effect,$cos_acc,$lsh_acc" >> $trained_filename
                amount_done=$(bc <<< "$amount_done+$amount_to_add")
                echo "$amount_done% done"
            done
        fi 
    done
done
