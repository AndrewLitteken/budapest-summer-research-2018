#!/bin/bash

curr_dir=$(pwd)
curr_set=$(echo $curr_dir | cut -d "/" -f 12)
echo "classes,samples,period,accuracy" > $curr_set'_matching_network_lsh_one_rest.csv' 
echo "classes,samples,training,planes,accuracy" > $curr_set'_matching_network_lsh_random.csv' 
echo "classes,samples,accuracy" > $curr_set'_matching_network_cosine.csv' 
for i in $(ls | grep o10 ); do
	file=$(echo $i | cut -d "." -f 1)
	third=$(echo $file | cut -d "_" -f 4)
	if [ $third == "one" ]; then
		classes=$(echo $file | cut -d "_" -f 5)		
		samples=$(echo $file | cut -d "_" -f 7)		
		period=$(echo $file | cut -d "_" -f 8)		
	elif [ $third == "random" ]; then
		classes=$(echo $file | cut -d "_" -f 5)		
		samples=$(echo $file | cut -d "_" -f 6)		
		training=$(echo $file | cut -d "_" -f 7)
		planes=$(echo $file | cut -d "_" -f 8)	
	else
		classes=$(echo $file | cut -d "_" -f 4)		
		samples=$(echo $file | cut -d "_" -f 5)		

	fi
	loss=$(cat $i | tail -n 5 | grep LOSS | cut -d " " -f 2)
	acc=$(cat $i | tail -n 5 | grep ACC | cut -d " " -f 2)
	echo $file
	if [ $third == "one" ]; then
		echo $classes','$samples','$period','$acc #>> $curr_set'_matching_network_lsh_one_rest.csv'
	elif [ $third == "random" ]; then
		echo $classes','$samples','$training','$planes','$acc #>> $curr_set'_matching_network_lsh_random.csv'
	else
		echo $classes','$samples','$acc #>> $curr_set'_matching_network_cosine.csv'
	fi
done
