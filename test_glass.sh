#!/bin/bash

for file in Matrices_real_target/glass/*
do
    for Q in 100 150 200
    do
	for I in 10 15 20
	do
            for K in 3 6 9 12 15 18
	    do
		for lambda in 0.0 0.2 0.5
		do
                    case $file in
                        Matrices_real_target/glass/glass_matrix_alphabet_NNd___1and2chains.csv)
                            for C in 0.9994 0.996 0.9998
                            do
                                ./a.out $Q $C $I $K $lambda Descriptors/glass-descriptors.csv $file 0
                            done
                            ;;
                        Matrices_real_target/glass/glass_matrix_alphabet_NNd___1and2and3chains.csv | Matrices_real_target/glass/glass_matrix_alphabet_NNdt__1and2chains.csv | Matrices_real_target/glass/glass_matrix_alphabet_NNdt__1and2and3chains.csv)
                            for C in 0.9993 0.9995 0.9997
                            do
                                ./a.out $Q $C $I $K $lambda Descriptors/glass-descriptors.csv $file 0
                            done
                            ;;
                        *)
                            echo -n "unknown"
                            ;;
                    esac
		done
	    done
	done
    done
done

rm a.out
