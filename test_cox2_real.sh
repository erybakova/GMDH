#!/bin/bash

for file in Matrices_real_target/cox2_real/*
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
                        Matrices_real_target/cox2_real/cox2_matrix_alphabet_NNd___UpTo2chains.csv | Matrices_real_target/cox2_real/cox2_matrix_alphabet_NNdb__UpTo2chains.csv | Matrices_real_target/cox2_real/cox2_matrix_alphabet_NNdb__UpTo3chains.csv | Matrices_real_target/cox2_real/cox2_matrix_alphabet_NNdbr_UpTo3chains.csv)
                            for C in 0.9 0.95 0.99
                            do
                                ./a.out $Q $C $I $K $lambda Descriptors/cox2-descriptors.csv $file 0
                            done
                            ;;
                        Matrices_real_target/cox2_real/cox2_matrix_alphabet_NNd___UpTo3chains.csv)
                            for C in 0.875 0.925 0.975
                            do
                                ./a.out $Q $C $I $K $lambda Descriptors/cox2-descriptors.csv $file 0
                            done
                            ;;
                        Matrices_real_target/cox2_real/cox2_matrix_alphabet_NNdbr_UpTo2chains.csv)
                            for C in 0.89 0.95 0.99
                            do
                                ./a.out $Q $C $I $K $lambda Descriptors/cox2-descriptors.csv $file 0
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
