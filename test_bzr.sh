#!/bin/bash

for file in Matrices_bin_target/bzr/*
do
    for Q in 100 150 200
    do
	for I in 10 15 20
	do
            for K in 3 6 9 12 15 18
	    do
		for lambda in 0
		do
                    case $file in
                        Matrices_bin_target/bzr/bzr_matrix_alphabet_NNd___UpTo2chains.csv | Matrices_bin_target/bzr/bzr_matrix_alphabet_NNdb__UpTo2chains.csv | Matrices_bin_target/bzr/bzr_matrix_alphabet_NNdb__UpTo3chains.csv | Matrices_bin_target/bzr/bzr_matrix_alphabet_NNdbr_UpTo3chains.csv)
                            for C in 0.97 0.98 0.99
                            do
                                ./a.out $Q $C $I $K $lambda Descriptors/bzr-descriptors.csv $file 1
                            done
                            ;;
                        Matrices_bin_target/bzr/bzr_matrix_alphabet_NNd___UpTo3chains.csv | Matrices_bin_target/bzr/bzr_matrix_alphabet_NNdbr_UpTo2chains.csv)
                            for C in 0.975 0.985 0.995
                            do
                                ./a.out $Q $C $I $K $lambda Descriptors/bzr-descriptors.csv $file 1
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
