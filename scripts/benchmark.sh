#!/bin/bash

set -u
shopt -s nullglob

num_runs=5

base_path="${1}"

bin="cargo run --release"

for input_name in $(exa -1 -s size "${base_path}/pd/")
do
    input_path="${base_path}/pd/${input_name}"

    for algo in "binary" "elias_fano"
    do
        bin_pd="${bin} pd_even pd ${algo} ${input_path} /dev/null"

        for run in $(seq 1 ${num_runs})
        do
            echo "Run ${run}: pd/${algo} (${input_path})"
            if ! ${bin_pd}; then
                echo "Failed"
            fi
        done
    done
done

naive_sizes="1K 2K 4K 8K 16K 32K 64K"
sparse_sizes="${naive_sizes} 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M"
cartesian_sizes="${sparse_sizes} 256M 512M"

algo="naive"
for input_size in ${naive_sizes}
do
    input_path="${base_path}/rmq/rmq.${input_size}"

    bin_rmq="${bin} rmq_even rmq ${algo} ${input_path} /dev/null"

    for run in $(seq 1 ${num_runs})
    do
        echo "Run ${run}: rmq/${algo} (${input_path})"
        if ! ${bin_rmq}; then
            echo "Failed"
        fi
    done
done

algo="sparse"
for input_size in ${sparse_sizes}
do
    input_path="${base_path}/rmq/rmq.${input_size}"

    bin_rmq="${bin} rmq_even rmq ${algo} ${input_path} /dev/null"

    for run in $(seq 1 ${num_runs})
    do
        echo "Run ${run}: rmq/${algo} (${input_path})"
        if ! ${bin_rmq}; then
            echo "Failed"
        fi
    done
done

algo="cartesian"
for input_size in ${cartesian_sizes}
do
    input_path="${base_path}/rmq/rmq.${input_size}"

    bin_rmq="${bin} rmq_even rmq ${algo} ${input_path} /dev/null"

    for run in $(seq 1 ${num_runs})
    do
        echo "Run ${run}: rmq/${algo} (${input_path})"
        if ! ${bin_rmq}; then
            echo "Failed"
        fi
    done
done
