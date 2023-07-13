#!/bin/bash

set -u
shopt -s nullglob

num_runs=5

base_path="${1}"

cargo build --release

bin="./target/release/ads-project"

naive_sizes="1K 2K 4K 8K 16K 32K 64K"
sparse_sizes="${naive_sizes} 128K 256K 512K 1M 2M 4M 8M 16M 32M 64M 128M"
cartesian_sizes="${sparse_sizes} 256M 512M"

for algo_sizes in "naive:${naive_sizes}" "sparse:${sparse_sizes}" "cartesian:${cartesian_sizes}"
do
    algo="${algo_sizes%:*}"
    input_sizes="${algo_sizes#*:}"

    for input_size in ${input_sizes}
    do
        input_path="${base_path}/rmq/rmq.${input_size}"

        bin_rmq="${bin} rmq_even rmq ${algo} ${input_path} /dev/null"

        for run in $(seq 1 ${num_runs})
        do
            echo "Run ${run}: rmq/${algo} (${input_path})"
            if ! ${bin_rmq}
            then
                echo "Failed"
            fi
        done
    done
done
