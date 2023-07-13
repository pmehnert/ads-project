#!/bin/bash

set -u
shopt -s nullglob

num_runs=5

base_path="${1}"
test_case="${2}"

cargo build --release

bin="./target/release/ads-project"

for input_name in $(exa -1 -s size "${base_path}/${test_case}/")
do
    input_path="${base_path}/${test_case}/${input_name}"

    for algo in "binary" "elias_fano"
    do
        bin_pd="${bin} ${test_case} pd ${algo} ${input_path} /dev/null"

        for run in $(seq 1 ${num_runs})
        do
            echo "Run ${run}: pd/${algo} (${input_path})"
            if ! ${bin_pd}
            then
                echo "Failed"
            fi
        done
    done
done
