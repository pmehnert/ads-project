#!/bin/bash

# todo maybe use sth like 2^i * 1000 for benchmarks instead 
# todo this way, inputs wouldn't be exact powers of two

bin="cargo run --release gen"

# 2^10 -- 2^30
test_sizes="
1000:1K
2000:2K
4000:4K
8000:8K
16000:16K
32000:32K
64000:64K
128000:128K
256000:256K
512000:512K
1024000:1M
2048000:2M
4096000:4M
8192000:8M
16384000:16M
32768000:32M
65536000:64M
131072000:128M
262144000:256M"

algo="${1}"
base_path="${3}"

mkdir -p "${base_path}"

for size in $test_sizes; do
    num_values=${size%:*}
    num_values_readable=${size#*:}
    num_queries=${2}
    input_path="${base_path}/${4}.${num_values_readable}"
    split_bits="${5}"
    ${bin} "${algo}" "${input_path}" ${num_values} ${num_queries} ${split_bits}
done
