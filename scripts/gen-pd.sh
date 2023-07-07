#!/bin/bash

bin="cargo run --release --bin gen-pd"

# 2^10 -- 2^30
test_sizes="
1024:1K
2048:2K
4096:4K
8192:8K
16384:16K
32768:32KiB
65536:64KiB
131072:128KiB
262144:256KiB
524288:512KiB
1048576:1M
2097152:2M
4194304:4M
8388608:8M
16777216:16M
33554432:32M
67108864:64M
134217728:128M
268435456:256M
536870912:512M
1073741824:1G
"

for size in $test_sizes; do
    num_values=${size%:*}
    human_readable=${size#*:}
    num_queries=1000000
    path_prefix="${1}"
    input_path="${path_prefix}/pd-${human_readable}.in"
    output_path="${path_prefix}/pd-${human_readable}.out"
    ${bin} "${input_path}" "${output_path}" ${num_values} ${num_queries}
done
