#!/bin/bash

set -u

run="cargo run --release"
check="cargo -Z unstable-options -C scripts run --release check"

name="${1}"
problem="${2}"
algo="${3}"
input="${4}"
output="${5}"

${run} "${name}" "${problem}" "${algo}" "${input}" "${output}"
${check} "${problem}" "${input}" "${output}"

