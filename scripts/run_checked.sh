#!/bin/bash

set -u

run="cargo run --release"
check="cargo -Z unstable-options -C scripts run --release check"

algo="${1}"
input="${2}"
output="${3}"

${run} "${algo}" "${input}" "${output}"
${check} "${algo}" "${input}" "${output}"

