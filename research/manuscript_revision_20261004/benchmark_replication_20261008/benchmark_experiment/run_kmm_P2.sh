#!/bin/bash
cd "$(dirname "$0")" || exit 1
python3 bench_harness.py --task yrd --protocol B --ks 7,9 --pass kmm
for s in $(seq 13 19); do python3 bench_harness.py --task yrd --protocol B --ks 11 --only $s --pass kmm; done
echo KMM_P2_DONE
