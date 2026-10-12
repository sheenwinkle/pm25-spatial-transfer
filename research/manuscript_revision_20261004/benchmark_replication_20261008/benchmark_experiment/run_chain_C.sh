#!/bin/bash
cd "$(dirname "$0")" || exit 1
for m in 3 6 12; do python3 bench_m.py --task yrd --m $m; done
for m in 3 6 12; do python3 bench_m.py --task aus --m $m; done
echo CHAIN_C_DONE
