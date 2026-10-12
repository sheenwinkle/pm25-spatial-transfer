#!/bin/bash
cd "$(dirname "$0")" || exit 1
for m in 3 6 12; do python3 bench_n.py --task yrd --m $m; done
echo CHAIN_D_DONE
