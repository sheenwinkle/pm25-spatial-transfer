#!/bin/bash
cd "$(dirname "$0")" || exit 1
python3 bench_harness.py --task yrd --protocol B --ks 9,11 --pass core
python3 bench_harness.py --task yrd --protocol S --ks 9,11 --pass core
python3 bench_harness.py --task aus --protocol S --pass core
python3 bench_harness.py --task yrd --protocol B --ks 9,11 --pass kmm
python3 bench_harness.py --task yrd --protocol S --ks 9,11 --pass kmm
python3 bench_harness.py --task aus --protocol S --pass kmm
echo CHAIN_B_DONE
