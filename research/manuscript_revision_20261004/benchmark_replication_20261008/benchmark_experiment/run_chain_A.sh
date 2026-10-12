#!/bin/bash
cd "$(dirname "$0")" || exit 1
python3 bench_harness.py --task yrd --protocol B --ks 5,7 --pass core
python3 bench_harness.py --task yrd --protocol S --ks 5,7 --pass core
python3 bench_harness.py --task aus --protocol B --pass core
python3 bench_harness.py --task yrd --protocol B --ks 5,7 --pass kmm
python3 bench_harness.py --task yrd --protocol S --ks 5,7 --pass kmm
python3 bench_harness.py --task aus --protocol B --pass kmm
echo CHAIN_A_DONE
