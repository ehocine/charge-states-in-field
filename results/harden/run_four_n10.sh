#!/bin/zsh
# Hardening: mu_4 at 0.1 T (D = 10 nm, Z = 2), where it lies 0.06 meV above Delta, with 10 states per channel
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
/usr/bin/time -l python3 scripts/f4_four.py results/four/D10_n10.jsonl 30 10 2 10 0.1 > results/harden/four_n10.log 2>&1
echo all done >> results/harden/four_n10.log
