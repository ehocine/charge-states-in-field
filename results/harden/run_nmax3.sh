#!/bin/zsh
# Hardening: n_max convergence of mu_3 in the window where it lies within 0.07 meV of Delta
# (gate D = 10 nm, Z = 2), two lanes.
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
S=scripts/f1_convergence.py; O=results/harden
( for B in 0.30 0.36 0.44; do python3 $S $O/nmax3_D10.jsonl 30 10 2 $B n_max3 12 16 20 24 28 32; done ) > $O/nmax3_a.log 2>&1 &
( for B in 0.32 0.40 0.25; do python3 $S $O/nmax3_D10.jsonl 30 10 2 $B n_max3 12 16 20 24 28 32; done ) > $O/nmax3_b.log 2>&1 &
wait
echo all done
