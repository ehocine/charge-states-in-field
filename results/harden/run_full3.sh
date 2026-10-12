#!/bin/zsh
# Hardening: three-electron CI in the complete box basis (n_max3 = 40 >= all states per channel)
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
S=scripts/f1_convergence.py; O=results/harden
for B in 0.30 0.25; do /usr/bin/time -l python3 $S $O/nmax3_D10.jsonl 30 10 2 $B n_max3 40; done > $O/full3.log 2>&1
echo all done >> $O/full3.log
