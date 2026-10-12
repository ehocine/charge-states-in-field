#!/bin/zsh
# Recheck of N = 3, 4 ground states after the Davidson start-vector fix (two lanes)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
S=scripts/h3_davidson_recheck.py; O=results/harden/recheck.jsonl
( python3 $S $O 3 D10 0.0 0.32 0.5 1.0 2.0 3.0 5.0; python3 $S $O 3 nogate 0.5 1.0 2.0 5.0 ) > results/harden/recheck_a.log 2>&1 &
( python3 $S $O 4 D10 0.1 0.3 0.5 1.0 3.0 5.0 ) > results/harden/recheck_b.log 2>&1 &
wait
echo all done > results/harden/recheck.done
