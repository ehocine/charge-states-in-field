#!/bin/zsh
# Rerun of all many-electron data with the rigorous sector bound (driver.FieldPoint.ground), 2026-10-10.
# Six lanes, one BLAS thread each.  Run from the repository root.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
BS="0.0 0.001 0.002 0.003 0.005 0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.75 0.9 1.0 1.25 1.5 1.75 2.0 2.5 3.0 3.5 4.0 4.5 5.0"
BZ1="0.0 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.75 1.0 1.25 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0"
BNS="0.3 0.5 1.0 2.0 3.0 4.0 5.0"
ZN=1.587817272185
mkdir -p results/rerun_logs
( python3 scripts/f2_scan.py results/scan/nogate.jsonl 30 inf $ZN ${=BS} > results/rerun_logs/A.log 2>&1 ) &
( python3 scripts/f2_scan.py results/scan/D10.jsonl 30 10 2.0 ${=BS} > results/rerun_logs/B.log 2>&1 ) &
( python3 scripts/f2_scan.py results/scan/Z1.jsonl 30 inf 1.0 ${=BZ1} > results/rerun_logs/C.log 2>&1;
  python3 scripts/f2_scan.py results/noBscreen/nogate.jsonl 30 inf $ZN ${=BNS} --no-field-screening >> results/rerun_logs/C.log 2>&1 ) &
( python3 scripts/f2_scan.py results/noBscreen/D10.jsonl 30 10 2.0 ${=BNS} --no-field-screening > results/rerun_logs/D.log 2>&1;
  python3 scripts/f4_four.py results/four/D10.jsonl 30 10 2.0 6 0.0 0.1 0.2 0.3 0.5 1.0 2.0 3.0 4.0 5.0 >> results/rerun_logs/D.log 2>&1;
  python3 scripts/f4_four.py results/four/D10.jsonl 30 10 2.0 8 0.3 1.0 3.0 5.0 >> results/rerun_logs/D.log 2>&1 ) &
( python3 scripts/f5_observables.py results/obs/nogate.jsonl 30 inf $ZN 0.0 0.1 0.3 0.5 1.0 2.0 3.0 5.0 > results/rerun_logs/E.log 2>&1;
  python3 scripts/f5_observables.py results/obs/D10.jsonl 30 10 2.0 0.0 0.1 0.3 0.5 1.0 2.0 3.0 5.0 >> results/rerun_logs/E.log 2>&1;
  python3 scripts/f5_observables.py results/obs/Z1.jsonl 30 inf 1.0 0.1 0.3 0.5 1.0 2.0 3.0 5.0 >> results/rerun_logs/E.log 2>&1 ) &
( C=results/convergence/conv.jsonl
  python3 scripts/f1_convergence.py $C 30 inf $ZN 1.0 n_ll 20 30 40 60 > results/rerun_logs/F.log 2>&1
  python3 scripts/f1_convergence.py $C 30 inf $ZN 1.0 js -0.5:2.5 -2.5:4.5 >> results/rerun_logs/F.log 2>&1
  python3 scripts/f1_convergence.py $C 30 inf $ZN 1.0 R_l 12 20 >> results/rerun_logs/F.log 2>&1
  python3 scripts/f1_convergence.py $C 30 inf $ZN 0.1 e_max0 5 6 8 >> results/rerun_logs/F.log 2>&1
  python3 scripts/f1_convergence.py $C 30 10 2.0 3.0 n_ll 20 30 40 60 >> results/rerun_logs/F.log 2>&1
  python3 scripts/f1_convergence.py $C 30 10 2.0 3.0 js -0.5:2.5 -2.5:4.5 -3.5:5.5 >> results/rerun_logs/F.log 2>&1
  python3 scripts/f1_convergence.py $C 30 10 2.0 3.0 n_max3 8 12 20 24 >> results/rerun_logs/F.log 2>&1
  python3 scripts/f1_convergence.py $C 30 10 2.0 3.0 R_l 12 20 >> results/rerun_logs/F.log 2>&1 ) &
wait
echo all done
