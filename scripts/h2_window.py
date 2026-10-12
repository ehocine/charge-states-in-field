"""Three-electron window with the gate (D = 10 nm, Z = 2) in the complete box basis.

    python3 scripts/h2_window.py          -> results/harden/window.json

Inputs: results/harden/three_sectors_D10.jsonl (scripts/h1_three_sectors.py, n_max = 16: energies of
the K' pair + K electron and of the all-K' state) and results/harden/nmax3_D10.jsonl (scripts/
f1_convergence.py, n_max3 = 12 ... 32 and 40, i.e. all states of every channel).  The truncation
error of mu_3 at n_max = 16 is taken from the complete-basis runs: for the K' pair + K branch at
0.25 and 0.30 T, for the all-K' branch (n_max = 32, converged to 1e-4 meV per step) at 0.32-0.40 T.
With these corrections the gap between the branches and mu_3 of the K' pair + K branch are
interpolated linearly between 0.30 and 0.32 T, giving the valley transition and the field where
mu_3 reaches Delta.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = os.path.join(ROOT, "results", "harden")
DELTA = 15.0


def load(name):
    return [json.loads(x) for x in open(os.path.join(H, name)) if x.strip()]


def main():
    sec = {round(r["B"], 6): r for r in load("three_sectors_D10.jsonl") if r["n_max"] == 16}
    conv = {}
    for r in load("nmax3_D10.jsonl"):
        conv[(round(r["B"], 6), int(float(r["value"])))] = r
    mixed = lambda r: len(set(r["N3"]["valleys"])) > 1
    # truncation error of mu_3 at n_max = 16
    d_mixed = [conv[(B, 40)]["mu3"] - conv[(B, 16)]["mu3"] for B in (0.25, 0.30)
               if (B, 40) in conv and mixed(conv[(B, 16)]) and mixed(conv[(B, 40)])]
    d_one = [conv[(B, 32)]["mu3"] - conv[(B, 16)]["mu3"] for B in (0.32, 0.36, 0.40)
             if (B, 32) in conv and not mixed(conv[(B, 16)])]
    dm, do = float(np.mean(d_mixed)), float(np.mean(d_one))
    # gap (all-K' minus K' pair + K) and mu_3 of the K' pair + K branch, complete basis
    Bs = [0.30, 0.32]
    gap16 = {B: sec[B]["allKp_M-5/2"] - sec[B]["Kp_pair+K"] for B in Bs}
    gap = {B: gap16[B] - dm + do for B in Bs}
    mu3_mixed = {0.30: conv[(0.30, 40)]["mu3"],
                 0.32: conv[(0.32, 16)]["mu3"] - gap16[0.32] + dm}     # n_max = 16 ground is all-K' at 0.32 T
    lin = lambda y, level: Bs[0] + (level - y[Bs[0]]) * (Bs[1] - Bs[0]) / (y[Bs[1]] - y[Bs[0]])
    B_valley = lin(gap, 0.0)
    B_cross = lin(mu3_mixed, DELTA)
    one_K = {B: conv[(B, 32)]["mu3"] for B in (0.32, 0.36, 0.40, 0.44) if (B, 32) in conv}
    res = {"dmu3_mixed_n16_to_full_meV": d_mixed, "dmu3_allKp_n16_to_n32_meV": d_one,
           "gap_n16_meV": gap16, "gap_full_meV": gap, "mu3_mixed_full_meV": mu3_mixed,
           "B_valley_T": B_valley, "B_mu3_Delta_T": B_cross, "mu3_allKp_n32_meV": one_K,
           "mu3_full_meV": {str(B): conv[(B, 40)]["mu3"] for B in (0.25, 0.30) if (B, 40) in conv}}
    json.dump(res, open(os.path.join(H, "window.json"), "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
