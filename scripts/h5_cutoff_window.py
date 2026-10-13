"""Cutoff dependence of mu_3 at the three-electron window (gate D = 10 nm, Z = 2).

    python3 scripts/h5_cutoff_window.py OUT.jsonl B N_LL1 [N_LL2 ...] [--all-kprime]

scripts/h2_window.py corrects mu_3 of the K' pair + K electron to the complete box basis at the
production cutoff (n_LL = 40 Landau levels, all ~38 states of every channel).  This script tests
the cutoff itself: with all states of every channel below n_LL Landau levels it computes the
two-electron ground state (K'K' singlet, M = -1) and the three-electron state of the window
(K' pair + K electron, singlet pair, M = -1/2; with --all-kprime the all-K' state, S_z = 1/2 sector,
M = -5/2, S = 1/2), and mu_3 = E_3 - E_2, in meV with spin Zeeman.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from fieldstates.ci import FieldCI  # noqa: E402
from fieldstates.driver import FieldPoint, zeeman  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

JS = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]


def main():
    all_kp = "--all-kprime" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out, B = args[0], float(args[1])
    mat = gapped_graphene(0.015)
    for n_ll in (float(x) for x in args[2:]):
        t = time.time()
        P = FieldPoint(mat, 2.0, B, D=10.0, js=JS, n_ll=n_ll)
        P.ti.compress(P.channels(None), tol=1e-8)
        e2, _ = FieldCI(P.ti, (-1, -1), 1, -1.0).davidson(tol=1e-9)
        c3 = FieldCI(P.ti, (-1, -1, -1), -1, -2.5) if all_kp else FieldCI(P.ti, (-1, -1, 1), 1, -0.5)
        e3, _ = c3.davidson(tol=1e-9)
        E2 = e2 * P.mev + zeeman(B, 0.0)
        E3 = e3 * P.mev + zeeman(B, 0.5)
        n_ch = {j: len(P.bases[-1].select(j)) for j in JS}
        rec = {"B": B, "n_ll": n_ll, "sector": "all-Kprime" if all_kp else "Kprime pair + K", "states_per_channel": n_ch, "size3": c3.size, "E2": E2, "E3": E3,
               "mu3": E3 - E2, "time": time.time() - t}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"B={B} n_LL={n_ll:g}: states/channel {min(n_ch.values())}-{max(n_ch.values())}, size {c3.size}, "
              f"mu3 = {rec['mu3']:.4f} meV  t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
