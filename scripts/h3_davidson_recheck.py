"""Recheck of the three- and four-electron ground states after the Davidson start-vector fix.

    python3 scripts/h3_davidson_recheck.py OUT.jsonl N SYSTEM B1 [B2 ...]     (N = 3 or 4; SYSTEM: nogate, D10)

The memory-lean Davidson solver started from a single configuration.  The Hamiltonian and the diagonal
preconditioner conserve the total spin, so where that configuration had pure spin (a doubly occupied
orbital) the iteration could not reach a lower state of another spin.  The solver now adds a small
deterministic random admixture.  This script recomputes the ground states with the settings of
scripts/f2_scan.py (N = 3, n_max3 = 16) or scripts/f4_four.py (N = 4, n4 = 6) and compares them
with the stored results (results/scan, results/four).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from fieldstates.driver import FieldPoint  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEMS = {"nogate": (np.inf, 1.587817272185), "D10": (10.0, 2.0)}
JS = list(np.arange(-1.5, 3.5 + 1e-9, 1.0))


def stored(N, system, B):
    if N == 3:
        path = os.path.join(ROOT, "results", "scan", f"{system}.jsonl")
        for x in open(path):
            r = json.loads(x)
            if np.isclose(r["B"], B):
                return r["N3"][0], r["N2_trunc"][0]["E"]
    else:
        path = os.path.join(ROOT, "results", "four", f"{system}.jsonl")
        for x in open(path):
            r = json.loads(x)
            if np.isclose(r["B"], B) and r.get("n4") == 6:
                return r["N4"] if isinstance(r["N4"], dict) else r["N4"][0], None
    return None, None


def main():
    out, N, system = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    D, Z = SYSTEMS[system]
    mat = gapped_graphene(0.015)
    for B in (float(b) for b in sys.argv[4:]):
        t = time.time()
        P = FieldPoint(mat, Z, B, D=D, js=JS, n_ll=40, R_l=15.0, e_max0=6.0)
        n_max = 16 if N == 3 else 6
        res = P.ground(N, n_max=n_max)
        g = res[0][0]
        old, _ = stored(N, system, B)
        rec = {"N": N, "system": system, "B": B, "n_max": n_max,
               "new": {k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items()},
               "old": old, "time": time.time() - t}
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        o = old or {}
        print(f"N={N} {system} B={B}: new E={g['E']:.4f} {list(g['valleys'])} M={g['M']} S={g['S']}  "
              f"old E={o.get('E', float('nan')):.4f} {o.get('valleys')} M={o.get('M')} S={o.get('S')}  "
              f"dE={g['E'] - o.get('E', np.nan):+.2e}  t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
