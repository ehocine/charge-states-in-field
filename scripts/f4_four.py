"""Fourth addition energy mu_4 = E_4 - E_3 at a reduced basis (n4 states per channel), with mu_3 at
the same truncation for calibration against the n_max3 results of f2_scan.py.

    python3 scripts/f4_four.py OUT.jsonl GAP D Z n4 B1 [B2 ...] [--js -1.5:3.5]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from fieldstates.driver import FieldPoint  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gap", type=float)
    ap.add_argument("D", type=float)
    ap.add_argument("Z", type=float)
    ap.add_argument("n4", type=int)
    ap.add_argument("B", type=float, nargs="+")
    ap.add_argument("--js", default="-1.5:3.5")
    ap.add_argument("--n_ll", type=float, default=40)
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.js.split(":"))
    js = list(np.arange(lo, hi + 1e-9, 1.0))
    mat = gapped_graphene(a.gap / 2000.0)
    for B in a.B:
        t = time.time()
        P = FieldPoint(mat, a.Z, B, D=a.D, js=js, n_ll=a.n_ll)
        rec = {"gap": a.gap, "D": a.D, "Z": a.Z, "B": B, "n4": a.n4, "js": js}
        res = {}
        for N in (2, 3, 4):
            r = P.ground(N, n_max=a.n4)
            res[N] = r[0][0]
            rec[f"N{N}"] = [{k: (list(v) if isinstance(v, tuple) else v) for k, v in x[0].items()} for x in r[:4]]
        rec["mu3"] = res[3]["E"] - res[2]["E"]
        rec["mu4"] = res[4]["E"] - res[3]["E"]
        rec["time"] = time.time() - t
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"B={B} n4={a.n4}: mu3={rec['mu3']:.3f} mu4={rec['mu4']:.3f} N4 {res[4]['valleys']} S={res[4]['S']} "
              f"M={res[4]['M']} t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
