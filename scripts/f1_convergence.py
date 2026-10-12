"""Convergence of the field CI in the basis parameters (Appendix).

    python3 scripts/f1_convergence.py OUT.jsonl GAP D Z B PARAM VALUE1 [VALUE2 ...]

PARAM: n_ll (Landau levels below the cutoff), R_l (box radius in units of l), js (channel window,
values like '-1.5:3.5'), n_max3 (states per channel for three electrons), e_max0 (low-field cutoff).
Writes the addition energies mu_1, mu_2, mu_3 (meV) and the ground-state sectors for each value,
others at the defaults of scripts/f2_scan.py.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from fieldstates.driver import FieldPoint  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

DEFAULTS = {"n_ll": 40, "R_l": 15.0, "js": "-1.5:3.5", "n_max3": 16, "e_max0": 6.0}


def js_of(s):
    lo, hi = (float(x) for x in s.split(":"))
    return list(np.arange(lo, hi + 1e-9, 1.0))


def main():
    out, gap, D, Z, B, param = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), \
        float(sys.argv[5]), sys.argv[6]
    mat = gapped_graphene(gap / 2000.0)
    for val in sys.argv[7:]:
        p = dict(DEFAULTS)
        p[param] = val if param == "js" else float(val)
        t = time.time()
        P = FieldPoint(mat, Z, B, D=D, js=js_of(p["js"]), n_ll=p["n_ll"], R_l=p["R_l"], e_max0=p["e_max0"])
        e1 = P.one_electron()["E"]
        r2 = P.ground(2)
        n3 = int(p["n_max3"])
        r2t = P.ground(2, n_max=n3)
        r3 = P.ground(3, n_max=n3)
        rec = {"gap": gap, "D": D, "Z": Z, "B": B, "param": param, "value": val, "settings": p,
               "mu1": e1, "mu2": r2[0][0]["E"] - e1, "mu2_trunc": r2t[0][0]["E"] - e1,
               "mu3": r3[0][0]["E"] - r2t[0][0]["E"],
               "N2": {k: (list(v) if isinstance(v, tuple) else v) for k, v in r2[0][0].items()},
               "N3": {k: (list(v) if isinstance(v, tuple) else v) for k, v in r3[0][0].items()},
               "n_states": {str(v): int(len(b.E)) for v, b in P.bases.items()}, "time": time.time() - t}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{param}={val}: mu1={rec['mu1']:.4f} mu2={rec['mu2']:.4f} mu2(n3)={rec['mu2_trunc']:.4f} "
              f"mu3={rec['mu3']:.4f} t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
