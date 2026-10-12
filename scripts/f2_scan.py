"""Ground states of N = 1, 2, 3 electrons on the impurity versus field.

    python3 scripts/f2_scan.py OUT.jsonl GAP_meV D_nm Z B1 [B2 ...] [--n_ll 40] [--n_max3 16]
                               [--js -1.5,...] [--no-field-screening]

Graphene on hBN (kappa = 2.5), impurity at d = 0.3 nm, Dirac-sea RPA screening in the field
(Landau-level polarization, fieldstates.screening), optional gate at distance D (inf: none).
One JSON line per field: one-electron levels of every channel (meV), and for N = 2 (full basis)
and N = 3 (n_max3 states per channel, with N = 2 at the same truncation for consistent addition
energies) every sector solved (valleys, exchange symmetry, M, spin S, energies in meV incl. spin
Zeeman).  Energies in meV.
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


def recs(results):
    return [{**{k: (list(v) if isinstance(v, tuple) else v) for k, v in r[0].items()}} for r in results]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("gap", type=float)
    ap.add_argument("D", type=float)
    ap.add_argument("Z", type=float)
    ap.add_argument("B", type=float, nargs="+")
    ap.add_argument("--n_ll", type=float, default=40)
    ap.add_argument("--R_l", type=float, default=15.0)
    ap.add_argument("--e_max0", type=float, default=6.0)
    ap.add_argument("--n_max3", type=int, default=16)
    ap.add_argument("--n_max2", type=int, default=0, help="states per channel for N = 2 (0: all)")
    ap.add_argument("--js", default="-1.5,-0.5,0.5,1.5,2.5,3.5")
    ap.add_argument("--no-field-screening", action="store_true")
    ap.add_argument("--skip3", action="store_true")
    a = ap.parse_args()
    mat = gapped_graphene(a.gap / 2000.0)
    js = [float(x) for x in a.js.split(",")]
    for B in a.B:
        t = time.time()
        P = FieldPoint(mat, a.Z, B, D=a.D, js=js, n_ll=a.n_ll, R_l=a.R_l, e_max0=a.e_max0,
                       field_screening=not a.no_field_screening)
        rec = {"gap_meV": a.gap, "D": a.D, "Z": a.Z, "B": B, "l_nm": P.l, "R_nm": P.R, "Emax_m": P.E_max / P.m,
               "n_ll": a.n_ll, "js": js, "field_screening": not a.no_field_screening,
               "n_states": {str(v): int(len(b.E)) for v, b in P.bases.items()},
               "sea_gap_meV": {f"{v:+d},{j:+.1f}": g * P.mev for v, b in P.bases.items() for j, g in b.sea_gap.items()},
               "N1": P.one_electron()}
        r2 = P.ground(2, n_max=(a.n_max2 or None))
        rec["N2"] = recs(r2)
        if not a.skip3:
            r2t = P.ground(2, n_max=a.n_max3)
            r3 = P.ground(3, n_max=a.n_max3)
            rec["n_max3"] = a.n_max3
            rec["N2_trunc"] = recs(r2t)
            rec["N3"] = recs(r3)
        rec["time"] = time.time() - t
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        mu1 = rec["N1"]["E"]
        mu2 = rec["N2"][0]["E"] - mu1
        msg = f"B={B:.3f} mu1={mu1:.3f} mu2={mu2:.3f}"
        if not a.skip3:
            mu3 = rec["N3"][0]["E"] - rec["N2_trunc"][0]["E"]
            msg += f" mu3={mu3:.3f} ({rec['N3'][0]['valleys']} S={rec['N3'][0]['S']})"
        print(msg + f" N2:{rec['N2'][0]['valleys']} S={rec['N2'][0]['S']} M={rec['N2'][0]['M']}  t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
