"""Hardening: energies of the competing three-electron states versus field (gate D = 10 nm, Z = 2).

    python3 scripts/h1_three_sectors.py OUT.jsonl N_MAX B1 [B2 ...]

K' pair + K electron (the zero-field ground state, M = -1/2) and all three in K' with M = -5/2
(orbitals j~ = 1/2, 1/2, 3/2, the ground state above 0.32 T) and M = -3/2 (third electron in the
second j~ = 1/2 orbital, the lowest all-in-one-valley state at B = 0), with the spin Zeeman
energy of S = 1/2 (meV).  The low-field slopes give the linear estimate of the valley transition.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fieldstates.ci import FieldCI  # noqa: E402
from fieldstates.driver import FieldPoint, zeeman  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

STATES = {"Kp_pair+K": ((-1, -1, 1), 1, -0.5), "allKp_M-5/2": ((-1, -1, -1), -1, -2.5),
          "allKp_M-3/2": ((-1, -1, -1), -1, -1.5)}


def main():
    out, n_max = sys.argv[1], int(sys.argv[2])
    mat = gapped_graphene(0.015)
    for B in (float(x) for x in sys.argv[3:]):
        t = time.time()
        P = FieldPoint(mat, 2.0, B, D=10.0)
        P.ti.compress(P.channels(n_max), tol=1e-8)
        rec = {"B": B, "n_max": n_max}
        for name, (valleys, sym, M) in STATES.items():
            e, _ = FieldCI(P.ti, valleys, sym, M).davidson(tol=1e-10)
            rec[name] = e * P.mev + zeeman(B, 0.5)
        rec["time"] = time.time() - t
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
