"""Finite-field moments M_N = -dE_N/dB (favourable valley K') at the settings of paper 2 (Table IV),
for comparison with the zero-field expectation values <P mu P> reported there.

    python3 scripts/f8_paper2_moments.py OUT.json D Z

E_1: K' level j~ = 1/2; E_2: K'K' singlet (full basis, and n_max = 20); E_3: K' singlet pair plus a
K electron (n_max = 20), the sector of M_3^(20) in paper 2.  Orbital part only (spin Zeeman removed).
Fields 0, 2 and 4 mT (linear: the two finite-difference slopes agree to < 1 mu_B).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from fieldstates.ci import FieldCI  # noqa: E402
from fieldstates.driver import FieldPoint  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

MU_B = 0.0578838
JS = (-1.5, -0.5, 0.5, 1.5, 2.5)


def energies(mat, Z, D, B):
    P = FieldPoint(mat, Z, B, D=D, js=JS, e_max0=6.0)
    b = P.bases[-1]
    out = {"E1": float(b.E[b.select(0.5)[0]]) * P.mev}
    P.ti.compress(P.channels(), tol=1e-9)
    out["E2"] = FieldCI(P.ti, (-1, -1), 1, -1.0).davidson(tol=1e-11)[0] * P.mev
    P.ti.compress(P.channels(n_max=20), tol=1e-9)
    out["E2_20"] = FieldCI(P.ti, (-1, -1), 1, -1.0).davidson(tol=1e-11)[0] * P.mev
    out["E3_20"] = FieldCI(P.ti, (-1, -1, 1), 1, -0.5).davidson(tol=1e-10)[0] * P.mev
    return out


def main():
    out_path, D, Z = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    mat = gapped_graphene(0.015)
    E = {B: energies(mat, Z, D, B) for B in (0.0, 0.002, 0.004)}
    res = {"D": D, "Z": Z, "energies_meV": {str(B): e for B, e in E.items()}, "moments_muB": {}}
    for k in ("E1", "E2", "E2_20", "E3_20"):
        s2 = (E[0.002][k] - E[0.0][k]) / 0.002
        s4 = (E[0.004][k] - E[0.0][k]) / 0.004
        res["moments_muB"][k] = {"2mT": -s2 / MU_B, "4mT": -s4 / MU_B}
    M = {k: res["moments_muB"][k]["4mT"] for k in res["moments_muB"]}
    res["slopes_meV_per_T"] = {"mu1": -M["E1"] * MU_B, "mu2": -(M["E2"] - M["E1"]) * MU_B,
                               "mu3": -(M["E3_20"] - M["E2_20"]) * MU_B}
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res["moments_muB"], indent=1), json.dumps(res["slopes_meV_per_T"]))


if __name__ == "__main__":
    main()
