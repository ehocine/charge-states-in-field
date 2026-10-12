"""Observables of the ground states at selected fields: sea-edge lift delta_N, Dyson orbital of the
1 -> 2 addition (quasiparticle weight, mean radius, density near the impurity), mean radii.

    python3 scripts/f5_observables.py OUT.jsonl GAP D Z B1 [B2 ...]

delta_N (meV): largest lift of the K zero-mode Landau level (top of the sea at -Delta) in the mean
field of the impurity and the N-electron ground state (fieldstates.observables.sea_edge_shift);
the charge-state diagram holds for E_F > -Delta + delta_N.  N = 2 uses the full basis, N = 3 the
n_max = 16 truncation.  Same settings as f2_scan.py.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from fieldstates import observables as ob  # noqa: E402
from fieldstates.driver import FieldPoint  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402


def main():
    out, gap, D, Z = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    mat = gapped_graphene(gap / 2000.0)
    js = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    for B in map(float, sys.argv[5:]):
        t = time.time()
        P = FieldPoint(mat, Z, B, D=D, js=js, n_ll=40)
        mev = P.mev
        rec = {"gap": gap, "D": D, "Z": Z, "B": B, "l_nm": P.l}
        o1 = P.one_electron()
        v1, j1 = o1["valley"], o1["j"]
        b1 = P.bases[v1]
        a0 = b1.select(j1)[0]
        F1, G1 = b1.F[a0], b1.G[a0]
        n1 = F1 ** 2 + G1 ** 2
        rec["N1"] = {"valley": v1, "j": j1, "r_mean_nm": ob.mean_radius(P, F1, G1),
                     "dens_1nm": ob.density_at(P, F1, G1, 1.0)}
        d1, _, _ = ob.sea_edge_shift(P, n1)
        rec["N1"]["delta_meV"] = d1 * P.m * mev
        r2 = P.ground(2)
        rec2, ci2, v2 = r2[0]
        n2, rbar2 = ci2.density(v2)
        d2, tops2, _ = ob.sea_edge_shift(P, n2)
        rec["N2"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in rec2.items()}
        rec["N2"].update({"r_mean_nm": rbar2, "delta_meV": d2 * P.m * mev})
        # Dyson orbital 1 -> 2: put the existing electron first in the sector's ordering
        if ci2.valleys[0] == v1:
            Zqp, FD, GD = ob.dyson_12(P, ci2, v2, v1, j1)
            rec["dyson12"] = {"Z_qp": Zqp, "r_mean_nm": ob.mean_radius(P, FD, GD),
                              "dens_1nm_rel": ob.density_at(P, FD, GD, 1.0) * Zqp / rec["N1"]["dens_1nm"]}
        r3 = P.ground(3, n_max=16)
        rec3, ci3, v3 = r3[0]
        n3, rbar3 = ci3.density(v3)
        d3, _, _ = ob.sea_edge_shift(P, n3)
        rec["N3"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in rec3.items()}
        rec["N3"].update({"r_mean_nm": rbar3, "delta_meV": d3 * P.m * mev})
        rec["time"] = time.time() - t
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"B={B}: delta1={d1*P.m*mev:.3f} delta2={d2*P.m*mev:.3f} delta3={d3*P.m*mev:.3f} meV  "
              f"r1={rec['N1']['r_mean_nm']:.1f} r2={rbar2:.1f} r3={rbar3:.1f}  "
              f"dyson12={rec.get('dyson12')}  t={rec['time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
