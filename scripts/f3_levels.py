"""Single-particle levels of the impurity in a field, both valleys (Fig. 1).

    python3 scripts/f3_levels.py OUT.json GAP D Z B1 [B2 ...]

For each field: the two lowest electron levels of the channels j = -1.5 ... 4.5 in K and K'
(meV), the highest sea state per channel, and the clean Landau levels.  Field-dependent screening.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from fieldstates.basis import FieldBasis, magnetic_length  # noqa: E402
from fieldstates.screening import impurity_shape  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

JS = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]


def main():
    out, gap, D, Z = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    mat = gapped_graphene(gap / 2000.0)
    m, mev = mat.m, mat.hbar_v * 1000
    data = json.load(open(out)) if os.path.exists(out) else {}
    for B in map(float, sys.argv[5:]):
        l = magnetic_length(B)
        R = min(30 / m, 15 * l)
        E_max = max(3 * m, np.sqrt(m * m + 6 / l ** 2) if np.isfinite(l) else 3 * m)
        v = impurity_shape(mat, 0.3, l, D)
        rec = {}
        for val in (1, -1):
            b = FieldBasis(v, Z, m, R=R, js=JS, E_max=E_max, r_core=0.3, l_B=l, valley=val, n_scan=300)
            rec[str(val)] = {f"{j:+.1f}": {"E": list(b.E[b.select(j)][:2] * mev), "sea_top": b.sea_top[j] * mev}
                             for j in JS}
        rec["landau"] = [float(np.sqrt(m * m + 2 * n / l ** 2) * mev) if np.isfinite(l) else m * mev for n in range(6)]
        data[f"{B:.4f}"] = rec
        json.dump(data, open(out, "w"), indent=1)
        print(B, {k: round(rec[k]["+0.5"]["E"][0], 3) for k in ("1", "-1")}, flush=True)


if __name__ == "__main__":
    main()
