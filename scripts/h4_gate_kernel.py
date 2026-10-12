"""Field scan of the gated impurity with the exact vacuum / hBN / metal gate kernel.

    python3 scripts/h4_gate_kernel.py OUT.jsonl GAP_meV D_nm Z B1 [B2 ...] [f2_scan options]

Papers 1-3 combine the gate with the mean dielectric constant, U, W ~ G(q)/eps(q) with
G = 1 - exp(-2qD) and kappa = (e_t + e_b)/2, which is exact only for e_t = e_b.  For vacuum above
the layer (e_t = 1) and hBN between layer and gate (e_b = 4) the exact electrostatics replaces
kappa/G by kappa_eff(q) = (e_t + e_b coth qD)/2 (paper 2, scripts/m11_gate_kernel.py).  This script
patches the impurity potential and the interaction of fieldstates.driver accordingly, keeping the
field-dependent sea polarization, and runs scripts/f2_scan.py, so that the characteristic fields
can be compared with results/scan/D10.jsonl.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import fieldstates.driver as driver  # noqa: E402
from fieldstates.screening import FieldScreenedImpurity  # noqa: E402
from collapse2d.potentials import polarization  # noqa: E402

E_TOP, E_BOT = 1.0, 4.0


def kappa_eff(q, D):
    x = np.asarray(q, float) * D
    return 0.5 * (E_TOP + E_BOT / np.tanh(np.where(x > 0, x, 1e-300)))


class ExactGateImpurity(FieldScreenedImpurity):
    """FieldScreenedImpurity with kappa/G replaced by the exact two-medium kernel."""

    def eps_exact(self, q):
        q = np.asarray(q, float)
        e = kappa_eff(q, self.D) + self.rK * q
        if self.N:
            e = e + 2 * np.pi * self.alpha0 * self.N * polarization(q, self.m) * self.ratio(q) \
                / np.where(q > 0, q, 1.0)
        return e

    def f(self, q):
        return np.exp(-np.asarray(q, float) * self.d) / self.eps_exact(q)


def impurity_shape(mat, d, l_B=np.inf, D=np.inf, field_screening=True):
    assert np.isfinite(D), "the exact kernel applies to a gated layer"
    return ExactGateImpurity(1.0, mat, d, l_B=(l_B if field_screening else np.inf), D=D)


def interaction_weight(mat, l_B=np.inf, D=np.inf, field_screening=True):
    P = ExactGateImpurity(1.0, mat, 1.0, l_B=(l_B if field_screening else np.inf), D=D, r_max=10.0, n_r=16)
    return lambda q: mat.alpha0 / P.eps_exact(np.asarray(q, float))


driver.impurity_shape = impurity_shape
driver.interaction_weight = interaction_weight

if __name__ == "__main__":
    import f2_scan
    f2_scan.main()
