"""Screening by the Landau-quantized Dirac sea: impurity potential and interaction in a field.

The static RPA dielectric function of papers 1-2,
    eps(q) = kappa + 2 pi alpha0 N Pi_0(q) G(q) / q,
becomes, in a field,
    eps_B(q) = kappa + 2 pi alpha0 N Pi_0(q) rho(q l, m l) G(q) / q,
with rho = (Pi_K + Pi_K') / (2 Pi_0) the ratio of the Landau-level polarization
(polarization.pi_field, summed over valleys, per spin) to its zero-field value.  rho depends only
on q l and m l; it tends to a constant as q -> 0 (both polarizations are ~ q^2) and to 1 for
q l >> 1 (short distances do not feel the field).  It is tabulated on 0.03 <= q l <= 8 and set to
1 beyond q l = 10 (smooth blend in between; the computed deviation there is below the
extrapolation error, ~1e-3).

Weak fields.  The truncated Landau-level sums reach their asymptotic regime only for
n_c >> (m l)^2; for m l > 5 (B < 14 mT at 2Delta = 30 meV) the cutoffs used are too small and give
a spurious rho > 1.  There the true deviation is negligible: |rho - 1| falls off as (m l)^-4
(8e-3 at m l = 1.85, 2e-4 at m l = 4.1), i.e. below 1e-4 for m l > 5, and rho = 1 is used.
"""
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

from .deps import collapse2d  # noqa: F401
from collapse2d.potentials import ScreenedImpurity
from .polarization import pi_field, pi_zero_field

ML_ZERO_FIELD = 5.0          # rho = 1 for m l above this (see module docstring)
QL_TABLE = np.concatenate([np.geomspace(0.03, 1.0, 16)[:-1], np.linspace(1.0, 8.0, 29)])
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "screening")


def ratio_table(ml, cache=True):
    """(q l, rho, error) for one value of m l; cached in results/screening."""
    path = os.path.join(_CACHE, "rho_ml%.6f.npz" % ml)
    if cache and os.path.exists(path):
        d = np.load(path)
        return d["ql"], d["rho"], d["err"]
    m, l = ml, 1.0                                    # units l = 1
    rho, err = [], []
    for ql in QL_TABLE:
        nc = max(400, int(4 * ql * ql) + 200)
        P, e = pi_field(ql, m, l, n_c=(nc, 2 * nc, 4 * nc))
        P0 = 2 * pi_zero_field(np.array([ql]), m)[0]
        rho.append(P / P0)
        err.append(e / P0)
    rho, err = np.array(rho), np.array(err)
    if cache:
        os.makedirs(_CACHE, exist_ok=True)
        np.savez(path, ql=QL_TABLE, rho=rho, err=err)
    return QL_TABLE, rho, err


class FieldRatio:
    """rho(q) for given m and l (callable, vectorised); rho = 1 for l = inf."""

    def __init__(self, m, l):
        self.m, self.l = float(m), float(l)
        if np.isinf(l) or m * l > ML_ZERO_FIELD:
            self._f = None
            return
        ql, rho, _ = ratio_table(m * l)
        self._lo, self._hi = ql[0], ql[-1]
        self._rho_lo = rho[0]
        self._f = PchipInterpolator(np.log(ql), rho)
        self._rho_hi = rho[-1]

    def __call__(self, q):
        q = np.asarray(q, float)
        if self._f is None:
            return np.ones_like(q)
        x = q * self.l
        out = np.empty_like(x)
        lo, hi = x <= self._lo, x >= self._hi
        mid = ~(lo | hi)
        out[lo] = self._rho_lo
        out[mid] = self._f(np.log(x[mid]))
        # blend the last tabulated value to 1 between q l = 8 and 10
        t = np.clip((x[hi] - self._hi) / 2.0, 0.0, 1.0)
        s = t * t * (3 - 2 * t)
        out[hi] = (1 - s) * self._rho_hi + s
        return out


class FieldScreenedImpurity(ScreenedImpurity):
    """collapse2d.potentials.ScreenedImpurity with the field-dependent sea polarization."""

    def __init__(self, Z, mat, d, l_B=np.inf, D=np.inf, rpa=True, **kw):
        self.ratio = FieldRatio(mat.m, l_B)
        super().__init__(Z, mat, d, D=D, rpa=rpa, **kw)

    def eps(self, q):
        q = np.asarray(q, dtype=float)
        e = self.kappa + self.rK * q
        if self.N:
            from collapse2d.potentials import polarization
            e = e + 2 * np.pi * self.alpha0 * self.N * polarization(q, self.m) * self.ratio(q) \
                / np.where(q > 0, q, 1.0) * self.G(q)
        return e


def impurity_shape(mat, d, l_B=np.inf, D=np.inf, field_screening=True):
    """Potential shape v(r) for unit charge (U = Z v), with or without the field in eps."""
    return FieldScreenedImpurity(1.0, mat, d, l_B=(l_B if field_screening else np.inf), D=D)


def interaction_weight(mat, l_B=np.inf, D=np.inf, field_screening=True):
    """w(q) = q W(q)/2 pi = alpha0 G(q)/eps_B(q)."""
    P = FieldScreenedImpurity(1.0, mat, 1.0, l_B=(l_B if field_screening else np.inf), D=D, r_max=10.0, n_r=16)
    return lambda q: mat.alpha0 * P.G(np.asarray(q, float)) / P.eps(np.asarray(q, float))
