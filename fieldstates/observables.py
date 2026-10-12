"""Observables of the field CI states.

Sea edge.  In a field the top of the Dirac sea is the K zero-mode Landau level, degenerate and
exactly at -Delta.  The no-pair Hamiltonian keeps it filled.  If the net potential of the impurity
and the bound electrons, U + V_H[n_N], is repulsive somewhere (N larger than the effective charge
inside some radius), K n = 0 states there are lifted above -Delta; for Fermi levels below the
lifted states they would empty, i.e. the sea would bind holes, which the CI does not describe.  The
lift delta_N is computed as the largest shift of the highest sea state over the K channels
j = -1/2 ... -j_far in the mean field h + V_H[n_N] (the electron density of the CI ground state);
the charge-state diagram is reliable for E_F > -Delta + delta_N.  The same quantity at B = 0 is the
binding of a valence-band hole by the complex.

Dyson orbital (1 -> 2).  Adding an electron to the one-electron ground state (orbital a0 of valley
v1) gives the amplitudes d_b = Psi_2[a0, b] of the two-electron state in the product representation
of FieldCI (spin singlet or opposite valleys), quasiparticle weight Z = sum |d_b|^2.
"""
import numpy as np

from .basis import FieldBasis
from .deps import chargestates  # noqa: F401
from chargestates.meanfield import TabulatedPotential, hartree_potential


class _Density:
    """Minimal stand-in for an Interaction whose basis holds one 'orbital' sqrt(n(r))."""

    def __init__(self, basis, n, q):
        class B:
            pass
        b = B()
        b.F = np.sqrt(np.maximum(n, 0.0))[None, :]
        b.G = np.zeros_like(b.F)
        b.grid, b.E_max, b.m = basis.grid, basis.E_max, basis.m
        self.basis, self.q = b, q


def hartree(P, n, r):
    """V_H(r) (nm^-1) of the radial density n (int n dr = N) on the basis grid of P."""
    basis = P.bases[1]
    return hartree_potential(_Density(basis, n, P.ti.q), [0], np.asarray(r, float), P.weight)


def sea_edge_shift(P, n, j_far=16.5, n_tab=240, valley=1):
    """Largest lift (units of m) above -m of the highest sea state of the K channels j <= -1/2 in the
    mean field U + V_H[n]; also returns per-channel values.  n: radial density of the N electrons."""
    m = P.m
    r_tab = np.geomspace(1e-6, P.R, n_tab)
    VH = hartree(P, n, r_tab)
    pot = TabulatedPotential(P.shape, P.Z, r_tab, VH)
    js = list(-np.arange(0.5, j_far + 1e-9, 1.0)) + [0.5, 1.5]
    b = FieldBasis(pot, 1.0, m, R=P.R, js=js, E_max=max(1.5 * m, 1.05 * np.sqrt(m * m + 2 / P.l ** 2)
                                                         if np.isfinite(P.l) else 1.5 * m),
                   r_core=P.d, l_B=P.l, valley=valley, n_scan=200)
    tops = {j: (t + m) / m for j, t in b.sea_top.items()}
    return max(tops.values()), tops, (r_tab, VH)


def dyson_12(P, ci, vec, v1, j1):
    """Dyson amplitudes for adding an electron to the one-electron ground state (valley v1, channel
    j1, lowest orbital) to give the two-electron state (ci, vec).  Returns Z_qp and the radial
    functions (F, G) of the added electron's orbital, in its valley's basis."""
    if ci.N != 2:
        raise ValueError("two-electron state expected")
    if ci.valleys[0] != v1:
        raise ValueError("electron 0 of the sector must be in the valley of the one-electron state")
    v2 = ci.valleys[1]
    b2 = P.ti.bases[v2]
    F = np.zeros(len(b2.grid.r))
    G = np.zeros_like(F)
    Zqp = 0.0
    for t, Psi in ci.full_tensor(vec).items():
        if t[0] != j1:
            continue
        d = Psi[0]
        idx = ci.ch[v2][t[1]]
        Zqp += float(d @ d)
        F += d @ b2.F[idx]
        G += d @ b2.G[idx]
    return Zqp, F, G


def mean_radius(P, F, G):
    g = P.bases[1].grid
    w = F * F + G * G
    return float((w * g.r) @ g.w / (w @ g.w))


def density_at(P, F, G, r0):
    """|psi|^2 summed over sublattices at radius r0 (angle-averaged), per nm^2, for a normalised
    orbital: (F^2 + G^2)/(2 pi r)."""
    g = P.bases[1].grid
    norm = (F * F + G * G) @ g.w
    k = int(np.searchsorted(g.r, r0))
    return float((F[k] ** 2 + G[k] ** 2) / norm / (2 * np.pi * g.r[k]))
