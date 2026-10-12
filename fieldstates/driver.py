"""Set-up of one field point and ground states for N = 1, 2, 3 over all valley/spin/M sectors.

Box and cutoffs.  Below ~0.1 T the box R = 30/m and E_max = 6m of paper 2 are kept.  In a field
the states of each channel are Landau-like and confined within ~l sqrt(2n); the box is reduced to
R = min(30/m, R_l l) and the cutoff raised to include n_ll Landau levels,
E_max = max(6m, sqrt(m^2 + 2 n_ll/l^2)).  The CI is converged in n_ll, R_l, the channel window and
(three electrons) the number of states per channel n_max (scripts/f1_convergence.py).

Spin.  Without intervalley exchange the energies do not depend on the spin configuration beyond
the exchange symmetry of each sector; the spin Zeeman energy g mu_B B S_z (g = 2) is added with the
largest spin S the sector allows (S_z = -S is lowest): N = 1: S = 1/2; same-valley pair: S = 0
(symmetric) or 1 (antisymmetric); opposite valleys: S = 1; three electrons: S = 1/2 for a singlet
pair plus one, 3/2 for a triplet pair plus one in the other valley, and for three in one valley
S = 3/2 if the state is fully antisymmetric in space, else 1/2.
"""
import json
import time

import numpy as np

from .basis import FieldBasis, magnetic_length
from .ci import FieldCI
from .interaction import TwoValleyInteraction
from .screening import impurity_shape, interaction_weight

MU_B = 0.0578838          # meV/T
G_SPIN = 2.0


def zeeman(B, S):
    """Spin Zeeman energy (meV) of the lowest S_z = -S member."""
    return -G_SPIN * MU_B * B * S


class FieldPoint:
    """Bases of both valleys and the interaction for one (material, Z, B, gate)."""

    def __init__(self, mat, Z, B, D=np.inf, d=0.3, js=(-1.5, -0.5, 0.5, 1.5, 2.5), n_ll=40, R_l=15.0,
                 e_max0=6.0, field_screening=True, n_scan=500, R=None, E_max=None):
        t0 = time.time()
        self.mat, self.Z, self.B, self.D, self.d = mat, float(Z), float(B), float(D), float(d)
        m = self.m = mat.m
        self.l = magnetic_length(B)
        self.R = R if R is not None else min(30.0 / m, R_l * self.l)
        self.E_max = E_max if E_max is not None else (
            max(e_max0 * m, np.sqrt(m * m + 2 * n_ll / self.l ** 2)) if np.isfinite(self.l) else e_max0 * m)
        self.js = list(js)
        self.shape = impurity_shape(mat, d, self.l, D, field_screening)
        self.bases = {v: FieldBasis(self.shape, Z, m, R=self.R, js=self.js, E_max=self.E_max, r_core=d,
                                    l_B=self.l, valley=v, n_scan=n_scan) for v in (1, -1)}
        self.weight = interaction_weight(mat, self.l, D, field_screening)
        self.ti = TwoValleyInteraction(self.bases, self.weight, r_core=d)
        self.mev = mat.hbar_v * 1000.0
        self.t_setup = time.time() - t0

    def channels(self, n_max=None, js=None):
        js = self.js if js is None else js
        return {v: {j: b.select(j)[:n_max] for j in js if len(b.select(j))} for v, b in self.bases.items()}

    # -- one electron ----------------------------------------------------------------------
    def one_electron(self):
        best = None
        levels = {}
        for v, b in self.bases.items():
            for j in self.js:
                idx = b.select(j)
                if len(idx):
                    levels[(v, j)] = float(b.E[idx[0]])
                    if best is None or b.E[idx[0]] < best[0]:
                        best = (float(b.E[idx[0]]), v, j)
        E = best[0] * self.mev + zeeman(self.B, 0.5)
        return {"E": E, "valley": best[1], "j": best[2], "S": 0.5,
                "levels_meV": {f"{v:+d},{j:+.1f}": e * self.mev for (v, j), e in sorted(levels.items())}}

    # -- sectors ---------------------------------------------------------------------------
    def sectors(self, N):
        """(valleys, sym, sym23) of every sector with N electrons."""
        if N == 2:
            return [((1, 1), 1, 0), ((1, 1), -1, 0), ((-1, -1), 1, 0), ((-1, -1), -1, 0), ((1, -1), 0, 0)]
        if N == 3:
            return [((1, 1, -1), 1, 0), ((1, 1, -1), -1, 0), ((-1, -1, 1), 1, 0), ((-1, -1, 1), -1, 0),
                    ((1, 1, 1), -1, 0), ((-1, -1, -1), -1, 0)]
        out = [((v,) * 4, -1, -1) for v in (1, -1)]
        out += [((1, 1, -1, -1), s1, s2) for s1 in (1, -1) for s2 in (1, -1)]
        out += [((v, v, v, -v), -1, 0) for v in (1, -1)]
        return out

    def _M_values(self, valleys):
        import itertools
        Ms = set()
        for t in itertools.product(*[self.js] * len(valleys)):
            Ms.add(round(sum(v * j for v, j in zip(valleys, t)), 1))
        return sorted(Ms)

    @staticmethod
    def _spin(valleys, sym, ci=None, vec=None, sym23=0):
        """Largest spin compatible with the sector and the state (see module docstring)."""
        if len(valleys) == 2:
            return 0.0 if sym == 1 else 1.0
        if len(valleys) == 3:
            if len(set(valleys)) == 2:
                return 0.5 if sym == 1 else 1.5
            return 1.5 if ci.antisymmetric_weight(vec) > 0.5 else 0.5
        if len(set(valleys)) == 1:                       # (up-up)(down-down): S_z = 0
            return ci.four_spin(vec)
        if sym23:                                        # one pair in each valley
            return (0.0 if sym == 1 else 1.0) + (0.0 if sym23 == 1 else 1.0)
        return 2.0 if ci.antisymmetric_weight(vec, first=3) > 0.5 else 1.0

    def ground(self, N, n_max=None, tol=1e-9, verbose=False):
        """Lowest N-electron state (N = 2, 3, 4) over all sectors, with spin Zeeman energy (meV);
        returns the solved sectors, sorted by energy.

        Sectors are solved in the order of their lowest diagonal element.  A sector is skipped only
        if a rigorous lower bound on its energy, its noninteracting minimum (FieldCI.min_one_body;
        W > 0) plus the largest spin Zeeman gain N/2, lies above the lowest energy found so far.
        (An earlier version skipped sectors by a window on the lowest diagonal element; that is
        unsafe, since the correlation lowering of compact configurations reaches ~1 m.)"""
        ch = self.channels(n_max)
        self.ti.compress(ch, tol=1e-8)
        cands = []
        for valleys, sym, sym23 in self.sectors(N):
            for M in self._M_values(valleys):
                ci = FieldCI(self.ti, valleys, sym, M, sym23=sym23)
                if ci.size == 0:
                    continue
                lb = ci.min_one_body()
                if np.isfinite(lb):
                    cands.append((ci.min_diagonal(), lb, valleys, sym, M, ci))
        z_best = zeeman(self.B, 0.5 * N)
        best = np.inf
        results, self.n_skipped = [], 0
        for d, lb, valleys, sym, M, ci in sorted(cands, key=lambda c: c[0]):
            if lb * self.mev + z_best > best:
                self.n_skipped += 1
                continue
            t = time.time()
            e, vec = ci.davidson(tol=tol)
            S = self._spin(valleys, sym, ci, vec, ci.sym23)
            rec = {"valleys": valleys, "sym": sym, "M": M, "E_orb": e * self.mev, "S": S,
                   "E": e * self.mev + zeeman(self.B, S), "size": ci.size, "it": ci.info["iterations"],
                   "res": ci.info["residual"], "t": time.time() - t}
            best = min(best, rec["E"])
            if verbose:
                print("   ", json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in rec.items()}), flush=True)
            results.append((rec, ci, vec))
        results.sort(key=lambda r: r[0]["E"])
        return results
