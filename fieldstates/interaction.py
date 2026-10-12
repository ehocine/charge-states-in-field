"""Interaction matrix elements between electrons of either valley in a field.

Each electron keeps its valley (intervalley exchange and scattering, lattice-scale terms, are
neglected as in paper 2), so every pair density is built from two orbitals of one valley and its
Hankel form factor A^l_ac(q) = int R_ac J_l(q r) dr (chargestates.interaction) is taken from that
valley's basis.  For electron 1 in valley v (a -> c) and electron 2 in valley v' (b -> d),

    <ab|W|cd> = int dq w(q) A^{v,l}_ac(q) A^{v',l}_bd(q),

with the selection rule s_v (j_a - j_c) = -s_v' (j_b - j_d) on the channel labels (s_K = +1,
s_K' = -1: a K' orbital is the complex conjugate of the K orbital at -B with label j~, physical
angular momentum -j~).  The q dimension is compressed jointly for both valleys: for each multipole
l the Gram matrix of sqrt(w) A^l summed over the channel pairs of both bases is diagonalised, so
that <ab|W|cd> = sum_k C^v_ac,k C^v'_bd,k with one set of q-profiles.
"""
import numpy as np

from .deps import chargestates  # noqa: F401
from chargestates.interaction import Interaction, q_nodes  # noqa: F401


class TwoValleyInteraction:
    """Compressed form factors for the bases {+1: K, -1: K'} (FieldBasis), shared q nodes."""

    def __init__(self, bases, weight, r_core=0.3, q=None, qw=None, dense_panel=8.0, n_log_panels=30):
        self.bases = bases
        b0 = next(iter(bases.values()))
        if q is None:
            k_max = max(np.sqrt(max(np.max(b.E) ** 2 - b.m ** 2, 0.0)) for b in bases.values())
            q, qw = q_nodes(b0.grid.R, k_max, r_core, dense_panel=dense_panel, n_log_panels=n_log_panels)
        self.q, self.qw = q, qw
        self.inter = {v: Interaction(b, weight, q=q, qw=qw, r_core=r_core) for v, b in bases.items()}
        self._C = {}

    def compress(self, channels, tol=1e-8):
        """channels: dict valley -> dict j -> state indices.  Sets one q basis per l for both valleys."""
        grams = {}
        for v, ch in channels.items():
            I = self.inter[v]
            sw = np.sqrt(I.wq)
            js = sorted(ch)
            for ja in js:
                for jc in js:
                    if jc < ja:
                        continue                               # A_ca = A_ac^T: same Gram contribution
                    l = int(round(abs(ja - jc)))
                    A = I.form_factors(ch[ja], ch[jc], l).reshape(-1, len(self.q)) * sw
                    grams[l] = grams.get(l, 0.0) + A.T @ A
        phi = {}
        for l, G in grams.items():
            lam, U = np.linalg.eigh(G)
            keep = lam > tol ** 2 * lam[-1]
            phi[l] = U[:, keep]
        for I in self.inter.values():
            I._phi = phi
        self.channels = channels
        self.ranks = {l: P.shape[1] for l, P in phi.items()}
        self._C = {}
        return self.ranks

    def C(self, v, ja, jc):
        """Compressed form factors of valley v, channels (ja, jc): array (k, a, c), cached."""
        key = (v, ja, jc)
        if key not in self._C:
            if (v, jc, ja) in self._C:
                self._C[key] = np.ascontiguousarray(self._C[(v, jc, ja)].transpose(0, 2, 1))
            else:
                ch = self.channels[v]
                l = int(round(abs(ja - jc)))
                Cf = self.inter[v].form_factors_compressed(ch[ja], ch[jc], l)
                self._C[key] = np.ascontiguousarray(Cf.transpose(2, 0, 1))
        return self._C[key]
