"""Configuration interaction for two and three electrons with arbitrary valley content.

    H = sum_i h_v(i) + sum_{i<k} W(i, k),   no-pair: each electron in the Furry space of its valley.

Electron i lives in valley v_i (+1: K, -1: K') and its orbitals are the FieldBasis states of that
valley, labelled by channel j_i.  Spin is handled through the exchange symmetry of the spatial
wave function: electrons 0 and 1 may be identical-valley partners with a symmetric (sym = +1,
spin singlet) or antisymmetric (sym = -1, parallel spins or the antisymmetric spatial partner of
a triplet) wave function; all other electrons are distinguishable (other valley, or the other spin
projection).  The sectors used:

    two electrons    (v, v) sym +1   same-valley singlet           S = 0
                     (v, v) sym -1   same-valley triplet           S = 1
                     (K, K') sym 0   opposite valleys              S = 0, 1 (degenerate)
    three electrons  (v, v, v') sym +-1   pair in v (S_pair = 0 or 1) + one electron in v' != v
                     (v, v, v) sym -1     up-up pair + down electron, all in v: S_z = 1/2, contains
                                          S = 1/2 and S = 3/2 (fully antisymmetric spatial part)
    four electrons   (v, v, v, v) sym -1, sym23 -1   up-up pair + down-down pair in one valley:
                                          S_z = 0 (S = 0, 1, 2)
                     (v, v, v', v') sym +-1, sym23 +-1   a pair in each valley
                     (v, v, v, v') sym -1, sym23 0   up-up-down in v + one electron in v'

Conserved: the physical angular momentum M = sum_i s_{v_i} j_i.  The vector is stored as full
product tensors Psi[a, b(, c)] per channel tuple (both orders of the 0-1 pair), and the 0 <-> 1
symmetry is imposed on the vector; H commutes with it, so no explicit exchange terms are needed.
Interaction matrix elements are sums over the jointly compressed q-profiles
(TwoValleyInteraction.C), so every two-body term is two tensor contractions.
"""
import itertools

import numpy as np


class FieldCI:
    def __init__(self, ti, valleys, sym, M, terms=None, sym23=0):
        """ti: compressed TwoValleyInteraction; valleys: tuple of +-1 per electron (2 to 4);
        sym: exchange symmetry of electrons 0, 1 (+1, -1, or 0 for distinguishable);
        sym23: the same for electrons 2, 3 (four electrons only)."""
        self.ti, self.valleys, self.sym, self.M = ti, tuple(int(v) for v in valleys), int(sym), float(M)
        self.sym23 = int(sym23)
        self.N = len(valleys)
        if self.N not in (2, 3, 4):
            raise ValueError("two to four electrons")
        if self.sym and self.valleys[0] != self.valleys[1]:
            raise ValueError("exchange symmetry needs two electrons of one valley")
        if self.sym23 and (self.N != 4 or self.valleys[2] != self.valleys[3]):
            raise ValueError("sym23 needs four electrons with electrons 2, 3 in one valley")
        self.pairs = [(i, k) for i in range(self.N) for k in range(i + 1, self.N)]
        self.terms = terms or {p: 1.0 for p in self.pairs}
        ch = ti.channels
        self.ch = ch
        labels = [sorted(ch[v]) for v in self.valleys]
        self.tuples = [t for t in itertools.product(*labels)
                       if np.isclose(sum(v * j for v, j in zip(self.valleys, t)), self.M)]
        self.shapes = [tuple(len(ch[v][j]) for v, j in zip(self.valleys, t)) for t in self.tuples]
        self.offs = np.cumsum([0] + [int(np.prod(s)) for s in self.shapes])
        self.size = int(self.offs[-1])
        self.index = {t: k for k, t in enumerate(self.tuples)}
        self.E = [{j: ti.bases[v].E[idx] for j, idx in ch[v].items()} for v in self.valleys]
        # couplings: for each pair (i, k), the (target, source) tuples differing only there
        self.links = {}
        for (i, k) in self.pairs:
            L = []
            for T, tT in enumerate(self.tuples):
                for S, tS in enumerate(self.tuples):
                    if all(tT[p] == tS[p] for p in range(self.N) if p not in (i, k)):
                        L.append((T, S))
            self.links[(i, k)] = L

    # -- helpers ---------------------------------------------------------------------------
    def _blocks(self, v):
        return [v[self.offs[k]:self.offs[k + 1]].reshape(self.shapes[k]) for k in range(len(self.tuples))]

    def _sym_pair(self, v, i, sign):
        out = np.empty_like(v)
        P = self._blocks(v)
        perm = list(range(self.N))
        perm[i], perm[i + 1] = i + 1, i
        for k, t in enumerate(self.tuples):
            tt = list(t)
            tt[i], tt[i + 1] = t[i + 1], t[i]
            X = P[self.index[tuple(tt)]].transpose(perm)
            out[self.offs[k]:self.offs[k + 1]] = (0.5 * (P[k] + sign * X)).ravel()
        return out

    def symmetrize(self, v):
        if self.sym:
            v = self._sym_pair(v, 0, self.sym)
        if self.sym23:
            v = self._sym_pair(v, 2, self.sym23)
        return v

    def _Cpair(self, i, jT, jS):
        return self.ti.C(self.valleys[i], jT, jS)

    def _contract(self, Ci, Ck, X, i, k):
        """sum_q sum_{A,B} Ci[q,a,A] Ck[q,b,B] X[..A(i)..B(k)..] -> array with a at i, b at k."""
        N = self.N
        Y = np.tensordot(Ci, X, axes=([2], [i]))               # (q, a, X axes without i)
        kk = 2 + (k if k < i else k - 1)
        Y = np.moveaxis(Y, kk, -1)                             # (q, a, rest..., B)
        Z = np.tensordot(Ck, Y, axes=([0, 2], [0, Y.ndim - 1]))  # (b, a, rest...)
        rest = [p for p in range(N) if p not in (i, k)]
        src = {k: 0, i: 1}
        src.update({p: 2 + n for n, p in enumerate(rest)})
        return Z.transpose([src[p] for p in range(N)])

    # -- H v -----------------------------------------------------------------------------
    def matvec(self, v):
        P = self._blocks(v)
        out = [np.zeros(s) for s in self.shapes]
        for k, t in enumerate(self.tuples):
            for i in range(self.N):
                e = self.E[i][t[i]]
                sh = [1] * self.N
                sh[i] = len(e)
                out[k] += e.reshape(sh) * P[k]
        for (i, k), L in self.links.items():
            s = self.terms[(i, k)]
            if not s:
                continue
            for T, S in L:
                tT, tS = self.tuples[T], self.tuples[S]
                Ci = self._Cpair(i, tT[i], tS[i])                  # (q, a, A)
                Ck = self._Cpair(k, tT[k], tS[k])                  # (q, b, B)
                out[T] += s * self._contract(Ci, Ck, P[S], i, k)
        return np.concatenate([o.ravel() for o in out])

    def diagonal(self):
        d = []
        for t, shp in zip(self.tuples, self.shapes):
            D = np.zeros(shp)
            dg = []
            for i in range(self.N):
                e = self.E[i][t[i]]
                sh = [1] * self.N
                sh[i] = len(e)
                D = D + e.reshape(sh)
                dg.append(np.einsum("qaa->qa", self._Cpair(i, t[i], t[i])))
            for (i, k) in self.pairs:
                s = self.terms[(i, k)]
                if not s:
                    continue
                V = dg[i].T @ dg[k]                                 # (a, b)
                sh = [1] * self.N
                sh[i], sh[k] = V.shape
                D = D + s * V.reshape(sh)
            d.append(D.ravel())
        return np.concatenate(d)

    def min_one_body(self):
        """Lowest noninteracting energy of the sector: the smallest sum of orbital energies over the
        allowed configurations (Pauli: an antisymmetric pair in one channel needs two different
        orbitals).  Since W(q) > 0, the projected interaction is positive semidefinite, so this is a
        rigorous lower bound on the lowest eigenvalue of the sector."""
        best = np.inf
        for t in self.tuples:
            lows = [self.E[i][t[i]] for i in range(self.N)]
            if any(len(e) == 0 for e in lows):
                continue
            tot, ok = 0.0, True
            used = set()
            for (i, sgn) in ((0, self.sym), (2, self.sym23)):
                if sgn == -1 and t[i] == t[i + 1]:
                    e = np.sort(lows[i])
                    if len(e) < 2:
                        ok = False
                        break
                    tot += e[0] + e[1]
                    used.update((i, i + 1))
            if not ok:
                continue
            tot += sum(float(np.min(lows[i])) for i in range(self.N) if i not in used)
            best = min(best, tot)
        return best

    def min_diagonal(self):
        """Lowest diagonal element among the configurations allowed by the symmetry (Pauli: no
        doubly occupied orbital in an antisymmetric pair)."""
        dg = self.diagonal()[self.allowed()]
        return float(dg.min()) if len(dg) else np.inf

    def allowed(self):
        """Mask of the product configurations that survive the exchange symmetry (Pauli)."""
        mask = np.ones(self.size, bool)
        for (i, sgn) in ((0, self.sym), (2, self.sym23)):
            if sgn != -1:
                continue
            for k, t in enumerate(self.tuples):
                if t[i] == t[i + 1]:
                    blk = mask[self.offs[k]:self.offs[k + 1]].reshape(self.shapes[k])
                    n = self.shapes[k][i]
                    idx = [slice(None)] * self.N
                    idx[i], idx[i + 1] = np.arange(n), np.arange(n)
                    blk[tuple(idx)] = False
        return mask

    # -- lowest eigenpair ------------------------------------------------------------------
    def davidson(self, tol=1e-9, max_iter=400, max_space=12, x0=None):
        """Lowest eigenpair by Davidson's method (diagonal preconditioner).  Memory-lean: the
        projected matrix is updated in place and the subspace is restarted from the current Ritz
        vector after max_space vectors, so at most 2 max_space vectors of the full size are stored
        (as in chargestates.ci4)."""
        diag = self.diagonal()
        if x0 is None:
            # start from the lowest configuration allowed by the exchange symmetry, plus a small
            # deterministic random admixture: H and the diagonal preconditioner conserve the total
            # spin, so a start vector of pure spin (e.g. a doubly occupied orbital) would confine
            # the iteration to that spin and miss a lower state of another spin (Hund's rule)
            ok = np.flatnonzero(self.allowed())
            if len(ok) == 0:
                raise ValueError("no configuration allowed by the exchange symmetry")
            x = np.zeros(self.size)
            x[ok[np.argmin(diag[ok])]] = 1.0
            x = self.symmetrize(x)
            noise = np.zeros(self.size)
            noise[ok] = np.random.default_rng(20261011).normal(size=len(ok))
            noise = self.symmetrize(noise)
            x = x / np.linalg.norm(x) + 0.1 * noise / np.linalg.norm(noise)
        else:
            x = self.symmetrize(np.array(x0, float))
        x /= np.linalg.norm(x)
        Vs, AVs = [x], [self.matvec(x)]
        Hs = np.array([[x @ AVs[0]]])
        res = np.inf
        for it in range(max_iter):
            ev, ey = np.linalg.eigh(0.5 * (Hs + Hs.T))
            theta, yv = ev[0], ey[:, 0]
            u = sum(c * v for c, v in zip(yv, Vs))
            Au = sum(c * v for c, v in zip(yv, AVs))
            r = Au - theta * u
            res = np.linalg.norm(r)
            if res < tol * max(1.0, abs(theta)):
                break
            den = theta - diag
            t = self.symmetrize(r / np.where(np.abs(den) < 1e-12, 1e-12, den))
            if len(Vs) >= max_space:
                nu = np.linalg.norm(u)
                Vs, AVs = [u / nu], [Au / nu]
                Hs = np.array([[Vs[0] @ AVs[0]]])
            # new direction: the preconditioned residual, or, where it lies in the subspace (e.g. a
            # diagonal H, for which it is parallel to u), the residual itself
            for cand in (t, self.symmetrize(r)):
                cand = cand / max(np.linalg.norm(cand), 1e-300)
                for _ in range(2):
                    for vv in Vs:
                        cand -= (vv @ cand) * vv
                nt = np.linalg.norm(cand)
                if nt > 1e-8:
                    break
            del r
            if nt <= 1e-8:
                break
            t = cand / nt
            At = self.matvec(t)
            k = len(Vs)
            H2 = np.empty((k + 1, k + 1))
            H2[:k, :k] = Hs
            H2[:k, k] = [v @ At for v in Vs]
            H2[k, :k] = [t @ Av for Av in AVs]
            H2[k, k] = t @ At
            Hs = H2
            Vs.append(t)
            AVs.append(At)
        self.info = {"iterations": it + 1, "residual": float(res)}
        self.energy, self.vector = float(theta), u
        return self.energy, u

    # -- analysis --------------------------------------------------------------------------
    def full_tensor(self, vec):
        return {t: P for t, P in zip(self.tuples, self._blocks(vec))}

    def antisymmetric_weight(self, vec, first=None):
        """Weight of the part of the state that is antisymmetric under all exchanges among the
        first three electrons (all in one valley, electrons 0, 1 already antisymmetric): the
        probability that the three form S = 3/2.  For N = 3 (first = 3 implied) this is the S = 3/2
        probability of the S_z = 1/2 state; 0 or 1 for an eigenstate."""
        if not (self.N >= 3 and len(set(self.valleys[:3])) == 1 and self.sym == -1):
            raise ValueError("needs three electrons of one valley with an antisymmetric pair")
        T = self.full_tensor(vec)
        tot = 0.0
        # A Psi = (1/3)[Psi - P_02 Psi - P_12 Psi] for Psi antisymmetric in (0, 1)
        rest = tuple(range(3, self.N))
        for t, P in T.items():
            P02 = T[(t[2], t[1], t[0]) + t[3:]].transpose((2, 1, 0) + rest)
            P12 = T[(t[0], t[2], t[1]) + t[3:]].transpose((0, 2, 1) + rest)
            A = (P - P02 - P12) / 3.0
            tot += np.sum(A * P)
        return float(tot / (vec @ vec))

    def four_spin(self, vec):
        """Spin of a four-electron state of one valley in the (up-up)(down-down) representation:
        the flip of all spins exchanges the two pairs, Psi[a,b,c,d] -> Psi[c,d,a,b], with eigenvalue
        (-1)^S; S = 2 requires a spatial wave function antisymmetric under all exchanges."""
        T = self.full_tensor(vec)
        par = sum(float(np.sum(P * T[(t[2], t[3], t[0], t[1])].transpose(2, 3, 0, 1))) for t, P in T.items())
        par /= float(vec @ vec)
        if par < 0:
            return 1.0
        anti = sum(float(np.sum(P * T[(t[0], t[2], t[1], t[3])].transpose(0, 2, 1, 3))) for t, P in T.items())
        return 2.0 if anti / float(vec @ vec) < -0.5 else 0.0

    def channel_weights(self):
        """Weight of each channel tuple in the current eigenvector."""
        return {t: float(np.sum(P * P)) for t, P in self.full_tensor(self.vector).items()}

    def density(self, vec=None):
        """Radial electron density n(r) (per r, i.e. int n dr = N) on the basis grid, summed over
        electrons, and the mean radius per electron."""
        vec = self.vector if vec is None else vec
        g = next(iter(self.ti.bases.values())).grid
        n = np.zeros(len(g.r))
        for t, P in self.full_tensor(vec).items():
            for i in range(self.N):
                b = self.ti.bases[self.valleys[i]]
                idx = self.ch[self.valleys[i]][t[i]]
                axes = tuple(a for a in range(self.N) if a != i)
                rho = np.tensordot(P, P, axes=(axes, axes))       # (a, a')
                F, G = b.F[idx], b.G[idx]
                n += np.einsum("ab,ar,br->r", rho, F, F) + np.einsum("ab,ar,br->r", rho, G, G)
        return n, float((n * g.r) @ g.w) / self.N
