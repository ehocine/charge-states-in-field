"""Static polarization of the gapped Dirac sea in a perpendicular magnetic field.

Units hbar v = 1.  Valley K, field B > 0 (electron charge -e), l = l_B:
    h_K = [[m, c a], [c a^dag, -m]],   c = sqrt(2)/l,
with Landau levels  (alpha |n-1>, beta |n>)  at  E = +-E_n,  E_n = sqrt(m^2 + 2n/l^2),  n >= 1,
and the zero mode (0, |0>) at E = -m (filled).  Valley K' (h_K' = h_K(-B)^*): levels
(alpha |n>, beta |n-1>) and the zero mode (|0>, 0) at E = +m (empty).  For the upper (+) and
lower (-) branch
    alpha_+ = sqrt((E_n + m)/2E_n),  beta_+ =  sqrt((E_n - m)/2E_n),
    alpha_- = sqrt((E_n - m)/2E_n),  beta_- = -sqrt((E_n + m)/2E_n).

The density operator e^{iq.r} acts on the cyclotron oscillator as a displacement with
|z|^2 = x = q^2 l^2/2; up to a phase common to all terms with the same n' - n,
    <n'|D|n> = g_{n' n}(x) = sqrt(n!/n'!) x^{(n'-n)/2} e^{-x/2} L_n^{(n'-n)}(x)    (n' >= n),
symmetric in (n, n').  The static polarization of one flavour of the insulating sea (all
negative-energy states filled) is

    Pi(q) = (1/2 pi l^2) sum_{occ s, empty s'} 2 |<s'|e^{iqr}|s>|^2 / (E_s' - E_s),

normalised as collapse2d.potentials.polarization (eps = kappa + 2 pi alpha0 N Pi / q at B = 0).
The double sum converges slowly (the high-energy tail is the field-independent short-distance
response); it is truncated at n, n' <= N_c and extrapolated in N_c^{-1/2} (see pi_field).
"""
import numpy as np
from scipy.special import gammaln

from .deps import collapse2d  # noqa: F401
from collapse2d.potentials import polarization as pi_zero_field


def laguerre_table(x, n_max):
    """g[d, n] = g_{n+d, n}(x) for 0 <= n, d and n + d <= n_max (zero elsewhere).

    Forward three-term recurrence in n for all d at once, normalised so that the values are
    O(1):  f_{n+1} = [(2n + 1 + d - x) f_n - sqrt(n (n + d)) f_{n-1}] / sqrt((n + 1)(n + 1 + d)).
    """
    d = np.arange(n_max + 1, dtype=float)
    out = np.zeros((n_max + 1, n_max + 1))
    with np.errstate(divide="ignore"):
        lnx = np.log(x) if x > 0 else -np.inf
    f0 = np.exp(0.5 * d * lnx - 0.5 * x - 0.5 * gammaln(d + 1)) if x > 0 else (d == 0).astype(float)
    fm = np.zeros_like(f0)
    f = f0
    out[:, 0] = f0
    for n in range(n_max):
        fn = ((2 * n + 1 + d - x) * f - np.sqrt(n * (n + d)) * fm) / np.sqrt((n + 1) * (n + 1 + d))
        fm, f = f, fn
        out[:, n + 1] = f
    # keep n + d <= n_max
    mask = (d[:, None] + np.arange(n_max + 1)[None, :]) <= n_max
    return out * mask


def _G(tab, a, b):
    """g_{a b} for integer arrays a, b (>= 0) from the table."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return tab[hi - lo, lo]


def pi_valleys(q, m, l, n_c):
    """(Pi_K, Pi_K') at wave vector q, truncated at Landau index n_c (one spin)."""
    x = 0.5 * (q * l) ** 2
    tab = laguerre_table(x, n_c + 1)
    n = np.arange(n_c + 1)
    E = np.sqrt(m * m + 2.0 * n / (l * l))
    ap, bp = np.sqrt((E + m) / (2 * E)), np.sqrt((E - m) / (2 * E))
    am, bm = np.sqrt((E - m) / (2 * E)), -np.sqrt((E + m) / (2 * E))
    N, Np = np.meshgrid(n, n, indexing="ij")          # N: valence index, Np: conduction index
    pref = 2.0 / (2 * np.pi * l * l)
    # valley K: valence (alpha_-|n-1>, beta_-|n>) for n >= 1, zero mode (0,|0>) at -m;
    #           conduction (alpha_+|n'-1>, beta_+|n'>), n' >= 1
    v1 = (N >= 1) & (Np >= 1)
    M = np.where(v1, ap[Np] * am[N] * _G(tab, np.maximum(Np - 1, 0), np.maximum(N - 1, 0)), 0.0) \
        + np.where(Np >= 1, bp[Np] * np.where(N >= 1, bm[N], 1.0) * _G(tab, Np, N), 0.0)
    den = E[Np] + np.where(N >= 1, E[N], m)
    PK = pref * np.sum(np.where(Np >= 1, M * M / den, 0.0))
    # valley K': valence (alpha_-|n>, beta_-|n-1>), n >= 1; conduction (alpha_+|n'>, beta_+|n'-1>),
    #            n' >= 1, and the zero mode (|0>, 0) at +m
    M = np.where(N >= 1, np.where(Np >= 1, ap[Np], 1.0) * am[N] * _G(tab, Np, N), 0.0) \
        + np.where(v1, bp[Np] * bm[N] * _G(tab, np.maximum(Np - 1, 0), np.maximum(N - 1, 0)), 0.0)
    den = np.where(Np >= 1, E[Np], m) + E[N]
    PKp = pref * np.sum(np.where(N >= 1, M * M / den, 0.0))
    return PK, PKp


def pi_field(q, m, l, n_c=(400, 800, 1600)):
    """Sum over both valleys, Pi_K + Pi_K' (one spin), extrapolated in the cutoff.

    The truncated sums behave as P(n_c) = P + a n_c^{-1/2} + b n_c^{-1}; three cutoffs fix P.
    Returns (P, estimated error) with the error the change from the two-point extrapolation.
    """
    vals = np.array([sum(pi_valleys(q, m, l, n)) for n in n_c])
    s = np.array(n_c, float) ** -0.5
    A = np.vstack([np.ones(3), s, s * s]).T
    P3 = np.linalg.solve(A, vals)[0]
    P2 = vals[-1] - s[-1] * (vals[-1] - vals[-2]) / (s[-1] - s[-2])
    return P3, abs(P3 - P2)
