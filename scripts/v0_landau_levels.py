"""V0 (2026-10-09): check of the radial Dirac equations in a perpendicular field (symmetric gauge), valley K, U = 0:
    F' = (j/r + r/2l^2) F - (E + m) G,   G' = -(j/r + r/2l^2) G + (E - m) F.
Pruefer: theta' = E - m cos2theta - (j/r + r/2l^2) sin2theta; box F(R) = 0; eigenvalues where
theta(R;E) = pi/2 + n pi.  Expected: E = +-sqrt(m^2 + 2n/l^2), n >= 1, and the n = 0 level at
E = -m for j <= -1/2 only (zero mode F = 0, G ~ r^{-j} exp(-r^2/4l^2)).

Usage: python3 scripts/v0_landau_levels.py   (about a minute; standalone, scipy only)
Result 2026-10-09: all expected levels reproduced to 6 digits (see PHYSICS.md Sec. 3)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

m, l, R = 1.0, 1.0, 9.0

def theta_R(E, j):
    r0 = 1e-6
    th0 = 0.0 if j > 0 else np.pi / 2          # regular start: F ~ r^j (j>0) or G ~ r^|j| (j<0)
    f = lambda r, t: [E - m * np.cos(2 * t[0]) - (j / r + r / (2 * l * l)) * np.sin(2 * t[0])]
    return solve_ivp(f, (r0, R), [th0], rtol=1e-11, atol=1e-12, method="DOP853").y[0, -1]

def levels(j, Emin=-3.1, Emax=3.1, n=1240):
    Es = np.linspace(Emin, Emax, n)
    th = np.array([theta_R(E, j) for E in Es])
    out = []
    for k in range(len(Es) - 1):
        a = np.floor((th[k] - np.pi / 2) / np.pi); b = np.floor((th[k + 1] - np.pi / 2) / np.pi)
        if b > a:
            t = np.pi / 2 + b * np.pi
            out.append(brentq(lambda E: theta_R(E, j) - t, Es[k], Es[k + 1], xtol=1e-12))
    return np.array(out)

exact = sorted([np.sqrt(m * m + 2 * n / l ** 2) for n in range(1, 4)])
print("expected |E| for n = 1..3:", np.round(exact, 6), " n = 0 at E = -m for j <= -1/2 (valley K, B > 0)")
for j in (-1.5, -0.5, 0.5, 1.5):
    ev = levels(j)
    print(f"j = {j:+.1f}:", np.round(ev, 6))
