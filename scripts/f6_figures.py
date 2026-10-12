"""Figures of the manuscript from the saved results.

    python3 scripts/f6_figures.py [RESULTS_SUBDIR]      (default: scan)

fig_levels   single-particle levels vs B, both valleys (results/levels/*.json)
fig_mu       addition energies mu_N(B) with ground-state valley/spin labels and the low-field
             slopes of paper 2; dotted: zero-field screening
fig_diagram  charge-state diagram in the (B, E_F) plane, with the sea-edge strip and, with the
             gate, the low-field four-electron strip (results/four)
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")
DELTA = 15.0
MU_B = 0.0578838
SYSTEMS = (("nogate", "no gate, $Z=0.95\\,Z_c$", (191.0, 564.0, 384.0)),
           ("D10", "gate $D=10$ nm, $Z=2$", (113.9, 410.1, 298.6)),
           ("Z1", "no gate, $Z=1$", None))


def load(path):
    if not os.path.exists(path):
        return []
    rows = [json.loads(x) for x in open(path) if x.strip()]
    best = {}
    for r in rows:
        best[round(r["B"], 6)] = r
    return [best[k] for k in sorted(best)]


def addition(rows):
    B = np.array([r["B"] for r in rows])
    mu1 = np.array([r["N1"]["E"] for r in rows])
    mu2 = np.array([r["N2"][0]["E"] for r in rows]) - mu1
    mu3 = np.array([r["N3"][0]["E"] - r["N2_trunc"][0]["E"] if "N3" in r else np.nan for r in rows])
    lab2 = [r["N2"][0] for r in rows]
    lab3 = [r["N3"][0] if "N3" in r else None for r in rows]
    return B, mu1, mu2, mu3, lab2, lab3


def fourth(key):
    """mu_4(B) (meV) from results/four/<key>*.jsonl, the largest basis at each field."""
    import glob
    best = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "four", f"{key}*.jsonl"))):
        for x in open(path):
            if x.strip():
                r = json.loads(x)
                b = round(r["B"], 6)
                if b not in best or r["n4"] > best[b]["n4"]:
                    best[b] = r
    B = np.array(sorted(best))
    return B, np.array([best[b]["mu4"] for b in B])


def sqrt_axis(ax, Bmax):
    ax.set_xscale("function", functions=(lambda x: np.sqrt(np.maximum(x, 0)), lambda x: x ** 2))
    ticks = [t for t in (0, 0.05, 0.2, 0.5, 1, 2, 3, 4, 5) if t <= Bmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlim(0, Bmax)


def crossing(B, y, level):
    s = np.sign(y - level)
    for k in range(len(B) - 1):
        if s[k] != s[k + 1] and s[k] != 0:
            return B[k] + (level - y[k]) * (B[k + 1] - B[k]) / (y[k + 1] - y[k])
    return np.nan


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else "scan"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    os.makedirs(FIG, exist_ok=True)
    fig_screening(plt)
    fig_levels(plt)
    data = {k: load(os.path.join(ROOT, "results", sub, f"{k}.jsonl")) for k, _, _ in SYSTEMS}
    systems = [s for s in SYSTEMS if data[s[0]]]

    # ---- addition energies ---------------------------------------------------------------
    fig, axs = plt.subplots(1, len(systems), figsize=(3.4 * len(systems), 2.9), sharey=True, squeeze=False)
    for ax, (key, title, moments) in zip(axs[0], systems):
        B, mu1, mu2, mu3, l2, l3 = addition(data[key])
        for y, c, name in ((mu1, "C0", "$\\mu_1$"), (mu2, "C1", "$\\mu_2$"), (mu3, "C2", "$\\mu_3$")):
            ax.plot(B, y, "-o", color=c, ms=2.5, lw=1.0, label=name)
        # linear continuation of the low-field slopes (fit on 2 mT <= B <= 10 mT, after the
        # two-electron valley transition); mu_3 only where the third electron is bound
        low = (B >= 0.0019) & (B <= 0.0101)
        if low.sum() >= 2:
            Bl = np.linspace(0, 0.5, 30)
            for y, c in ((mu1, "C0"), (mu2, "C1"), (mu3, "C2")):
                if c == "C2" and np.all(y[low] > DELTA):
                    continue
                k, y0 = np.polyfit(B[low], y[low], 1)
                ax.plot(Bl, y0 + k * Bl, ":", color=c, lw=0.8)
        ax.axhspan(DELTA, 80, color="0.85", lw=0)
        ax.axhspan(-80, -DELTA, color="0.85", lw=0)
        ax.set_ylim(-55, 25)
        sqrt_axis(ax, B.max())
        ax.set_xlabel("$B$ (T)")
        ax.set_title(title, fontsize=9)
    axs[0][0].set_ylabel("addition energy (meV)")
    axs[0][0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_mu.{ext}"), dpi=200)

    # ---- charge-state diagram ------------------------------------------------------------
    cols = ["#f2f2f2", "#d4e6f4", "#9ecae1", "#5a9fd0", "#3a7fbf"]
    fig, axs = plt.subplots(1, len(systems), figsize=(3.4 * len(systems), 2.9), sharey=True, squeeze=False)
    for ax, (key, title, _) in zip(axs[0], systems):
        B, mu1, mu2, mu3, l2, l3 = addition(data[key])
        Bs = np.linspace(B.min(), B.max(), 1200)
        lines = [np.interp(Bs, B, y) for y in (mu1, mu2, mu3)]
        B4, mu4 = fourth(key)
        if len(B4) >= 2:
            lines.append(np.interp(Bs, B4, mu4, right=np.inf))
        bounds = [np.full_like(Bs, -DELTA)] + [np.clip(y, -DELTA, DELTA) for y in lines] + [np.full_like(Bs, DELTA)]
        for n in range(len(lines) + 1):
            lo, hi = bounds[n], np.maximum(bounds[n + 1], bounds[n])
            ax.fill_between(Bs, lo, hi, color=cols[n], lw=0)
            h = hi - lo
            if n == 4 and h.max() > 0:
                k = int(np.argmax(h))
                ax.annotate("$N=4$", (Bs[k], DELTA - 0.4), xytext=(0.02 * Bs.max(), 9.0), fontsize=7,
                            ha="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="0.2"))
            elif h.max() > 1.0:
                k = int(np.argmax(h)) if n != 2 else int(np.argmax(h * (Bs > 0.3 * Bs.max())))
                xk = min(max(Bs[k], 0.004 * Bs.max()), 0.8 * Bs.max())
                if n == 0:
                    xk = 0.35 * crossing(Bs, lines[0], -DELTA) if np.isfinite(crossing(Bs, lines[0], -DELTA)) else xk
                    k = int(np.searchsorted(Bs, xk))
                yk = 0.5 * (lo[k] + hi[k])
                ax.text(xk, yk, f"$N={n}$", ha="center", va="center", fontsize=8)
        for y in lines[:3]:
            ax.plot(Bs, np.clip(y, -DELTA, DELTA), "k-", lw=0.8)
        if len(lines) > 3:
            y = np.where(lines[3] < DELTA, lines[3], np.nan)
            ax.plot(Bs, y, "k-", lw=0.6)
        obs = os.path.join(ROOT, "results", "obs", f"{key}.jsonl")
        if os.path.exists(obs):
            rows = [json.loads(x) for x in open(obs) if x.strip()]
            rows = sorted({r["B"]: r for r in rows}.values(), key=lambda r: r["B"])
            Bo = np.array([r["B"] for r in rows])
            # validity: in the region of N electrons (mu_N < E_F < mu_{N+1}) the diagram needs
            # E_F > -Delta + delta_N; hatch the part of each region below that line
            for N, lo_line, hi_line in ((2, lines[1], lines[2]), (3, lines[2], None)):
                dN = np.interp(Bs, Bo, np.array([max(r[f"N{N}"]["delta_meV"], 0.0) for r in rows]))
                lo = np.maximum(np.clip(lo_line, -DELTA, DELTA), -DELTA)
                top = np.minimum(-DELTA + dN, DELTA if hi_line is None else np.clip(hi_line, -DELTA, DELTA))
                ok = top > lo
                ax.fill_between(Bs, lo, np.where(ok, top, lo), where=ok, facecolor="none", edgecolor="0.25",
                                hatch="////", lw=0)
        sqrt_axis(ax, B.max())
        ax.set_ylim(-DELTA, DELTA)
        ax.set_xlabel("$B$ (T)")
        ax.set_title(title, fontsize=9)
        for name, y in (("mu1", mu1), ("mu2", mu2), ("mu3", mu3)):
            print(key, name, "crosses -Delta at", crossing(B, y, -DELTA), "+Delta at", crossing(B, y, DELTA))
    axs[0][0].set_ylabel("$E_F$ (meV)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_diagram.{ext}"), dpi=200)


def fig_levels(plt):
    """Lowest level of each channel vs B in K (dashed) and K' (solid), clean Landau levels (grey)."""
    files = [(k, t) for k, t, _ in SYSTEMS[:2] if os.path.exists(os.path.join(ROOT, "results", "levels", f"{k}.json"))]
    if not files:
        return
    fig, axs = plt.subplots(1, len(files), figsize=(3.4 * len(files), 3.0), sharey=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, (key, title) in zip(axs[0], files):
        d = json.load(open(os.path.join(ROOT, "results", "levels", f"{key}.json")))
        Bs = sorted(d, key=float)
        B = np.array([float(b) for b in Bs])
        LL = np.array([d[b]["landau"] for b in Bs])
        for n in range(1, LL.shape[1]):
            ax.plot(B, LL[:, n], "-", color="0.75", lw=0.6)
            ax.plot(B, -LL[:, n], "-", color="0.75", lw=0.6)
        js = ["-1.5", "-0.5", "+0.5", "+1.5", "+2.5", "+3.5", "+4.5"]
        for i, j in enumerate(js):
            c = cmap(i / (len(js) - 1))
            for val, ls in (("-1", "-"), ("1", "--")):
                y = np.array([d[b][val][j]["E"][0] if d[b][val][j]["E"] else np.nan for b in Bs])
                ax.plot(B, y, ls, color=c, lw=1.0, label=(f"$j,\\tilde{{\\jmath}}={float(j):g}$" if val == "-1" else None))
        ax.axhspan(DELTA, 200, color="0.92", lw=0)
        ax.axhspan(-200, -DELTA, color="0.92", lw=0)
        ax.set_ylim(-60, 50)
        ax.set_xlim(0, B.max())
        ax.set_xlabel("$B$ (T)")
        ax.set_title(title, fontsize=9)
    axs[0][0].set_ylabel("$E$ (meV)")
    axs[0][0].legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_levels.{ext}"), dpi=200)


def fig_screening(plt):
    """rho = Pi_B/Pi_0 vs q l and eps_B/eps_0 vs q for 2Delta = 30 meV (fieldstates.screening)."""
    sys.path.insert(0, ROOT)
    from fieldstates.screening import ratio_table, FieldScreenedImpurity
    from fieldstates.basis import magnetic_length
    from collapse2d.materials import gapped_graphene
    mat = gapped_graphene(0.015)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6))
    q = np.geomspace(1e-3, 1.0, 300)
    eps0 = FieldScreenedImpurity(1.0, mat, 1.0, l_B=np.inf, r_max=10.0, n_r=16).eps(q)
    for B, c in ((0.3, "C0"), (1.0, "C1"), (3.0, "C2")):
        l = magnetic_length(B)
        ql, rho, err = ratio_table(mat.m * l)
        axs[0].plot(ql, rho, "-", color=c, lw=1.0, label=f"{B:g} T ($m l_B={mat.m * l:.2f}$)")
        epsB = FieldScreenedImpurity(1.0, mat, 1.0, l_B=l, r_max=10.0, n_r=16).eps(q)
        axs[1].plot(q / mat.m, epsB / eps0, "-", color=c, lw=1.0)
    axs[0].axhline(1, color="0.5", lw=0.5)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("$q l_B$")
    axs[0].set_ylabel("$\\Pi_B/\\Pi_0$")
    axs[0].legend(fontsize=7)
    axs[1].axhline(1, color="0.5", lw=0.5)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("$q/m$")
    axs[1].set_ylabel("$\\varepsilon_B/\\varepsilon_0$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_screening.{ext}"), dpi=200)


if __name__ == "__main__":
    main()
