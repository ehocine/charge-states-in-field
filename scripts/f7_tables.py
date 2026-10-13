"""LaTeX table bodies for the manuscript (paper/tables/*.tex) from the saved results.

    python3 scripts/f7_tables.py

tab_conv_body.tex    convergence of mu_2, mu_3 in the basis parameters (results/convergence)
tab_fields_body.tex  characteristic fields of the scans (results/scan; three-electron window with the
                     gate corrected to the complete basis, results/harden/window.json)
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def minus(text):
    """Typeset the minus signs of numbers (text hyphen -> math minus)."""
    import re
    return re.sub(r"(?<![\w$-])-(?=\d)", "$-$", text)    # not after a digit or inside "--"
TAB = os.path.join(ROOT, "paper", "tables")
DELTA = 15.0


def conv_table():
    rows = [json.loads(x) for x in open(os.path.join(ROOT, "results", "convergence", "conv.jsonl")) if x.strip()]
    names = {"n_ll": "$n_{\\rm LL}$", "js": "channels", "R_l": "$R/l_B$", "n_max3": "$n_{\\max}$ ($N=3$)",
             "e_max0": "$E_{\\max}/m$"}
    out = []
    for (D, B, label) in ((np.inf, 1.0, "no gate, $0.95\\,Z_c$, 1\\,T"), (10.0, 3.0, "$D=10$\\,nm, $Z=2$, 3\\,T"),
                          (np.inf, 0.1, "no gate, $0.95\\,Z_c$, 0.1\\,T")):
        sel = [r for r in rows if np.isclose(r["B"], B) and (r["D"] == D or (np.isinf(D) and r["D"] > 1e9))]
        if not sel:
            continue
        out.append(f"\\multicolumn{{4}}{{l}}{{{label}}}\\\\")
        for p in ("n_ll", "js", "R_l", "n_max3", "e_max0"):
            sub = [r for r in sel if r["param"] == p]
            if not sub:
                continue
            for r in sub:
                v = r["value"].replace(":", "\\dots") if p == "js" else r["value"]
                out.append(f"{names[p]} & {v} & {r['mu2']:.3f} & {r['mu3']:.3f}\\\\")
        out.append("\\hline")
    # three-electron window with the gate: states per channel up to the complete basis (results/harden)
    hp = os.path.join(ROOT, "results", "harden", "nmax3_D10.jsonl")
    if os.path.exists(hp):
        hrows = [json.loads(x) for x in open(hp) if x.strip()]
        for B in (0.30, 0.32):
            sel = sorted((r for r in hrows if np.isclose(r["B"], B)), key=lambda r: float(r["value"]))
            if not sel:
                continue
            out.append(f"\\multicolumn{{4}}{{l}}{{$D=10$\\,nm, $Z=2$, {B:.2f}\\,T}}\\\\")
            for r in sel:
                n = int(float(r["value"]))
                if n in (12, 16, 24, 32, 40):
                    v = "all" if n >= 40 else str(n)
                    out.append(f"$n_{{\\max}}$ ($N=3$) & {v} & {r['mu2']:.3f} & {r['mu3']:.3f}\\\\")
            # all states below a higher Landau-level cutoff (scripts/h5_cutoff_window.py; mu_3 only)
            cp = os.path.join(ROOT, "results", "harden", "cutoff_window_D10.jsonl")
            if os.path.exists(cp):
                for r in (json.loads(x) for x in open(cp) if x.strip()):
                    if np.isclose(r["B"], B) and B < 0.31 and r["n_ll"] > 40 \
                            and r.get("sector", "Kprime pair + K") == "Kprime pair + K":   # same branch as above
                        out.append(f"all, $n_{{\\rm LL}}$ & {r['n_ll']:g} & -- & {r['mu3']:.3f}\\\\")
            out.append("\\hline")
    open(os.path.join(TAB, "tab_conv_body.tex"), "w").write(minus("\n".join(out[:-1]) + "\n"))


def crossings(B, y, level):
    """All fields where y crosses level (linear interpolation)."""
    out = []
    for k in range(len(B) - 1):
        if (y[k] - level) * (y[k + 1] - level) < 0:
            out.append(B[k] + (level - y[k]) * (B[k + 1] - B[k]) / (y[k + 1] - y[k]))
    return out


def crossing(B, y, level):
    c = crossings(B, y, level)
    return c[0] if c else np.nan


def sector_energy(rec, valleys, sym):
    E = [s["E"] for s in rec if tuple(s["valleys"]) == tuple(valleys) and s["sym"] == sym]
    return min(E) if E else np.nan


def valley_field_2(rows):
    """Field where the K'K' singlet overtakes the K-K' pair (two electrons), interpolated."""
    B = np.array([r["B"] for r in rows])
    d = np.array([sector_energy(r["N2"], (1, -1), 0) - sector_energy(r["N2"], (-1, -1), 1) for r in rows])
    return crossing(B, d, 0.0), d[0]


def scan(name):
    path = os.path.join(ROOT, "results", "scan", f"{name}.jsonl")
    rows = {}
    for x in open(path):
        if x.strip():
            r = json.loads(x)
            rows[round(r["B"], 6)] = r
    return [rows[k] for k in sorted(rows)]


def first_change(rows, key, f):
    """Field interval where f(ground-state record) changes first."""
    prev = None
    for r in rows:
        if key not in r:
            continue
        val = f(r[key][0])
        if prev is not None and val != prev[1]:
            return prev[0], r["B"]
        prev = (r["B"], val)
    return None


def fields_table():
    """Characteristic fields: valley transitions of the two- and three-electron ground states, the
    fields where mu_1, mu_2 fall below -Delta, and where mu_3 crosses +Delta."""
    out, info = [], {}
    for name, label in (("nogate", "no gate, $Z=0.95\\,Z_c$"), ("D10", "$D=10$\\,nm, $Z=2$"), ("Z1", "no gate, $Z=1$")):
        if not os.path.exists(os.path.join(ROOT, "results", "scan", f"{name}.jsonl")):
            continue
        rows = scan(name)
        B = np.array([r["B"] for r in rows])
        mu1 = np.array([r["N1"]["E"] for r in rows])
        mu2 = np.array([r["N2"][0]["E"] for r in rows]) - mu1
        mu3 = np.array([r["N3"][0]["E"] - r["N2_trunc"][0]["E"] for r in rows])
        b2, split0 = valley_field_2(rows)
        # three electrons: from a pair plus one in the other valley to all three in K' (only
        # meaningful where the third electron is bound, mu_3 < Delta)
        v3 = first_change(rows, "N3", lambda s: len(set(s["valleys"])))
        if v3 is not None and np.all(mu3[(B >= v3[0]) & (B <= v3[1])] > DELTA) and np.all(mu3 > DELTA):
            v3 = None
        c3 = crossings(B, mu3, DELTA)
        wp = os.path.join(ROOT, "results", "harden", "window.json")
        if name == "D10" and os.path.exists(wp):
            # complete-basis correction of the three-electron window (scripts/h2_window.py)
            w = json.load(open(wp))
            v3 = (w["B_valley_T"], w["B_valley_T"])
            c3 = [w["B_mu3_Delta_T"]] + [x for x in c3 if x > 0.4]
        info[name] = {"B_valley2": b2, "split0_meV": split0, "valley3": v3, "B_mu1": crossing(B, mu1, -DELTA),
                      "B_mu2": crossing(B, mu2, -DELTA), "mu3_cross_Delta": c3,
                      "mu3_min_above_0.5T": float(np.min(mu3[B >= 0.5])) if np.any(B >= 0.5) else None}
        f = lambda x: "--" if x is None or not np.isfinite(x) else (f"{x:.2f}" if x >= 0.1 else f"{x * 1000:.1f}\\,mT")
        g = lambda t: "--" if t is None else (f"{t[0]:.2f}" if np.isclose(t[0], t[1]) else
                                              f"{t[0]:.2f}--{t[1]:.2f}" if t[0] >= 0.01 else f"{t[0]*1000:.0f}--{t[1]*1000:.0f}\\,mT")
        f3 = lambda x: f"{x:.3f}" if 0.1 <= x < 0.4 else f(x)
        c3s = ", ".join(f3(x) for x in c3) if c3 else ("never (above $\\Delta$)" if np.all(mu3 > DELTA) else "--")
        out.append(f"{label} & {f(b2)} & {g(v3)} & {f(info[name]['B_mu1'])} & {f(info[name]['B_mu2'])} & {c3s}\\\\")
    open(os.path.join(TAB, "tab_fields_body.tex"), "w").write(minus("\n".join(out) + "\n"))
    json.dump(info, open(os.path.join(ROOT, "results", "scan", "fields.json"), "w"), indent=1, default=float)
    print(json.dumps(info, indent=1, default=float))


def screening_table():
    """mu_1, mu_2 (meV) with the Landau-level screening and with the zero-field screening."""
    out = []
    for name, label in (("nogate", "no gate"), ("D10", "$D=10$\\,nm")):
        path = os.path.join(ROOT, "results", "noBscreen", f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        ref = {round(r["B"], 6): r for r in scan(name)}
        rows = {}
        for x in open(path):
            if x.strip():
                r = json.loads(x)
                rows[round(r["B"], 6)] = r
        first = True
        for B in sorted(rows):
            if B not in ref:
                continue
            a, b = ref[B], rows[B]
            mu = lambda r: (r["N1"]["E"], r["N2"][0]["E"] - r["N1"]["E"])
            (m1, m2), (n1, n2) = mu(a), mu(b)
            out.append(f"{label if first else ''} & {B:g} & {m1:.2f} & {n1:.2f} & {m2:.2f} & {n2:.2f}\\\\")
            first = False
    open(os.path.join(TAB, "tab_screen_body.tex"), "w").write(minus("\n".join(out) + "\n"))


def sts_table():
    """Mean radius of the one-electron orbital, the 1 -> 2 Dyson orbital, its quasiparticle weight and
    the intensity of the 1 -> 2 addition peak at r = 1 nm relative to the 0 -> 1 peak; sea-edge lift."""
    out = []
    for name, label in (("nogate", "no gate"), ("D10", "$D=10$\\,nm"), ("Z1", "$Z=1$")):
        path = os.path.join(ROOT, "results", "obs", f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        rows = {}
        for x in open(path):
            if x.strip():
                r = json.loads(x)
                rows[r["B"]] = r
        first = True
        for B in sorted(rows):
            r = rows[B]
            d = r.get("dyson12")
            if d is None:
                continue
            lb = r["l_nm"]
            out.append(f"{label if first else ''} & {B:g} & {lb:.1f} & {r['N1']['r_mean_nm']:.1f} & {d['r_mean_nm']:.1f} & "
                       f"{d['Z_qp']:.3f} & {d['dens_1nm_rel']:.2f} & {max(r['N2']['delta_meV'], 0):.1f}\\\\")
            first = False
    open(os.path.join(TAB, "tab_sts_body.tex"), "w").write(minus("\n".join(out) + "\n"))


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    conv_table()
    fields_table()
    screening_table()
    sts_table()
    for f in sorted(os.listdir(TAB)):
        print("==", f)
        print(open(os.path.join(TAB, f)).read())
