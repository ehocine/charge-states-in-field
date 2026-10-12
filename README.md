# Charge states of a near-critical Coulomb impurity in gapped graphene in a magnetic field

Code, data and figures for

> E. Hocine, *Magnetic-field-driven charging and valley polarization of a near-critical Coulomb
> impurity in gapped graphene* (manuscript, 2026; REVTeX/PRB format)

Third paper of a series:

- paper 1, *Atomic collapse in gapped two-dimensional Dirac materials*, code `collapse2d`
  ([ehocine/dirac-collapse](https://github.com/ehocine/dirac-collapse), release 1.0.0,
  [doi:10.5281/zenodo.23123320](https://doi.org/10.5281/zenodo.23123320));
- paper 2, *Charge states of a near-critical Coulomb impurity in gapped graphene: Correlation and
  gate control*, code `chargestates`
  ([ehocine/impurity-charge-states](https://github.com/ehocine/impurity-charge-states), release 1.0.4,
  [doi:10.5281/zenodo.23325384](https://doi.org/10.5281/zenodo.23325384)), on which this package builds.

Repository: https://github.com/ehocine/charge-states-in-field · Archive (all releases; 1.0.1 is the version of the manuscript): [doi:10.5281/zenodo.23317748](https://doi.org/10.5281/zenodo.23317748) · License: MIT (see `LICENSE`) · Citation: `CITATION.cff`

## Quick start

```bash
pip install -r requirements.txt                      # numpy, scipy, mpmath, matplotlib, pytest
git clone --branch 1.0.4 https://github.com/ehocine/impurity-charge-states external/impurity-charge-states
git clone --branch 1.0.0 https://github.com/ehocine/dirac-collapse external/dirac-collapse
make test                                            # regression tests (~5 min)
make figures                                         # figures and generated tables from results/ (~1 min)
```

`chargestates` is found in `external/impurity-charge-states` (or via `CHARGESTATES_ROOT`), and it
finds `collapse2d` in `external/dirac-collapse` (or via `COLLAPSE2D_ROOT`); see `fieldstates/deps.py`.

## Model and method in brief

No-pair (Furry-picture) Hamiltonian for up to three (four) electrons bound to a charge Ze at 0.3 nm
above gapped graphene in a perpendicular field B. The field enters the radial Dirac equation exactly
(symmetric gauge, one basis per valley, K' = complex conjugate of K at -B), and the impurity
potential and the electron-electron interaction are screened by the static polarization of the
Landau-quantized Dirac sea (double Landau-level sum, extrapolated in the cutoff). The positive-energy
space is defined by the Pruefer index of the box eigenstates (the K zero mode at -Delta belongs to
the sea), with the field in the projector. CI in every valley/spin/angular-momentum sector with a
matrix-free Davidson solver and jointly q-compressed form factors; spin Zeeman energy included.

## Layout

```
fieldstates/
  deps.py          locates chargestates (release 1.0.4), which locates collapse2d
  polarization.py  static polarization of the gapped sea in a field (Landau-level sums)
  screening.py     eps_B(q), field-dependent impurity potential and interaction
  basis.py         Furry basis in a field (Pruefer index, valley-dependent box condition)
  interaction.py   two-valley form factors with joint q compression
  ci.py            CI for 2-4 electrons with arbitrary valley content and pair symmetries
  driver.py        one field point: bases, interaction, ground states over all sectors, Zeeman
  observables.py   Hartree potential of CI densities, lift of the K zero mode (validity edge),
                   Dyson orbital 1 -> 2
tests/             test_fieldstates.py (Landau levels, B -> 0 limit vs chargestates, orbital-moment
                   slope, polarization limits, hermiticity, N = 4 noninteracting limit, ...)
scripts/
  v0_landau_levels.py   standalone check of the radial equations in a field
  f1_convergence.py     convergence in n_LL, channels, R/l_B, n_max, E_max (results/convergence)
  f2_scan.py            addition energies and ground-state sectors vs B (results/scan, results/noBscreen)
  f3_levels.py          single-particle levels vs B (results/levels)
  f4_four.py            fourth addition energy (results/four)
  f5_observables.py     validity edge, Dyson orbitals, radii (results/obs)
  f6_figures.py         figures
  f7_tables.py          generated tables (paper/tables)
  f8_paper2_moments.py  finite-field moments for paper 2 (results/paper2_moments)
  h1_three_sectors.py   three-electron sectors (K' pair + K, all K') vs B with the gate (results/harden)
  h2_window.py          three-electron window corrected to the complete basis (results/harden/window.json)
  h3_davidson_recheck.py  N = 3, 4 ground states recomputed after the Davidson start-vector fix
  h4_gate_kernel.py     field scan of Z = 2, D = 10 nm with the exact vacuum/hBN/metal gate kernel
                        (results/harden/exact_gate_D10.jsonl)
results/           JSON outputs; results/screening caches the polarization ratio tables
figures/           figures of the manuscript
paper/tables/      generated table bodies
```

## Reproducing the data

| step | command | time (laptop, 4 threads) |
|---|---|---|
| convergence | `scripts/f1_convergence.py` (parameters as in the convergence table) | ~1.5 h |
| scans | `scripts/f2_scan.py results/scan/{nogate,D10,Z1}.jsonl 30 {inf,10,inf} {1.587817272185,2.0,1.0} B...` | ~1.5 h per system |
| zero-field screening | `scripts/f2_scan.py ... --no-field-screening` (results/noBscreen) | ~20 min per system |
| levels, four electrons, observables | `scripts/f3_levels.py`, `f4_four.py`, `f5_observables.py` | ~1.5 h |
| three-electron window | `scripts/h1_three_sectors.py`; `results/harden/run_nmax3.sh`, `run_full3.sh` (n_max up to all states) | ~3 h |
| figures and tables | `make figures` | ~1 min |

The field grids and settings are stored in every result record.
