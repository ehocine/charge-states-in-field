"""Validation of fieldstates (PLAN.md Sec. 4): V0-V3 and internal consistency."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fieldstates.basis import FieldBasis, magnetic_length  # noqa: E402
from fieldstates.ci import FieldCI  # noqa: E402
from fieldstates.interaction import TwoValleyInteraction  # noqa: E402
from fieldstates.polarization import pi_field, pi_zero_field, pi_valleys  # noqa: E402
from chargestates.ci import TwoElectronCI  # noqa: E402
from chargestates.furry import FurryBasis  # noqa: E402
from chargestates.interaction import Interaction, interaction_weight  # noqa: E402
from chargestates.observables import moment_one, MU_B_MEV_PER_T  # noqa: E402
from collapse2d.analysis import make_solver  # noqa: E402
from collapse2d.materials import gapped_graphene  # noqa: E402

MAT = gapped_graphene(0.015)
M = MAT.m
Z = 0.95 * 1.6713866023
JS = [-0.5, 0.5, 1.5]


@pytest.fixture(scope="module")
def shape():
    return make_solver(MAT, 0.3, "rpa").v


def small_basis(shape, B=0.0, valley=1, bc=None, Z=Z, E_max=3.0, R=15.0):
    return FieldBasis(shape, Z, M, R=R / M, js=JS, E_max=E_max * M, r_core=0.3, n_scan=200,
                      l_B=magnetic_length(B), valley=valley, bc=bc)


def test_polarization_zero_field_limit():
    """Landau-level polarization (both valleys) -> 2 Pi_0 for l >> 1/m."""
    for q in (0.3, 1.0):
        P, err = pi_field(q, 1.0, 4.0)
        assert abs(P / (2 * pi_zero_field(np.array([q]), 1.0)[0]) - 1) < 3e-4


def test_polarization_valley_symmetry():
    """Pi_K = Pi_K' (particle-hole symmetry maps the K zero mode at -m onto the K' one at +m)."""
    PK, PKp = pi_valleys(0.7, 1.0, 1.5, 300)
    assert abs(PK - PKp) < 1e-12 * max(PK, 1)


def test_zero_field_equals_furry(shape):
    """V1: at B = 0 the Pruefer-index Furry space is that of chargestates (E > -m)."""
    a = FurryBasis(shape, Z, M, R=15 / M, js=JS, E_max=3 * M, r_core=0.3, n_scan=200)
    b = small_basis(shape)
    assert len(a.E) == len(b.E)
    assert np.max(np.abs(np.sort(a.E) - np.sort(b.E))) < 1e-10 * M


def test_landau_levels(shape):
    """V0/V1: without impurity, +-sqrt(m^2 + 2n/l^2); the n = 0 level at +m belongs to K' only."""
    l = magnetic_length(1.0)
    exact = np.sqrt(M * M + 2 * np.arange(0, 5) / l ** 2)
    for valley in (1, -1):
        b = small_basis(shape, B=1.0, valley=valley, Z=0.0, E_max=5.0, R=12.0)
        for j in JS:
            E = b.E[b.select(j)]
            assert np.all(np.min(np.abs(E[:, None] - exact[None, :]), axis=1) < 1e-8 * M)
        has_zero = any(np.isclose(b.E[b.select(j)][0], M, atol=1e-8 * M) for j in JS)
        assert has_zero == (valley == -1)


def test_low_field_slope_is_orbital_moment(shape):
    """V2: dE1/dB at B -> 0 equals -/+ the orbital moment of the B = 0 level (paper 2)."""
    b0 = small_basis(shape)
    mu = moment_one(b0, b0.select(0.5)[0])                     # in mu_B, valley K
    dB = 2e-3
    for valley in (1, -1):
        e = [small_basis(shape, B=s * dB, valley=valley) for s in (1,)]
        E1 = e[0].E[e[0].select(0.5)[0]]
        slope = (E1 - b0.E[b0.select(0.5)[0]]) * MAT.hbar_v * 1000 / dB / MU_B_MEV_PER_T
        assert abs(slope + valley * mu) < 0.02 * abs(mu)


def test_ci_zero_field_matches_chargestates(shape):
    """V1: FieldCI at B = 0 reproduces chargestates.TwoElectronCI (singlet and K-K')."""
    b = small_basis(shape, bc="F")
    w = interaction_weight(MAT)
    I = Interaction(b, w)
    ti = TwoValleyInteraction({1: b, -1: b}, w, q=I.q, qw=I.qw)
    ti.compress({v: {j: b.select(j) for j in JS} for v in (1, -1)}, tol=1e-10)
    for sec, val, sym, Mtot in (("singlet", (1, 1), 1, 1.0), ("intervalley", (1, -1), 0, 0.0)):
        ref = TwoElectronCI(b, I, sec, Mtot).solve(1, method="dense")[0]
        e, _ = FieldCI(ti, val, sym, Mtot).davidson(tol=1e-11)
        assert abs(e - ref) < 1e-8 * M


def test_kprime_boundary_condition_zero_field(shape):
    """At B = 0 the K' box (G(R) = 0) and the K box (F(R) = 0) give the same bound level and
    two-electron energies to box accuracy."""
    bK, bKp = small_basis(shape, valley=1), small_basis(shape, valley=-1)
    assert abs(bK.E[bK.select(0.5)[0]] - bKp.E[bKp.select(0.5)[0]]) < 1e-9 * M
    w = interaction_weight(MAT)
    ti = TwoValleyInteraction({1: bK, -1: bKp}, w)
    ti.compress({1: {j: bK.select(j) for j in JS}, -1: {j: bKp.select(j) for j in JS}}, tol=1e-9)
    eK, _ = FieldCI(ti, (1, 1), 1, 1.0).davidson(tol=1e-11)
    eKp, _ = FieldCI(ti, (-1, -1), 1, -1.0).davidson(tol=1e-11)
    assert abs(eK - eKp) < 2e-3 * M


def test_hermitian_and_symmetry(shape):
    """H is symmetric in every sector; the same-valley three-electron S = 3/2 weight is 0 or 1."""
    b = small_basis(shape, B=0.5, valley=-1, E_max=2.5, R=10.0)
    bK = small_basis(shape, B=0.5, valley=1, E_max=2.5, R=10.0)
    ti = TwoValleyInteraction({1: bK, -1: b}, interaction_weight(MAT))
    ti.compress({1: {j: bK.select(j)[:4] for j in JS}, -1: {j: b.select(j)[:4] for j in JS}}, tol=1e-10)
    rng = np.random.default_rng(1)
    for val, sym, Mtot in (((1, -1), 0, 0.0), ((-1, -1), -1, -2.0), ((1, 1, -1), 1, 0.5),
                           ((-1, -1, -1), -1, -2.5)):
        c = FieldCI(ti, val, sym, Mtot)
        x, y = c.symmetrize(rng.normal(size=c.size)), c.symmetrize(rng.normal(size=c.size))
        assert abs(x @ c.matvec(y) - y @ c.matvec(x)) < 1e-10 * np.linalg.norm(x) * np.linalg.norm(y)
    c = FieldCI(ti, (-1, -1, -1), -1, -2.5)
    e, v = c.davidson(tol=1e-10)
    w3 = c.antisymmetric_weight(v)
    assert min(w3, 1 - w3) < 1e-6


def test_four_electrons_noninteracting_and_hermitian(shape):
    """N = 4: without interaction the energy is the sum of the four lowest allowed orbital energies;
    with interaction H is symmetric on the symmetrised space; generic contraction = N = 3 path."""
    b = small_basis(shape, B=1.0, valley=-1, E_max=4.0, R=10.0)
    bK = small_basis(shape, B=1.0, valley=1, E_max=4.0, R=10.0)
    ti = TwoValleyInteraction({1: bK, -1: b}, interaction_weight(MAT))
    ti.compress({1: {j: bK.select(j)[:4] for j in JS}, -1: {j: b.select(j)[:4] for j in JS}}, tol=1e-10)
    off = {p: 0.0 for p in [(i, k) for i in range(4) for k in range(i + 1, 4)]}
    # (up-up)(down-down) in K': lowest is j~ = 1/2 (x2) and the next two orbitals (x2)
    E = sorted(np.concatenate([b.E[b.select(j)][:4] for j in JS]))
    best = min(FieldCI(ti, (-1,) * 4, -1, Mt, terms=off, sym23=-1).davidson(tol=1e-10)[0]
               for Mt in (-3.0, -4.0, -2.0, -5.0))
    assert abs(best - 2 * (E[0] + E[1])) < 1e-9 * M
    rng = np.random.default_rng(2)
    c = FieldCI(ti, (-1, -1, -1, -1), -1, -4.0, sym23=-1)
    x, y = c.symmetrize(rng.normal(size=c.size)), c.symmetrize(rng.normal(size=c.size))
    assert abs(x @ c.matvec(y) - y @ c.matvec(x)) < 1e-10 * np.linalg.norm(x) * np.linalg.norm(y)


def test_field_ratio_weak_field_and_limits():
    """rho = 1 at B = 0 and for m l > 5 (weak fields); rho < 1 at small q l, -> 1 at large q l."""
    from fieldstates.screening import FieldRatio
    q = np.geomspace(1e-4, 1.0, 50)
    assert np.all(FieldRatio(M, np.inf)(q) == 1.0)
    assert np.all(FieldRatio(M, 6.0 / M)(q) == 1.0)
    r = FieldRatio(M, magnetic_length(1.0))
    assert r(np.array([0.3 / magnetic_length(1.0)]))[0] < 0.9
    assert abs(r(np.array([20.0 / magnetic_length(1.0)]))[0] - 1) < 1e-12


def test_sector_lower_bound(shape):
    """FieldCI.min_one_body is the exact noninteracting ground energy of a sector (Pauli included)
    and a lower bound on the interacting one (W > 0), for two, three and four electrons."""
    b = small_basis(shape, B=0.5, valley=-1, E_max=2.5, R=10.0)
    bK = small_basis(shape, B=0.5, valley=1, E_max=2.5, R=10.0)
    ti = TwoValleyInteraction({1: bK, -1: b}, interaction_weight(MAT))
    ti.compress({1: {j: bK.select(j)[:4] for j in JS}, -1: {j: b.select(j)[:4] for j in JS}}, tol=1e-10)
    for val, sym, Mt, s23 in (((-1, -1), -1, -1.0, 0), ((-1, -1, 1), 1, -0.5, 0), ((-1, -1, -1), -1, -2.5, 0),
                              ((-1, -1, -1, -1), -1, -4.0, -1)):
        c = FieldCI(ti, val, sym, Mt, sym23=s23)
        off = {p: 0.0 for p in c.pairs}
        e0, _ = FieldCI(ti, val, sym, Mt, terms=off, sym23=s23).davidson(tol=1e-11)
        e, _ = c.davidson(tol=1e-10)
        assert abs(c.min_one_body() - e0) < 1e-9 * M
        assert c.min_one_body() < e


def test_davidson_matches_dense_in_field(shape):
    """The (memory-lean) Davidson eigenvalue equals the lowest eigenvalue of the dense Hamiltonian on
    the exchange-symmetric space, at B = 1 T, for two, three and four electrons in several sectors."""
    b = small_basis(shape, B=1.0, valley=-1, E_max=2.5, R=10.0)
    bK = small_basis(shape, B=1.0, valley=1, E_max=2.5, R=10.0)
    ti = TwoValleyInteraction({1: bK, -1: b}, interaction_weight(MAT))
    ti.compress({1: {j: bK.select(j)[:3] for j in JS}, -1: {j: b.select(j)[:3] for j in JS}}, tol=1e-10)
    shift = 100.0 * M          # pushes the symmetric eigenvalues below the zeros of the complement
    for val, sym, Mtot, s23 in (((-1, -1), 1, -1.0, 0), ((1, -1), 0, 0.0, 0), ((-1, -1, 1), 1, -0.5, 0),
                                ((-1, -1, -1), -1, -2.5, 0), ((-1, -1, -1, -1), -1, -4.0, -1),
                                ((-1, -1, 1, 1), 1, 0.0, 1)):
        c = FieldCI(ti, val, sym, Mtot, sym23=s23)
        if c.size == 0:
            continue
        P = np.array([c.symmetrize(e) for e in np.eye(c.size)]).T
        H = np.array([c.matvec(P[:, i]) for i in range(c.size)]).T
        A = P.T @ (H - shift * P)
        e_dense = np.linalg.eigvalsh(0.5 * (A + A.T))[0] + shift
        e_dav, _ = c.davidson(tol=1e-11)
        assert abs(e_dav - e_dense) < 1e-8 * M, (val, sym, Mtot, e_dav, e_dense)
