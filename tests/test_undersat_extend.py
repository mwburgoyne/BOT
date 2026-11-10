"""Tests for the compact undersaturated-FVF reconstruction tool."""

import os

import numpy as np
import pytest

from botkit import SurfaceFluids, read_excel
from botkit import pipeline
from botkit.model import Config, Diagnostics
from botkit.qc import detect_insufficient_undersaturated
from botkit.undersat_extend import (
    b_to_z,
    branch_is_under_defined,
    cubic_z,
    cacb_from_anchor_slope,
    cg_psc_anchored,
    compact_oil_bo,
    curvature_snr,
    extend_bg_compressibility,
    fit_cacb,
    gas_moles_per_mscf,
    oil_moles_per_stb,
    reconstruct_b,
    z_to_b,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")
SURFACE = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)
T = 200.0 + 459.67  # deg R


def test_b_z_roundtrip_is_exact():
    n = oil_moles_per_stb(0.8, SURFACE)
    p = np.array([2000.0, 3000.0, 5000.0])
    b = np.array([1.45, 1.42, 1.38])
    z = b_to_z(b, p, n, T)
    np.testing.assert_allclose(z_to_b(z, p, n, T), b, rtol=1e-12)


def test_fit_then_reconstruct_recovers_a_synthetic_branch():
    # build a synthetic undersaturated oil branch from known constants, fit it
    # back, and check the cubic reproduces Bo to fitting precision.
    n = oil_moles_per_stb(1.2, SURFACE)
    CA_true, CB_true = 1.5, 0.02
    p = np.linspace(2500.0, 9000.0, 12)
    z = np.array([cubic_z(CA_true, CB_true, pi, "oil") for pi in p])
    b = z_to_b(z, p, n, T)

    CA, CB = fit_cacb(p, b, n, "oil", T)
    b_rec = reconstruct_b(CA, CB, p, n, "oil", T)
    err = np.abs(b_rec / b - 1.0)
    assert err.max() < 1e-3


def test_anchor_slope_beats_constant_co_on_synthetic_branch():
    # a curved branch from known constants; reconstruct from the anchor + the
    # bubble-point slope only, and confirm the cubic tracks the curvature far
    # better than a flat-compressibility expansion.
    import botkit.undersat_extend as U

    n = oil_moles_per_stb(1.0, SURFACE)
    CA_true, CB_true = 2.0, 0.025
    psat = 3000.0
    p = np.linspace(psat, 12000.0, 20)
    b = z_to_b(np.array([U.cubic_z(CA_true, CB_true, pi, "oil") for pi in p]),
               p, n, T)
    bo_sat = b[0]
    # bubble-point compressibility from the true branch (a correlation/EOS stand-in)
    co_pb = -(b[1] - b[0]) / (b[0] * (p[1] - p[0]))

    sol = cacb_from_anchor_slope(psat, bo_sat, co_pb, n, "oil", T)
    assert sol is not None
    CA, CB = sol
    b_cubic = reconstruct_b(CA, CB, p, n, "oil", T)
    b_const = bo_sat * np.exp(-co_pb * (p - psat))

    e_cubic = np.abs(b_cubic / b - 1.0).mean()
    e_const = np.abs(b_const / b - 1.0).mean()
    assert e_cubic < e_const
    assert e_cubic < 0.01  # well under 1 percent


def test_compact_oil_bo_anchors_node_and_is_monotone():
    n = oil_moles_per_stb(0.9, SURFACE)
    psat, bo_sat, rs = 2500.0, 1.5, 0.9
    co_pb = 1.5e-5
    p_q = np.linspace(2700.0, 8000.0, 10)
    bo, (CA, CB) = compact_oil_bo(psat, rs, bo_sat, SURFACE, T, p_q, co_pb=co_pb)
    assert np.all(np.isfinite(bo))
    # undersaturated Bo falls with pressure and stays below the saturated value
    assert np.all(np.diff(bo) < 0)
    assert bo[0] < bo_sat
    # node anchoring: Bo extrapolated back to psat equals bo_sat
    bo_at_psat, _ = compact_oil_bo(psat, rs, bo_sat, SURFACE, T,
                                   np.array([psat]), co_pb=co_pb)
    assert bo_at_psat[0] == pytest.approx(bo_sat, rel=1e-9)


def test_gas_compressibility_integration_reduces_to_ideal():
    # with the anchor compressibility equal to the ideal-gas value, the integral
    # collapses to Bg ~ 1/p exactly.
    psc = 14.696
    p_anchor, bg_anchor = 300.0, 8.0
    # fine grid so the trapezoidal integral of c_g converges to the analytic ln
    p_q = np.linspace(300.0, 2000.0, 400)
    cg_up = 1.0 / 1000.0  # = ideal c_g at p_up so the whole curve is ideal 1/p
    bg = extend_bg_compressibility(p_q, p_anchor, bg_anchor, cg_up, 1000.0, psc)
    np.testing.assert_allclose(bg, bg_anchor * p_anchor / p_q, rtol=1e-4)


def test_cg_psc_anchored_hits_both_endpoints():
    cg_up, p_up, psc = 9.0e-4, 1000.0, 14.696
    assert cg_psc_anchored(np.array([psc]), cg_up, p_up, psc)[0] == \
        pytest.approx(1.0 / psc)
    assert cg_psc_anchored(np.array([p_up]), cg_up, p_up, psc)[0] == \
        pytest.approx(cg_up)


def test_curvature_snr_high_for_curved_low_for_linear():
    n = gas_moles_per_mscf(0.0, SURFACE)
    p = np.linspace(900.0, 5000.0, 12)
    # a strongly curved Z(p) -> high S/N
    z_curved = 1.0 - 1e-4 * p + 4e-8 * p ** 2
    b_curved = z_to_b(z_curved, p, n, T)
    # a near-straight Z(p) -> low S/N
    z_flat = 1.0 - 1e-4 * p
    b_flat = z_to_b(z_flat, p, n, T)
    assert curvature_snr(p, b_curved, n, T) > curvature_snr(p, b_flat, n, T)


def test_branch_under_defined_predicate():
    psat = 2000.0
    assert branch_is_under_defined(psat, np.empty((0, 3)))
    assert branch_is_under_defined(psat, np.array([[2100.0, 1.4, 0.5]]))  # 1 row
    # two clustered rows just above psat do not span
    clustered = np.array([[2050.0, 1.45, 0.5], [2100.0, 1.44, 0.51]])
    assert branch_is_under_defined(psat, clustered)
    # two rows reaching well above psat do span
    spanned = np.array([[2100.0, 1.45, 0.5], [4000.0, 1.40, 0.55]])
    assert not branch_is_under_defined(psat, spanned)


def test_detector_fires_on_thin_branch():
    table = read_excel(DATA, surface=SURFACE)
    # strip every undersaturated oil branch to a single row
    table.pvto.usat = [r[:1] if r.shape[0] else r for r in table.pvto.usat]
    diag = Diagnostics()
    detect_insufficient_undersaturated(table, diag)
    assert any(a.kind == "insufficient_undersaturated" for a in diag.anomalies)


def _synthetic_locus():
    p = np.array([500.0, 1000.0, 2000.0, 3000.0])
    rs = np.array([0.2, 0.5, 1.0, 1.5])
    rv = np.array([0.01, 0.02, 0.03, 0.04])
    bo = np.array([1.2, 1.3, 1.45, 1.6])
    bg = np.array([6.0, 3.0, 1.6, 1.1])
    uo = np.array([1.0, 0.8, 0.6, 0.5])
    ug = np.array([0.02, 0.025, 0.03, 0.035])
    return p, rs, rv, bo, bg, uo, ug


def test_extend_low_compressibility_bg_path():
    from botkit.extend_low import extend_below_pmin
    args = _synthetic_locus()
    low = extend_below_pmin(*args, SURFACE, bg_method="compressibility")
    assert low is not None
    assert "c_g integration" in low["bg_basis"]
    bg_e = low["bg"]                       # ascending grid psc .. just below p1
    assert np.all(bg_e > 0)
    assert np.all(np.diff(bg_e) < 0)       # p rises -> Bg falls
    assert np.all(bg_e > args[4][0])       # below p1, so Bg exceeds the anchor Bg
    assert bg_e[0] > bg_e[-1]              # ideal-style rise toward psc

    # the compressibility path genuinely differs from 1/Bg interpolation
    low_i = extend_below_pmin(*args, SURFACE, bg_method="interp")
    assert not np.allclose(low["bg"], low_i["bg"], rtol=1e-3)


def test_extend_low_compressibility_falls_back_when_cg_nonpositive():
    from botkit.extend_low import extend_below_pmin
    p, rs, rv, bo, bg, uo, ug = _synthetic_locus()
    bg = np.array([3.0, 6.0, 1.6, 1.1])    # Bg rising with p at the bottom -> c_g<=0
    low = extend_below_pmin(p, rs, rv, bo, bg, uo, ug, SURFACE,
                            bg_method="compressibility")
    assert "c_g integration" not in low["bg_basis"]
    assert any("compressibility Bg unavailable" in f for f in low["flags"])


def test_pipeline_low_bg_compressibility_builds():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    cfg.bg_low_method = "compressibility"
    res = pipeline.build(table, cfg)
    e = res.extended
    assert e.pvtg.p.min() == pytest.approx(cfg.psc)
    assert np.all(e.pvtg.bg > 0)
    assert any("c_g integration" in c.reason for c in res.changes.changes)


def test_pipeline_compact_method_builds_and_logs():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    cfg.undersaturated_method = "compact"
    res = pipeline.build(table, cfg)
    e = res.extended
    # branches still physical: monotone Bo with positive compressibility
    for i, rows in enumerate(e.pvto.usat):
        if rows.shape[0] < 2:
            continue
        bo = np.concatenate([[e.pvto.bo[i]], rows[:, 1]])
        assert np.all(np.diff(bo) <= 1e-9), f"Bo not monotone at node {i}"
    # the compact reconstruction ran and was recorded
    assert any("compact cubic" in c.action for c in res.changes.changes)


# --- low-side extension when the table already carries an Rs = 0 psc node ---

def _locus_with_psc_node(psc_value):
    p, rs, rv, bo, bg, uo, ug = _synthetic_locus()
    return (np.concatenate(([psc_value], p)), np.concatenate(([0.0], rs)),
            np.concatenate(([0.05], rv)), np.concatenate(([1.02], bo)),
            np.concatenate(([200.0], bg)), np.concatenate(([1.5], uo)),
            np.concatenate(([0.012], ug)))


@pytest.mark.parametrize("psc_value", [14.696, 14.696000000000002])
def test_extend_low_skips_table_that_reaches_psc(psc_value):
    # The K-value anchor is the lowest Rs>0 node, but a table whose lowest node
    # already sits at psc needs no extension: prepending a second psc node gave
    # non-ascending saturated pressures and a PCHIP "strictly increasing" error.
    from botkit.extend_low import extend_below_pmin
    assert extend_below_pmin(*_locus_with_psc_node(psc_value), SURFACE) is None


def test_extend_low_grid_stays_below_lowest_node():
    # An Rs = 0 node above psc (dead-oil vertex at 50 psia): K is anchored at the
    # lowest Rs>0 node, yet the new nodes must stop below the lowest input node.
    from botkit.extend_low import extend_below_pmin
    args = list(_locus_with_psc_node(50.0))
    low = extend_below_pmin(*args, SURFACE)
    assert low is not None
    assert np.all(low["p"] < 50.0) and low["p"][0] == pytest.approx(14.696)
    assert low["p_top"] == 50.0 and low["p1"] == 500.0
    assert any("lowest Rs>0 node" in f for f in low["flags"])


def _example_with_psc_node():
    from botkit.model import BlackOilTable, PVTGTable, PVTOTable
    t = read_excel(DATA, surface=SURFACE)
    o, g = t.pvto, t.pvtg
    pvto = PVTOTable(rs=np.r_[0.0, o.rs], p=np.r_[14.696, o.p],
                     bo=np.r_[1.03, o.bo], uo=np.r_[2.0 * o.uo[0], o.uo],
                     usat=[np.empty((0, 3))] + list(o.usat))
    pvtg = PVTGTable(p=np.r_[14.696, g.p], rv=np.r_[0.05, g.rv],
                     bg=np.r_[g.bg[0] * g.p[0] / 14.696, g.bg],
                     ug=np.r_[0.98 * g.ug[0], g.ug],
                     usat=[np.empty((0, 3))] + list(g.usat))
    return BlackOilTable(pvto, pvtg, SURFACE)


def test_pipeline_builds_when_table_reaches_psc():
    table = _example_with_psc_node()
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    e = pipeline.build(table, cfg).extended
    for arr in (e.pvto.p, e.pvtg.p):
        assert np.all(np.diff(arr) > 0)            # strictly ascending, one psc node
        assert arr[0] == pytest.approx(14.696)
    # the stored psc node survives untouched (honour input data)
    assert e.pvtg.rv[0] == pytest.approx(0.05)


def test_enforce_monotonic_cgr_is_opt_in():
    # Off by default: the low-pressure retrograde rise in saturated Rv is real
    # and is kept; flattening it is a simulator-compliance choice the user makes.
    assert Config().enforce_monotonic_cgr is False


def test_gas_branches_reach_dry_gas_per_node():
    # Each node's undersaturated gas branch runs from its own saturated Rv down
    # to Rv = 0 (the dry-gas end of a PVTG table). A single grid spanning the
    # whole locus left low-Rv nodes with one row or none and never reached 0.
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    e = pipeline.build(table, cfg).extended
    n = cfg.n_undersaturated_nodes
    for rv_sat, rows in zip(e.pvtg.rv, e.pvtg.usat):
        if rv_sat <= 0:
            continue
        assert rows.shape[0] == n                 # same resolution on every node
        assert rows[-1, 0] == 0.0                 # ends at dry gas
        assert np.all(np.diff(rows[:, 0]) < 0) and rows[0, 0] < rv_sat
        assert np.all(rows[:, 1] > 0) and np.all(rows[:, 2] > 0)


def _example_cut_at_rv_minimum():
    # Drop the nodes below the example's Rv minimum (800 psia): the trusted locus
    # is then monotone, and only the low-side extension creates a reversal.
    from botkit.model import BlackOilTable, PVTGTable, PVTOTable
    t = read_excel(DATA, surface=SURFACE)
    o, g = t.pvto, t.pvtg
    mo, mg = o.p >= 800.0, g.p >= 800.0
    pvto = PVTOTable(rs=o.rs[mo], p=o.p[mo], bo=o.bo[mo], uo=o.uo[mo],
                     usat=[u for u, k in zip(o.usat, mo) if k])
    pvtg = PVTGTable(p=g.p[mg], rv=g.rv[mg], bg=g.bg[mg], ug=g.ug[mg],
                     usat=[u for u, k in zip(g.usat, mg) if k])
    return BlackOilTable(pvto, pvtg, SURFACE)


def test_enforce_monotonic_cgr_floors_extension_created_reversal():
    table = _example_cut_at_rv_minimum()
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    cfg.enforce_monotonic_cgr = True
    res = pipeline.build(table, cfg)
    rv = res.extended.pvtg.rv
    assert np.all(np.diff(rv) >= -1e-15)          # monotone after the opt-in fix
    assert any("Flattened saturated Rv" in c.action for c in res.changes.changes)
    # default (off) keeps the extension's rising r_s toward psc
    cfg.enforce_monotonic_cgr = False
    rv_off = pipeline.build(table, cfg).extended.pvtg.rv
    assert rv_off[0] > rv_off.min()
