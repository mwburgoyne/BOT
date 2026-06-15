"""Viscosity tests: per-node Vc regression, reliability gate, continuity."""

import os

import numpy as np

from botkit import SurfaceFluids, read_excel
from botkit.model import Config
from botkit import pipeline
from botkit.kvalues import phase_properties
from botkit.viscosity import regress_node_densities

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")
SURFACE = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)


def test_node_density_reliability_drops_low_pressure():
    table = read_excel(DATA, surface=SURFACE)
    fr = pipeline.fit(table, Config())
    den_co, den_cg, ok = regress_node_densities(fr.props, fr.lbc)
    # the two lowest-pressure nodes are ill-conditioned and dropped
    assert not ok[0] and not ok[1]
    # the rest are reliable, with physical critical densities
    assert np.all(ok[2:])
    assert np.all((den_co[ok] > 5) & (den_co[ok] < 30))
    assert np.all((den_cg[ok] > 5) & (den_cg[ok] < 30))


def test_extension_viscosity_continuous_and_monotone():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    res = pipeline.build(table, cfg)
    e = res.extended
    # reproduces the observed viscosity at the join
    j = int(np.argmin(np.abs(e.pvto.p - 1800.0)))
    measured = float(table.pvto.uo[table.pvto.p == 1800.0][0])
    assert abs(e.pvto.uo[j] - measured) < 5e-3
    # monotone from the lowest measured node upward (toward the critical point
    # oil viscosity falls and gas viscosity rises). Below the lowest node the
    # low-side extension toward psc reverses the oil trend (the dead-oil rise),
    # which is a separate, physical regime checked below.
    pmin0 = float(table.pvto.p.min())
    hi = e.pvto.p >= pmin0 - 1e-6
    assert np.all(np.diff(e.pvto.uo[hi]) < 0)
    assert np.all(np.diff(e.pvtg.ug[hi]) > 0)


def test_low_side_extension_to_psc():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    e = pipeline.build(table, cfg).extended
    # the locus now reaches psc with the convention/anchored endpoints
    assert abs(e.pvto.p.min() - cfg.psc) < 1e-6
    assert e.pvto.rs[0] == 0.0                  # Rs = 0 at psc (stock-tank)
    assert abs(e.pvto.bo[0] - cfg.bo_psc_anchor) < 1e-9   # Bo anchored to 1.0
    assert e.pvtg.bg[0] > e.pvtg.bg[1] > 0      # Bg largest at psc, positive
    assert e.pvto.uo[0] > 0 and e.pvtg.ug[0] > 0          # finite viscosities
    # Rs stays monotone increasing across the joined low + high locus
    assert np.all(np.diff(e.pvto.rs) >= -1e-9)


def test_monotonic_cgr_governs_low_side_rv():
    # Default enforces monotonic saturated Rv: the low-side extension nodes are
    # flattened to the retrograde floor. Toggling the flag off keeps the
    # physically realistic non-monotonic retrograde rise toward psc.
    table = read_excel(DATA, surface=SURFACE)
    cfg_on = Config(); cfg_on.auto_apply_fixes = True   # enforce_monotonic_cgr True
    cfg_off = Config(); cfg_off.auto_apply_fixes = True
    cfg_off.enforce_monotonic_cgr = False
    e_on = pipeline.build(table, cfg_on).extended
    e_off = pipeline.build(table, cfg_off).extended

    pmin0 = float(table.pvtg.p.min())
    low_on = e_on.pvtg.p < pmin0
    low_off = e_off.pvtg.p < pmin0
    assert low_on.any() and low_off.any()
    # default: low-side Rv is flat (monotone, at the floor)
    assert np.allclose(e_on.pvtg.rv[low_on], e_on.pvtg.rv[low_on][0])
    assert np.all(np.diff(e_on.pvtg.rv) >= -1e-12)
    # flag off: low-side Rv rises toward psc (non-monotone over the full table)
    assert e_off.pvtg.rv[0] > e_off.pvtg.rv[low_off][-1]
    assert not np.all(np.diff(e_off.pvtg.rv) >= -1e-12)


def test_whitson_mode_skips_psc_extension():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.whitson_mode = True
    e = pipeline.build(table, cfg).extended
    # no low-side extension: the locus still starts at the lowest measured node
    assert abs(e.pvto.p.min() - float(table.pvto.p.min())) < 1e-6
