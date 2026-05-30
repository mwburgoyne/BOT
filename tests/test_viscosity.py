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
    # monotone across the whole extended locus
    assert np.all(np.diff(e.pvto.uo) < 0)
    assert np.all(np.diff(e.pvtg.ug) > 0)
