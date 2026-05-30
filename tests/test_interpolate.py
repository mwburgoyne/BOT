"""Interpolation-layer tests: node exactness, composition basis, monotonicity."""

import os

import numpy as np
import pytest

from botkit import SurfaceFluids, read_excel
from botkit.interpolate import SaturatedInterpolator, densify_saturated
from botkit.qc import _shared_locus

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")
SURFACE = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)


def _trusted_interp():
    table = read_excel(DATA, surface=SURFACE)
    loc = _shared_locus(table)
    m = loc["p"] <= 1800.0001
    return SaturatedInterpolator(loc["p"][m], loc["rs"][m], loc["rv"][m],
                                 loc["bo"][m], loc["bg"][m], loc["uo"][m],
                                 loc["ug"][m], SURFACE), {k: v[m] for k, v in loc.items()}


def test_reproduces_input_nodes_exactly():
    interp, loc = _trusted_interp()
    out = interp.evaluate(loc["p"])
    for key in ("rs", "rv", "bo", "bg", "uo", "ug"):
        np.testing.assert_allclose(out[key], loc[key], rtol=1e-9, atol=1e-12,
                                   err_msg=f"{key} not reproduced at nodes")


def test_compositions_bounded():
    interp, loc = _trusted_interp()
    # evaluate on a fine grid; gas mole fractions stay in (0, 1)
    grid = np.linspace(loc["p"].min(), loc["p"].max(), 200)
    out = interp.evaluate(grid)
    assert np.all((out["xg"] > 0) & (out["xg"] < 1))
    assert np.all((out["yg"] > 0) & (out["yg"] < 1))


def test_rs_monotone_between_nodes():
    interp, loc = _trusted_interp()
    grid = np.linspace(loc["p"].min(), loc["p"].max(), 300)
    rs = interp.evaluate(grid)["rs"]
    # Rs is monotone increasing in pressure on the trusted locus
    assert np.all(np.diff(rs) > -1e-9)


def test_densify_preserves_nodes():
    interp, loc = _trusted_interp()
    dense = densify_saturated(interp, loc["p"], per_gap=3)
    # original node pressures are a subset of the densified grid
    for p in loc["p"]:
        assert np.any(np.isclose(dense["p"], p))
    # densified Rs is monotone
    assert np.all(np.diff(dense["rs"]) > -1e-9)
