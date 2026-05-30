"""EOS tests: cubic solver, global tune, per-node shifts, pressure match."""

import os

import numpy as np
import pytest

from botkit import SurfaceFluids, read_excel
from botkit.eos import (
    eos_pressure_match_error,
    initial_parameters,
    molar_volume,
    tune_global,
    tune_local_shifts,
    zroots,
)
from botkit.kvalues import phase_properties

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")
SURFACE = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)


def _props_trimmed(cut=1800.0):
    table = read_excel(DATA, surface=SURFACE)
    full = phase_properties(table)
    p = table.pvto.p
    m = p <= cut * 1.0001
    out = {"p": p[m]}
    for k in ("vo", "vg", "xo", "xg", "yo", "yg"):
        out[k] = full[k][m]
    return out


def test_zroots_returns_real_roots():
    roots = zroots(a=1.0e4, b=1.0, p=2000.0, T=680.0)
    assert roots.size in (1, 2)
    assert np.all(np.isfinite(roots))


def test_global_then_local_matches_pressures_in_extension_region():
    props = _props_trimmed()
    params0 = initial_parameters(SURFACE, T=680.0)
    params, so_g, sg_g = tune_global(props, params0, regress_temperature=False)

    so, sg = tune_local_shifts(props, params)
    # the extension anchors on the upper nodes; require an exact match there
    err_top = eos_pressure_match_error(props, params, so, sg, top_n=6)
    assert err_top < 0.01, f"upper-node EOS pressure error {err_top:.3%} too high"
    # molar-volume error is well-conditioned everywhere (no liquid-stiffness blowup)
    from botkit.eos import molar_volume_match_error
    err_v = molar_volume_match_error(props, params, so, sg)
    assert err_v < 0.20, f"molar-volume error {err_v:.3%} unexpectedly large"


def test_regularized_shifts_are_smoother():
    props = _props_trimmed()
    params0 = initial_parameters(SURFACE, T=680.0)
    params, _, _ = tune_global(props, params0)

    so0, sg0 = tune_local_shifts(props, params, smoothness=0.0)
    so1, sg1 = tune_local_shifts(props, params, smoothness=1.0)

    def roughness(s):
        return float(np.sum(np.abs(np.diff(s, 2))))

    # regularization markedly smooths the gas shift trend
    assert roughness(sg1) < 0.5 * roughness(sg0)

    # in-sample molar-volume cost stays small (well under 1% on the oil branch)
    from botkit.eos import molar_volume
    oil_err = max(
        abs(molar_volume(props["p"][i], props["xo"][i], props["xg"][i],
                         params, so1[i], sg1[i]) - props["vo"][i]) / props["vo"][i]
        for i in range(len(props["p"]))
    )
    assert oil_err < 0.01


def test_smoothness_zero_matches_pointwise():
    props = _props_trimmed()
    params0 = initial_parameters(SURFACE, T=680.0)
    params, _, _ = tune_global(props, params0)
    so_a, sg_a = tune_local_shifts(props, params, smoothness=0.0)
    so_b, sg_b = tune_local_shifts(props, params)  # default is 0.0
    np.testing.assert_allclose(so_a, so_b)
    np.testing.assert_allclose(sg_a, sg_b)


def test_trend_extrapolator_anchors_exactly():
    from botkit.eos import trend_extrapolator
    p = np.array([800.0, 1200.0, 1600.0, 1800.0])
    s = np.array([0.01, 0.02, 0.035, 0.043])
    for transform in ("linear", "log", "recip", "sqrt"):
        f, slope = trend_extrapolator(p, s, n_trend=3, transform=transform)
        # collapses to the last point exactly (no offset at the join)
        assert float(f(p[-1])) == pytest.approx(s[-1], abs=1e-12)
        # extrapolates beyond the last node along the trend (non-flat)
        assert float(f(3000.0)) != pytest.approx(s[-1])
        # within range it reproduces interior nodes
        assert float(f(1200.0)) == pytest.approx(0.02, abs=1e-9)


def test_trend_extrapolator_perfectly_linear_input():
    from botkit.eos import trend_extrapolator
    # a series exactly linear in log p must extrapolate exactly under transform=log
    p = np.array([1000.0, 1400.0, 1800.0])
    s = 0.5 * np.log(p) - 3.0
    f, _ = trend_extrapolator(p, s, n_trend=3, transform="log")
    assert float(f(2500.0)) == pytest.approx(0.5 * np.log(2500.0) - 3.0, rel=1e-9)


def test_molar_volume_positive():
    props = _props_trimmed()
    params0 = initial_parameters(SURFACE, T=680.0)
    v = molar_volume(props["p"][0], props["xo"][0], props["xg"][0],
                     params0, 0.0, 0.0)
    assert v > 0
