"""K-value tests: App. A round-trip and App. B convergence pressure."""

import os

import numpy as np
import pytest

from botkit import SurfaceFluids, read_excel
from botkit.kvalues import (
    convergence_pressure,
    kvalues,
    phase_properties,
    rs_rv_from_kvalues,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")
SURFACE = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)


def test_kvalue_roundtrip_on_example():
    table = read_excel(DATA, surface=SURFACE)
    rs, rv = table.pvto.rs, table.pvtg.rv
    kv = kvalues(rs, rv, SURFACE)
    rs_back, rv_back = rs_rv_from_kvalues(kv["ko"], kv["kg"], SURFACE)
    np.testing.assert_allclose(rs_back, rs, rtol=1e-9)
    np.testing.assert_allclose(rv_back, rv, rtol=1e-9)


def test_mole_fractions_sum_to_one():
    kv = kvalues(np.array([0.5, 1.0]), np.array([0.002, 0.006]), SURFACE)
    np.testing.assert_allclose(kv["xo"] + kv["xg"], 1.0)
    np.testing.assert_allclose(kv["yo"] + kv["yg"], 1.0)


def test_phase_properties_positive():
    table = read_excel(DATA, surface=SURFACE)
    props = phase_properties(table)
    for key in ("deno", "deng", "vo", "vg", "mwo", "mwg"):
        assert np.all(props[key] > 0), key


def test_convergence_pressure_recovers_synthetic_value():
    # Build K-values that are exactly log-linear in log p and cross 1 at Pk.
    pk_true = 6000.0
    p = np.array([2000.0, 3000.0, 4000.0, 5000.0])
    # ko approaches 1 from above, kg from below, both = 1 at pk_true
    ko = (pk_true / p) ** 0.30   # > 1 for p < pk
    kg = (p / pk_true) ** 0.20   # < 1 for p < pk
    pk = convergence_pressure(p, ko, kg, n_nodes=4)
    assert pk == pytest.approx(pk_true, rel=1e-6)


def test_convergence_pressure_above_table():
    table = read_excel(DATA, surface=SURFACE)
    # use only the trustworthy shared saturated locus (<= 1800 psia)
    mask = table.pvto.p <= 1800.0
    kv = kvalues(table.pvto.rs[mask], table.pvtg.rv[mask], SURFACE)
    pk = convergence_pressure(table.pvto.p[mask], kv["ko"], kv["kg"], n_nodes=4)
    assert pk > table.pvto.p[mask].max()  # convergence is above the data
