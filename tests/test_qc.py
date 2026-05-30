"""QC detector tests: fires on seeded defects, quiet on clean data."""

import os

import numpy as np
import pytest

from botkit import SurfaceFluids, read_excel
from botkit.model import BlackOilTable, Diagnostics, PVTGTable, PVTOTable, Severity
from botkit.qc import (
    detect_monotonicity,
    detect_saturated_compressibility,
    detect_undersaturated_compressibility,
    run_qc,
    shared_saturated_pressure,
)
from botkit.report import diagnostics_to_json, diagnostics_to_markdown

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")
SURFACE = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)


def test_detects_1800_misalignment_on_example():
    table = read_excel(DATA, surface=SURFACE)
    cut = shared_saturated_pressure(table)
    assert cut == pytest.approx(1800.0)

    diag, suggestions = run_qc(table)
    kinds = {a.kind for a in diag.anomalies}
    assert "pressure_misalignment" in kinds
    assert suggestions["saturated_cut"] == pytest.approx(1800.0)

    # report renders without error and is non-trivial
    md = diagnostics_to_markdown(diag, suggestions)
    assert "QC report" in md and "1800" in md
    js = diagnostics_to_json(diag, suggestions)
    assert "pressure_misalignment" in js


def _clean_table():
    """A small monotone, internally-consistent saturated table (no defects)."""
    p = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    rs = np.array([0.30, 0.50, 0.70, 0.90, 1.10])
    bo = np.array([1.20, 1.30, 1.40, 1.50, 1.60])
    bg = np.array([5.0, 2.5, 1.7, 1.3, 1.05])
    rv = np.array([0.0030, 0.0016, 0.0020, 0.0030, 0.0042])
    uo = np.array([0.60, 0.47, 0.38, 0.33, 0.29])
    ug = np.array([0.012, 0.013, 0.015, 0.017, 0.019])
    pvto = PVTOTable(rs=rs, p=p, bo=bo, uo=uo)
    pvtg = PVTGTable(p=p, rv=rv, bg=bg, ug=ug)
    return BlackOilTable(pvto=pvto, pvtg=pvtg, surface=SURFACE)


def test_quiet_on_monotone_clean_data():
    table = _clean_table()
    diag = Diagnostics()
    detect_monotonicity(table, diag)
    # no Rs/Bo/Bg monotonicity complaints on clean data
    assert not [a for a in diag.anomalies if a.kind == "monotonicity"]


def test_fires_on_injected_non_monotone_rs():
    table = _clean_table()
    table.pvto.rs[3] = 0.45  # Rs dips, breaking monotonicity
    diag = Diagnostics()
    detect_monotonicity(table, diag)
    assert any(a.kind == "monotonicity" for a in diag.anomalies)


def test_fires_on_injected_bo_step():
    # a Bo step change produces a negative/!discontinuous oil compressibility
    table = _clean_table()
    table.pvto.bo[2] = 1.80  # large step up at one node
    diag = Diagnostics()
    detect_saturated_compressibility(table, diag)
    assert any(a.kind in ("negative_saturated_compressibility",
                          "compressibility_discontinuity")
               for a in diag.anomalies)


def test_negative_total_compressibility_sweep_on_example_locus():
    # on the trimmed (<=1800) shared locus, no negative saturated compressibility
    table = read_excel(DATA, surface=SURFACE)
    diag = Diagnostics()
    detect_saturated_compressibility(table, diag, cut=1800.0)
    assert not [a for a in diag.anomalies
                if a.kind == "negative_saturated_compressibility"]


def test_gas_above_oil_compressibility_ordering_flags_violation():
    # craft a node where gas total compressibility dips below oil's
    table = _clean_table()
    table.pvtg.bg[2] = 1.69  # flatten Bg slope -> small gas compressibility
    diag = Diagnostics()
    detect_saturated_compressibility(table, diag)
    # ordering check available; may or may not trip on this synthetic set,
    # so assert the detector runs and the kind is reachable when violated.
    table.pvtg.bg = np.array([5.0, 2.5, 2.49, 1.3, 1.05])  # near-flat mid slope
    diag2 = Diagnostics()
    detect_saturated_compressibility(table, diag2)
    assert any(a.kind == "compressibility_ordering" for a in diag2.anomalies)


def test_cgr_reversal_flag_controls_framing():
    table = read_excel(DATA, surface=SURFACE)
    # enforce -> WARN + a truncation floor suggested
    diag_on, sug_on = run_qc(table, enforce_monotonic_cgr=True)
    cgr_on = [a for a in diag_on.anomalies if a.kind == "cgr_reversal"]
    assert cgr_on and cgr_on[0].severity is Severity.WARN
    assert "cgr_floor" in sug_on

    # leave-as-is -> INFO + no truncation floor suggested
    diag_off, sug_off = run_qc(table, enforce_monotonic_cgr=False)
    cgr_off = [a for a in diag_off.anomalies if a.kind == "cgr_reversal"]
    assert cgr_off and cgr_off[0].severity is Severity.INFO
    assert "cgr_floor" not in sug_off


def test_undersaturated_compressibility_positivity():
    table = _clean_table()
    # attach a bad undersaturated oil branch where Bo rises with pressure
    table.pvto.usat[0] = np.array([[3000.0, 1.21, 0.46], [3500.0, 1.23, 0.45]])
    diag = Diagnostics()
    detect_undersaturated_compressibility(table, diag)
    assert any(a.kind == "negative_undersaturated_compressibility"
               for a in diag.anomalies)
