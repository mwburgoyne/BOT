"""End-to-end pipeline test on the example fluid."""

import os
import tempfile

import numpy as np

from botkit import SurfaceFluids, read_eclipse, read_excel, write_eclipse
from botkit import pipeline
from botkit.model import Config

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")
SURFACE = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)


def test_qc_stage_stops_for_review_by_default():
    table = read_excel(DATA, surface=SURFACE)
    res = pipeline.run(table, Config())  # auto_apply_fixes defaults False
    assert res.extended is None          # stopped after QC
    assert res.suggestions["saturated_cut"] == 1800.0
    assert any(a.kind == "pressure_misalignment" for a in res.diagnostics.anomalies)


def test_full_build_extends_and_fills():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    res = pipeline.build(table, cfg)
    e = res.extended

    # discarded the untrusted tail (cut at 1800) and extended above it
    assert res.info["cut"] == 1800.0
    assert e.pvto.p.max() > 1800.0
    # saturated Rs strictly increasing; EOS upper-node match is tight
    assert np.all(np.diff(e.pvto.rs) > 0)
    assert res.info["eos_pressure_error"] < 0.01
    # undersaturated branches were filled
    assert any(b.shape[0] > 0 for b in e.pvto.usat)

    # extended deck round-trips through the Eclipse reader/writer
    with tempfile.NamedTemporaryFile("w", suffix=".inc", delete=False) as fh:
        fh.write(write_eclipse(e))
        tmp = fh.name
    try:
        back = read_eclipse(tmp)
    finally:
        os.unlink(tmp)
    assert back.pvto.n == e.pvto.n
    np.testing.assert_allclose(back.pvto.rs, e.pvto.rs, rtol=1e-4)


def test_undersaturated_oil_branches_physical_by_construction():
    # anchored fill -> every branch monotone with positive compressibility,
    # without any clipping or regeneration
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    res = pipeline.build(table, cfg)
    e = res.extended
    for i, rows in enumerate(e.pvto.usat):
        if rows.shape[0] < 2:
            continue
        bo = np.concatenate([[e.pvto.bo[i]], rows[:, 1]])
        uo = np.concatenate([[e.pvto.uo[i]], rows[:, 2]])
        assert np.all(np.diff(bo) <= 1e-12), f"Bo not monotone at node {i}"
        assert np.all(np.diff(uo) >= -1e-12), f"uo not monotone at node {i}"
    # nothing was clipped or regenerated, and no wrong-direction flag fired
    assert not [a for a in res.diagnostics.anomalies
                if a.kind == "negative_undersaturated_compressibility"]


def test_low_pressure_branch_has_physical_compressibility():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    res = pipeline.build(table, cfg)
    e = res.extended
    i = int(np.argmin(np.abs(e.pvto.p - 100.0)))
    rows = np.vstack([[e.pvto.p[i], e.pvto.bo[i]], e.pvto.usat[i][:, :2]])
    bo, p = rows[:, 1], rows[:, 0]
    c_o = np.mean(-np.diff(np.log(bo)) / np.diff(p))
    assert 1e-7 < c_o < 1e-5             # physical, not over-steep
    assert np.all(np.diff(bo) <= 1e-12)  # monotone


def test_change_log_records_fixes_with_reasons():
    from botkit import changes_to_markdown, changes_to_text
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    res = pipeline.build(table, cfg)

    actions = " ".join(c.action for c in res.changes.changes)
    # the substantive fixes are recorded
    assert "Discarded" in actions and "1800" in actions      # trim
    assert "convergence pressure" in actions.lower()         # extension
    assert "Rv" in actions                                   # cgr truncation
    # every change carries a non-empty reason
    assert all(c.reason for c in res.changes.changes)
    # renderers work and round-trip through the dict form
    assert "change summary" in changes_to_markdown(res.changes).lower()
    assert "Why:" in changes_to_text(res.changes)
    assert len(res.changes.to_dict()["changes"]) == len(res.changes)


def test_change_log_empty_when_clean(tmp_path):
    # a clean, aligned, monotone table needs no corrective changes
    import numpy as np
    from botkit.model import BlackOilTable, PVTGTable, PVTOTable
    p = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    pvto = PVTOTable(rs=[0.3, 0.5, 0.7, 0.9, 1.1], p=p,
                     bo=[1.2, 1.3, 1.4, 1.5, 1.6], uo=[0.6, 0.47, 0.38, 0.33, 0.29])
    pvtg = PVTGTable(p=p, rv=[0.006, 0.004, 0.003, 0.0024, 0.002],
                     bg=[5.0, 2.5, 1.7, 1.3, 1.05], ug=[0.012, 0.013, 0.015, 0.017, 0.019])
    table = BlackOilTable(pvto=pvto, pvtg=pvtg, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 4000.0
    cfg.enforce_monotonic_cgr = False  # monotone Rv already
    res = pipeline.build(table, cfg)
    # no trim (aligned), no cgr fix; at most an extension record
    assert not any("Discarded" in c.action for c in res.changes.changes)


def test_manual_point_replacement():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    cfg.manual_replace_pressures = (600.0,)
    res = pipeline.build(table, cfg)
    # the 600 psia node's properties were replaced by interpolation
    assert res.fit.repairs
    keys = {k for (_p, k, _o, _n) in res.fit.repairs}
    assert {"rs", "bo", "rv", "bg"} <= keys
    assert any("Replaced saturated" in c.action for c in res.changes.changes)


def test_output_pressure_grid_resamples():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 4000.0
    grid = (100.0, 150.0, 200.0, 300.0, 600.0, 1200.0, 1800.0, 2500.0)
    cfg.output_pressures = grid
    res = pipeline.build(table, cfg)
    e = res.extended
    np.testing.assert_allclose(e.pvto.p, sorted(grid))
    assert np.all(np.diff(e.pvto.rs) > 0)          # monotone after resample
    assert any(b.shape[0] > 0 for b in e.pvto.usat)  # branches filled at new nodes


def test_output_pressure_out_of_range_flagged():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 4000.0
    cfg.output_pressures = (200.0, 600.0, 99000.0)  # last is above the model range
    res = pipeline.build(table, cfg)
    assert any(a.kind == "output_pressure_out_of_range"
               for a in res.diagnostics.anomalies)
    assert res.extended.pvto.p.max() < 99000.0


def test_eos_gate_independent_of_shift_smoothing():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    cfg.shift_smoothness = 1.0  # smoothing must not make the gate trip
    res = pipeline.build(table, cfg)
    assert res.info["eos_trusted"] is True
    assert res.info["eos_pressure_error"] < 1e-3
    assert not any(a.kind == "eos_fallback" for a in res.diagnostics.anomalies)


def test_cgr_floor_applied_when_enforced():
    table = read_excel(DATA, surface=SURFACE)
    cfg = Config()
    cfg.auto_apply_fixes = True
    cfg.convergence_pressure_Pk = 6000.0
    cfg.enforce_monotonic_cgr = True
    res = pipeline.build(table, cfg)
    rv = res.extended.pvtg.rv
    # below the retrograde minimum, Rv is floored (non-decreasing low-P side)
    i_min = int(np.argmin(rv))
    assert np.all(rv[:i_min] == rv[i_min]) or i_min == 0
