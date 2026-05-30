"""I/O tests: Excel read and Eclipse write -> read round-trip."""

import os

import numpy as np
import pytest

from botkit import (
    BlackOilTable,
    PVTGTable,
    PVTOTable,
    SurfaceFluids,
    read_eclipse,
    read_excel,
    write_eclipse,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "PVTO&PVTG_example.xlsx")


def test_read_excel_shapes():
    table = read_excel(DATA)
    assert table.pvto.n == 32
    assert table.pvtg.n == 32
    # saturated arrays are aligned and finite
    assert np.all(np.isfinite(table.pvto.bo))
    assert np.all(np.isfinite(table.pvtg.bg))
    # first oil node matches the workbook
    assert table.pvto.rs[0] == pytest.approx(0.07947, rel=1e-6)
    assert table.pvto.p[0] == pytest.approx(100.0)


def test_eclipse_roundtrip_saturated():
    surface = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)
    table = read_excel(DATA, surface=surface)

    deck = write_eclipse(table, header="round-trip test")
    assert "PVTO" in deck and "PVTG" in deck and "DENSITY" in deck

    # parse it back
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".inc", delete=False) as fh:
        fh.write(deck)
        tmp = fh.name
    try:
        back = read_eclipse(tmp)
    finally:
        os.unlink(tmp)

    np.testing.assert_allclose(back.pvto.rs, table.pvto.rs, rtol=1e-5)
    np.testing.assert_allclose(back.pvto.p, table.pvto.p, rtol=1e-5)
    np.testing.assert_allclose(back.pvto.bo, table.pvto.bo, rtol=1e-5)
    np.testing.assert_allclose(back.pvtg.rv, table.pvtg.rv, rtol=1e-5)
    np.testing.assert_allclose(back.pvtg.bg, table.pvtg.bg, rtol=1e-5)
    # surface densities recovered from the DENSITY keyword
    assert back.surface.st_oil_density == pytest.approx(49.87)
    assert back.surface.st_gas_density == pytest.approx(0.0689)


def test_eclipse_roundtrip_with_undersaturated():
    # one saturated oil node with two undersaturated rows
    pvto = PVTOTable(
        rs=[0.5, 1.0],
        p=[1000.0, 2000.0],
        bo=[1.30, 1.50],
        uo=[0.47, 0.33],
        usat=[
            np.array([[1500.0, 1.29, 0.49], [2500.0, 1.28, 0.51]]),
            np.empty((0, 3)),
        ],
    )
    pvtg = PVTGTable(
        p=[1000.0, 2000.0],
        rv=[0.0016, 0.0069],
        bg=[2.54, 1.20],
        ug=[0.0134, 0.0179],
        usat=[
            np.array([[0.0010, 2.60, 0.0130]]),
            np.empty((0, 3)),
        ],
    )
    table = BlackOilTable(pvto=pvto, pvtg=pvtg,
                          surface=SurfaceFluids(49.87, 0.0689))

    import tempfile
    deck = write_eclipse(table)
    with tempfile.NamedTemporaryFile("w", suffix=".inc", delete=False) as fh:
        fh.write(deck)
        tmp = fh.name
    try:
        back = read_eclipse(tmp)
    finally:
        os.unlink(tmp)

    # undersaturated rows preserved on node 0, none on node 1
    np.testing.assert_allclose(back.pvto.usat[0], pvto.usat[0], rtol=1e-5)
    assert back.pvto.usat[1].shape == (0, 3)
    np.testing.assert_allclose(back.pvtg.usat[0], pvtg.usat[0], rtol=1e-5)


def test_surface_fluid_constants():
    # derived constants reproduce the notebook's surface-fluid setup
    s = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)
    assert s.oil_mw == pytest.approx(240 - 2.22 * (141.5 / (49.87 / 62.428) - 131.5))
    assert s.Lg == pytest.approx(1 / 0.3795)
    assert s.mult == pytest.approx(1000 / 5.6146)
    assert s.Co == pytest.approx(s.mult * s.Lo / s.Lg)
