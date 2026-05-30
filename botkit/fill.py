"""Fill undersaturated oil and gas branches for each saturated node.

For oil, pressure rises above the bubble point at constant Rs; for gas, the
vapourized ratio Rv falls below its saturated value at constant pressure.  Each
undersaturated property is regenerated from the tuned EOS and LBC viscosity.
Method: Singh & Whitson, SPE 109596 (2007).
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

from .eos import EOSParameters, molar_volume
from .model import SurfaceFluids
from .viscosity import LBCParameters


def undersaturated_pressures(min_psat: float, Pk: float, n: int) -> np.ndarray:
    """Pressure nodes above the bubble point on a square-root progression."""
    incr = ((Pk - 1) ** 0.5 - min_psat ** 0.5) / n
    return np.array([(min_psat ** 0.5 + i * incr) ** 2 for i in range(1, n + 1)])


def undersaturated_rv_grid(min_rv: float, max_rv: float, n: int) -> np.ndarray:
    """Reducing Rv values below the saturated value (square-root progression)."""
    incr = (max_rv ** 0.5 - min_rv ** 0.5) / n
    return np.array([(max_rv ** 0.5 - i * incr) ** 2 for i in range(n + 1)])


def fill_oil_branch(psat: float, rs: float, bo_sat: float, uo_sat: float,
                    params: EOSParameters, lbc: LBCParameters,
                    surface: SurfaceFluids, p_grid: np.ndarray,
                    so_node: float, sg_node: float) -> np.ndarray:
    """Undersaturated oil rows [P, Bo, uo] above ``psat`` at constant Rs.

    The Peneloux volume shift is held at the node's bubble-point value
    (``so_node``, ``sg_node``) because the composition is fixed along the
    branch; the EOS pressure response then supplies the physical
    (positive) undersaturated compressibility.
    """
    s = surface
    xo = s.Co / (rs * s.mult + s.Co)
    xg = 1.0 - xo
    mwo = s.oil_mw * xo + s.gas_mw * xg

    # Anchor Bo and uo to the measured saturated values at the bubble point and
    # let the EOS density and LBC viscosity supply only the pressure *ratio*.
    # This makes the branch continuous with the saturated node and monotone by
    # construction (rho rises -> Bo falls, uo rises), independent of any small
    # EOS/LBC offset from the measured node - so no post-hoc clipping is needed.
    rho_o_sat = mwo / molar_volume(psat, xo, xg, params, so_node, sg_node)
    uo_lbc_sat = lbc.phase_viscosity(xo, xg, rho_o_sat)
    rows = [[psat, bo_sat, uo_sat]]
    for p in p_grid:
        if p <= psat:
            continue
        rho_o = mwo / molar_volume(p, xo, xg, params, so_node, sg_node)
        bo = bo_sat * rho_o_sat / rho_o
        uo = uo_sat * lbc.phase_viscosity(xo, xg, rho_o) / uo_lbc_sat
        rows.append([p, bo, uo])
    return np.array(rows)


def fill_gas_branch(psat: float, rv_sat: float, bg_sat: float, ug_sat: float,
                    params: EOSParameters, lbc: LBCParameters,
                    surface: SurfaceFluids, rv_grid: np.ndarray,
                    so_node: float, sg_node: float) -> np.ndarray:
    """Undersaturated gas rows [Rv, Bg, ug] below ``rv_sat`` at constant pressure.

    The volume shift is held at the node's value (constant pressure along the
    branch), consistent with the undersaturated oil treatment.
    """
    s = surface
    so, sg = so_node, sg_node
    rows = [[rv_sat, bg_sat, ug_sat]]
    for rv in rv_grid:
        if rv >= rv_sat:
            continue
        yo = s.Lo * rv / (s.Lg + s.Lo * rv)
        yg = 1.0 - yo
        mwg = s.oil_mw * yo + s.gas_mw * yg
        rho_g = mwg / molar_volume(psat, yo, yg, params, so, sg)
        bg = (s.st_gas_density + s.st_oil_density * rv / s.mult) / rho_g * s.mult
        ug = lbc.phase_viscosity(yo, yg, rho_g)
        rows.append([rv, bg, ug])
    return np.array(rows)
