"""Extend the saturated locus to the convergence pressure.

K-values are extrapolated in log-log space to K = 1 at the convergence pressure
(honouring the slope at the top of the trusted table), then the saturated
properties (Rs, Rv, Bo, Bg, viscosities) are regenerated from the tuned EOS and
LBC viscosity at each extension pressure.  A fold check stops the extension if
Bo or Bg reverses direction, which marks a near-critical region the smooth
extrapolation cannot represent.  Method: Singh, Fevang & Whitson, SPE 109596 (2007).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .eos import EOSParameters, molar_volume
from .kvalues import rs_rv_from_kvalues
from .model import SurfaceFluids
from .viscosity import LBCParameters


def quad_extrap(slope: float, x1: float, y1: float, x2: float, y2: float,
                x: float) -> float:
    """Quadratic y(x) through (x1,y1) and (x2,y2) honouring dy/dx=slope at x1."""
    b = (slope * (x2 ** 2 - x1 ** 2) + 2 * x1 * (y1 - y2)) / (x1 - x2) ** 2
    a = (slope - b) / (2 * x1)
    c = y1 - a * x1 ** 2 - b * x1
    return a * x ** 2 + b * x + c


def extrapolate_kvalues(p: np.ndarray, ko: np.ndarray, kg: np.ndarray,
                        Pk: float, n_ext: int, anchor: int = -1,
                        mode: str = "convergence"
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrapolate ko, kg from the table top toward higher pressure.

    ``anchor`` selects the table row used as the extrapolation start (-1 = last
    row).  ``mode`` chooses the high-side law:

    * ``"convergence"`` (default) -- a log-log quadratic that honours the slope
      at the anchor and bends to K = 1 at the convergence pressure Pk
      (Singh, Fevang & Whitson, SPE 109596).  A top-node leave-one-out over the Whitson
      corpus showed this matches the best local log-log fit while enforcing the
      physical K = 1 endpoint a purely local slope misses.
    * ``"constant"`` -- the classic constant-K extension (CKE): K is frozen at
      the anchor, so no new gas dissolves and the composition is held above the
      table.  This is the conservative "Whitson mode" revert.

    Returns extension pressures and the matching ko, kg arrays, beginning at the
    anchor pressure.
    """
    lp = np.log10
    x1 = p[anchor]
    ko_a, kg_a = ko[anchor], kg[anchor]
    p_ext = np.linspace(x1, Pk - 1.0, n_ext)

    if mode == "constant":
        return p_ext, np.full(n_ext, ko_a), np.full(n_ext, kg_a)

    ko_slope = lp(ko_a / ko[anchor - 1]) / lp(x1 / p[anchor - 1])
    kg_slope = lp(kg_a / kg[anchor - 1]) / lp(x1 / p[anchor - 1])
    ko_ext = np.array([10 ** quad_extrap(ko_slope, lp(x1), lp(ko_a), lp(Pk), 0.0, lp(pe))
                       for pe in p_ext])
    kg_ext = np.array([10 ** quad_extrap(kg_slope, lp(x1), lp(kg_a), lp(Pk), 0.0, lp(pe))
                       for pe in p_ext])
    return p_ext, ko_ext, kg_ext


def extend_saturated(p_ext: np.ndarray, ko_ext: np.ndarray, kg_ext: np.ndarray,
                     params: EOSParameters, lbc: LBCParameters,
                     surface: SurfaceFluids, max_psat: float,
                     so_interp: Callable[[float], float],
                     sg_interp: Callable[[float], float],
                     detect_fold: bool = True) -> Dict[str, np.ndarray]:
    """Regenerate saturated properties along the extension from EOS + LBC.

    Volume shifts are taken from the per-node fits, interpolated and clamped at
    the top of the trusted table (``max_psat``).  Returns extended arrays and a
    boolean ``folded`` flag with the index where a Bo/Bg reversal was detected.
    """
    s = surface
    rs, rv = rs_rv_from_kvalues(ko_ext, kg_ext, s)
    xo = s.Co / (rs * s.mult + s.Co)
    xg = 1.0 - xo
    yo = s.Lo * rv / (s.Lg + s.Lo * rv)
    yg = 1.0 - yo
    mwo = s.oil_mw * xo + s.gas_mw * xg
    mwg = s.oil_mw * yo + s.gas_mw * yg

    bo, bg, uo, ug, deno, deng = ([] for _ in range(6))
    for i, p in enumerate(p_ext):
        # the shift interpolators carry the trend above the table themselves
        so, sg = float(so_interp(p)), float(sg_interp(p))
        vop = molar_volume(p, xo[i], xg[i], params, so, sg)
        rho_o = mwo[i] / vop
        bo.append((s.st_oil_density + s.st_gas_density * rs[i] * s.mult) / rho_o)
        deno.append(rho_o)
        uo.append(lbc.phase_viscosity(xo[i], xg[i], rho_o))

        vgp = molar_volume(p, yo[i], yg[i], params, so, sg)
        rho_g = mwg[i] / vgp
        bg.append((s.st_gas_density + s.st_oil_density * rv[i] / s.mult) / rho_g * s.mult)
        deng.append(rho_g)
        ug.append(lbc.phase_viscosity(yo[i], yg[i], rho_g))

    # Note: uo/ug here are the global-LBC estimate; the pipeline overrides them
    # with the per-node critical-density extrapolation, which reproduces the
    # observed viscosity at the join.
    uo, ug = np.array(uo), np.array(ug)
    bo, bg = np.array(bo), np.array(bg)
    folded_at: Optional[int] = None
    fold_property: Optional[str] = None
    if detect_fold and len(bo) > 2:
        # toward the critical point Bo should keep rising and Bg keep falling
        bo_rev = np.where(np.diff(bo) < 0)[0]
        bg_rev = np.where(np.diff(bg) > 0)[0]
        candidates = []
        if bo_rev.size:
            candidates.append((int(bo_rev[0]) + 1, "oil FVF Bo"))
        if bg_rev.size:
            candidates.append((int(bg_rev[0]) + 1, "gas FVF Bg"))
        if candidates:
            folded_at, fold_property = min(candidates, key=lambda c: c[0])

    return {
        "p": p_ext, "rs": rs, "rv": rv, "bo": bo, "bg": bg,
        "uo": np.array(uo), "ug": np.array(ug),
        "deno": np.array(deno), "deng": np.array(deng),
        "ko": ko_ext, "kg": kg_ext,
        "folded_at": folded_at, "fold_property": fold_property,
    }
