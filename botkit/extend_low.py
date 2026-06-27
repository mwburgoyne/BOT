"""Extend the saturated locus *below* the lowest measured pressure, down to the
standard-condition pressure psc.

Simulators drive the bottom-hole flowing pressure toward psc (~14.7 psia) in
unconstrained history matching, so a Black-Oil Table must return sensible
Rs, Rv, Bo, Bg and viscosities there.  Many tables stop well short.  This module
continues the table down to psc using the K-value machinery the companion
``bopvt-lookup`` study validated on a seven-fluid Whitson-school corpus
(leave-one-out):

* **K-values** are extrapolated below the lowest Rs>0 node p1 with origin poles
  (finite at psc, no clamp): the gas pseudocomponent K_g = K_g(p1)*(p1/p)^2
  (Curtis's K_g.p^2, exponent 2 deployment-best) and the oil pseudocomponent
  K_o = K_o(p1)*(p1/p)^0.5 (n_o = 0.5).  Rs, Rv are then recovered from the pair
  by the binary VLE bijection (the two-component compositions are fixed by the
  K-values alone -- Gibbs F = C - 2 = 0), i.e. :func:`rs_rv_from_kvalues`.
  Rs -> 0 at psc by the stock-tank convention; r_s stays finite.
* **Bo** is anchored to 1.0 at psc (small thermal expansion ignored) and filled
  by 1/Bo interpolation.
* **Bg** is anchored at psc to a Z = 1 ideal-gas value when a reservoir
  temperature is known; otherwise the gas Z is effectively frozen and Bg(psc) is
  taken by the isothermal pressure ratio Bg(p1)*(p1/psc) (T cancels).  Filled by
  1/Bg interpolation by default, or (``bg_method="compressibility"``) by
  integrating the gas compressibility down from the lowest node,
  Bg = Bg(p1)*exp(-INT c_g dp), with c_g bounded-interpolated in 1/p between the
  exact ideal value c_g(psc) = 1/psc and the node -- the deployable low-pressure
  gas method of the companion study, a strict improvement on the naive 1/p rule
  that reduces to it in the ideal limit.
* **viscosities** continue the mobilities 1/(Bo*uo) (log p) and 1/(Bg*ug)
  (plain p) -- the established interpolation coordinates -- divided by the
  anchored B.  Flagged as extrapolated; LBC is *not* used down here because its
  low-pressure critical-density fit is the very thing the build drops as
  ill-conditioned.

Reference: Singh, Fevang & Whitson, SPE 109596 (2007), App. A; companion ``bopvt-lookup``
findings of 6-7 Jun 2026 (bottom K_g origin pole; below-p_min r_s rule).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.interpolate import PchipInterpolator

from .kvalues import kvalues, rs_rv_from_kvalues
from .model import FT3_PER_BBL, GAS_MOLAR_VOLUME_MSCF, R_FIELD, SurfaceFluids
from .undersat_extend import extend_bg_compressibility


def ideal_bg(temperature_R: float, p: float) -> float:
    """Ideal-gas (Z = 1) gas FVF in rb/Mscf at pressure ``p`` (psia).

    Bg = Z R T / (p * ft3_per_bbl * Mscf_per_lbmol); consistent with the unit
    constants in :mod:`botkit.model`.
    """
    return R_FIELD * temperature_R / (p * FT3_PER_BBL * GAS_MOLAR_VOLUME_MSCF)


def extend_below_pmin(p, rs, rv, bo, bg, uo, ug, surface: SurfaceFluids, *,
                      psc: float = 14.696, n_nodes: int = 8,
                      kg_exp: float = 2.0, ko_exp: float = 0.5,
                      bo_psc: float = 1.0,
                      reservoir_temperature: Optional[float] = None,
                      bg_method: str = "interp"
                      ) -> Optional[Dict[str, object]]:
    """Build the saturated locus from just below p1 down to and including psc.

    ``p`` and the property arrays are the assembled saturated locus (ascending in
    pressure assumed after the internal sort).  Returns a dict of extension
    arrays (ascending, ending just below the lowest input node) plus a list of
    flag strings, or ``None`` when the table already reaches psc.
    """
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    p = p[order]
    rs, rv = np.asarray(rs, float)[order], np.asarray(rv, float)[order]
    bo, bg = np.asarray(bo, float)[order], np.asarray(bg, float)[order]
    uo, ug = np.asarray(uo, float)[order], np.asarray(ug, float)[order]

    # anchor at the lowest node carrying dissolved gas (the last positive-K_o node)
    pos = np.where(rs > 0.0)[0]
    a = int(pos[0]) if pos.size else 0
    p1 = float(p[a])
    if p1 <= psc * 1.0001:
        return None

    kv = kvalues(rs, rv, surface)
    ko1, kg1 = float(kv["ko"][a]), float(kv["kg"][a])
    flags = []

    # low-side grid: psc .. p1, log-spaced, including psc but excluding p1
    grid = np.exp(np.linspace(np.log(psc), np.log(p1), n_nodes + 1))[:-1]

    # K-value origin-pole extrapolation, then the binary VLE bijection for Rs, Rv
    ratio = p1 / grid
    kg_e = kg1 * ratio ** kg_exp
    ko_e = ko1 * ratio ** ko_exp
    bij = (ko_e < 1.0) & (kg_e > ko_e)          # the mathematical bijection guard
    rs_e = np.zeros_like(grid)
    rv_e = np.full_like(grid, rv[a])
    if np.any(bij):
        rs_b, rv_b = rs_rv_from_kvalues(ko_e[bij], kg_e[bij], surface)
        rs_e[bij] = np.maximum(rs_b, 0.0)
        rv_e[bij] = rv_b
    if not np.all(bij):
        # below the pressure where K_o reaches 1 the bijection is undefined; hold
        # r_s at the last valid value and keep Rs = 0 there (it is ~0 anyway).
        last = np.where(bij)[0]
        if last.size:
            rv_e[~bij] = rv_e[last[0]]
        flags.append("K-value bijection undefined near psc on some node(s); r_s held")
    rs_e[0] = 0.0                                # Rs = 0 at psc, the stock-tank convention

    # Bo: anchor 1.0 at psc, then 1/Bo interpolation (plain p, the oil coordinate)
    xp = np.concatenate(([psc], p))
    inv_bo = PchipInterpolator(xp, np.concatenate(([1.0 / bo_psc], 1.0 / bo)),
                              extrapolate=True)
    bo_e = 1.0 / inv_bo(grid)

    # Bg: Z = 1 ideal anchor when T known, else isothermal pressure ratio
    if reservoir_temperature:
        bg_psc = ideal_bg(reservoir_temperature, psc)
        bg_basis = "Z=1 ideal-gas anchor (reservoir temperature supplied)"
    else:
        bg_psc = bg[a] * (p1 / psc)
        bg_basis = "frozen-Z pressure ratio Bg(p1)*p1/psc (no reservoir temperature)"

    # local gas compressibility at the anchor node, from the saturated Bg locus
    cg_p1 = (-(np.log(bg[a + 1]) - np.log(bg[a])) / (p[a + 1] - p[a])
             if a + 1 < len(p) else np.nan)
    if bg_method == "compressibility" and np.isfinite(cg_p1) and cg_p1 > 0:
        # Bg = Bg(p1) * exp(-INTEGRAL c_g dp), with c_g bounded-interpolated in
        # 1/p between the exact ideal-gas value c_g(psc) = 1/psc and the anchor.
        # Anchored at the data node p1 so the extension joins the locus, and
        # well-conditioned (a few-% c_g error integrates to a few-% Bg, no
        # amplification). Reduces to the Z=1 ideal Bg ~ 1/p in the low-p limit.
        # bopvt-lookup gas_lowp_methods.py (27 Jun 2026); validated 1-4% vs the
        # exact EOS where the naive 1/p rule sat at 8-21%.
        q = np.concatenate(([p1], grid[::-1]))           # descend p1 -> psc
        bg_chain = extend_bg_compressibility(q, p1, float(bg[a]), float(cg_p1),
                                             p1, psc)
        bg_e = bg_chain[1:][::-1]                         # back to ascending grid
        bg_psc = float(bg_e[0])
        bg_basis = ("psc-anchored c_g integration (bounded in 1/p) "
                    "Bg=Bg(p1)*exp(-INT c_g dp)")
    else:
        if bg_method == "compressibility":
            flags.append("compressibility Bg unavailable (need >=2 bottom nodes "
                         "with positive c_g); used 1/Bg interpolation")
        inv_bg = PchipInterpolator(xp, np.concatenate(([1.0 / bg_psc], 1.0 / bg)),
                                  extrapolate=True)
        bg_e = 1.0 / inv_bg(grid)

    # viscosities: continue the mobilities and divide out the anchored B
    mob_o = PchipInterpolator(np.log(p), 1.0 / (bo * uo), extrapolate=True)
    mob_g = PchipInterpolator(p, 1.0 / (bg * ug), extrapolate=True)
    uo_e = 1.0 / (mob_o(np.log(grid)) * bo_e)
    ug_e = 1.0 / (mob_g(grid) * bg_e)
    if np.any(uo_e <= 0) or np.any(ug_e <= 0):
        flags.append("mobility extrapolation produced a non-physical viscosity; "
                     "review the bottom rows")

    return {
        "p": grid, "rs": rs_e, "rv": rv_e, "bo": bo_e, "bg": bg_e,
        "uo": uo_e, "ug": ug_e, "p1": p1, "bg_basis": bg_basis,
        "bg_psc": bg_psc, "flags": flags,
    }
