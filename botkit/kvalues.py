"""Two-pseudocomponent K-value formulation and convergence pressure.

The surface-oil / surface-gas K-value relations are the modified-black-oil
formulation of Whitson & Torp, JPT 1983 (SPE 10067) -- mapping the black-oil
ratios (Rs, Rv) to two-pseudocomponent mole fractions and equilibrium ratios
(ko, kg) -- with the two-component construction of Coats (SPE 50990).  The
black-oil ratios are mapped to mole fractions and K-values; the inverse recovers
(Rs, Rv) from a pair of K-values.  Singh, Fevang & Whitson, SPE 109596 (2007),
restate these transforms (App. A) and add the consistent convergence-pressure
estimate used here: extrapolating log K versus log p to K = 1 (App. B).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .model import BlackOilTable, SurfaceFluids


def kvalues(rs: np.ndarray, rv: np.ndarray,
            surface: SurfaceFluids) -> Dict[str, np.ndarray]:
    """K-values and component mole fractions from (Rs, Rv).

    Returns ko, kg (oil/gas pseudocomponent equilibrium ratios) and the liquid
    and vapour mole fractions xo, xg, yo, yg.  Modified-black-oil transforms of
    Whitson & Torp (JPT 1983, SPE 10067); restated in Singh, Fevang & Whitson,
    SPE 109596, App. A.
    """
    rs = np.asarray(rs, dtype=float)
    rv = np.asarray(rv, dtype=float)
    Co, mult, Lo, Lg = surface.Co, surface.mult, surface.Lo, surface.Lg

    ko = (1.0 + rs * mult / Co) / (1.0 + 1.0 / (Co * rv / mult))
    with np.errstate(divide="ignore"):
        # at the Rs = 0 vertex (psc) kg -> inf is the correct limit; the
        # composition basis (xo/xg/yo/yg) carries the bottom edge, so the pole
        # is harmless here.
        kg = (1.0 + Co / (rs * mult)) / (1.0 + Co * (rv / mult))
    xo = Co / (rs * mult + Co)
    xg = 1.0 - xo
    yo = Lo * rv / (Lg + Lo * rv)
    yg = 1.0 - yo
    return {"ko": ko, "kg": kg, "xo": xo, "xg": xg, "yo": yo, "yg": yg}


def rs_rv_from_kvalues(ko: np.ndarray, kg: np.ndarray,
                       surface: SurfaceFluids) -> Tuple[np.ndarray, np.ndarray]:
    """Recover (Rs, Rv) from a pair of K-values (inverse of :func:`kvalues`).

    Singh, Fevang & Whitson, SPE 109596, App. A.
    """
    ko = np.asarray(ko, dtype=float)
    kg = np.asarray(kg, dtype=float)
    Co, mult = surface.Co, surface.mult
    rs = Co * (1.0 - ko) / (kg - 1.0) / mult
    rv = ko * (kg - 1.0) / (Co * kg * (1.0 - ko)) * mult
    return rs, rv


def phase_properties(table: BlackOilTable) -> Dict[str, np.ndarray]:
    """Densities and molar volumes of the saturated phases.

    Combines the K-value mole fractions with the table's Bo/Bg to give phase
    molecular weights, mass densities, and molar volumes used by the EOS
    regression.  Singh, Fevang & Whitson, SPE 109596, App. A.
    """
    s = table.surface
    o, g = table.pvto, table.pvtg
    if s is None:
        raise ValueError("phase_properties requires surface fluids")

    kv = kvalues(o.rs, g.rv, s)
    xo, yo = kv["xo"], kv["yo"]
    mwo = s.oil_mw * xo + s.gas_mw * (1.0 - xo)
    mwg = s.oil_mw * yo + s.gas_mw * (1.0 - yo)

    deno = (s.st_oil_density + s.st_gas_density * o.rs * s.mult) / o.bo
    deng = (s.st_gas_density + s.st_oil_density * (g.rv / s.mult)) / (g.bg / s.mult)

    return {
        **kv,
        "mwo": mwo, "mwg": mwg,
        "deno": deno, "deng": deng,
        "vo": mwo / deno, "vg": mwg / deng,
    }


def convergence_pressure(p: np.ndarray, ko: np.ndarray, kg: np.ndarray,
                         n_nodes: int = 2) -> float:
    """Convergence pressure by log K vs log p extrapolation to K = 1.

    Linear fit of log10(K) against log10(p) through the top ``n_nodes`` saturated
    nodes, solved for K = 1 (log K = 0), for both the oil and gas pseudocomponents;
    the two estimates are averaged.  Singh, Fevang & Whitson, SPE 109596, App. B.  The
    canonical method uses the top *two* nodes (the last log-K slope), which is the
    default; using more nodes is a least-squares smoothing of that slope.  This
    must be computed on the trusted locus only, i.e. after the misaligned tail has
    been removed.  The result may be sanity-checked against the K-value trend and
    overridden.

    The oil and gas roots disagree increasingly away from the critical point and
    run away for lean fluids (one K-line flattens, its root lands at millions of
    psia or near zero); :func:`convergence_pressure_crossing` reaches the same
    target in one bounded condition and is the default.
    """
    p = np.asarray(p, dtype=float)
    n = min(n_nodes, len(p))
    if n < 2:
        raise ValueError("convergence_pressure needs at least two nodes")

    lp = np.log10(p[-n:])

    def solve_for_unity(k: np.ndarray) -> float:
        lk = np.log10(np.abs(k[-n:]))
        slope, intercept = np.polyfit(lp, lk, 1)
        if slope == 0:
            return np.inf
        # log K = slope * log p + intercept = 0  ->  log p = -intercept / slope
        return 10.0 ** (-intercept / slope)

    pk_oil = solve_for_unity(ko)
    pk_gas = solve_for_unity(kg)
    return float(np.mean([pk_oil, pk_gas]))


def convergence_pressure_crossing(p: np.ndarray, mwo: np.ndarray, mwg: np.ndarray,
                                  n_nodes: int = 4) -> float:
    """Convergence pressure as the oil/gas average-MW crossing (single root).

    The two reservoir phases are mixtures of the same surface gas (light) and
    stock-tank oil (heavy), so each phase's average molecular weight is the
    composition coordinate of the two-component (Coats, SPE 50990) model.  As
    pressure rises toward the critical point the oil phase takes up gas and its
    average MW ``mwo`` falls, while the gas phase takes up heavy ends and ``mwg``
    rises; at criticality the phases are identical, so x_g = y_g and the two
    average MWs coincide.  Their crossing therefore locates the convergence
    pressure in a *single* condition, from the table data plus the surface
    molecular weights, with no EOS or LBC.

    This is the molar-volume (M/gamma) coordinate of Whitson's note: molar volume
    is the one quantity ideal (Amagat) mixing preserves, so a fluid and all its
    splits ride one line indexed by average MW.  It reaches the same target as
    Singh, Fevang & Whitson (SPE 109596, App. B) but in one bounded root rather
    than two K-roots averaged -- where the oil and gas roots blow up away from
    critical, the crossing stays finite.

    Each branch is fitted linearly in ``p`` through the top ``n_nodes`` saturated
    nodes (the validated probe used four) and solved for the crossing.  The fit is
    valid only where the geometry is physical -- oil MW falling, gas MW rising, and
    the crossing above the top node.  ``nan`` is returned otherwise so the caller
    can fall back to :func:`convergence_pressure`.

    Note: this locates the two-component criticality consistently, but which root
    is closest to the true p_c is not yet validated against an EOS critical point
    (the PhazeComp band run is built to answer that); it is the consistency-cleaner
    estimate, not a proven-more-accurate one.
    """
    p = np.asarray(p, dtype=float)
    mwo = np.asarray(mwo, dtype=float)
    mwg = np.asarray(mwg, dtype=float)
    n = min(n_nodes, len(p))
    if n < 2:
        raise ValueError("convergence_pressure_crossing needs at least two nodes")

    pt = p[-n:]
    slope_o, int_o = np.polyfit(pt, mwo[-n:], 1)   # oil MW falls with p
    slope_g, int_g = np.polyfit(pt, mwg[-n:], 1)   # gas MW rises with p

    denom = slope_o - slope_g
    if denom == 0:
        return float("nan")
    pk = (int_g - int_o) / denom
    # criticality must lie above the table, with the branches converging
    if not np.isfinite(pk) or pk <= pt[-1] or slope_o >= 0 or slope_g <= 0:
        return float("nan")
    return float(pk)
