"""Two-pseudocomponent K-value formulation and convergence pressure.

Implements the surface-oil / surface-gas K-value relations and the
convergence-pressure estimate of Singh & Whitson, SPE 109596 (2007),
Appendices A and B.  The black-oil ratios (Rs, Rv) are mapped to component
mole fractions and equilibrium ratios (ko, kg); the inverse recovers (Rs, Rv)
from a pair of K-values.  Convergence pressure is found by extrapolating
log K versus log p to K = 1 (App. B).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .model import BlackOilTable, SurfaceFluids


def kvalues(rs: np.ndarray, rv: np.ndarray,
            surface: SurfaceFluids) -> Dict[str, np.ndarray]:
    """K-values and component mole fractions from (Rs, Rv).

    Returns ko, kg (oil/gas pseudocomponent equilibrium ratios) and the liquid
    and vapour mole fractions xo, xg, yo, yg.  Singh & Whitson, SPE 109596,
    App. A.
    """
    rs = np.asarray(rs, dtype=float)
    rv = np.asarray(rv, dtype=float)
    Co, mult, Lo, Lg = surface.Co, surface.mult, surface.Lo, surface.Lg

    ko = (1.0 + rs * mult / Co) / (1.0 + 1.0 / (Co * rv / mult))
    kg = (1.0 + Co / (rs * mult)) / (1.0 + Co * (rv / mult))
    xo = Co / (rs * mult + Co)
    xg = 1.0 - xo
    yo = Lo * rv / (Lg + Lo * rv)
    yg = 1.0 - yo
    return {"ko": ko, "kg": kg, "xo": xo, "xg": xg, "yo": yo, "yg": yg}


def rs_rv_from_kvalues(ko: np.ndarray, kg: np.ndarray,
                       surface: SurfaceFluids) -> Tuple[np.ndarray, np.ndarray]:
    """Recover (Rs, Rv) from a pair of K-values (inverse of :func:`kvalues`).

    Singh & Whitson, SPE 109596, App. A.
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
    regression.  Singh & Whitson, SPE 109596, App. A.
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
    the two estimates are averaged.  Singh & Whitson, SPE 109596, App. B.  The
    canonical method uses the top *two* nodes (the last log-K slope), which is the
    default; using more nodes is a least-squares smoothing of that slope.  This
    must be computed on the trusted locus only, i.e. after the misaligned tail has
    been removed.  The result may be sanity-checked against the K-value trend and
    overridden.
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
