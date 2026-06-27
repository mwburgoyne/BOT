"""Compact undersaturated-FVF reconstruction for under-defined branches.

When an input table leaves an undersaturated branch insufficiently defined - no
undersaturated rows at all, a single row, or a couple of clustered rows that do
not span the pressure range - the saturated point plus a curvature model is
enough to reconstruct or extend the branch.  This module ports the compact
two-constant cubic of Milan Kratky's note and its low-pressure companion,
evaluated in the sibling ``bopvt-lookup`` project, into the BOT_util unit
conventions.

The method casts a black-oil branch as a two-pseudocomponent (surface-oil +
surface-gas) mixture, converts the FVF to a reservoir molar density, and reduces
each undersaturated branch (constant Rs for oil, constant rs for gas) to two
SRK-derived constants ``CA = a_m/(RT)^2`` and ``CB = b_m/(RT)`` through the cubic

    Z^3 - Z^2 + (A - B - B^2) Z - A*B = 0,   A = CA*p,  B = CB*p.

This compositional view of a black-oil table is established prior art
(Whitson & Torp, SPE 10067, 1983; Coats, Thomas & Pierson, SPE 50990, 1998,
who trace it to Coats 1980); the per-branch two-constant reduction with a value
+ bubble-point-slope inversion for the data-less case is the operational twist.
The cubic is the Soave-Redlich-Kwong form (Soave, Chem. Eng. Sci. 27, 1972).

Where to use which (validated on one PR79 volatile-oil temperature ladder):

* Oil, undersaturated rows present but sparse -> fit ``CA, CB`` to them and
  reconstruct/extend; honours the measured rows.  Wins over interpolation only
  when sparse (dense data: interpolating 1/Bo ties or beats it).
* Oil, no undersaturated rows -> ``cacb_from_anchor_slope`` solves ``CA, CB``
  from ``Bo_sat`` and the bubble-point compressibility ``c_o(pb)`` (from a tuned
  EOS or a Vasquez-Beggs/McCain correlation); the cubic supplies the curvature.
  ~0.06% mean vs ~3% for the textbook constant-c_o expansion, ~50x better, and
  graceful (+-20% c_o -> ~1-2% branch error).
* Gas, low dewpoint, no undersaturated rows -> the cubic is ill-conditioned at a
  near-ideal low-pressure anchor; use ``extend_bg_compressibility`` instead:
  ``Bg(p) = Bg0 * exp(-INTEGRAL c_g dp)`` with ``c_g`` bounded-interpolated in
  ``1/p`` between the exact ideal-gas value ``c_g(psc) = 1/psc`` and the first
  trusted point.  ~1-4% vs ~28-56% for the cubic and ~8-21% for naive 1/p.
* Whether the curvature licenses the cubic at all is the changeover criterion
  (``curvature_snr``): keep the cubic only when its quadratic term clears the
  data noise; otherwise the slope (compressibility) is all that is supported.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import fsolve, least_squares

from .model import FT3_PER_BBL, R_FIELD, SurfaceFluids

__all__ = [
    "oil_moles_per_stb",
    "gas_moles_per_mscf",
    "cubic_z",
    "b_to_z",
    "z_to_b",
    "fit_cacb",
    "reconstruct_b",
    "cacb_from_anchor_slope",
    "curvature_snr",
    "cg_psc_anchored",
    "extend_bg_compressibility",
    "compact_oil_bo",
    "branch_is_under_defined",
]


# --- two-pseudocomponent mole accounting (BOT_util surface conventions) ------

def oil_moles_per_stb(rs: float, surface: SurfaceFluids) -> float:
    """Total reservoir-oil moles per stock-tank barrel along an oil branch.

    ``Lo`` lbmol of stock-tank oil per bbl plus ``rs*Lg`` lbmol of dissolved
    surface gas (rs in Mscf/bbl, ``Lg`` lbmol/Mscf).  Constant along an
    undersaturated oil branch because the composition is fixed.
    """
    return surface.Lo + rs * surface.Lg


def gas_moles_per_mscf(rv: float, surface: SurfaceFluids) -> float:
    """Total reservoir-gas moles per Mscf of surface gas along a gas branch.

    ``Lg`` lbmol of surface gas per Mscf plus ``rv*Lo`` lbmol of vapourized
    stock-tank oil (rv in bbl/Mscf, ``Lo`` lbmol/bbl).
    """
    return surface.Lg + rv * surface.Lo


def b_to_z(b: np.ndarray, p: np.ndarray, n_moles: float, T: float) -> np.ndarray:
    """Compressibility factor from a formation volume factor.

    ``Z = p * V_res / (n R T)`` with the reservoir volume of one surface unit
    ``V_res = B * FT3_PER_BBL`` (B in rb/STB for oil, rb/Mscf for gas) and
    ``n`` the moles in that surface unit.
    """
    return p * b * FT3_PER_BBL / (n_moles * R_FIELD * T)


def z_to_b(z: np.ndarray, p: np.ndarray, n_moles: float, T: float) -> np.ndarray:
    """Formation volume factor from a compressibility factor (inverse of b_to_z)."""
    return z * n_moles * R_FIELD * T / (p * FT3_PER_BBL)


# --- the SRK two-constant cubic ---------------------------------------------

def _cubic_coeffs(CA: float, CB: float, p: float):
    # SRK: Z^3 - Z^2 + (A - B - B^2) Z - A B = 0,  A = CA*p,  B = CB*p
    A = CA * p
    B = CB * p
    return [1.0, -1.0, A - B - B * B, -A * B]


def cubic_z(CA: float, CB: float, p: float, phase: str) -> float:
    """Selected real root of the SRK cubic: smallest for oil, largest for gas.

    Returns NaN when no positive real root exists (a degenerate branch); callers
    must treat NaN as "no value" and not clamp to a spurious root.
    """
    roots = np.roots(_cubic_coeffs(CA, CB, p))
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-7 and r.real > 0)
    if not real:
        return np.nan
    return real[0] if phase == "oil" else real[-1]


def fit_cacb(p: np.ndarray, b: np.ndarray, n_moles: float, phase: str,
             T: float, x0: Optional[Tuple[float, float]] = None
             ) -> Tuple[float, float]:
    """Least-squares fit of ``(CA, CB)`` reproducing Z(p) along one branch.

    Data-seeded and multi-start to avoid the cubic's local minima.  For a liquid
    Z -> CB*p at high p, so the slope ``median(Z/p)`` seeds CB.
    """
    p = np.asarray(p, dtype=float)
    z = b_to_z(np.asarray(b, dtype=float), p, n_moles, T)

    def resid(theta):
        CA, CB = theta
        out = []
        for pi, zi in zip(p, z):
            zp = cubic_z(CA, CB, pi, phase)
            out.append((zp - zi) / zi if np.isfinite(zp) else 10.0)
        return out

    cb0 = float(np.median(z / p))
    seeds = [] if x0 is None else [list(x0)]
    seeds += [[0.5, cb0], [1.0, cb0], [2.0, cb0 * 0.8], [0.1, cb0], [5.0, cb0]]
    hi_cb = 1e-1 if phase == "oil" else 1e-2
    best = None
    for s in seeds:
        try:
            sol = least_squares(resid, s, method="trf",
                                bounds=([0, 0], [1e3, hi_cb]), max_nfev=8000)
        except Exception:
            continue
        if best is None or sol.cost < best[1]:
            best = (sol.x, sol.cost)
    if best is None:
        raise RuntimeError("CA,CB fit failed to converge")
    return float(best[0][0]), float(best[0][1])


def reconstruct_b(CA: float, CB: float, p_query: np.ndarray, n_moles: float,
                  phase: str, T: float) -> np.ndarray:
    """Formation volume factor at ``p_query`` from stored ``(CA, CB)``."""
    z = np.array([cubic_z(CA, CB, float(pi), phase) for pi in np.atleast_1d(p_query)])
    return z_to_b(z, np.asarray(p_query, dtype=float), n_moles, T)


def cacb_from_anchor_slope(psat: float, b_sat: float, c_sat: float,
                           n_moles: float, phase: str, T: float
                           ) -> Optional[Tuple[float, float]]:
    """Solve ``(CA, CB)`` from the saturated value and its compressibility.

    The data-less case: one saturated point ``(psat, b_sat)`` is one equation
    for two unknowns; the missing constraint is the saturation-pressure
    compressibility ``c_sat = -(1/B) dB/dp`` (``c_o(pb)`` for oil, ``c_g(pd)``
    for gas), supplied by a tuned EOS or a correlation.  The cubic then carries
    the curvature (how the compressibility falls with pressure).  Returns None
    if the multi-start solve fails to find a positive root pair.
    """
    h = max(1.0, 0.001 * psat)
    slope_t = -c_sat * b_sat                       # dB/dp at the saturation point
    z_anchor = b_to_z(b_sat, psat, n_moles, T)
    cb0 = z_anchor / psat                          # liquid Z ~ CB*p near anchor

    def eqs(x):
        CA, CB = x
        b0 = reconstruct_b(CA, CB, np.array([psat]), n_moles, phase, T)[0]
        bph = reconstruct_b(CA, CB, np.array([psat + h]), n_moles, phase, T)[0]
        bmh = reconstruct_b(CA, CB, np.array([psat - h]), n_moles, phase, T)[0]
        r0 = (b0 - b_sat) / b_sat if np.isfinite(b0) else 10.0
        s = (bph - bmh) / (2 * h)
        r1 = (s - slope_t) / abs(slope_t) if np.isfinite(s) else 10.0
        return [r0, r1]

    best = None
    for ca0 in (0.05, 0.2, 0.5, 1.0, 2.0, 5.0):
        x, _info, ier, _msg = fsolve(eqs, [ca0, cb0], full_output=True)
        if ier == 1 and x[0] > 0 and x[1] > 0:
            resid = float(np.hypot(*eqs(x)))
            if best is None or resid < best[1]:
                best = (x, resid)
    if best is None:
        return None
    return float(best[0][0]), float(best[0][1])


# --- changeover criterion: does the curvature license the cubic? ------------

def curvature_snr(p: np.ndarray, b: np.ndarray, n_moles: float, T: float,
                  noise: float = 0.002) -> float:
    """Signal-to-noise of the cubic's quadratic Z term over the branch.

    The slope (compressibility) is always recoverable; the curvature - the p^2
    term that separates CA from CB - is the marginal information that licenses
    the cubic.  Returns ``|c2| * span^2 / (noise * mean(Z))`` with ``c2`` the
    quadratic coefficient of a parabola fit to Z(p).  Keep the cubic when this
    clears ~2; below it, drop to compressibility integration (the slope only).
    """
    p = np.asarray(p, dtype=float)
    z = b_to_z(np.asarray(b, dtype=float), p, n_moles, T)
    if len(p) < 3:
        return 0.0
    c2 = np.polyfit(p, z, 2)[0]
    span = float(p.max() - p.min())
    return abs(c2) * span * span / (noise * float(np.mean(z)))


# --- low-pressure gas: psc-anchored compressibility integration -------------

def cg_psc_anchored(p: np.ndarray, cg_up: float, p_up: float,
                    psc: float = 14.696) -> np.ndarray:
    """Bounded interpolation of ``c_g`` in ``1/p`` between psc and the first point.

    Anchored at the exact ideal-gas value ``c_g(psc) = 1/psc`` and the first
    trusted compressibility ``(p_up, cg_up)``.  Bounded (interpolation, never
    extrapolation) so it cannot overshoot below the dewpoint.
    """
    p = np.asarray(p, dtype=float)
    x0, x1 = 1.0 / psc, 1.0 / p_up
    return (1.0 / psc) + (cg_up - 1.0 / psc) * ((1.0 / p) - x0) / (x1 - x0)


def extend_bg_compressibility(p_query: np.ndarray, p_anchor: float,
                              bg_anchor: float, cg_up: float, p_up: float,
                              psc: float = 14.696) -> np.ndarray:
    """Gas FVF down a low-dewpoint branch by integrating ``c_g``.

    ``Bg(p) = Bg(anchor) * exp(-INTEGRAL_anchor^p c_g dp')`` with ``c_g`` from
    ``cg_psc_anchored``.  Well-conditioned because it never separates CA from CB:
    a few-% error in ``c_g`` integrates to a few-% error in ``Bg``, no
    amplification.  In the ideal limit it reduces to ``Bg ~ 1/p``.  ``p_query``
    must be sorted and start at ``p_anchor``.
    """
    p = np.asarray(p_query, dtype=float)
    cg = cg_psc_anchored(p, cg_up, p_up, psc)
    # proper trapezoidal integral of c_g dp from the anchor (not a single c_g*dp)
    ln_ratio = np.concatenate([[0.0],
                               -np.cumsum((cg[1:] + cg[:-1]) / 2 * np.diff(p))])
    return bg_anchor * np.exp(ln_ratio)


# --- branch-completion driver and under-definition test ---------------------

def branch_is_under_defined(psat: float, rows: np.ndarray, *,
                            min_rows: int = 2, min_span_frac: float = 0.5
                            ) -> bool:
    """True when an undersaturated branch is too thin to interpolate reliably.

    Under-defined = fewer than ``min_rows`` rows, or a pressure span below
    ``min_span_frac * psat`` (clustered points that do not bracket the range).
    These are the branches the compact reconstruction is reserved for; a
    well-sampled branch is better served by interpolation.
    """
    if rows is None or rows.shape[0] < min_rows:
        return True
    span = float(rows[:, 0].max() - psat)
    return span < min_span_frac * psat


def compact_oil_bo(psat: float, rs: float, bo_sat: float, surface: SurfaceFluids,
                   T: float, p_query: np.ndarray, *,
                   measured: Optional[np.ndarray] = None,
                   co_pb: Optional[float] = None) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Undersaturated oil Bo at ``p_query`` for an under-defined branch.

    Sources ``(CA, CB)`` from whatever is available: a fit to measured
    undersaturated rows when they span the range, else the anchor + ``c_o(pb)``
    inversion.  The result is ratio-anchored to ``bo_sat`` at ``psat`` so the
    branch is continuous with the measured node and the assumed oil molecular
    weight cancels (its only role is the density ratio, where it divides out).

    Returns ``(Bo at p_query, (CA, CB))``.  Raises ValueError if neither a
    spanning measured branch nor ``co_pb`` is supplied.
    """
    n = oil_moles_per_stb(rs, surface)
    p_query = np.asarray(p_query, dtype=float)

    if measured is not None and not branch_is_under_defined(psat, measured):
        p = np.concatenate([[psat], measured[:, 0]])
        b = np.concatenate([[bo_sat], measured[:, 1]])
        CA, CB = fit_cacb(p, b, n, "oil", T)
    elif co_pb is not None:
        sol = cacb_from_anchor_slope(psat, bo_sat, co_pb, n, "oil", T)
        if sol is None:
            raise ValueError(f"anchor+slope solve failed at psat={psat:g}")
        CA, CB = sol
    else:
        raise ValueError("need spanning measured rows or co_pb to reconstruct")

    bo = reconstruct_b(CA, CB, p_query, n, "oil", T)
    bo_at_psat = reconstruct_b(CA, CB, np.array([psat]), n, "oil", T)[0]
    if np.isfinite(bo_at_psat) and bo_at_psat > 0:
        bo = bo * (bo_sat / bo_at_psat)            # honour the measured node
    return bo, (CA, CB)
