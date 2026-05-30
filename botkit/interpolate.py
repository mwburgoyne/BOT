"""Monotone (PCHIP) interpolation of the saturated Black-Oil locus.

This is the "bot_interpolation" layer used to honour and interpolate *between*
the input data points; the EOS is reserved for creating genuinely new data
(extension above the table, undersaturated branches).

Following the consolidated interpolation findings of the companion lookup work,
the solution-ratio quantities are interpolated as *compositions* -- the gas
mole fraction in the liquid (x_g) and in the vapour (y_g) -- the
"composition-everywhere" method, and Rs / Rv are formed from the interpolated
compositions at the query.  Interpolating the bounded, regular compositions
avoids the K = y_g / x_g coordinate singularity as x_g -> 0 (the Rs = 0 edge)
that interpolating K-values or Rs / Rv directly would suffer.

Each quantity uses the per-quantity abscissa fixed by the companion study's
Whitson-corpus leave-one-out validation: x_g in p^1.5 (Rs ~ p^1.5 empirically,
which rebuilds the Rs -> 0 bottom segment to single-digit % error versus
100-200% in log p), y_g in log p.  Volumetrics use reciprocal families that
interpolate cleanly and stay positive: oil 1/Bo in plain p with oil mobility
1/(Bo*uo) in log p (uo recovered as the log-p mobility / the plain-p Bo); gas
1/Bg and gas mobility 1/(Bg*ug) in plain p.  PCHIP is shape-preserving, so every
input node is reproduced exactly and monotone data stays monotone.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.interpolate import PchipInterpolator

from .kvalues import kvalues
from .model import SurfaceFluids


def rs_from_xg(xg: np.ndarray, surface: SurfaceFluids) -> np.ndarray:
    """Solution GOR from the gas mole fraction in the liquid (inverts Singh App. A)."""
    # xo = Co / (Rs*mult + Co), xo = 1 - xg  ->  Rs = Co*xg / ((1-xg)*mult)
    xo = 1.0 - xg
    return surface.Co * xg / (xo * surface.mult)


def rv_from_yg(yg: np.ndarray, surface: SurfaceFluids) -> np.ndarray:
    """Vapourized OGR from the gas mole fraction in the vapour (inverts Singh App. A)."""
    # yo = Lo*Rv / (Lg + Lo*Rv), yo = 1 - yg  ->  Rv = Lg*(1-yg) / (Lo*yg)
    yo = 1.0 - yg
    return surface.Lg * yo / (surface.Lo * yg)


class SaturatedInterpolator:
    """Honour-the-data interpolation of the saturated locus.

    Built from the trusted saturated arrays; evaluates Rs, Rv, Bo, Bg, uo, ug at
    any pressure inside the locus range, reproducing the input nodes exactly.
    """

    def __init__(self, p, rs, rv, bo, bg, uo, ug, surface: SurfaceFluids):
        p = np.asarray(p, dtype=float)
        order = np.argsort(p)
        p = p[order]
        rs, rv = np.asarray(rs)[order], np.asarray(rv)[order]
        bo, bg = np.asarray(bo)[order], np.asarray(bg)[order]
        uo, ug = np.asarray(uo)[order], np.asarray(ug)[order]

        self.surface = surface
        self.pmin, self.pmax = float(p[0]), float(p[-1])
        lp = np.log(p)
        p15 = p ** 1.5

        kv = kvalues(rs, rv, surface)
        # compositions: gas mole fraction in liquid (xg) in p^1.5, vapour (yg) in log p
        self._xg = PchipInterpolator(p15, kv["xg"], extrapolate=True)
        self._yg = PchipInterpolator(lp, kv["yg"], extrapolate=True)
        # oil volumetrics: 1/Bo in plain p, oil mobility 1/(Bo*uo) in log p
        self._inv_bo = PchipInterpolator(p, 1.0 / bo, extrapolate=True)
        self._inv_bo_uo = PchipInterpolator(lp, 1.0 / (bo * uo), extrapolate=True)
        # gas volumetrics in plain p (1/Bg and gas mobility 1/(Bg*ug))
        self._inv_bg = PchipInterpolator(p, 1.0 / bg, extrapolate=True)
        self._inv_bg_ug = PchipInterpolator(p, 1.0 / (bg * ug), extrapolate=True)

    def evaluate(self, p) -> Dict[str, np.ndarray]:
        """Saturated properties at pressure(s) ``p`` from the interpolants."""
        p = np.asarray(p, dtype=float)
        lp = np.log(p)
        p15 = p ** 1.5
        xg = self._xg(p15)
        yg = self._yg(lp)
        bo = 1.0 / self._inv_bo(p)
        uo = 1.0 / (self._inv_bo_uo(lp) * bo)
        bg = 1.0 / self._inv_bg(p)
        ug = 1.0 / (self._inv_bg_ug(p) * bg)
        return {
            "rs": rs_from_xg(xg, self.surface),
            "rv": rv_from_yg(yg, self.surface),
            "bo": bo, "bg": bg, "uo": uo, "ug": ug,
            "xg": xg, "yg": yg,
        }

    def in_range(self, p) -> np.ndarray:
        """Whether pressure(s) lie within the interpolation (no extrapolation)."""
        p = np.asarray(p, dtype=float)
        return (p >= self.pmin) & (p <= self.pmax)


def densify_saturated(interp: SaturatedInterpolator, p_nodes: np.ndarray,
                      per_gap: int = 0) -> Dict[str, np.ndarray]:
    """Insert ``per_gap`` PCHIP-interpolated nodes (log-spaced) between data nodes.

    Honours the original nodes exactly and fills the gaps with monotone
    interpolated values rather than the EOS, so the densified locus carries no
    EOS low-pressure pathology.
    """
    p_nodes = np.sort(np.asarray(p_nodes, dtype=float))
    if per_gap <= 0:
        return interp.evaluate(p_nodes)
    pts = []
    for a, b in zip(p_nodes[:-1], p_nodes[1:]):
        seg = np.exp(np.linspace(np.log(a), np.log(b), per_gap + 2))
        pts.append(seg[:-1])
    pts.append([p_nodes[-1]])
    grid = np.concatenate(pts)
    out = interp.evaluate(grid)
    out["p"] = grid
    return out
