"""Two-pseudocomponent Peng-Robinson EOS with Peneloux volume shift.

Models the reservoir oil and gas as mixtures of two pseudocomponents (surface
oil and surface gas).  Each phase pressure is reproduced from the table molar
volume through the Peng-Robinson equation with a Peneloux volume translation;
the component parameters and per-node volume shifts are regressed to match the
observed molar volumes.

References:
  * Peng, D.-Y. & Robinson, D.B. (1976/1978), Peng-Robinson EOS and the
    high-acentric-factor alpha modification.
  * Peneloux, A., Rauzy, E. & Freze, R. (1982), consistent volume translation.
  * Critical-property correlations for the surface pseudocomponents follow the
    forms used in Singh & Whitson, SPE 109596 (2007).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .model import R_FIELD, SurfaceFluids

WATER_SG_REF = 62.4
AIR_MW = 28.966


def _pr_alpha_m(omega: float) -> float:
    """Peng-Robinson alpha-function m parameter (1978 high-omega form)."""
    if omega <= 0.49:
        return 0.37464 + 1.54226 * omega - 0.26992 * omega ** 2
    return 0.3796 + 1.485 * omega - 0.1644 * omega ** 2 + 0.01667 * omega ** 3


def _pr_ab(Tc: float, Pc: float, omega: float, T: float) -> Tuple[float, float]:
    m = _pr_alpha_m(omega)
    alpha = (1.0 + m * (1.0 - (T / Tc) ** 0.5)) ** 2
    a = 0.45724 * R_FIELD ** 2 * Tc ** 2 * alpha / Pc
    b = 0.07780 * R_FIELD * Tc / Pc
    return a, b


def ab_oil(oil_mw: float, oil_density: float, T: float):
    """Initial PR a, b for the surface-oil pseudocomponent (T in deg R)."""
    sgo = oil_density / WATER_SG_REF
    Tc = 608 + 364 * np.log10(oil_mw - 71.2) + (2450 * np.log10(oil_mw) - 3800) * np.log10(sgo)
    Pc = 1188 - 431 * np.log10(oil_mw - 61.1) + (2319 - 852 * np.log10(oil_mw - 53.7)) * (sgo - 0.8)
    omega = 0.000003 * oil_mw ** 2 + 0.004 * oil_mw - 0.039
    a, b = _pr_ab(Tc, Pc, omega, T)
    return a, b, Tc, Pc


def ab_gas(gas_mw: float, gas_density: float, T: float):
    """Initial PR a, b for the surface-gas pseudocomponent (T in deg R)."""
    sg = gas_mw / AIR_MW
    if sg < 0.75:
        Tc = 168 + 325 * sg - 12.5 * sg ** 2
        Pc = 667 + 15 * sg - 37.5 * sg ** 2
    else:
        Tc = 187 + 330 * sg - 71.5 * sg ** 2
        Pc = 706 + 51.7 * sg - 11.1 * sg ** 2
    omega = 0.1637 * sg - 0.0792
    a, b = _pr_ab(Tc, Pc, omega, T)
    return a, b, Tc, Pc


def zroots(a: float, b: float, p: float, T: float) -> np.ndarray:
    """Real roots of the Peng-Robinson cubic in Z (middle root dropped)."""
    A = a * p / (R_FIELD * T) ** 2
    B = b * p / (R_FIELD * T)
    coeffs = [1.0, B - 1.0, A - 3 * B ** 2 - 2 * B, -(A * B - B ** 2 - B ** 3)]
    roots = np.roots(coeffs)
    real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
    if len(real) == 3:
        real = [real[0], real[2]]
    return np.array(real)


def _a_mix(ao: float, fo: float, ag: float, fg: float) -> float:
    # van der Waals one-fluid mixing, zero binary interaction
    return ao * fo ** 2 + ag * fg ** 2 + 2 * (ao * ag) ** 0.5 * fo * fg


def _b_mix(bo: float, fo: float, bg: float, fg: float) -> float:
    return bo * fo + bg * fg


@dataclass
class EOSParameters:
    """Tuned two-component PR parameters and reservoir temperature."""

    ao: float
    bo: float
    ag: float
    bg: float
    T: float  # deg R

    def as_tuple(self):
        return self.ao, self.bo, self.ag, self.bg, self.T


def molar_volume(p: float, fo: float, fg: float, params: EOSParameters,
                 so: float, sg: float) -> float:
    """Observed molar volume predicted by the EOS (Peneloux-translated).

    Solves the PR cubic for the largest Z root, forms the EOS molar volume, and
    subtracts the volume translation to return the *observed* molar volume that
    is compared with the table value.
    """
    a = _a_mix(params.ao, fo, params.ag, fg)
    b = _b_mix(params.bo, fo, params.bg, fg)
    Z = zroots(a, b, p, params.T).max()
    veos = Z * R_FIELD * params.T / p
    return veos - (fo * so * params.bo + fg * sg * params.bg)


def phase_pressure(v: float, fo: float, fg: float, params: EOSParameters,
                   so: float, sg: float) -> float:
    """PR pressure for a phase given its observed molar volume."""
    veos = v + fo * so * params.bo + fg * sg * params.bg
    a = _a_mix(params.ao, fo, params.ag, fg)
    b = _b_mix(params.bo, fo, params.bg, fg)
    return R_FIELD * params.T / (veos - b) - a / (veos * (veos + b) + b * (veos - b))


def initial_parameters(surface: SurfaceFluids, T: float) -> EOSParameters:
    """Starting PR parameters from the surface-fluid correlations."""
    ao, bo, _, _ = ab_oil(surface.oil_mw, surface.st_oil_density, T)
    ag, bg, _, _ = ab_gas(surface.gas_mw, surface.st_gas_density, T)
    return EOSParameters(ao=ao, bo=bo, ag=ag, bg=bg, T=T)


def tune_global(props: Dict[str, np.ndarray], params0: EOSParameters,
                regress_temperature: bool = False) -> Tuple[EOSParameters, float, float]:
    """Regress the component parameters to the observed molar volumes.

    Replaces the original loose Nelder-Mead (xatol = 350 psi) with a bounded
    least-squares fit over the per-node oil/gas *relative* molar-volume
    residuals.  Relative residuals weight the small oil volumes and large gas
    volumes equally; the per-node volume shifts (regressed separately) then
    match each node exactly, so no global shift is fitted here.  Temperature is
    fixed unless ``regress_temperature``.  Returns the tuned parameters and the
    (zero) global shifts kept for API symmetry.
    """
    p = props["p"]
    vo, vg = props["vo"], props["vg"]
    xo, xg, yo, yg = props["xo"], props["xg"], props["yo"], props["yg"]

    # scale parameters to O(1) for the solver
    scale = np.array([params0.ao, params0.bo, params0.ag, params0.bg])

    def unpack(z):
        ao, bo, ag, bg = z[:4] * scale
        T = z[4] if regress_temperature else params0.T
        return EOSParameters(ao, bo, ag, bg, T)

    def residuals(z):
        prm = unpack(z)
        res = []
        for i in range(len(p)):
            res.append((molar_volume(p[i], xo[i], xg[i], prm, 0.0, 0.0) - vo[i]) / vo[i])
            res.append((molar_volume(p[i], yo[i], yg[i], prm, 0.0, 0.0) - vg[i]) / vg[i])
        return res

    z0 = [1.0, 1.0, 1.0, 1.0]
    lo = [0.3, 0.3, 0.3, 0.3]
    hi = [3.0, 3.0, 3.0, 3.0]
    if regress_temperature:
        z0.append(params0.T)
        lo.append(params0.T * 0.7)
        hi.append(params0.T * 1.3)

    sol = least_squares(residuals, z0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    return unpack(sol.x), 0.0, 0.0


def tune_local_shifts(props: Dict[str, np.ndarray], params: EOSParameters,
                      bound: float = 0.2, smoothness: float = 0.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-node Peneloux volume shifts matching the observed molar volumes.

    With ``smoothness = 0`` each node is solved independently, reproducing the
    oil and gas molar volumes essentially exactly (the original behaviour).

    With ``smoothness > 0`` all nodes are solved jointly and a roughness penalty
    (the squared second difference of so(p) and sg(p)) is added, yielding smooth,
    monotone, confidently interpolatable / extrapolatable shift trends.  The
    penalty trades a small in-sample molar-volume error for much smoother
    controls; the cost grows gently with the penalty weight.  Relative
    molar-volume residuals weight the small oil and large gas volumes equally.
    """
    p = props["p"]
    vo, vg = props["vo"], props["vg"]
    xo, xg, yo, yg = props["xo"], props["xg"], props["yo"], props["yg"]
    n = len(p)

    if smoothness <= 0.0:
        so_tab, sg_tab = [], []
        for i in range(n):
            def res(s):
                return [
                    molar_volume(p[i], xo[i], xg[i], params, s[0], s[1]) - vo[i],
                    molar_volume(p[i], yo[i], yg[i], params, s[0], s[1]) - vg[i],
                ]
            sol = least_squares(res, [0.0, 0.0],
                                bounds=([-bound, -bound], [bound, bound]),
                                xtol=1e-12, ftol=1e-12)
            so_tab.append(sol.x[0])
            sg_tab.append(sol.x[1])
        return np.array(so_tab), np.array(sg_tab)

    # joint regularized solve: variables [so(0..n-1), sg(0..n-1)]
    w = np.sqrt(smoothness)

    def residuals(z):
        so, sg = z[:n], z[n:]
        out = []
        for i in range(n):
            out.append((molar_volume(p[i], xo[i], xg[i], params, so[i], sg[i]) - vo[i]) / vo[i])
            out.append((molar_volume(p[i], yo[i], yg[i], params, so[i], sg[i]) - vg[i]) / vg[i])
        out.extend((w * np.diff(so, 2)).tolist())
        out.extend((w * np.diff(sg, 2)).tolist())
        return out

    sol = least_squares(residuals, np.zeros(2 * n),
                        bounds=(-bound - 0.05, bound + 0.05),
                        xtol=1e-12, ftol=1e-12)
    return sol.x[:n], sol.x[n:]


# abscissa transforms for the parameter-trend extrapolation. Any monotone g
# keeps the extrapolation anchored at the last point exactly, since
# g(p) - g(p_last) = 0 there. Empirically the oil volume-shift trend linearises
# in log p (or 1/p) and the gas shift in plain pressure.
_ABSCISSA = {
    "linear": (np.asarray, "p"),
    "log": (np.log, "log p"),
    "recip": (lambda p: 1.0 / np.asarray(p, dtype=float), "1/p"),
    "sqrt": (np.sqrt, "sqrt p"),
}


def trend_extrapolator(p_nodes: np.ndarray, s_tab: np.ndarray, n_trend: int = 3,
                       transform: str = "linear"):
    """Anchored linear extrapolator for a per-node parameter trend.

    Within the node range the parameter is interpolated (piecewise-linear)
    through the fitted values.  Above the top node it is extended along the
    local trend of the last ``n_trend`` points in the chosen abscissa
    ``transform`` (``linear`` = p, ``log`` = log p, ``recip`` = 1/p,
    ``sqrt`` = sqrt p):

        s(p) = s_last + slope * (g(p) - g(p_last)),

    where ``g`` is the transform and the slope is the least-squares slope of
    (s - s_last) against (g(p) - g(p_last)) over the last ``n_trend`` points -- a
    line through the last point.  By construction the extrapolation passes
    through the last valid point exactly (no offset at the join) and follows the
    established trend rather than freezing flat.  A transform that linearises the
    trend (log p / 1/p for the oil shift, plain p for the gas shift) gives the
    most defensible extrapolation.  Returns the callable and the fitted slope.
    """
    g = _ABSCISSA[transform][0]
    p_nodes = np.asarray(p_nodes, dtype=float)
    s_tab = np.asarray(s_tab, dtype=float)
    p_last, s_last = p_nodes[-1], s_tab[-1]
    g_last = float(g(np.array([p_last]))[0])
    k = min(max(2, n_trend), len(p_nodes))
    dg = g(p_nodes[-k:]) - g_last
    ds = s_tab[-k:] - s_last
    denom = float(np.sum(dg * dg))
    slope = float(np.sum(dg * ds) / denom) if denom > 0 else 0.0

    def f(p):
        p = np.asarray(p, dtype=float)
        below = np.interp(p, p_nodes, s_tab)
        above = s_last + slope * (g(p) - g_last)
        return np.where(p > p_last, above, below)

    return f, slope


def eos_pressure_match_error(props: Dict[str, np.ndarray], params: EOSParameters,
                             so, sg, top_n: Optional[int] = None) -> float:
    """Maximum relative error between EOS phase pressure and table pressure.

    Used by the fallback gate: if this exceeds the configured tolerance the
    EOS-dependent paths are disabled in favour of K-value/correlation methods.

    The extension regenerates properties above the top of the table, so the gate
    that matters is the fit over the upper anchor nodes; pass ``top_n`` to
    restrict the metric to the highest ``top_n`` pressures.  (At the lowest
    pressures the liquid branch is so stiff that a negligible molar-volume
    residual inflates into a large pressure error, yet those nodes are retained
    from the table and never regenerated.)
    """
    p = props["p"]
    xo, xg, yo, yg = props["xo"], props["xg"], props["yo"], props["yg"]
    so = np.broadcast_to(so, p.shape)
    sg = np.broadcast_to(sg, p.shape)
    idx = range(len(p)) if top_n is None else range(max(0, len(p) - top_n), len(p))
    err = 0.0
    for i in idx:
        po = phase_pressure(props["vo"][i], xo[i], xg[i], params, so[i], sg[i])
        pg = phase_pressure(props["vg"][i], yo[i], yg[i], params, so[i], sg[i])
        err = max(err, abs(po - p[i]) / p[i], abs(pg - p[i]) / p[i])
    return float(err)


def molar_volume_match_error(props: Dict[str, np.ndarray], params: EOSParameters,
                             so, sg) -> float:
    """Maximum relative molar-volume error over all nodes (well-conditioned).

    A stiffness-free alternative consistency metric: unlike the pressure error,
    relative molar-volume error stays bounded on the liquid branch.
    """
    p = props["p"]
    xo, xg, yo, yg = props["xo"], props["xg"], props["yo"], props["yg"]
    so = np.broadcast_to(so, p.shape)
    sg = np.broadcast_to(sg, p.shape)
    err = 0.0
    for i in range(len(p)):
        vo = molar_volume(p[i], xo[i], xg[i], params, so[i], sg[i])
        vg = molar_volume(p[i], yo[i], yg[i], params, so[i], sg[i])
        err = max(err, abs(vo - props["vo"][i]) / props["vo"][i],
                  abs(vg - props["vg"][i]) / props["vg"][i])
    return float(err)
