"""Lohrenz-Bray-Clark (LBC) viscosity for the two-pseudocomponent system.

Reference: Lohrenz, J., Bray, B.G. & Clark, C.R. (1964), "Calculating
Viscosities of Reservoir Fluids from Their Compositions."  The dilute-gas
viscosity, viscosity-reducing parameter, and pseudocritical density are mixed
linearly across the surface-oil / surface-gas pseudocomponents and tuned to the
table viscosities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy.optimize import minimize

from .model import R_FIELD, SurfaceFluids

# Default LBC fourth-degree polynomial coefficients (Lohrenz et al., 1964)
LBC_COEFFS = (0.10230, 0.023364, 0.058533, -0.040758, 0.0093324)
ZC_PR = 0.3074  # Peng-Robinson critical compressibility


def lbc_viscosity(u_dilute: float, xi: float, reduced_density: float,
                  coeffs=LBC_COEFFS) -> float:
    """LBC viscosity (cP) from dilute-gas viscosity, xi and reduced density."""
    poly = (coeffs[0] + coeffs[1] * reduced_density
            + coeffs[2] * reduced_density ** 2
            + coeffs[3] * reduced_density ** 3
            + coeffs[4] * reduced_density ** 4)
    return (poly ** 4 - 1e-4) / xi + u_dilute


def _dilute_viscosity(Tc: float, Pc: float, mw: float, T: float):
    """Dilute-gas viscosity and viscosity-reducing parameter for a component."""
    xi = 5.35 * (Tc / mw ** 3 / Pc ** 4) ** (1 / 6)
    Tr = T / Tc
    if Tr <= 1.5:
        u = 34e-5 * Tr ** 0.94 / xi
    else:
        u = 17.78e-5 * (4.58 * Tr - 1.67) ** (5 / 8) / xi
    return u, xi


@dataclass
class LBCParameters:
    """Tuned LBC mixing parameters for the two pseudocomponents."""

    den_co: float   # oil pseudocomponent critical density
    den_cg: float   # gas pseudocomponent critical density
    xi_o: float     # oil viscosity-reducing parameter
    xi_g: float     # gas viscosity-reducing parameter
    u_o: float      # oil dilute-gas viscosity
    u_g: float      # gas dilute-gas viscosity
    coeffs: tuple = field(default=LBC_COEFFS)

    def phase_viscosity(self, fo: float, fg: float, phase_density: float) -> float:
        rho_pc = self.den_co * fo + self.den_cg * fg
        xi = self.xi_o * fo + self.xi_g * fg
        u = self.u_o * fo + self.u_g * fg
        return lbc_viscosity(u, xi, phase_density / rho_pc, self.coeffs)


def initial_lbc(surface: SurfaceFluids, T: float,
                Tco: float, Pco: float, Tcg: float, Pcg: float) -> LBCParameters:
    """Starting LBC parameters from the surface-fluid critical properties."""
    mwo, mwg = surface.oil_mw, surface.gas_mw
    sgo = surface.st_oil_density / 62.4
    vco = 21.573 + 0.015122 * mwo - 27.656 * sgo + 0.070615 * mwo * sgo
    den_co = mwo / vco
    vcg = ZC_PR * R_FIELD * Tcg / Pcg
    den_cg = mwg / vcg
    u_o, xi_o = _dilute_viscosity(Tco, Pco, mwo, T)
    u_g, xi_g = _dilute_viscosity(Tcg, Pcg, mwg, T)
    return LBCParameters(den_co, den_cg, xi_o, xi_g, u_o, u_g)


def regress_lbc(props: Dict[str, np.ndarray], params0: LBCParameters) -> LBCParameters:
    """Tune the six LBC mixing parameters to the table viscosities.

    Minimises the summed relative error of oil and gas viscosity over all nodes,
    matching the global regression of the original notebook.
    """
    xo, xg, yo, yg = props["xo"], props["xg"], props["yo"], props["yg"]
    uo, ug = props["uo"], props["ug"]
    deno, deng = props["deno"], props["deng"]

    def err(z):
        p = LBCParameters(*z, coeffs=params0.coeffs)
        e = 0.0
        for i in range(len(xo)):
            vo = p.phase_viscosity(xo[i], xg[i], deno[i])
            vg = p.phase_viscosity(yo[i], yg[i], deng[i])
            e += abs(vo - uo[i]) / uo[i] + abs(vg - ug[i]) / ug[i]
        return e

    z0 = [params0.den_co, params0.den_cg, params0.xi_o, params0.xi_g,
          params0.u_o, params0.u_g]
    res = minimize(err, z0, method="nelder-mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 10000})
    return LBCParameters(*res.x, coeffs=params0.coeffs)


def lbc_match_error(props: Dict[str, np.ndarray], params: LBCParameters) -> float:
    """Maximum relative viscosity error over all nodes."""
    xo, xg, yo, yg = props["xo"], props["xg"], props["yo"], props["yg"]
    uo, ug = props["uo"], props["ug"]
    deno, deng = props["deno"], props["deng"]
    err = 0.0
    for i in range(len(xo)):
        vo = params.phase_viscosity(xo[i], xg[i], deno[i])
        vg = params.phase_viscosity(yo[i], yg[i], deng[i])
        err = max(err, abs(vo - uo[i]) / uo[i], abs(vg - ug[i]) / ug[i])
    return float(err)
