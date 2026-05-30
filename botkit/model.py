"""Core data structures for the Black-Oil Table toolkit.

A Black-Oil Table is represented as a saturated PVTO branch (oil) and a
saturated PVTG branch (gas), each optionally carrying undersaturated rows per
saturated node.  Surface-fluid properties and the two-pseudocomponent mixing
constants follow Singh & Whitson, SPE 109596 (2007).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

# Universal gas constant in field units, psia-ft^3 / (R-lbmol)
R_FIELD = 10.73146

# Standard molar volume of gas, Mscf/lbmol (~379.5 scf/lbmol at 60 F, 14.696 psia)
GAS_MOLAR_VOLUME_MSCF = 0.3795

# Cubic feet per stock-tank barrel
FT3_PER_BBL = 5.6146

# Density of water, lbm/ft^3, used to convert oil density to specific gravity
WATER_DENSITY = 62.428


def stock_tank_oil_mw(st_oil_density: float) -> float:
    """Estimate stock-tank oil molecular weight from its density.

    Linear API-gravity correlation MW = 240 - 2.22 * API, with
    API = 141.5 / SG - 131.5 and SG referenced to water at 62.428 lbm/ft^3.
    Used as a fallback when a measured oil molecular weight is not supplied.
    """
    api = 141.5 / (st_oil_density / WATER_DENSITY) - 131.5
    return 240.0 - 2.22 * api


@dataclass
class SurfaceFluids:
    """Stock-tank fluid densities and derived mixing constants.

    The mixing constants (Lo, Lg, Mult, Co) convert between the engineering
    black-oil ratios (Rs, Rv) and the two-pseudocomponent mole fractions used
    by the K-value formulation of Singh & Whitson, SPE 109596 (2007), App. A.
    """

    st_oil_density: float      # stock-tank oil density, lbm/ft^3
    st_gas_density: float      # stock-tank gas density, lbm/ft^3
    oil_mw: Optional[float] = None  # measured stock-tank oil MW; correlated if None

    # derived (populated in __post_init__)
    gas_mw: float = field(init=False)
    Lo: float = field(init=False)
    Lg: float = field(init=False)
    mult: float = field(init=False)
    Co: float = field(init=False)

    def __post_init__(self) -> None:
        if self.oil_mw is None:
            self.oil_mw = stock_tank_oil_mw(self.st_oil_density)
        # lbmol of stock-tank oil per barrel
        self.Lo = FT3_PER_BBL * self.st_oil_density / self.oil_mw
        # lbmol of surface gas per Mscf
        self.Lg = 1.0 / GAS_MOLAR_VOLUME_MSCF
        # bbl -> Mscf scaling
        self.mult = 1000.0 / FT3_PER_BBL
        # composite constant relating Rs/Rv to mole fractions (Singh App. A)
        self.Co = self.mult * self.Lo / self.Lg
        # surface-gas molecular weight from its density
        self.gas_mw = self.st_gas_density * 1000.0 / self.Lg


@dataclass
class PVTOTable:
    """Saturated oil branch plus optional undersaturated oil rows.

    Saturated arrays are aligned by index.  ``usat[i]`` holds the undersaturated
    rows for saturated node ``i`` as an (m, 3) array of columns [P, Bo, uo],
    with pressures above the node's saturation pressure ``p[i]``.  An empty
    array means no undersaturated data for that node.
    """

    rs: np.ndarray    # saturated solution GOR, Mscf/bbl
    p: np.ndarray     # saturation (bubble-point) pressure, psia
    bo: np.ndarray    # saturated oil formation volume factor, rb/stb
    uo: np.ndarray    # saturated oil viscosity, cP
    usat: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rs = np.asarray(self.rs, dtype=float)
        self.p = np.asarray(self.p, dtype=float)
        self.bo = np.asarray(self.bo, dtype=float)
        self.uo = np.asarray(self.uo, dtype=float)
        if not self.usat:
            self.usat = [np.empty((0, 3)) for _ in self.p]

    @property
    def n(self) -> int:
        return len(self.p)


@dataclass
class PVTGTable:
    """Saturated gas branch plus optional undersaturated gas rows.

    ``usat[i]`` holds the undersaturated rows for saturated node ``i`` as an
    (m, 3) array of columns [Rv, Bg, ug], with vapourized oil-gas ratios below
    the node's saturated value.
    """

    p: np.ndarray     # dew-point pressure, psia
    rv: np.ndarray    # saturated vapourized oil-gas ratio, bbl/Mscf
    bg: np.ndarray    # saturated gas formation volume factor, rb/Mscf
    ug: np.ndarray    # saturated gas viscosity, cP
    usat: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.p = np.asarray(self.p, dtype=float)
        self.rv = np.asarray(self.rv, dtype=float)
        self.bg = np.asarray(self.bg, dtype=float)
        self.ug = np.asarray(self.ug, dtype=float)
        if not self.usat:
            self.usat = [np.empty((0, 3)) for _ in self.p]

    @property
    def n(self) -> int:
        return len(self.p)


@dataclass
class BlackOilTable:
    """A Black-Oil Table: paired PVTO and PVTG branches and surface fluids."""

    pvto: PVTOTable
    pvtg: PVTGTable
    surface: Optional[SurfaceFluids] = None


# --- Configuration -------------------------------------------------------

AUTO = "auto"  # sentinel: derive the value from the QC detectors


@dataclass
class Config:
    """Workflow configuration.

    Every choice that was an interactive prompt in the original notebook is an
    explicit field here.  ``AUTO`` defers the value to a detector-derived
    default.
    """

    # QC / table-trimming
    saturated_cut = AUTO            # highest shared saturated pressure to keep
    enforce_monotonic_cgr: bool = True
    co_trend_tol: float = 0.5       # flag undersat oil branches whose c_o departs
                                    # from the smooth c_o(psat) trend by > this frac
    manual_replace_pressures: tuple = ()  # pressures whose saturated node is
                                          # replaced by interpolation (refit after)

    # extension
    convergence_pressure_Pk = AUTO  # Singh App. B analytical value if AUTO
    convergence_pressure_nodes: int = 2  # top-N nodes for App. B (2 = canonical)
    first_extrap_node: int = -1     # index from the table end to anchor extrapolation
    n_extension_nodes: int = 15
    n_undersaturated_nodes: int = 10
    output_pressures: tuple = ()    # if set, the output saturated locus is built
                                    # at exactly these pressures (resampled from
                                    # the interpolated + EOS-extended model)
    extrapolate_shift_trend: bool = False  # hold the volume shift flat above the
                                           # table (the Peneloux shift is
                                           # pressure-independent); projecting the
                                           # fitted per-node trend forward
                                           # suppresses the near-critical Bo rise
    shift_trend_points: int = 3            # last-N points for the trend slope when
                                           # extrapolate_shift_trend is True
    oil_shift_abscissa: str = "log"       # transform linearising the oil shift trend
    gas_shift_abscissa: str = "linear"    # transform linearising the gas shift trend
    truncate_at_fold: bool = True         # stop the extension at a near-critical fold

    # EOS
    eos_fallback_tol: float = 0.05  # disable EOS paths if it misses table by >5%
    reservoir_temperature: Optional[float] = None  # deg R; required unless regressed
    regress_temperature: bool = False              # opt-in: treat T as unknown
    shift_smoothness: float = 0.0   # >0 regularizes the volume-shift trends
                                    # (smooth, extrapolatable; small in-sample cost)

    # workflow
    auto_apply_fixes: bool = False  # False = stop for human approval after QC


# --- Diagnostics ---------------------------------------------------------

class Severity(Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass
class Anomaly:
    kind: str
    location: str
    severity: Severity
    message: str
    suggested_fix: str = ""


@dataclass
class Diagnostics:
    anomalies: List[Anomaly] = field(default_factory=list)

    def add(self, anomaly: Anomaly) -> None:
        self.anomalies.append(anomaly)

    def by_severity(self, severity: Severity) -> List[Anomaly]:
        return [a for a in self.anomalies if a.severity is severity]

    def to_dict(self) -> dict:
        return {
            "anomalies": [
                {
                    "kind": a.kind,
                    "location": a.location,
                    "severity": a.severity.value,
                    "message": a.message,
                    "suggested_fix": a.suggested_fix,
                }
                for a in self.anomalies
            ]
        }


@dataclass
class Change:
    """A change the pipeline applied to the table, with its justification."""

    action: str   # what was changed
    reason: str   # why it was changed
    detail: str = ""


@dataclass
class ChangeLog:
    """Ordered record of the fixes applied during a build."""

    changes: List[Change] = field(default_factory=list)

    def add(self, action: str, reason: str, detail: str = "") -> None:
        self.changes.append(Change(action=action, reason=reason, detail=detail))

    def __len__(self) -> int:
        return len(self.changes)

    def to_dict(self) -> dict:
        return {"changes": [
            {"action": c.action, "reason": c.reason, "detail": c.detail}
            for c in self.changes
        ]}
