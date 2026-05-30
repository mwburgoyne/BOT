"""Automated quality-control detectors for Black-Oil Tables.

Each detector inspects the table and appends :class:`Anomaly` records to a
:class:`Diagnostics` object; none mutate the table.  The detectors replace the
plot-eyeballing and interactive point-removal of the original notebook:

  * pressure misalignment between PVTO and PVTG (a prior bad extension),
  * monotonicity of Rs, Bo, Bg and the retrograde Rv minimum,
  * saturated-compressibility sign / derivative discontinuities,
  * Bo-Rs linearity of the saturated oil branch,
  * undersaturated oil-viscosity linearity in log pressure,
  * low-pressure CGR (Rv) reversal that Eclipse rejects.

The derivative / saturated-compressibility check follows the black-oil
consistency relations of Singh & Whitson, SPE 109596 (2007); negative saturated
compressibility as a marker of a missing or corrupt node is a standard
Whitson-school diagnostic.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .model import Anomaly, BlackOilTable, Diagnostics, Severity


def shared_saturated_pressure(table: BlackOilTable,
                              rtol: float = 1e-4) -> float:
    """Highest pressure present in both the PVTO and PVTG saturated branches.

    The last shared saturated pressure is the originally-tested saturated
    fluid; pressures above it that appear in only one branch usually mark a
    prior non-equilibrium extension.
    """
    po = table.pvto.p
    pg = table.pvtg.p
    shared = [p for p in po if np.any(np.isclose(p, pg, rtol=rtol))]
    if not shared:
        raise ValueError("PVTO and PVTG share no saturated pressures")
    return float(max(shared))


def _shared_locus(table: BlackOilTable, rtol: float = 1e-4) -> Dict[str, np.ndarray]:
    """Saturated rows on pressures shared by both branches, sorted by pressure."""
    o, g = table.pvto, table.pvtg
    rows = []
    for i, p in enumerate(o.p):
        j = np.where(np.isclose(p, g.p, rtol=rtol))[0]
        if j.size:
            k = j[0]
            rows.append((p, o.rs[i], g.rv[k], o.bo[i], g.bg[k], o.uo[i], g.ug[k]))
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows)
    keys = ["p", "rs", "rv", "bo", "bg", "uo", "ug"]
    return {k: arr[:, i] for i, k in enumerate(keys)}


def detect_pressure_misalignment(table: BlackOilTable, diag: Diagnostics,
                                 rtol: float = 1e-4) -> float:
    """Flag saturated rows above the last shared PVTO/PVTG pressure."""
    cut = shared_saturated_pressure(table, rtol)
    n_o = int(np.sum(table.pvto.p > cut * (1 + rtol)))
    n_g = int(np.sum(table.pvtg.p > cut * (1 + rtol)))
    if n_o or n_g:
        diag.add(Anomaly(
            kind="pressure_misalignment",
            location=f"P > {cut:g} psia",
            severity=Severity.WARN,
            message=(f"PVTO and PVTG saturated pressures diverge above "
                     f"{cut:g} psia ({n_o} oil, {n_g} gas rows). This usually "
                     f"marks a prior non-equilibrium extension."),
            suggested_fix=f"Discard saturated rows above {cut:g} psia "
                          f"(set Config.saturated_cut={cut:g}).",
        ))
    return cut


def _flag_non_monotonic(name: str, p: np.ndarray, y: np.ndarray,
                        increasing: bool, diag: Diagnostics) -> None:
    dy = np.diff(y)
    bad = dy <= 0 if increasing else dy >= 0
    if np.any(bad):
        idx = np.where(bad)[0]
        spans = ", ".join(f"{p[i]:g}->{p[i+1]:g}" for i in idx[:6])
        diag.add(Anomaly(
            kind="monotonicity",
            location=f"{name} at P = {spans}",
            severity=Severity.WARN,
            message=(f"Saturated {name} is not {'increasing' if increasing else 'decreasing'} "
                     f"with pressure at {len(idx)} interval(s)."),
            suggested_fix="Inspect/smooth the flagged rows or trim the misaligned tail.",
        ))


def detect_monotonicity(table: BlackOilTable, diag: Diagnostics,
                        cut: Optional[float] = None) -> None:
    """Check saturated Rs, Bo (increasing) and Bg (decreasing) with pressure."""
    loc = _shared_locus(table)
    p = loc["p"]
    if cut is not None:
        m = p <= cut * 1.0001
        loc = {k: v[m] for k, v in loc.items()}
        p = loc["p"]
    if len(p) < 3:
        return
    _flag_non_monotonic("Rs", p, loc["rs"], increasing=True, diag=diag)
    _flag_non_monotonic("Bo", p, loc["bo"], increasing=True, diag=diag)
    _flag_non_monotonic("Bg", p, loc["bg"], increasing=False, diag=diag)


def detect_saturated_compressibility(table: BlackOilTable, diag: Diagnostics,
                                     cut: Optional[float] = None,
                                     jump_factor: float = 5.0) -> Dict[str, np.ndarray]:
    """Flag negative or discontinuous saturated total compressibilities.

    Uses the saturated oil/gas total-compressibility relations (Singh &
    Whitson, SPE 109596, consistency form).  A sign change to negative, or a
    jump larger than ``jump_factor`` times the neighbouring magnitude, signals
    a corrupt or missing saturated node.
    """
    loc = _shared_locus(table)
    p = loc["p"]
    if cut is not None:
        m = p <= cut * 1.0001
        loc = {k: v[m] for k, v in loc.items()}
        p = loc["p"]
    if len(p) < 3:
        return {}

    rs, rv, bo, bg = loc["rs"], loc["rv"], loc["bo"], loc["bg"]
    dp = np.diff(p)
    dBo, dBg = np.diff(bo) / dp, np.diff(bg) / dp
    dRs, dRv = np.diff(rs) / dp, np.diff(rv) / dp
    denom = 1.0 - rs[1:] * rv[1:]
    tc_oil = 1.0 / bo[1:] * (-dBo + dRs * (bg[1:] - rv[1:] * bo[1:]) / denom)
    tc_gas = 1.0 / bg[1:] * (-dBg + dRv * (bo[1:] - rs[1:] * bg[1:]) / denom)

    for name, tc in (("oil", tc_oil), ("gas", tc_gas)):
        neg = np.where(tc < 0)[0]
        if neg.size:
            ps = ", ".join(f"{p[i+1]:g}" for i in neg[:6])
            diag.add(Anomaly(
                kind="negative_saturated_compressibility",
                location=f"{name} compressibility at P = {ps} psia",
                severity=Severity.ERROR,
                message=(f"Saturated {name} total compressibility is negative at "
                         f"{neg.size} node(s) - a derivative discontinuity "
                         f"indicating a corrupt or missing node."),
                suggested_fix="Trim the misaligned tail or insert a saturated "
                              "node to restore a smooth derivative.",
            ))
        # discontinuity: large relative jumps
        mag = np.abs(tc)
        if len(mag) >= 3:
            jumps = np.where(mag[1:] > jump_factor * np.maximum(mag[:-1], 1e-12))[0]
            if jumps.size:
                ps = ", ".join(f"{p[i+2]:g}" for i in jumps[:6])
                diag.add(Anomaly(
                    kind="compressibility_discontinuity",
                    location=f"{name} compressibility at P = {ps} psia",
                    severity=Severity.WARN,
                    message=(f"Saturated {name} compressibility jumps by more "
                             f"than {jump_factor:g}x at {jumps.size} node(s)."),
                    suggested_fix="Inspect the flagged rows for a step change.",
                ))

    # ordering: gas is more compressible than oil, converging to equality near
    # the critical / convergence point. Flag nodes (away from the top of the
    # locus) where c_g,sat < c_o,sat with both positive.
    both_pos = (tc_oil > 0) & (tc_gas > 0)
    interior = np.arange(len(tc_oil)) < len(tc_oil) - 1  # exclude the top node
    bad = np.where(both_pos & interior & (tc_gas < tc_oil))[0]
    if bad.size:
        ps = ", ".join(f"{p[i+1]:g}" for i in bad[:6])
        diag.add(Anomaly(
            kind="compressibility_ordering",
            location=f"P = {ps} psia",
            severity=Severity.WARN,
            message=(f"Saturated gas total compressibility is below the oil "
                     f"value at {bad.size} interior node(s), which is "
                     f"non-physical away from the critical point."),
            suggested_fix="Review the flagged rows; near-equality is only "
                          "expected approaching the convergence pressure.",
        ))
    return {"p": p[1:], "tc_oil": tc_oil, "tc_gas": tc_gas}


def detect_undersaturated_compressibility(table: BlackOilTable,
                                          diag: Diagnostics) -> None:
    """Flag non-positive undersaturated oil compressibility.

    Along an undersaturated oil branch (constant Rs, rising pressure) the
    isothermal compressibility c_o = -(1/Bo) dBo/dp must stay positive; a
    non-positive value marks a corrupt undersaturated row.
    """
    o = table.pvto
    for i, rows in enumerate(o.usat):
        if rows.shape[0] < 1:
            continue
        # include the saturated anchor so the saturated->first-undersaturated
        # step (Bo must fall as pressure rises above the bubble point) is checked
        anchor = np.array([[o.p[i], o.bo[i], o.uo[i]]])
        rows = np.vstack([anchor, rows])
        p, bo = rows[:, 0], rows[:, 1]
        order = np.argsort(p)
        p, bo = p[order], bo[order]
        c_o = -np.diff(bo) / np.diff(p) / bo[1:]
        # flag only genuinely wrong-direction (rising Bo) rows; an exactly flat
        # segment from monotonic enforcement is Eclipse-valid and not flagged
        bad = np.where(c_o < -1e-9)[0]
        if bad.size:
            diag.add(Anomaly(
                kind="negative_undersaturated_compressibility",
                location=f"undersaturated oil branch at Psat = {o.p[i]:g} psia",
                severity=Severity.ERROR,
                message=(f"Undersaturated oil Bo rises with pressure at "
                         f"{bad.size} row(s) (it must fall above the bubble point)."),
                suggested_fix="Enable enforce_undersaturated_monotonic or "
                              "regenerate this branch by interpolation.",
            ))


def detect_bo_rs_linearity(table: BlackOilTable, diag: Diagnostics,
                           cut: Optional[float] = None,
                           resid_tol: float = 0.02) -> None:
    """Flag saturated rows whose Bo departs from the Bo-Rs trend.

    Saturated Bo is closely linear in Rs; a row with relative residual above
    ``resid_tol`` against the least-squares Bo(Rs) line is flagged.  (Standing
    oil-shrinkage consistency heuristic.)
    """
    o = table.pvto
    rs, bo, p = o.rs, o.bo, o.p
    if cut is not None:
        m = p <= cut * 1.0001
        rs, bo, p = rs[m], bo[m], p[m]
    if len(rs) < 3:
        return
    slope, intercept = np.polyfit(rs, bo, 1)
    fit = slope * rs + intercept
    resid = np.abs(bo - fit) / bo
    bad = np.where(resid > resid_tol)[0]
    if bad.size:
        ps = ", ".join(f"{p[i]:g}" for i in bad[:6])
        diag.add(Anomaly(
            kind="bo_rs_linearity",
            location=f"P = {ps} psia",
            severity=Severity.INFO,
            message=(f"Saturated Bo departs from the Bo-Rs line by > "
                     f"{resid_tol:.0%} at {bad.size} row(s) (max "
                     f"{resid.max():.1%})."),
            suggested_fix="Review the flagged rows for measurement error.",
        ))


def detect_undersat_viscosity_loglog(table: BlackOilTable, diag: Diagnostics,
                                     resid_tol: float = 0.03) -> None:
    """Flag undersaturated oil viscosity that departs from log-p linearity."""
    o = table.pvto
    for i, rows in enumerate(o.usat):
        if rows.shape[0] < 3:
            continue
        p, uo = rows[:, 0], rows[:, 2]
        if np.any(uo <= 0):
            continue
        slope, intercept = np.polyfit(np.log(p), np.log(uo), 1)
        fit = np.exp(slope * np.log(p) + intercept)
        resid = np.abs(uo - fit) / uo
        if np.any(resid > resid_tol):
            diag.add(Anomaly(
                kind="undersat_viscosity_loglog",
                location=f"undersaturated oil branch at Psat = {o.p[i]:g} psia",
                severity=Severity.INFO,
                message=(f"Undersaturated oil viscosity departs from log-p "
                         f"linearity (max {resid.max():.1%})."),
                suggested_fix="Review the undersaturated viscosity rows.",
            ))


def detect_cgr_reversal(table: BlackOilTable, diag: Diagnostics,
                        cut: Optional[float] = None,
                        enforce_monotonic: bool = True) -> Optional[float]:
    """Report a low-pressure rise in saturated Rv (CGR reversal).

    Saturated Rv falls to a retrograde minimum then rises; below that minimum
    the rising Rv with falling pressure is *physically real* (a genuine
    retrograde feature), but most commercial simulators require monotonic
    saturated Rv and reject it.

    The reversal is always reported.  Its severity and suggested action depend
    on ``enforce_monotonic``: when True the rows will be truncated to the
    minimum for simulator compliance; when False they are retained as the more
    physically representative form (and may be rejected by some simulators).
    Returns the minimum Rv (the truncation floor) when a reversal is present.
    """
    loc = _shared_locus(table)
    p, rv = loc["p"], loc["rv"]
    if cut is not None:
        m = p <= cut * 1.0001
        p, rv = p[m], rv[m]
    if len(rv) < 3:
        return None
    i_min = int(np.argmin(rv))
    if i_min > 0:  # there are points below the minimum on the low-pressure side
        if enforce_monotonic:
            severity = Severity.WARN
            fix = (f"Truncate saturated Rv below {p[i_min]:g} psia to the "
                   f"minimum {rv[i_min]:g} and drop their undersaturated rows "
                   f"(enforce_monotonic_cgr=True).")
        else:
            severity = Severity.INFO
            fix = ("Retained as physically representative "
                   "(enforce_monotonic_cgr=False); note some simulators reject "
                   "non-monotonic saturated Rv.")
        diag.add(Anomaly(
            kind="cgr_reversal",
            location=f"P <= {p[i_min]:g} psia",
            severity=severity,
            message=(f"Saturated Rv reaches its minimum {rv[i_min]:g} at "
                     f"{p[i_min]:g} psia; {i_min} lower-pressure node(s) have "
                     f"higher Rv. This is a real retrograde feature but most "
                     f"commercial simulators require monotonic saturated Rv."),
            suggested_fix=fix,
        ))
        return float(rv[i_min])
    return None


def run_qc(table: BlackOilTable, cut: Optional[float] = None,
           enforce_monotonic_cgr: bool = True
           ) -> Tuple[Diagnostics, Dict[str, float]]:
    """Run all detectors and return diagnostics plus derived suggestions.

    ``cut`` overrides the auto-detected saturated-cut pressure.
    ``enforce_monotonic_cgr`` controls how a CGR reversal is framed and whether
    a truncation floor is suggested.  The returned suggestions dict carries the
    recommended ``saturated_cut`` and, when enforcing, the ``cgr_floor``
    (minimum Rv) for the monotonicity fix.
    """
    diag = Diagnostics()
    detected_cut = detect_pressure_misalignment(table, diag)
    use_cut = cut if cut is not None else detected_cut
    detect_monotonicity(table, diag, cut=use_cut)
    detect_saturated_compressibility(table, diag, cut=use_cut)
    detect_undersaturated_compressibility(table, diag)
    detect_bo_rs_linearity(table, diag, cut=use_cut)
    detect_undersat_viscosity_loglog(table, diag)
    cgr_floor = detect_cgr_reversal(table, diag, cut=use_cut,
                                    enforce_monotonic=enforce_monotonic_cgr)

    suggestions: Dict[str, float] = {"saturated_cut": use_cut}
    if cgr_floor is not None and enforce_monotonic_cgr:
        suggestions["cgr_floor"] = cgr_floor
    return diag, suggestions
