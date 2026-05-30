"""End-to-end orchestration: QC, fit, extend, fill, and assemble.

The workflow mirrors the original notebook but is configuration-driven and
emits diagnostics instead of interactive prompts:

    QC  ->  resolve cut / Pk  ->  tune EOS + LBC  ->  fallback gate
        ->  extend saturated locus to Pk  ->  fill undersaturated branches
        ->  enforce monotonic CGR (optional)  ->  assemble extended table

When ``Config.auto_apply_fixes`` is False the QC stage stops for human review;
call :func:`build` after inspecting the report (optionally editing the Config).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from . import eos as _eos
from . import viscosity as _visc
from .extend import extend_saturated, extrapolate_kvalues
from .fill import (enforce_oil_branch_monotonic, fill_gas_branch,
                   fill_oil_branch, fill_oil_branch_constant_c,
                   oil_branch_compressibilities, undersaturated_pressures,
                   undersaturated_rv_grid)
from .interpolate import SaturatedInterpolator
from .kvalues import convergence_pressure, kvalues, phase_properties
from .model import (AUTO, Anomaly, BlackOilTable, ChangeLog, Config,
                    Diagnostics, PVTGTable, PVTOTable, Severity)
from .qc import _shared_locus, detect_undersaturated_compressibility, run_qc


@dataclass
class FitResult:
    """Tuned models and resolved settings for the trusted (trimmed) locus."""

    trusted: Dict[str, np.ndarray]
    props: Dict[str, np.ndarray]
    params: _eos.EOSParameters
    lbc: _visc.LBCParameters
    so_tab: np.ndarray
    sg_tab: np.ndarray
    Pk: float
    cut: float
    eos_pressure_error: float
    eos_trusted: bool


@dataclass
class PipelineResult:
    diagnostics: Diagnostics
    suggestions: Dict[str, float]
    fit: Optional[FitResult] = None
    extended: Optional[BlackOilTable] = None
    info: Dict[str, object] = field(default_factory=dict)
    changes: ChangeLog = field(default_factory=ChangeLog)


def _resolve(value, auto_value):
    return auto_value if value is AUTO else value


def qc_stage(table: BlackOilTable, config: Config) -> PipelineResult:
    """Run QC and return diagnostics plus suggestions (no table changes)."""
    diag, sug = run_qc(table, cut=_resolve(config.saturated_cut, None),
                       enforce_monotonic_cgr=config.enforce_monotonic_cgr)
    return PipelineResult(diagnostics=diag, suggestions=sug)


def fit(table: BlackOilTable, config: Config, diag: Optional[Diagnostics] = None
        ) -> FitResult:
    """Trim to the trusted locus and tune the EOS + LBC models on it."""
    if table.surface is None:
        raise ValueError("fit requires surface fluids on the table")
    s = table.surface
    diag = diag if diag is not None else Diagnostics()

    loc = _shared_locus(table)
    cut = _resolve(config.saturated_cut, float(max(loc["p"])))
    m = loc["p"] <= cut * 1.0001
    trusted = {k: v[m] for k, v in loc.items()}

    kv = kvalues(trusted["rs"], trusted["rv"], s)
    props = {
        "p": trusted["p"], "uo": trusted["uo"], "ug": trusted["ug"],
        **kv,
    }
    mwo = s.oil_mw * kv["xo"] + s.gas_mw * kv["xg"]
    mwg = s.oil_mw * kv["yo"] + s.gas_mw * kv["yg"]
    props["deno"] = (s.st_oil_density + s.st_gas_density * trusted["rs"] * s.mult) / trusted["bo"]
    props["deng"] = (s.st_gas_density + s.st_oil_density * (trusted["rv"] / s.mult)) / (trusted["bg"] / s.mult)
    props["vo"] = mwo / props["deno"]
    props["vg"] = mwg / props["deng"]

    # Singh App. B convergence pressure on the trusted (trimmed) locus only
    Pk = _resolve(config.convergence_pressure_Pk,
                  convergence_pressure(trusted["p"], kv["ko"], kv["kg"],
                                       n_nodes=config.convergence_pressure_nodes))

    T = config.reservoir_temperature if config.reservoir_temperature else 680.0
    params0 = _eos.initial_parameters(s, T)
    params, _, _ = _eos.tune_global(props, params0,
                                    regress_temperature=config.regress_temperature)

    # Judge the EOS itself on the exact per-node fit, so the fallback gate is
    # independent of any shift smoothing chosen for the extension.
    so_exact, sg_exact = _eos.tune_local_shifts(props, params, smoothness=0.0)
    err = _eos.eos_pressure_match_error(props, params, so_exact, sg_exact, top_n=6)
    eos_trusted = err <= config.eos_fallback_tol

    # Working shifts: optionally smoothed for confident interpolation/extrapolation.
    if config.shift_smoothness > 0:
        so_tab, sg_tab = _eos.tune_local_shifts(props, params,
                                                smoothness=config.shift_smoothness)
    else:
        so_tab, sg_tab = so_exact, sg_exact
    if not eos_trusted:
        diag.add(Anomaly(
            kind="eos_fallback",
            location="extension",
            severity=Severity.WARN,
            message=(f"Tuned EOS misses the upper-locus pressures by "
                     f"{err:.1%} (> {config.eos_fallback_tol:.0%}); EOS-based "
                     f"regeneration is disabled in favour of K-value/log-log "
                     f"extrapolation."),
            suggested_fix="Review the table consistency or relax eos_fallback_tol.",
        ))

    # tune LBC (used for viscosity regeneration)
    Tco, Pco = _eos.ab_oil(s.oil_mw, s.st_oil_density, params.T)[2:]
    Tcg, Pcg = _eos.ab_gas(s.gas_mw, s.st_gas_density, params.T)[2:]
    lbc0 = _visc.initial_lbc(s, params.T, Tco, Pco, Tcg, Pcg)
    lbc = _visc.regress_lbc(props, lbc0)

    return FitResult(trusted=trusted, props=props, params=params, lbc=lbc,
                     so_tab=so_tab, sg_tab=sg_tab, Pk=Pk, cut=cut,
                     eos_pressure_error=err, eos_trusted=eos_trusted)


def build(table: BlackOilTable, config: Config,
          result: Optional[PipelineResult] = None) -> PipelineResult:
    """Full build: fit, extend to Pk, fill undersaturated branches, assemble."""
    if result is None:
        result = qc_stage(table, config)
    diag = result.diagnostics
    s = table.surface

    fr = fit(table, config, diag)
    result.fit = fr
    trusted, props = fr.trusted, fr.props
    cl = result.changes

    # record the trim of the untrusted tail (rows above the cut in either branch)
    n_o_drop = int(np.sum(table.pvto.p > fr.cut * 1.0001))
    n_g_drop = int(np.sum(table.pvtg.p > fr.cut * 1.0001))
    if n_o_drop or n_g_drop:
        cl.add(
            action=f"Discarded {n_o_drop} PVTO and {n_g_drop} PVTG saturated "
                   f"row(s) above {fr.cut:g} psia, keeping a "
                   f"{len(trusted['p'])}-node trusted locus.",
            reason="PVTO and PVTG saturated pressures diverge above this point, "
                   "which indicates a prior non-equilibrium extension; only the "
                   "trusted shared locus is retained.",
        )
    if not fr.eos_trusted:
        cl.add(
            action="EOS-based regeneration flagged for fallback.",
            reason=f"The tuned EOS misses the upper-locus pressures by "
                   f"{fr.eos_pressure_error:.1%}, above the {config.eos_fallback_tol:.0%} "
                   f"tolerance; treat the EOS-generated data with caution.",
        )

    # volume-shift interpolators. Above the table the shifts either follow their
    # local trend (anchored at the last valid point, no offset) or hold flat.
    psat = trusted["p"]
    if config.extrapolate_shift_trend:
        so_interp, _ = _eos.trend_extrapolator(psat, fr.so_tab,
                                               config.shift_trend_points,
                                               transform=config.oil_shift_abscissa)
        sg_interp, _ = _eos.trend_extrapolator(psat, fr.sg_tab,
                                               config.shift_trend_points,
                                               transform=config.gas_shift_abscissa)
    else:
        def so_interp(p):
            return np.interp(p, psat, fr.so_tab)
        def sg_interp(p):
            return np.interp(p, psat, fr.sg_tab)

    # extrapolate K-values and regenerate saturated properties to Pk
    p_ext, ko_ext, kg_ext = extrapolate_kvalues(
        psat, props["ko"], props["kg"], fr.Pk, config.n_extension_nodes,
        anchor=config.first_extrap_node)
    ext = extend_saturated(p_ext, ko_ext, kg_ext, fr.params, fr.lbc, s,
                           max_psat=float(psat.max()),
                           so_interp=so_interp, sg_interp=sg_interp)
    if ext["folded_at"] is not None:
        fold = ext["folded_at"]
        diag.add(Anomaly(
            kind="eos_fold",
            location=f"extension node {fold} (P~{p_ext[fold]:.0f} psia)",
            severity=Severity.WARN,
            message="Bo or Bg reversed during EOS extension - a near-critical "
                    "fold the smooth extrapolation cannot represent. The "
                    "extension is truncated below the fold.",
            suggested_fix="Lower the convergence pressure to extend smoothly to "
                          "the critical region without folding.",
        ))
        if config.truncate_at_fold:
            cl.add(
                action=f"Truncated the EOS extension at ~{p_ext[fold]:.0f} psia.",
                reason="Bo or Bg reversed (a near-critical fold) that the smooth "
                       "K-value extrapolation cannot represent; nodes beyond the "
                       "fold are dropped.",
            )
            for k in ("p", "rs", "rv", "bo", "bg", "uo", "ug"):
                ext[k] = ext[k][:fold]

    pk_source = ("Singh App. B analytical estimate"
                 if config.convergence_pressure_Pk is AUTO else "user-specified value")
    cl.add(
        action=f"Extended the saturated tables to a convergence pressure "
               f"Pk = {fr.Pk:g} psia ({len(ext['p'])} extension node(s)).",
        reason=f"Convergence pressure taken from the {pk_source}; saturated "
               f"properties above the table are regenerated from the tuned "
               f"PR79 + Peneloux EOS and LBC viscosity.",
    )

    # assemble the extended saturated locus: trusted rows up to the anchor,
    # then the regenerated extension
    anchor = config.first_extrap_node
    keep = slice(0, len(psat) + anchor if anchor < 0 else anchor)
    sat_p = np.concatenate([psat[keep], ext["p"]])
    sat_rs = np.concatenate([trusted["rs"][keep], ext["rs"]])
    sat_rv = np.concatenate([trusted["rv"][keep], ext["rv"]])
    sat_bo = np.concatenate([trusted["bo"][keep], ext["bo"]])
    sat_bg = np.concatenate([trusted["bg"][keep], ext["bg"]])
    sat_uo = np.concatenate([trusted["uo"][keep], ext["uo"]])
    sat_ug = np.concatenate([trusted["ug"][keep], ext["ug"]])

    # optional simulator-compliance fix: truncate non-monotonic low-P Rv
    if config.enforce_monotonic_cgr and "cgr_floor" in result.suggestions:
        floor = result.suggestions["cgr_floor"]
        i_min = int(np.argmin(sat_rv))
        if i_min > 0:
            cl.add(
                action=f"Flattened saturated Rv at {i_min} node(s) below "
                       f"{sat_p[i_min]:g} psia to the retrograde minimum "
                       f"{floor:g} and removed their undersaturated gas lines.",
                reason="The low-pressure rise in Rv is physically real but most "
                       "commercial simulators require monotonic saturated Rv "
                       "(enforce_monotonic_cgr).",
            )
        sat_rv[:i_min] = floor

    # fill undersaturated branches. Each branch is at fixed composition, so its
    # volume shift is held constant at the node's value (interpolated within the
    # table, trend-extrapolated above it) - this gives a physical undersaturated
    # compressibility rather than the spurious shape the saturated-locus shift
    # trend would inject at constant composition.
    p_grid = undersaturated_pressures(float(sat_p.min()), fr.Pk,
                                      config.n_undersaturated_nodes)
    rv_grid = undersaturated_rv_grid(float(sat_rv.min()), float(sat_rv.max()),
                                     config.n_undersaturated_nodes)
    so_node = np.asarray(so_interp(sat_p), dtype=float)
    sg_node = np.asarray(sg_interp(sat_p), dtype=float)

    # first pass: EOS oil branches and their (c_o, c_mu). Branches whose
    # compressibility is non-physically steep (the low-pressure blind spot) are
    # flagged against the table median; the reliable branches then define a
    # c_o(p) / c_mu(p) trend that supplies a locally interpolated compressibility
    # for each flagged node rather than a single table-wide average.
    oil_eos, co_list, cmu_list = [], [], []
    for i in range(len(sat_p)):
        rows = fill_oil_branch(sat_p[i], sat_rs[i], sat_bo[i], sat_uo[i],
                               fr.params, fr.lbc, s, p_grid,
                               so_node[i], sg_node[i])[1:]  # drop saturated dup
        oil_eos.append(rows)
        c_o, c_mu = oil_branch_compressibilities(rows, sat_bo[i], sat_uo[i], sat_p[i])
        co_list.append(c_o)
        cmu_list.append(c_mu)

    valid = [(sat_p[i], co_list[i], cmu_list[i]) for i in range(len(sat_p))
             if co_list[i] is not None and co_list[i] > 0]
    median_co = float(np.median([c for _, c, _ in valid])) if valid else 0.0
    factor = config.hybrid_co_factor
    reliable = [(p, c, cm) for (p, c, cm) in valid if c <= factor * median_co]
    rel_p = np.array([p for p, _, _ in reliable])
    rel_co = np.array([c for _, c, _ in reliable])
    rel_cmu = np.array([cm for _, _, cm in reliable])

    o_usat, g_usat = [], []
    regenerated, enforced = [], 0
    for i in range(len(sat_p)):
        oil_rows = oil_eos[i]
        c_o = co_list[i]
        is_blind_spot = (config.hybrid_undersaturated_oil and oil_rows.shape[0] > 0
                         and c_o is not None and median_co > 0
                         and c_o > factor * median_co and rel_p.size >= 1)
        if is_blind_spot:
            # compressibility interpolated from the surrounding reliable nodes
            # (np.interp clamps to the nearest reliable node beyond their range)
            c_o_use = float(np.interp(sat_p[i], rel_p, rel_co))
            c_mu_use = float(np.interp(sat_p[i], rel_p, rel_cmu))
            oil_rows = fill_oil_branch_constant_c(sat_p[i], sat_bo[i], sat_uo[i],
                                                  p_grid, c_o_use, c_mu_use)
            diag.add(Anomaly(
                kind="undersaturated_regenerated",
                location=f"undersaturated oil branch at Psat = {sat_p[i]:g} psia",
                severity=Severity.INFO,
                message="EOS undersaturated oil compressibility was non-physically "
                        "steep (low-pressure blind spot); regenerated by the "
                        "reciprocal-family law with c_o interpolated from the "
                        f"surrounding reliable nodes (c_o={c_o_use:.2e}/psi).",
                suggested_fix="",
            ))
            regenerated.append(sat_p[i])
        if config.enforce_undersaturated_monotonic and oil_rows.shape[0] > 0:
            fixed = enforce_oil_branch_monotonic(oil_rows, sat_bo[i], sat_uo[i])
            if not np.array_equal(fixed, oil_rows):
                enforced += 1
            oil_rows = fixed
        o_usat.append(oil_rows)
        if config.enforce_monotonic_cgr and i < int(np.argmin(sat_rv)):
            g_usat.append(np.empty((0, 3)))  # truncated nodes carry no usat gas
        else:
            g_usat.append(fill_gas_branch(
                sat_p[i], sat_rv[i], sat_bg[i], sat_ug[i], fr.params, fr.lbc, s,
                rv_grid, so_node[i], sg_node[i])[1:])

    if regenerated:
        nodes = ", ".join(f"{p:g}" for p in regenerated)
        cl.add(
            action=f"Regenerated {len(regenerated)} undersaturated oil branch(es) "
                   f"(Psat = {nodes} psia) by the reciprocal-family law.",
            reason="The two-component EOS is unreliable in the low-pressure blind "
                   "spot and produced a non-physically steep compressibility; the "
                   "branches were rebuilt with c_o interpolated from the "
                   "surrounding reliable nodes.",
        )
    if enforced:
        cl.add(
            action=f"Enforced monotonicity on {enforced} undersaturated oil "
                   f"branch(es) (Bo non-increasing, uo non-decreasing).",
            reason="Above the bubble point oil compresses; any residual "
                   "wrong-direction step is clipped for a deck-legal, physical "
                   "branch (enforce_undersaturated_monotonic).",
        )

    pvto = PVTOTable(rs=sat_rs, p=sat_p, bo=sat_bo, uo=sat_uo, usat=o_usat)
    pvtg = PVTGTable(p=sat_p, rv=sat_rv, bg=sat_bg, ug=sat_ug, usat=g_usat)
    result.extended = BlackOilTable(pvto=pvto, pvtg=pvtg, surface=s)

    # output-side QC: flag any generated undersaturated rows that came out
    # non-physical (e.g. the low-pressure oil branch in the EOS blind spot) so
    # they are surfaced for review rather than silently emitted.
    detect_undersaturated_compressibility(result.extended, diag)

    # honour-the-data interpolator over the trusted locus (for lookup/QC/plots)
    result.info["interpolator"] = SaturatedInterpolator(
        trusted["p"], trusted["rs"], trusted["rv"], trusted["bo"],
        trusted["bg"], trusted["uo"], trusted["ug"], s)
    result.info.update({"Pk": fr.Pk, "cut": fr.cut,
                        "eos_pressure_error": fr.eos_pressure_error,
                        "eos_trusted": fr.eos_trusted,
                        "n_saturated": len(sat_p)})
    return result


def write_deck(result: PipelineResult, path: Optional[str] = None,
               title: str = "Black-Oil Table extended to convergence pressure") -> str:
    """Write the extended Eclipse deck with the change summary in its header.

    The header records the method and the corrective changes (with reasons) that
    were applied, so the rationale travels with the deck.
    """
    from .io import write_eclipse
    from .report import changes_to_text

    if result.extended is None:
        raise ValueError("no extended table to write; run build() first")
    header = (f"{title}\n"
              f"Method: Singh & Whitson, SPE 109596 (2007)\n"
              f"Pk = {result.info.get('Pk', '?'):g} psia; "
              f"trusted cut = {result.info.get('cut', '?'):g} psia\n\n"
              f"{changes_to_text(result.changes)}")
    return write_eclipse(result.extended, path, header=header)


def run(table: BlackOilTable, config: Optional[Config] = None) -> PipelineResult:
    """Convenience entry point.

    Always runs QC.  If ``Config.auto_apply_fixes`` is True the full build runs;
    otherwise only the QC stage runs and the caller inspects the report before
    calling :func:`build`.
    """
    config = config or Config()
    result = qc_stage(table, config)
    if config.auto_apply_fixes:
        result = build(table, config, result)
    return result
