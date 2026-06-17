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
from .extend_low import extend_below_pmin
from .fill import (fill_gas_branch, fill_oil_branch, undersaturated_pressures,
                   undersaturated_rv_grid)
from .interpolate import SaturatedInterpolator
from .kvalues import (convergence_pressure, convergence_pressure_crossing,
                      kvalues, phase_properties)
from .model import (AUTO, Anomaly, BlackOilTable, ChangeLog, Config,
                    Diagnostics, PVTGTable, PVTOTable, Severity)
from .qc import (_shared_locus, detect_undersaturated_compressibility,
                 detect_undersaturated_compressibility_trend,
                 discontinuous_above_cut, loo_predict, run_qc)


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
    repairs: list = field(default_factory=list)
    den_co_fn: object = None  # extrapolator for the oil critical density vs p
    den_cg_fn: object = None  # extrapolator for the gas critical density vs p


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

    # optional repair of user-named spurious saturated points: replace the node's
    # values with the leave-one-out interpolation through its neighbours, then let
    # the EOS refit to the corrected locus.
    repairs = []
    for pr in config.manual_replace_pressures:
        j = int(np.argmin(np.abs(trusted["p"] - pr)))
        for key in ("rs", "rv", "bo", "bg", "uo", "ug"):
            old = float(trusted[key][j])
            trusted[key][j] = loo_predict(trusted["p"], trusted[key], j)
            repairs.append((float(trusted["p"][j]), key, old,
                            float(trusted[key][j])))

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

    # Convergence pressure on the trusted (trimmed) locus only. The default
    # "crossing" method locates the single Pk where the oil/gas average MWs meet
    # (two-component criticality); it falls back to the Singh App. B two-K-root
    # average when the crossing geometry is degenerate, which is also the explicit
    # "singh" choice.
    pk_singh = convergence_pressure(trusted["p"], kv["ko"], kv["kg"],
                                    n_nodes=config.convergence_pressure_nodes)
    pk_auto = pk_singh
    if config.pk_method == "crossing":
        pk_cross = convergence_pressure_crossing(trusted["p"], mwo, mwg)
        if np.isfinite(pk_cross):
            pk_auto = pk_cross
        else:
            diag.add(Anomaly(
                kind="pk_crossing_fallback",
                location="convergence pressure",
                severity=Severity.INFO,
                message=("The oil/gas average-MW crossing was degenerate (branches "
                         "not converging above the table); the Singh App. B "
                         "two-root average was used for Pk instead."),
                suggested_fix="Inspect the top-node K-value trend, or set "
                              "Config.convergence_pressure_Pk explicitly.",
            ))
    Pk = _resolve(config.convergence_pressure_Pk, pk_auto)

    if config.reservoir_temperature:
        T = config.reservoir_temperature
    else:
        T = 680.0  # deg R (220 F) default when neither supplied nor regressed
        if not config.regress_temperature:
            diag.add(Anomaly(
                kind="assumed_temperature",
                location="EOS regression",
                severity=Severity.WARN,
                message=f"No reservoir temperature supplied; assuming {T:.0f} deg R "
                        f"({T - 460:.0f} deg F). Set Config.reservoir_temperature, "
                        f"or regress_temperature=True, for a fluid-specific value.",
                suggested_fix="Provide Config.reservoir_temperature (deg R).",
            ))
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

    # Per-node viscosity critical densities (VcVis, as rho_c = M/Vc_vis)
    # reproducing the observed viscosities, with the ill-conditioned low-pressure
    # nodes dropped. VcVis is a component property, so above the table it is held
    # flat at the last reliable value (reproducing the observed viscosity at the
    # join); the viscosity then varies through the EOS phase density and
    # composition. The per-node trend can optionally be extrapolated instead.
    den_co, den_cg, ok = _visc.regress_node_densities(props, lbc,
                                                      tol=config.vc_reliability_tol)
    pr = props["p"][ok]
    if pr.size >= 2 and config.extrapolate_vc_trend:
        den_co_fn, _ = _eos.trend_extrapolator(pr, den_co[ok], 4, config.oil_vc_abscissa)
        den_cg_fn, _ = _eos.trend_extrapolator(pr, den_cg[ok], 4, config.gas_vc_abscissa)
    elif pr.size >= 1:  # hold VcVis flat at the last reliable node (clamped)
        den_co_fn = lambda p, x=pr, y=den_co[ok]: np.interp(p, x, y)
        den_cg_fn = lambda p, x=pr, y=den_cg[ok]: np.interp(p, x, y)
    else:  # no reliable nodes; fall back to the global LBC densities
        den_co_fn = lambda p, v=lbc.den_co: np.full_like(np.asarray(p, float), v)
        den_cg_fn = lambda p, v=lbc.den_cg: np.full_like(np.asarray(p, float), v)

    return FitResult(trusted=trusted, props=props, params=params, lbc=lbc,
                     so_tab=so_tab, sg_tab=sg_tab, Pk=Pk, cut=cut,
                     eos_pressure_error=err, eos_trusted=eos_trusted,
                     repairs=repairs, den_co_fn=den_co_fn, den_cg_fn=den_cg_fn)


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
        disc = discontinuous_above_cut(table, fr.cut)
        if disc:
            why = (f"A discontinuity in the {', '.join(disc)} above this pressure "
                   f"(a direction reversal that should not occur above the "
                   f"saturation pressure) marks a prior non-equilibrium "
                   f"extension; only the trusted locus below it is retained.")
        else:
            why = ("PVTO and PVTG no longer share saturated pressures above this "
                   "point, marking a prior non-equilibrium extension; only the "
                   "trusted locus below it is retained.")
        cl.add(
            action=f"Discarded {n_o_drop} PVTO and {n_g_drop} PVTG saturated "
                   f"row(s) above {fr.cut:g} psia, keeping a "
                   f"{len(trusted['p'])}-node trusted locus.",
            reason=why,
        )
    if fr.repairs:
        for pres, key, old, new in fr.repairs:
            cl.add(
                action=f"Replaced saturated {key.upper()} at {pres:g} psia: "
                       f"{old:.5g} -> {new:.5g}.",
                reason="User-specified point removed and replaced by interpolation "
                       "through its neighbours; EOS refit to the corrected locus.",
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

    # extrapolate K-values and regenerate saturated properties to Pk. Whitson
    # mode forces the classic constant-K (CKE) high-side law.
    k_mode = "constant" if config.whitson_mode else config.kvalue_extension
    p_ext, ko_ext, kg_ext = extrapolate_kvalues(
        psat, props["ko"], props["kg"], fr.Pk, config.n_extension_nodes,
        anchor=config.first_extrap_node, mode=k_mode)
    ext = extend_saturated(p_ext, ko_ext, kg_ext, fr.params, fr.lbc, s,
                           max_psat=float(psat.max()),
                           so_interp=so_interp, sg_interp=sg_interp)

    # Recompute the extension viscosity from LBC with the per-node critical
    # densities extrapolated along their trend (den_co in log p, den_cg in 1/p).
    # This reproduces the observed viscosity at the join and extends a smooth,
    # physical critical-density trend rather than a single global value.
    kve = kvalues(ext["rs"], ext["rv"], s)
    dco_e = np.asarray(fr.den_co_fn(ext["p"]), dtype=float)
    dcg_e = np.asarray(fr.den_cg_fn(ext["p"]), dtype=float)
    ext["uo"] = np.array([_visc.viscosity_from_densities(
        fr.lbc, dco_e[i], dcg_e[i], kve["xo"][i], kve["xg"][i], ext["deno"][i])
        for i in range(len(ext["p"]))])
    ext["ug"] = np.array([_visc.viscosity_from_densities(
        fr.lbc, dco_e[i], dcg_e[i], kve["yo"][i], kve["yg"][i], ext["deng"][i])
        for i in range(len(ext["p"]))])
    if ext["folded_at"] is not None:
        fold = ext["folded_at"]
        prop = ext.get("fold_property") or "Bo or Bg"
        diag.add(Anomaly(
            kind="eos_fold",
            location=f"extension node {fold} (P~{p_ext[fold]:.0f} psia)",
            severity=Severity.WARN,
            message=(f"The extension toward the convergence pressure became "
                     f"non-physical near the critical point at ~{p_ext[fold]:.0f} "
                     f"psia: {prop} reversed direction (it should change "
                     f"monotonically toward the critical point). The K-value "
                     f"extrapolation cannot capture the sharp near-critical "
                     f"curvature, so the extension is stopped just below it."),
            suggested_fix="Lower the convergence pressure so the extension reaches "
                          "the critical region without reversing.",
        ))
        if config.truncate_at_fold:
            cl.add(
                action=f"Stopped the extension at ~{p_ext[fold]:.0f} psia "
                       f"(dropped the nodes above it).",
                reason=f"Above this pressure the {prop} reversed direction, which "
                       f"is non-physical approaching the critical point; the "
                       f"K-value extrapolation cannot represent the sharp "
                       f"near-critical curvature there.",
            )
            for k in ("p", "rs", "rv", "bo", "bg", "uo", "ug"):
                ext[k] = ext[k][:fold]

    if config.convergence_pressure_Pk is not AUTO:
        pk_source = "user-specified value"
    elif config.pk_method == "crossing":
        pk_source = ("oil/gas average-MW crossing (single two-component "
                     "criticality root)")
    else:
        pk_source = "Singh App. B two-root average"
    k_law = ("the K-values held constant (classic constant-K extension)"
             if k_mode == "constant"
             else "the K-values extended to K=1 at Pk")
    cl.add(
        action=f"Extended the saturated tables to a convergence pressure "
               f"Pk = {fr.Pk:g} psia ({len(ext['p'])} extension node(s)).",
        reason=f"Convergence pressure taken from the {pk_source}; with {k_law}, "
               f"saturated properties above the table are regenerated from the "
               f"tuned PR79 + Peneloux EOS and LBC viscosity.",
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

    # low-side extension: continue the saturated locus below the lowest measured
    # pressure down to psc (Rs/Rv from the K-value origin poles + binary VLE
    # bijection; Bo anchored to 1.0; Bg anchored at psc by Z=1 or pressure ratio;
    # viscosities by mobility continuation). Off in Whitson mode.
    if config.extend_to_psc and not config.whitson_mode:
        T_for_bg = (config.reservoir_temperature if config.reservoir_temperature
                    else (fr.params.T if config.regress_temperature else None))
        low = extend_below_pmin(
            sat_p, sat_rs, sat_rv, sat_bo, sat_bg, sat_uo, sat_ug, s,
            psc=config.psc, n_nodes=config.n_low_extension_nodes,
            kg_exp=config.kg_low_pole_exp, ko_exp=config.ko_low_pole_exp,
            bo_psc=config.bo_psc_anchor, reservoir_temperature=T_for_bg)
        if low is not None:
            sat_p = np.concatenate([low["p"], sat_p])
            sat_rs = np.concatenate([low["rs"], sat_rs])
            sat_rv = np.concatenate([low["rv"], sat_rv])
            sat_bo = np.concatenate([low["bo"], sat_bo])
            sat_bg = np.concatenate([low["bg"], sat_bg])
            sat_uo = np.concatenate([low["uo"], sat_uo])
            sat_ug = np.concatenate([low["ug"], sat_ug])
            cl.add(
                action=f"Extended the saturated tables below {low['p1']:g} psia "
                       f"down to psc = {config.psc:g} psia "
                       f"({len(low['p'])} node(s)).",
                reason=f"Simulators drive flowing pressure toward psc, so the "
                       f"table is continued there: Rs/Rv from the K-value origin "
                       f"poles (K_g.(p1/p)^{config.kg_low_pole_exp:g}, "
                       f"K_o.(p1/p)^{config.ko_low_pole_exp:g}) via the binary VLE "
                       f"bijection with Rs=0 at psc; Bo anchored to "
                       f"{config.bo_psc_anchor:g}; Bg by {low['bg_basis']}; "
                       f"viscosities by mobility continuation.",
            )
            for f in low["flags"]:
                diag.add(Anomaly(
                    kind="low_side_extension",
                    location=f"below {low['p1']:g} psia",
                    severity=Severity.WARN,
                    message=f"Low-side extension toward psc: {f}.",
                    suggested_fix="Review the bottom saturated rows or set "
                                  "extend_to_psc=False.",
                ))

    # optional resample of the saturated locus onto a user-specified pressure
    # grid (e.g. to refine the rapidly-changing low-pressure region). The grid is
    # served by the honour-the-data interpolator over the assembled trusted+EOS
    # locus; requested pressures outside that range are dropped and flagged.
    if config.output_pressures:
        full = SaturatedInterpolator(sat_p, sat_rs, sat_rv, sat_bo, sat_bg,
                                     sat_uo, sat_ug, s)
        grid = np.array(sorted(set(float(p) for p in config.output_pressures)))
        inside = (grid >= sat_p.min() - 1e-6) & (grid <= sat_p.max() + 1e-6)
        if not np.all(inside):
            dropped = ", ".join(f"{p:g}" for p in grid[~inside])
            diag.add(Anomaly(
                kind="output_pressure_out_of_range",
                location=f"P = {dropped} psia",
                severity=Severity.WARN,
                message=(f"Requested output pressure(s) outside the model range "
                         f"[{sat_p.min():g}, {sat_p.max():g}] psia were dropped."),
                suggested_fix="Raise the convergence pressure to extend the range.",
            ))
        grid = grid[inside]
        ev = full.evaluate(grid)
        sat_p = grid
        sat_rs, sat_rv = ev["rs"], ev["rv"]
        sat_bo, sat_bg = ev["bo"], ev["bg"]
        sat_uo, sat_ug = ev["uo"], ev["ug"]
        cl.add(
            action=f"Resampled the saturated locus onto {len(grid)} "
                   f"user-specified pressure(s).",
            reason="Output pressures requested; properties taken by "
                   "composition-everywhere interpolation of the trusted + "
                   "EOS-extended locus.",
        )

    # optional simulator-compliance fix: truncate non-monotonic low-P Rv
    cgr_idx = 0  # number of low-pressure nodes flattened to the retrograde minimum
    if config.enforce_monotonic_cgr and "cgr_floor" in result.suggestions:
        floor = result.suggestions["cgr_floor"]
        cgr_idx = int(np.argmin(sat_rv))
        if cgr_idx > 0:
            cl.add(
                action=f"Flattened saturated Rv at {cgr_idx} node(s) below "
                       f"{sat_p[cgr_idx]:g} psia to the retrograde minimum "
                       f"{floor:g} and removed their undersaturated gas lines.",
                reason="The low-pressure rise in Rv is physically real but most "
                       "commercial simulators require monotonic saturated Rv "
                       "(enforce_monotonic_cgr).",
            )
        sat_rv[:cgr_idx] = floor

    # fill undersaturated branches. Each branch is at fixed composition, so its
    # volume shift is held at the node's value, and Bo / uo are anchored to the
    # measured saturated values with the EOS density and LBC viscosity supplying
    # only the pressure ratio. The branches are therefore continuous with the
    # saturated node and monotone by construction - no post-hoc clipping or
    # blind-spot regeneration is needed; the output-side QC below still flags any
    # residual non-physical branch for review.
    p_grid = undersaturated_pressures(float(sat_p.min()), fr.Pk,
                                      config.n_undersaturated_nodes)
    rv_grid = undersaturated_rv_grid(float(sat_rv.min()), float(sat_rv.max()),
                                     config.n_undersaturated_nodes)
    so_node = np.asarray(so_interp(sat_p), dtype=float)
    sg_node = np.asarray(sg_interp(sat_p), dtype=float)

    o_usat, g_usat = [], []
    for i in range(len(sat_p)):
        o_usat.append(fill_oil_branch(
            sat_p[i], sat_rs[i], sat_bo[i], sat_uo[i], fr.params, fr.lbc, s,
            p_grid, so_node[i], sg_node[i])[1:])  # drop saturated dup
        if config.enforce_monotonic_cgr and i < cgr_idx:
            g_usat.append(np.empty((0, 3)))  # flattened nodes carry no usat gas
        else:
            g_usat.append(fill_gas_branch(
                sat_p[i], sat_rv[i], sat_bg[i], sat_ug[i], fr.params, fr.lbc, s,
                rv_grid, so_node[i], sg_node[i])[1:])

    pvto = PVTOTable(rs=sat_rs, p=sat_p, bo=sat_bo, uo=sat_uo, usat=o_usat)
    pvtg = PVTGTable(p=sat_p, rv=sat_rv, bg=sat_bg, ug=sat_ug, usat=g_usat)
    result.extended = BlackOilTable(pvto=pvto, pvtg=pvtg, surface=s)

    # output-side QC: flag any generated undersaturated rows that came out
    # non-physical (wrong-direction compressibility, or a branch whose c_o
    # sticks out from the smooth trend) so they are surfaced for review rather
    # than silently emitted or auto-altered.
    detect_undersaturated_compressibility(result.extended, diag)
    detect_undersaturated_compressibility_trend(result.extended, diag,
                                                tol=config.co_trend_tol)

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
              f"Method: modified black-oil K-values (Whitson & Torp, SPE 10067, "
              f"1983; Coats, SPE 50990); consistent table modification per "
              f"Singh, Fevang & Whitson, SPE 109596 (2007)\n"
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
