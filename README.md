# BOT: Black-Oil Table QC and Extension Toolkit

`botkit` quality-controls a Black-Oil Table (PVTO/PVTG), fills in missing
undersaturated data, and extends the saturated tables to the convergence
pressure. It is a tested Python re-implementation of the original
`BOT_Extrapolation` notebook. The manual, plot-driven steps are replaced by
automated detectors that flag problems for review, and a change log records every
fix and the reason for it.

The table is recast as a two-pseudocomponent (surface-oil plus surface-gas)
K-value system — the modified-black-oil formulation of Whitson and Torp (JPT
1983, SPE 10067) and Coats (SPE 50990) — and the consistent table modification
(negative-compressibility QC, extension to the convergence pressure,
near-critical consistency) follows Singh, Fevang and Whitson, SPE 109596 (2007).
The K-values are extended to the convergence pressure, and a two-component
Peng-Robinson EOS
with Peneloux volume shifts and Lohrenz-Bray-Clark viscosity regenerates the
volumetric and viscosity properties. Around that core the toolkit adds automated
QC, an analytical convergence pressure, an EOS regression with a fallback gate, a
monotone interpolation layer that honours the input data, and physically grounded
handling of the undersaturated branches and the near-critical region.

## What it does

1. Reads PVTO/PVTG from Excel or an Eclipse deck.
2. Runs QC detectors and reports anomalies with suggested fixes.
3. Trims the untrusted tail above the last shared PVTO/PVTG pressure.
4. Computes the convergence pressure analytically on the trimmed locus (the
   oil/gas average-MW crossing by default; Singh App. B as the fallback).
5. Tunes a two-component PR79 + Peneloux EOS and LBC viscosity to the data.
6. Extends the saturated tables up to the convergence pressure (K to K=1 at Pk),
   stopping at a near-critical fold.
7. Extends the saturated tables down to psc (K-value origin poles + binary VLE
   bijection; Bo, Bg and viscosity anchored at psc).
8. Fills the undersaturated oil and gas branches.
9. Writes an Eclipse PVTO/PVTG deck with the change summary in the header.

## Quick start

```python
from botkit import SurfaceFluids, read_excel, run_qc
from botkit.model import Config
from botkit import pipeline
from botkit.report import diagnostics_to_markdown, changes_to_markdown

surface = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)  # lbm/ft3
table = read_excel("data/PVTO&PVTG_example.xlsx", surface=surface)

# QC only: review the report before anything is changed
diag, suggestions = run_qc(table)
print(diagnostics_to_markdown(diag, suggestions))

# Build the extended table. Pk defaults to the average-MW crossing (see below).
cfg = Config()
cfg.reservoir_temperature = 680.0   # deg R; omit to get a flagged 680 R default
cfg.auto_apply_fixes = True
result = pipeline.build(table, cfg)

print(changes_to_markdown(result.changes))
pipeline.write_deck(result, "EXTENDED_PVT.inc")
```

With `auto_apply_fixes = False` (the default) `pipeline.run` stops after QC so the
report can be reviewed before any data is generated.

## Required and optional inputs

Stock-tank oil and gas densities are required. They set the two-pseudocomponent
mixing constants and are passed on `SurfaceFluids(st_oil_density, st_gas_density)`,
in lbm/ft3. A measured stock-tank oil molecular weight can be supplied as
`oil_mw`; otherwise it is correlated from the oil density.

Reservoir temperature is optional, on `Config.reservoir_temperature` in deg R. If
it is not supplied and `regress_temperature` is False, the build assumes 680 R
(220 F) and records an `assumed_temperature` warning, because temperature affects
the EOS. Set `regress_temperature = True` to treat it as an unknown and regress
it, as the original notebook did.

## QC detectors

Detectors diagnose without changing the table. Fixes are applied only with
`auto_apply_fixes` or by an explicit option.

| Detector | What it catches |
|---|---|
| `pressure_misalignment` | PVTO and PVTG saturated pressures diverge above the last shared point (a prior bad extension) |
| `monotonicity` | Rs and Bo not rising, or Bg not falling, with pressure |
| `negative_saturated_compressibility` | negative saturated total compressibility (a derivative discontinuity) |
| `compressibility_discontinuity` | a large jump in saturated compressibility |
| `compressibility_ordering` | gas total compressibility below oil away from the critical point |
| `bo_rs_linearity` | saturated Bo off the Bo-Rs trend (an off-trend oil point) |
| `negative_undersaturated_compressibility` | undersaturated Bo rising with pressure |
| `undersat_viscosity_loglog` | undersaturated viscosity off log-p linearity |
| `undersaturated_co_outlier` | a branch whose c_o sticks out from the smooth c_o(Psat) trend |
| `cgr_reversal` | low-pressure retrograde Rv that most simulators reject |
| `assumed_temperature` | no reservoir temperature supplied; 680 R assumed |

The total-compressibility check uses the saturated oil and gas total
compressibilities (the SPE 109596 consistency form, with the dRs/dp and dRv/dp
mass-transfer terms). A negative value, a large jump, or gas below oil each
indicates a corrupt or missing node.

Note on outlier detection: a leave-one-out interpolation residual is not a
reliable outlier signal for properties that curve steeply at low pressure, such
as Bg and Rv, because a good low-pressure point can sit well off a fit that omits
it. Spurious saturated points are instead caught by the trend and consistency
detectors above (Bo-Rs linearity, monotonicity, compressibility, c_o trend), and
acted on by manual replacement.

## Editing the table

Manual point replacement. `Config.manual_replace_pressures` takes a list of
pressures. The node nearest each one has its saturated Rs, Rv, Bo, Bg, uo and ug
replaced by a PCHIP interpolation through its neighbours, and the EOS is refit to
the corrected locus. Use this to remove a point you judge spurious.

Output pressure grid. `Config.output_pressures` takes a list of saturated
pressures. The output saturated locus is built at exactly those pressures by
interpolating the trusted-plus-extended model (composition-everywhere PCHIP within
the data, EOS extension above it). Use this to refine the rapidly-changing
low-pressure region, where coarse spacing can give a simulator trouble with linear
interpolation. Pressures outside the model range are dropped and flagged.

## Configuration

`Config` in `botkit.model`. `AUTO` defers a value to the detectors.

QC and trimming:

| Option | Default | Meaning |
|---|---|---|
| `saturated_cut` | `AUTO` | highest shared saturated pressure to keep |
| `enforce_monotonic_cgr` | `True` | flatten low-pressure saturated Rv for simulator compliance |
| `co_trend_tol` | `0.5` | flag an undersaturated branch whose c_o departs this fraction from the trend |
| `manual_replace_pressures` | `()` | pressures whose saturated node is replaced by interpolation |

Extension (high side, above the table toward Pk):

| Option | Default | Meaning |
|---|---|---|
| `convergence_pressure_Pk` | `AUTO` | `AUTO` computes Pk from `pk_method` |
| `pk_method` | `"crossing"` | `"crossing"` = oil/gas average-MW crossing (single bounded root); `"singh"` = SPE 109596 App. B two-K-root average (also the auto-fallback) |
| `convergence_pressure_nodes` | `2` | top-N nodes for the Singh App. B fit (2 is the canonical last slope) |
| `first_extrap_node` | `-1` | table row from the end that anchors the extrapolation |
| `n_extension_nodes` | `15` | saturated extension nodes to Pk |
| `kvalue_extension` | `"convergence"` | high-side K law: `"convergence"` extends K to K=1 at Pk (default); `"constant"` freezes K (classic CKE) |
| `n_undersaturated_nodes` | `10` | undersaturated rows per branch |
| `output_pressures` | `()` | resample the saturated locus onto these pressures |
| `extrapolate_shift_trend` | `False` | hold the volume shift flat above the table; True projects the fitted per-node trend |
| `shift_trend_points` | `3` | last-N points that set the trend slope when the trend option is on |
| `oil_shift_abscissa` | `"log"` | transform that linearises the oil-shift trend (`log`/`recip`/`linear`/`sqrt`) |
| `gas_shift_abscissa` | `"linear"` | transform that linearises the gas-shift trend |
| `truncate_at_fold` | `True` | stop the extension at a near-critical Bo/Bg fold |

Extension (low side, below the table down to psc):

| Option | Default | Meaning |
|---|---|---|
| `extend_to_psc` | `True` | continue the saturated locus below the lowest measured pressure down to psc |
| `psc` | `14.696` | standard-condition pressure, psia |
| `n_low_extension_nodes` | `8` | nodes inserted between psc and the lowest measured pressure |
| `kg_low_pole_exp` | `2.0` | gas K-value origin-pole exponent, K_g = K_g(p1)*(p1/p)^exp |
| `ko_low_pole_exp` | `0.5` | oil K-value origin-pole exponent, K_o = K_o(p1)*(p1/p)^exp |
| `bo_psc_anchor` | `1.0` | Bo at psc (small thermal expansion ignored) |

Master revert:

| Option | Default | Meaning |
|---|---|---|
| `whitson_mode` | `False` | True restores the classic conservative behaviour: constant-K high side (CKE) and no low-side extension to psc |

EOS:

| Option | Default | Meaning |
|---|---|---|
| `eos_fallback_tol` | `0.05` | flag EOS fallback if it misses the upper locus by more than this |
| `reservoir_temperature` | `None` | reservoir temperature in deg R |
| `regress_temperature` | `False` | treat temperature as an unknown and regress it |
| `shift_smoothness` | `0.0` | roughness penalty on the volume-shift trends (0 is exact per node) |

Workflow:

| Option | Default | Meaning |
|---|---|---|
| `auto_apply_fixes` | `False` | False stops after QC; True runs the full build |

## Design notes

Honour the input data. Detectors flag and propose; the trusted saturated nodes
are passed to the output unchanged. The EOS is used only to create new data (the
extension and the undersaturated branches), never to overwrite a measured point.
Manual replacement is the one exception, and only at pressures the user names.

Interpolate compositions, not K-values or Rs/Rv. Between data points the solution
ratios are interpolated as the gas mole fractions x_g and y_g, and Rs and Rv are
formed from them at the query. The compositions are bounded and regular
everywhere. K = y_g/x_g has a singularity as x_g goes to zero (the Rs = 0 edge),
and interpolating Rs/Rv directly inherits that. PCHIP is monotone and exact at the
nodes.

Convergence pressure. By default (`pk_method = "crossing"`) Pk is the oil/gas
average-molecular-weight crossing. Both reservoir phases are mixtures of the same
surface oil and gas, so each phase's average MW is its position on the
two-component composition axis; toward the critical point the oil-phase MW falls
and the gas-phase MW rises, and where they meet the phases are identical
(x_g = y_g) — that pressure is Pk. Mechanically the two phase average MWs are each
extrapolated as a straight line in pressure through the top few trusted nodes, and
Pk is the single pressure where the lines cross — one bounded root, from the table
and the surface MWs alone, with no EOS. This is the molar-volume (M/γ) coordinate
of Whitson's note: average MW is the axis because molar volume is what ideal
(Amagat) mixing preserves; the linear-in-pressure reach is the empirical step, the
single-coordinate analogue of Singh's linear log-K-against-log-p extrapolation.

Setting `pk_method = "singh"` instead uses the SPE 109596 App. B method: log K
against log p extrapolated to K = 1 through the top two nodes (the last slope),
done separately for the oil and gas pseudocomponents and averaged — two roots that
diverge for lean fluids. It is also the automatic fallback when the crossing
geometry is degenerate. Either method is computed on the trusted locus after the
bad tail is removed, and either can be overridden via `convergence_pressure_Pk`.

High-side K law. Above the table the K-values are extended to K=1 at Pk by default
(`kvalue_extension = "convergence"`): a log-log quadratic that honours the slope at
the anchor and bends to the K=1 endpoint. A top-node leave-one-out over a
seven-fluid Whitson-school corpus showed this matches the best purely local log-log fit
(median K_g error 0.0 percent, K_o 0.3 percent) while enforcing the physical K=1 at
Pk that a local slope misses; freezing K (the classic constant-K extension, CKE) is
the worst, 1.2 and 3.1 percent at the very first step and compounding upward.
`kvalue_extension = "constant"` selects CKE, and `whitson_mode = True` forces it.

Low-side extension to psc. Simulators drive the flowing pressure toward psc in
unconstrained history matching, so by default (`extend_to_psc = True`) the saturated
locus is continued below the lowest measured pressure down to psc. Rs and Rv come
from the K-value origin poles validated in the companion `bopvt-lookup` study on the
Whitson corpus: below the lowest Rs>0 node p1 the gas pseudocomponent follows
K_g = K_g(p1)*(p1/p)^2 (Curtis's K_g.p^2) and the oil pseudocomponent
K_o = K_o(p1)*(p1/p)^0.5, both finite poles at the origin with no clamp, and Rs, Rv
are recovered from the pair by the binary VLE bijection (the two-component
compositions are fixed by the K-values alone, Gibbs F = C - 2 = 0). Rs goes to zero
at psc by the stock-tank convention. Bo is anchored to 1.0 at psc (small thermal
expansion ignored) and filled by 1/Bo interpolation. Bg is anchored at psc to the
Z=1 ideal-gas value when a reservoir temperature is supplied, otherwise the gas Z is
effectively frozen and Bg(psc) is taken by the isothermal pressure ratio
Bg(p1)*(p1/psc), then filled by 1/Bg interpolation. Viscosities continue the
mobilities 1/(Bo*uo) and 1/(Bg*ug) in their established coordinates, divided by the
anchored B, and are flagged as extrapolated; LBC is not used here because its
low-pressure VcVis fit is the very thing the build drops as ill-conditioned.

EOS regression. The global PR fit uses per-node oil and gas molar-volume
residuals. The large low-pressure gas molar volumes cannot be matched by a +/-0.2
volume shift (the two-component EOS only reaches Z about 0.84 there), so they are
reproduced through the component a and b, while the small high-pressure gaps are
matched by the per-node shifts. The match over the upper nodes, which the
extension anchors on, is essentially exact.

Fallback gate. After tuning, the EOS phase pressures are compared with the table
over the upper anchor nodes. If the error exceeds `eos_fallback_tol` the
EOS-dependent paths are flagged. The gate is evaluated on the exact per-node
shifts, so the decision reflects EOS quality and is independent of any shift
smoothing chosen for the extension. The pressure metric uses the upper nodes
because the liquid branch is so stiff at low pressure that a negligible
molar-volume residual becomes a large pressure error there, and those nodes are
kept from the table and never regenerated.

Undersaturated branches. Each branch is at fixed composition, so its Peneloux
shift, mixed a and b, molecular weight and LBC mixing parameters are held at the
node value; only pressure varies. Bo and uo are anchored to the measured saturated
values, with the EOS density and LBC viscosity supplying the pressure ratio. The
branches are therefore continuous with the saturated node and monotone by
construction. No clipping or special-case regeneration is needed. The earlier
notebook produced wrong-direction low-pressure points (it varied the shift along
the saturated-locus trend at fixed composition) and removed them by hand; the
anchoring removes the cause.

Volume shift above the table. The Peneloux volume translation is
pressure-independent by definition, so the shift is held flat (at the last fitted
value) for the extension. The per-node shift varies with pressure in the fitted
region only because the two-component EOS absorbs model error against real data,
and projecting that trend forward lowers the oil molar volume and suppresses the
near-critical rise in Bo, which then turns over. Holding the shift flat gives the
correct character: Bo rises with an increasing slope toward the critical point.
The trend option (`extrapolate_shift_trend = True`) remains available; it is
fitted in the abscissa that linearises it, log p for the oil shift and plain p for
the gas shift, and anchored so it passes through the last fitted point exactly.

Shift smoothing. `shift_smoothness` adds a roughness penalty to the shift trends.
Zero reproduces the exact per-node fit. A small value gives smooth, monotone shift
trends at a sub-1% in-sample molar-volume cost, which extrapolate more
defensibly than a noisy fit.

Viscosity. The LBC viscosity is `f(rho / rho_pc)`, where `rho` is the phase
density from the EOS and `rho_pc` is the viscosity pseudo-critical density set by
the per-component VcVis. VcVis is regressed at each node to reproduce the
observed viscosities (the ill-conditioned low-pressure nodes are dropped), then
held flat above the table because VcVis is a component property; the per-node
variation is the LBC absorbing model error, like the volume shift. The extension
viscosity therefore reproduces the observed value at the join and varies above it
through the EOS density and composition. Setting `extrapolate_vc_trend = True`
extrapolates the VcVis trend instead (oil in log p, gas in 1/p).

Near-critical fold. Extending to a high Pk drives the fluid toward its critical
point, where the K-value extrapolation can fold (Bo or Bg reversing). The fold is
detected and the extension truncated below it.

Monotonic Rv. The retrograde low-pressure rise in saturated Rv is real, but most
simulators reject it. With `enforce_monotonic_cgr` on (the default), the
low-pressure saturated Rv is flattened to its retrograde minimum and the
undersaturated gas lines on those nodes are dropped; this covers both the original
in-table reversal and the steep retrograde rise the low-side extension produces
toward psc (the K-value bijection gives a finite, rising r_s there). With it off,
the full non-monotonic profile is kept as the more physically realistic behaviour
and reported as a note. The K-values and compositions are intermediate and not
written to the deck, and the measured Bg and ug are honoured, so nothing is
recomputed from the flattened Rv.

No interactive prompts. Every decision that was an `input()` in the notebook is a
`Config` field with an auto-derived default, so a run is reproducible.

## Change summary

When the pipeline corrects or extends a table it records each change with its
reason. `result.changes` is a `ChangeLog`. Render it with `changes_to_markdown` or
`changes_to_text`, and `pipeline.write_deck` puts the summary in the Eclipse deck
header so the reasoning travels with the deck.

## Package layout

```
botkit/
  model.py        data structures, Config, Diagnostics, ChangeLog, constants
  io.py           Excel and Eclipse PVTO/PVTG read and write
  kvalues.py      modified-black-oil K-value transforms (App. A); convergence pressure (average-MW crossing or Singh App. B)
  eos.py          PR79 plus Peneloux; regression; fallback metrics; trend extrapolation
  viscosity.py    Lohrenz-Bray-Clark viscosity
  interpolate.py  composition-everywhere PCHIP layer
  qc.py           QC detectors; leave-one-out node prediction
  extend.py       high-side K-value extrapolation to Pk, EOS regeneration, fold detection
  extend_low.py   low-side extension to psc (K-value origin poles + bijection, anchors)
  fill.py         undersaturated branches, anchored to the measured node
  report.py       markdown and JSON diagnostics; change summary; plots
  pipeline.py     orchestration: QC, fit, extend, fill, assemble, write
tests/            pytest suite
data/             example PVTO/PVTG workbook
notebooks/        BOT_QC_and_Extension.ipynb demonstrator
```

Run the tests with `pytest`.

## Attribution

Implemented methods are credited to their publications in the code: the
two-pseudocomponent (modified-black-oil) K-value formulation of Whitson and Torp
(JPT 1983, SPE 10067) and Coats (SPE 50990); the consistent table modification
and convergence-pressure extension of Singh, Fevang and Whitson, SPE 109596
(2007); Peng and Robinson (1976/1978); Peneloux, Rauzy and Freze (1982);
Lohrenz, Bray and Clark (1964); and the Standing-era oil-property correlations.
The composition-everywhere interpolation basis and the per-phase abscissa choices
follow the companion Black-Oil PVT lookup work.
