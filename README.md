# BOT — Black-Oil Table QC & Extension Toolkit

`botkit` quality-controls a Black-Oil Table (PVTO/PVTG), fills in missing
undersaturated data, and extends the saturated tables to the convergence
pressure. It is a reproducible, tested re-implementation of the approach in the
original `BOT_Extrapolation` notebook, with the manual, plot-driven steps
replaced by automated detectors that flag problems for a single human review and
a change-log that records every fix and why it was made.

The numerical method follows **Singh & Whitson, SPE 109596 (2007)**: the table
is recast as a two-pseudocomponent (surface-oil + surface-gas) K-value system,
the K-values are extended to the convergence pressure, and a two-component
Peng-Robinson equation of state with Péneloux volume shifts plus Lohrenz-Bray-Clark
viscosity regenerates the volumetric and viscosity properties. Around that core
the toolkit adds automated QC, analytical convergence-pressure estimation, a
robust EOS regression with a fallback gate, a monotone "honour-the-data"
interpolation layer, and physically-grounded handling of the undersaturated
branches and the near-critical region.

---

## Contents

- [Quick start](#quick-start)
- [The method](#the-method)
- [Workflow](#workflow)
- [Quality-control detectors](#quality-control-detectors)
- [Change summary](#change-summary)
- [Configuration options](#configuration-options)
- [Design decisions and why](#design-decisions-and-why)
- [Package layout](#package-layout)
- [Attribution](#attribution)

---

## Quick start

```python
from botkit import SurfaceFluids, read_excel, run_qc
from botkit.model import Config
from botkit import pipeline
from botkit.report import diagnostics_to_markdown, changes_to_markdown

# 1. Load the table and define the surface fluids
surface = SurfaceFluids(st_oil_density=49.87, st_gas_density=0.0689)  # lbm/ft3
table = read_excel("data/PVTO&PVTG_example.xlsx", surface=surface)

# 2. Run QC and review the report (no changes made yet)
diag, suggestions = run_qc(table)
print(diagnostics_to_markdown(diag, suggestions))

# 3. Configure and build the extended table (Pk defaults to the Singh App. B fit)
cfg = Config()
cfg.auto_apply_fixes = True            # proceed past QC
result = pipeline.build(table, cfg)

# 4. Read what was changed and why, then write the deck (summary in the header)
print(changes_to_markdown(result.changes))
pipeline.write_deck(result, "EXTENDED_PVT.inc")
```

By default (`auto_apply_fixes = False`) `pipeline.run` stops after QC so the
diagnostics can be reviewed before anything is generated.

---

## The method

### Two-pseudocomponent K-values (Singh & Whitson App. A)

Each saturated row `(Rs, Rv, Bo, Bg)` is mapped, using the stock-tank fluid
densities, to the equilibrium ratios and mole fractions of two pseudocomponents
— "surface oil" and "surface gas". The liquid composition is `(x_o, x_g)` and
the vapour composition `(y_o, y_g)`, with K-values `K_o = y_o/x_o` etc. This is
the bridge between the engineering ratios and a compositional description.

### Convergence pressure (Singh & Whitson App. B)

At the convergence pressure `Pk` all K-values tend to unity. `Pk` is estimated
analytically by fitting `log K` against `log p` through the top saturated nodes
(the canonical method uses the top two — the last log-K slope) and solving for
`K = 1`, replacing the original notebook's hand-guessed value. It is computed on
the trusted locus only — i.e. **after** the misaligned tail is removed — and may
be overridden.

### EOS regeneration (PR79 + Péneloux + LBC)

A two-component Peng-Robinson EOS (Peng & Robinson, 1976/1978) is regressed to
the observed phase molar volumes, with a per-node Péneloux volume translation
(Péneloux et al., 1982) that matches each node, and Lohrenz-Bray-Clark viscosity
(Lohrenz, Bray & Clark, 1964) tuned to the table viscosities. The tuned EOS is
used to generate **new** data: the saturated extension above the table and the
undersaturated branches.

### Honour-the-data interpolation (the `bot_interpolation` layer)

Between existing data points the toolkit does **not** use the EOS; it uses
monotone PCHIP interpolation that passes through every input node exactly. The
solution-ratio quantities are interpolated as **compositions** (`x_g`, `y_g`) in
`log p` — the "composition-everywhere" basis — and `Rs`, `Rv` are formed from
the interpolated compositions at the query. Volumetrics use reciprocal families
(oil `1/Bo`, `1/(Bo·μo)` in `log p`; gas `1/Bg`, `1/(Bg·μg)` in plain `p`).

---

## Workflow

`pipeline.build` runs these stages, each emitting diagnostics rather than
prompting:

1. **QC** — automated detectors scan the input table and produce a
   severity-ranked anomaly report with suggested fixes and a recommended
   `saturated_cut`.
2. **Trim** — discard the untrusted tail above the last shared PVTO/PVTG
   pressure (a prior non-equilibrium extension), keeping only the trusted
   saturated locus. Everything downstream uses the trimmed data only.
3. **K-values & Pk** — App. A transforms; App. B analytical convergence
   pressure on the trusted locus (unless overridden).
4. **EOS + LBC tuning** — global PR regression, per-node volume shifts, LBC
   viscosity; a fallback gate judges whether the EOS is trustworthy.
5. **Extend** — extrapolate K-values to `Pk` and regenerate the saturated
   properties from the EOS + LBC, truncating at a near-critical fold.
6. **Fill** — generate undersaturated oil (rising p, constant Rs) and gas
   (falling Rv, constant p) branches, with physically-grounded shift handling
   and a reciprocal-family fallback for the low-pressure blind spot.
7. **Enforce & assemble** — optional monotonic-Rv (simulator) compliance and
   monotone undersaturated branches; assemble and (optionally) write the deck.
8. **Output QC** — re-scan the generated branches and flag anything residual.

Every corrective action taken in steps 2–7 is recorded in the
[change summary](#change-summary).

---

## Quality-control detectors

All detectors diagnose without mutating the table (the "honour input data"
principle); fixes are applied only on approval or with `auto_apply_fixes`.

| Detector | What it catches | Original manual equivalent |
|---|---|---|
| `pressure_misalignment` | PVTO/PVTG saturated pressures diverge above the last shared point (a prior bad extension) | eyeballing the plot, discarding rows |
| `monotonicity` | Rs/Bo not rising, Bg not falling with pressure | visual inspection |
| `negative_saturated_compressibility` | negative saturated total compressibility (a derivative discontinuity / corrupt node) | — |
| `compressibility_discontinuity` | a large jump in saturated compressibility | — |
| `compressibility_ordering` | gas total compressibility below oil's away from the critical point (non-physical) | — |
| `bo_rs_linearity` | saturated Bo departing from the Bo–Rs trend | — |
| `negative_undersaturated_compressibility` | undersaturated Bo rising with pressure | manual point removal |
| `undersat_viscosity_loglog` | undersaturated viscosity off log-p linearity | — |
| `cgr_reversal` | low-pressure retrograde Rv that most simulators reject | the cell-33 truncation |

The total-compressibility check uses the saturated oil/gas total
compressibilities (the SPE 109596 consistency form, including the `dRs/dp` and
`dRv/dp` mass-transfer terms); a negative value, a large jump, or gas-below-oil
ordering each indicates a corrupt or missing node.

---

## Change summary

When the pipeline corrects or extends a table it records each change together
with its justification. `result.changes` is a `ChangeLog`; render it with
`changes_to_markdown` / `changes_to_text`, and `pipeline.write_deck` embeds the
summary in the Eclipse deck header so the rationale travels with the deck:

```
-- Change summary (6 applied):
-- 1. Discarded 22 PVTO and 22 PVTG saturated row(s) above 1800 psia, keeping a 10-node trusted locus.
--    Why: PVTO and PVTG saturated pressures diverge above this point ... a prior non-equilibrium extension.
-- 2. Truncated the EOS extension at ~3000 psia.
--    Why: Bo or Bg reversed (a near-critical fold) the smooth K-value extrapolation cannot represent.
-- 3. Extended the saturated tables to a convergence pressure Pk = 8195 psia (3 extension node(s)).
-- 4. Flattened saturated Rv at 4 node(s) below 800 psia to the retrograde minimum 0.001354 ...
--    Why: ... most commercial simulators require monotonic saturated Rv.
-- 5. Regenerated 2 undersaturated oil branch(es) (Psat = 100, 200 psia) by the reciprocal-family law.
-- 6. Enforced monotonicity on 6 undersaturated oil branch(es).
```

---

## Configuration options

`Config` (in `botkit.model`). `AUTO` defers a value to the detectors.

**QC / trimming**

| Option | Default | Meaning |
|---|---|---|
| `saturated_cut` | `AUTO` | highest shared saturated pressure to keep; `AUTO` = the detected value |
| `enforce_monotonic_cgr` | `True` | make low-pressure saturated Rv monotone for simulator compliance |
| `enforce_undersaturated_monotonic` | `True` | force undersaturated oil Bo to fall and μo to rise with pressure |
| `hybrid_undersaturated_oil` | `True` | regenerate blind-spot oil branches by the reciprocal-family law |
| `hybrid_co_factor` | `3.0` | regenerate a branch when its `c_o` exceeds this multiple of the table median |

**Extension**

| Option | Default | Meaning |
|---|---|---|
| `convergence_pressure_Pk` | `AUTO` | `AUTO` = Singh App. B analytical value |
| `convergence_pressure_nodes` | `2` | top-N nodes for the App. B fit (2 = canonical last-slope) |
| `first_extrap_node` | `-1` | table row (from the end) anchoring the extrapolation |
| `n_extension_nodes` | `15` | number of saturated extension nodes to `Pk` |
| `n_undersaturated_nodes` | `10` | undersaturated rows per branch (square-root progression) |
| `extrapolate_shift_trend` | `True` | extend volume shifts along their trend (vs hold flat) |
| `shift_trend_points` | `3` | last-N points defining the trend slope |
| `oil_shift_abscissa` | `"log"` | transform linearising the oil-shift trend (`log`/`recip`/`linear`/`sqrt`) |
| `gas_shift_abscissa` | `"linear"` | transform linearising the gas-shift trend |
| `truncate_at_fold` | `True` | stop the extension at a near-critical Bo/Bg fold |

**EOS**

| Option | Default | Meaning |
|---|---|---|
| `eos_fallback_tol` | `0.05` | disable EOS paths if it misses the upper locus by more than this |
| `reservoir_temperature` | `None` | reservoir temperature (deg R); required unless regressed |
| `regress_temperature` | `False` | opt-in: treat temperature as an unknown and regress it |
| `shift_smoothness` | `0.0` | roughness penalty on the volume-shift trends (`0` = exact per node) |

**Workflow**

| Option | Default | Meaning |
|---|---|---|
| `auto_apply_fixes` | `False` | `False` stops after QC for human review; `True` runs the full build |

---

## Design decisions and why

**Honour input data; diagnose, do not silently edit.** Detectors flag problems
and propose fixes; the trusted saturated nodes are passed through to the output
unchanged. The EOS is only ever used to create *new* data (the extension and the
undersaturated branches), never to overwrite a measured point. Every applied
change is recorded with its reason.

**Bad data is removed before anything is fitted.** The trim to the trusted locus
is the first step; the convergence pressure, the EOS/LBC regression and the
extension all use the trimmed data only, so a prior botched extension cannot
contaminate the new one.

**Interpolate compositions, not K-values or Rs/Rv.** Between data points the
solution ratios are interpolated as the gas mole fractions `x_g`, `y_g` (the
"composition-everywhere" basis), and `Rs`/`Rv` are recovered from them. The
compositions are bounded and regular everywhere, whereas `K = y_g/x_g` has a
coordinate singularity as `x_g → 0` (the `Rs = 0` edge) and interpolating
`Rs`/`Rv` directly inherits that. PCHIP is monotone and exact at the nodes.

**Analytical convergence pressure, the canonical way.** `Pk` comes from the
App. B `log K` vs `log p` extrapolation through the top two nodes (the last
slope) rather than a hand-guess, on the trusted locus, while remaining
overridable — the K-value trend is still available to sanity-check it.

**Robust EOS regression, weighted to where shifts cannot help.** The global PR
fit uses the per-node oil/gas molar-volume residuals (replacing the original
loose Nelder-Mead, `xatol = 350` psi). The large low-pressure gas molar volumes
genuinely cannot be matched by a ±0.2 volume shift (the two-component EOS reaches
only `Z ≈ 0.84` there), so they are reproduced by the component `a, b`, while the
small high-pressure gaps are matched by the per-node shifts. The upper-locus
match — the region the extension anchors on — is essentially exact (~1e-7).

**Fallback gate on the EOS itself.** After tuning, the EOS phase pressures are
compared with the table over the upper anchor nodes; if the error exceeds
`eos_fallback_tol` the EOS-dependent paths are flagged for fallback. The gate is
evaluated on the *exact* per-node shifts so the decision reflects the EOS
quality and is independent of any shift smoothing chosen for the extension. (The
pressure metric is taken on the upper nodes because the liquid branch is so stiff
at low pressure that a negligible molar-volume residual inflates into a large
pressure error there, yet those nodes are kept from the table and never
regenerated.)

**Volume shifts held constant along undersaturated branches.** An undersaturated
branch is at fixed composition, so its Péneloux shift — and with it the mixed EOS
`a, b`, the molecular weight and the LBC mixing parameters — is held at the
node's bubble-point value; only pressure varies, through the EOS. Varying the
shift by the *saturated-locus* trend along a constant-composition branch injects
a spurious, composition-driven volume change that pushes Bo the wrong way — the
artifact the original notebook removed by hand. Holding the shift constant yields
a physical, positive undersaturated compressibility by construction.

**Reciprocal-family regeneration for the low-pressure blind spot.** At the
lowest saturated nodes the two-component EOS is unreliable. Where the generated
oil branch comes out with a non-physically steep compressibility (`c_o` above
`hybrid_co_factor` × the table median), the branch is regenerated from a
constant-compressibility reciprocal-family law `Bo = Bo_b · exp(-c_o (p - p_b))`.
The compressibility `c_o` is **interpolated from the surrounding reliable nodes**
at that node's pressure (not a single table-wide average), so it reflects the
local trend. This is the EOS-for-new-data / interpolation-for-consistency hybrid
applied to the one region the EOS cannot serve.

**Trend-informed, anchored shift extrapolation above the table.** Above the
table the volume shifts follow the local trend of the last few points, anchored
so the extrapolation passes through the last valid point exactly (no offset at
the join) rather than freezing flat. The trend is fitted in the abscissa that
linearises it — empirically `log p` (or `1/p`) for the oil shift and plain `p`
for the gas shift, echoing the per-phase abscissa rules. Any monotone transform
preserves the exact-anchor property.

**Optional shift smoothing for confident extrapolation.** `shift_smoothness`
adds a roughness penalty to the volume-shift trends. `0` reproduces the exact
per-node fit; a small value gives smooth, monotone, confidently extrapolatable
shift trends at a sub-1% in-sample molar-volume cost — useful precisely because
a smooth trend extrapolates more defensibly than a noisy one.

**Near-critical fold detection and truncation.** Extending to a high `Pk` drives
the fluid toward its critical point, where the smooth K-value extrapolation can
fold (Bo or Bg reversing). The fold is detected and the extension truncated below
it, avoiding non-physical near-critical output.

**Monotonic Rv for simulators is a choice, not a default mangling.** The
retrograde low-pressure rise in saturated Rv is physically real but rejected by
most commercial simulators. With `enforce_monotonic_cgr` on, the low-pressure
saturated Rv is truncated to its retrograde minimum (the real minimum value) and
the now-meaningless undersaturated gas lines on those nodes are dropped — a
deck-legal, defensible shape. With it off, the physically representative reversal
is retained and reported as an informational note. The curtailment is a
localized Rv-only approximation: the K-values and compositions are intermediate
(not written to the deck) and the measured Bg/ug are honoured, so nothing is
recomputed from the flattened Rv.

**No interactive prompts.** Every decision that was an `input()` in the original
notebook is a `Config` field with an auto-derived default, so a run is fully
reproducible and scriptable.

---

## Package layout

```
botkit/
  model.py        data structures, Config, Diagnostics, ChangeLog, constants
  io.py           Excel + Eclipse PVTO/PVTG read/write
  kvalues.py      Singh App. A transforms; App. B convergence pressure
  eos.py          PR79 + Péneloux; regression; fallback metrics; trend extrapolation
  viscosity.py    Lohrenz-Bray-Clark viscosity
  interpolate.py  composition-everywhere PCHIP layer (honour-the-data)
  qc.py           automated QC detectors
  extend.py       K-value extrapolation + EOS regeneration + fold detection
  fill.py         undersaturated branches + reciprocal-family fallback
  report.py       markdown / JSON diagnostics; change summary; plots
  pipeline.py     orchestration (QC -> fit -> extend -> fill -> assemble -> write)
tests/            pytest suite (run with `pytest`)
data/             example PVTO/PVTG workbook
```

---

## Attribution

The implemented methods are credited to their original publications in the code:
Singh & Whitson, SPE 109596 (2007); Peng & Robinson (1976/1978); Péneloux, Rauzy
& Fréze (1982); Lohrenz, Bray & Clark (1964); and the Standing-era oil-property
correlations. The composition-everywhere interpolation basis and the per-phase
abscissa choices follow the consolidated findings of the companion Black-Oil PVT
lookup work.
