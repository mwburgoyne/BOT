# Black-Oil Table QC & Extension Toolkit — Design

A modernization of the `BOT_Extrapolation-v5` notebook into a reusable, tested
toolkit plus a clean demonstrator notebook. The goal is a **semi-automated**
workflow that quality-controls a Black-Oil Table (BOT), fills missing data, and
extends the saturated tables to convergence pressure — replacing the manual,
plot-eyeballing, `input()`-driven steps of the original with automated detectors
that *flag* problems for a one-pass human approval.

The numerical method is unchanged in spirit: **Singh & Whitson, SPE 109596
(2007)** — convert the table to a two-pseudocomponent (surface oil + surface
gas) K-value system, extrapolate K-values to the convergence pressure, and use a
two-component Peng-Robinson EOS with per-node volume shifts plus Lohrenz-Bray-Clark
viscosity to regenerate volumetric and viscosity properties. What changes is the
robustness, automation, attribution, and reproducibility around that method.

## Design principles

1. **Honour input data.** Detectors diagnose and *propose*; they never silently
   mutate user nodes. The pipeline applies a fix only after approval (or when a
   config flag opts into it).
2. **Self-contained.** No imports from sibling projects. Methods are
   re-implemented here so the repository can be published cleanly and stand alone.
   Lessons from prior work are folded in; code is not copied.
3. **Attribute the science.** Every implemented method/correlation carries a
   comment or docstring crediting the original author(s) and publication.
4. **No AI fingerprints.** No AI/assistant signatures, trailers, or
   "generated-by" notices anywhere in code, notebook, docs, or commits.
5. **Config over prompts.** All decisions that were interactive `input()` calls
   become explicit config fields with sensible auto-derived defaults.
6. **Fail loud, fall back gracefully.** Where the EOS is untrustworthy, gate it
   off and fall back to analytical K-value / correlation extrapolation, with the
   decision logged.

## Package layout

```
BOT_util/
  botkit/
    __init__.py
    io.py            # read PVTO/PVTG (Excel, Eclipse .inc); write Eclipse deck
    model.py         # dataclasses: SurfaceFluids, BlackOilTable, Diagnostics, Config
    kvalues.py       # Singh App. A transforms; App. B convergence pressure
    eos.py           # 2-component PR79 + Peneloux volume shift; cubic Z solver
    viscosity.py     # Lohrenz-Bray-Clark (LBC) viscosity
    qc.py            # automated QC detectors -> ranked anomaly report
    extend.py        # extend saturated locus to Pk (K-value + EOS, fallback gate)
    fill.py          # undersaturated oil/gas fill; monotonicity enforcement
    report.py        # QC plots + machine-readable JSON/markdown summary
    pipeline.py      # orchestration: qc -> approve -> kvalues -> eos -> extend -> fill -> write
  notebooks/
    BOT_QC_and_Extension.ipynb   # narrative demonstrator that calls botkit
  tests/
    test_io.py
    test_kvalues.py
    test_eos.py
    test_qc.py
    test_roundtrip.py
  data/
    PVTO&PVTG_example.xlsx        # existing example fluid
  DESIGN.md
  README.md
  pyproject.toml
```

## Data model (`model.py`)

- `SurfaceFluids` — stock-tank oil/gas densities, derived molar masses and the
  Singh mixing constants (`Lo`, `Lg`, `Mult`, `Co`). Optional measured oil MW
  override; otherwise the existing API-based correlation, cited.
- `BlackOilTable` — saturated locus (P, Rs, Rv, Bo, Bg, μo, μg) plus
  undersaturated branches per saturated node, held as structured arrays rather
  than the notebook's parallel Python lists.
- `Config` — every previously-interactive choice as a field, each with an
  `auto` sentinel that triggers the detector-derived default:
  - `saturated_cut` (`auto` = detector-found last shared saturated pressure)
  - `convergence_pressure_Pk` (`auto` = Singh App. B analytical value)
  - `first_extrap_node` (was `last_row`)
  - `n_extension_nodes`, `n_undersaturated_nodes`
  - `eos_fallback_tol` (default 5% — gate the EOS off above this)
  - `enforce_monotonic_cgr` (the cell-33 retrograde fix, generalized)
  - `auto_apply_fixes` (False = approval required; True = apply + log)
  - `reservoir_temperature` (required/known by default; set `regress_temperature=True` to treat it as an unknown and regress it, as the original notebook did)
- `Diagnostics` — list of `Anomaly(kind, location, severity, message,
  suggested_fix)`; serializable to JSON.

## Stage specifications

### 1. I/O (`io.py`)
Read PVTO/PVTG from the Excel workbook (existing format) and from Eclipse
`PVTO`/`PVTG` keyword decks. Write the Eclipse deck (the existing cell-35
formatter, refactored and de-globalized). Round-trip read→write→read is a test.

### 2. QC detectors (`qc.py`) — the core new value
Each detector returns zero or more `Anomaly` records. These replace the
notebook's manual eyeballing and `input()` removal cells.

- **Pressure alignment** — find the highest pressure shared by PVTO and PVTG
  (the originally-tested saturated point); flag rows above it as a likely prior
  bad extension. (Automates the manual "discard above 1800 psia" step.)
- **Monotonicity** — Rs, Bo increasing with P; Bg decreasing; Rv reaching a
  minimum then increasing (retrograde). Flag violations with the offending
  index range.
- **Saturated-compressibility / derivative-discontinuity scan** — compute the
  saturated-locus pseudo-compressibilities (the `TcOil`/`TcGas` derivative terms
  already in cell 11) and flag sign reversals / discontinuities. This is the
  principled, automatable version of "which rows look anomalous," and it
  pinpoints botched extension breaks. *(Concept: derivative-discontinuity /
  negative saturated compressibility — Whitson PVT literature.)*
- **Bo–Rs linearity** — saturated Bo is near-linear in Rs; large residuals flag
  inconsistent rows. *(Standing-era shrinkage QC heuristic.)*
- **Undersaturated viscosity log-log linearity** — undersaturated μo is
  near-linear in log p; flag departures.
- **CGR reversal** — non-monotonic saturated Rv at low pressure that would make
  Eclipse error; propose the truncate-to-minimum fix (generalized cell 33).

Output: a severity-ranked anomaly report (markdown + JSON) with a proposed fix
per anomaly. In semi-automated mode the pipeline pauses here for approval.

### 3. K-values & convergence pressure (`kvalues.py`)
- Singh App. A transforms: (Rs, Rv, Bo, Bg, surface props) →
  K-values, mole fractions (xo, xg, yo, yg), molar masses, densities, molar
  volumes. (The cell-8 algebra, documented and cited to SPE 109596 App. A.)
- **Convergence pressure** via Singh App. B: linear fit of log K vs log p through
  the top saturated nodes, solved for K = 1 — an *analytical* Pk that replaces
  the hand-guessed `Pk = 6000`. Config can override; the notebook still plots the
  K-trend so the value can be sanity-checked.
- K-value extrapolation to Pk: keep the existing slope-honoring quadratic in
  log-log space as the default, with the App. B linear extrapolation available
  as the robust fallback.

### 4. EOS & viscosity (`eos.py`, `viscosity.py`)
- Two-component Peng-Robinson (1979) with the cubic Z-root solver (cell 12),
  cited to Peng & Robinson (1976/1979). Component a, b first guesses from the
  existing API/gravity correlations, cited.
- Per-node Peneloux volume shifts (the cell-16 local `so`, `sg` regression),
  cited to Péneloux et al. (1982). Global regression first (replacing the loose
  `xatol=350` Nelder-Mead with a tighter, bounded scheme), then per-node refine.
- **Auto-fallback gate**: after tuning, re-evaluate the EOS at every input
  saturated node; if max relative error in pressure (or K) exceeds
  `eos_fallback_tol`, disable EOS-dependent extension/fill and fall back to
  K-value extrapolation + correlations, logging the decision.
- **Fold detection**: during EOS extension, stop if Bo or Bg reverses direction
  (near-critical fold the smooth extrapolation cannot represent), and flag.
- LBC viscosity (Lohrenz, Bray & Clark, 1964) with the global coefficient
  regression and per-node ρc match (cells 18-20), cited.

### 5. Extend & fill (`extend.py`, `fill.py`)
- Extend the saturated locus from `first_extrap_node` to Pk over
  `n_extension_nodes`, regenerating Bo, Bg, Rs, Rv, μo, μg from the tuned EOS +
  LBC (cell 22), or from the fallback path when gated.
- Fill undersaturated oil (constant Rs, rising P) and gas (reducing Rv at
  constant P) branches (cells 24-25), with monotonicity and CGR-reversal
  enforcement applied automatically rather than via `input()`.

### 6. Report & pipeline (`report.py`, `pipeline.py`)
- QC plot grid (Bo, Bg, Rs, Rv, μo, μg with table / extrapolated / undersaturated
  overlays) — the cell-27/37 plots, parameterized.
- `pipeline.run(config)` orchestrates: load → QC → (approve) → K-values → Pk →
  EOS tune → gate → extend → fill → enforce → write deck → report. Returns the
  extended `BlackOilTable` and `Diagnostics`. The notebook drives this stage by
  stage so each step is inspectable.

## Human-in-the-loop approval

Default (`auto_apply_fixes=False`): the pipeline runs QC, prints/returns the
ranked anomaly report, and stops. The user reviews, then either accepts the
proposed fixes wholesale or edits the `Config` (e.g. pins `saturated_cut`,
overrides `Pk`, lists specific rows to drop) and re-runs. In the notebook this is
a single review cell — no `input()` prompts. `auto_apply_fixes=True` applies all
proposed fixes and logs them, for batch use.

## Testing (`tests/`)

- `test_io` — Excel read; Eclipse write→read round-trip preserves values.
- `test_kvalues` — App. A transforms invert correctly; App. B Pk recovers a
  known value on a synthetic monotone K-trend.
- `test_eos` — cubic solver returns correct roots; tuned EOS reproduces input
  saturated pressures within tolerance on the example fluid.
- `test_qc` — each detector fires on a seeded defect (inject a non-monotone Rv,
  a derivative break at a known row, a Bo–Rs outlier) and stays silent on clean
  data.
- `test_roundtrip` — full pipeline on `PVTO&PVTG_example.xlsx` produces a
  monotone, Eclipse-valid extended deck; key values within tolerance of the
  notebook's published result.

## Attribution & conventions

Method citations carried in code (author, year, paper):
Singh & Whitson, SPE 109596 (2007); Peng & Robinson (1976/1979);
Péneloux, Rauzy & Fréze (1982); Lohrenz, Bray & Clark (1964); Standing (oil
property correlations). No AI/assistant attribution anywhere. Commits and any PR
text carry no AI trailers.

## Proposed build order

1. `model.py`, `io.py`, and the example round-trip test (foundation).
2. `kvalues.py` with Singh App. A + App. B analytical Pk (the first automation win).
3. `qc.py` detectors + report, validated against the known 1800-psia defect.
4. `eos.py` + `viscosity.py` with the fallback gate and fold detection.
5. `extend.py` + `fill.py` + pipeline orchestration.
6. `BOT_QC_and_Extension.ipynb` demonstrator + README.

Each step is independently testable and reviewable before the next.
```
