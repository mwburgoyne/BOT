"""Supercritical-band EOS-validation deck (2026-06-16).

WHY. The four band methods (CKE, KEL/Singh-logK, TSY, molar-volume) all
extrapolate the SATURATED LOCUS above the table's top saturation pressure
p_sat,max up toward the convergence/critical pressure. We want EOS truth for
two things: (1) the critical-pressure LOCATOR -- Singh App. B fits log(K_o)
and log(K_g) vs log(p) separately and averages two disagreeing roots, while
the molar-volume method solves the single joint condition M_bar_oil =
M_bar_gas (x_g=y_g, K_o=K_g=1, R_s.r_s=1e6); and (2) the saturated-locus
SHAPE from p_sat,max up to p_c.

KEY PHYSICS -> deck design. Above the cricondenbar the original fluid is
single-phase, so a depletion ladder of ONE fluid cannot populate the band.
The well-posed EOS truth is

  (a) the EOS CRITICAL POINT of the fluid family -- p_c, T_c, critical
      composition -- from a P-T phase envelope of a near-critical sibling
      (PhazeComp prints "Critical Point: <T>, <P>" and the cricondenbar);

  (b) the SATURATED LOCUS traced up toward p_c by the SATURATION PRESSURE
      of a ladder of recombination siblings spanning oil-like (low psat)
      to near-critical (high psat, approaching p_c). Each sibling's psat is
      one saturated-locus point at that pressure; near-critical siblings
      give the high-pressure locus points that live in the band.

Construction (mirrors gen_phazecomp_recomb_ladders.py):
  1. Flash GOR2000 at REF_P / 200 F (two-phase) and snapshot the equilibrium
     liquid OILREF (~GOR700 oil, psat~2000) and vapour GASREF.
  2. Recombine OILREF + GASREF at a spread of mole ratios -> sibling fluids
     of rising psat. Low GASREF fraction = oil-like; high fraction pushes
     toward (and past) the critical composition. The ladder is chosen so the
     family brackets the critical mix.
  3. For each sibling:
       - SATP at 200 F  -> p_sat of that sibling (one saturated-locus point).
       - a FLASH at SAT_FRAC*p_sat (both phases present) with BOTH equilibrium
         phases separated through the BOT 2-stage surface separator:
            OIL SEP (EQL) -> Rs = CumGOR, Bo = 1/LiqVol  (bubble-side rail)
            GAS SEP (EQV) -> rs = 1e6/CumGOR             (dew-side rail)
         so the harvester recovers Rs/Bo/rs (hence x_g/y_g/K, M_bar) on the
         locus point just below psat.
  4. For the near-critical siblings, extra two-phase FLASHes at a few
     pressures just below their psat (phase props in the near-critical
     region), plus a P-T envelope for the EOS-true critical point.

PhazeComp keywords (verified in ~/projects/phazecomp/PhazeComp User Guide/
PhazeComp_KnowledgeBase.md [calculations]/[envelopes]):
  SATP / PSAT / BUBP / PBUB -> upper saturation pressure (bubble for oils,
                               upper dew for gases). Header line in the .Out:
                               "One Phase at Temperature = 200 F, Pressure =
                               <psat> psia:".
  P-T ID "title": P_vs_T    -> traces the two-phase boundary and prints
                               "Critical Point: <T>, <P>", "Cricondenbar:
                               <P>", "Cricondentherm: <T>" for the FEED.

Per-flashpoint label: NOTE "===FLASHPOINT=== FLUID=<name> P=<p> ..."; the DLE
separator IDs carry "OIL SEP"/"GAS SEP" + "FLUID=<name>" so the harvester keys
fluid + phase (same scheme as the recomb deck). SATP/P-T blocks carry their own
NOTE tags so the harvester can locate them by FLUID=.

VERSION NOTE. P-T phase envelopes -- the only thing that prints the EOS-true
Critical Point -- were added in PhazeComp 2.0.0. Version 1.8.1 has NO
critical-point calculation (only SATP/BUBP/DEWP saturation calcs), so the deck's
envelope blocks are gated behind EMIT_ENVELOPES (default False -> a 1.8.1-runnable
deck that traces the saturated locus only; the proxy validation). Set
EMIT_ENVELOPES=True and regenerate when running 2.0.0+ to get the true p_c.

Run it (on whichever version) and bring the .Out back to phz/band-validation/.
Then run the band scorer (curtis_band_eos_validation.py).

Run:  python3 scripts/gen_phazecomp_band_validation.py
"""

import os

TRES_F = 200.0          # reservoir T for the whole locus (matches the BOT corpus)
SEP_T = 60.0
SEP_P1 = 100.0
SEP_P2 = 14.696
REF_P = 2000.0          # reference flash pressure for the OILREF/GASREF split
SAT_FRAC = 0.998        # flash just below psat so both phases exist on the locus

# P-T phase envelopes (which print the EOS-true Critical Point) were added in
# PhazeComp 2.0.0; version 1.8.1 has NO critical-point calculation -- its only
# saturation calcs are SATP/BUBP/DEWP. Leave EMIT_ENVELOPES False for a deck that
# runs on 1.8.1 (saturated locus only -> proxy validation against the highest
# near-critical sibling psat). Set it True only when running PhazeComp >= 2.0.0,
# which adds the four P-T blocks that yield the true p_c (the full scorer).
EMIT_ENVELOPES = False

# Recombination ladder: GASREF mole fraction from oil-like up toward (past) the
# critical composition. SIB45 (frac 0.45) reached psat 3838 in the recomb deck;
# GOR2000 (the original feed) is psat 4110; the critical mix sits above. Pushing
# the GASREF fraction toward ~0.6-0.7 should bracket the critical composition,
# so the top siblings are near-critical and a couple cross to the dew side.
# (fraction GASREF, sibling label).  OILREF fraction = 1 - frac.
LADDER_FRACS = [
    0.00,   # OILREF itself (psat ~2000, oil-like anchor)
    0.15,
    0.25,
    0.35,
    0.45,
    0.52,
    0.58,
    0.62,
    0.66,
    0.70,
    0.74,   # likely past the critical composition (dew-side / near-critical)
]

# Siblings flagged near-critical get extra sub-psat flashes + a P-T envelope.
# By fraction: the top three of the ladder.
NEAR_CRIT_FRACS = LADDER_FRACS[-3:]
# Extra two-phase flash pressures BELOW psat for near-critical siblings, as
# fractions of that sibling's psat (phase properties approaching criticality).
SUBPSAT_FRACS = [0.99, 0.97, 0.94, 0.90]


def sib_name(frac):
    """Stable, harvester-friendly fluid label, e.g. 0.45 -> SIB045, 0.00 -> OILREF."""
    if frac <= 1e-9:
        return "OILREF"
    return f"SIB{int(round(frac * 1000)):03d}"


SURFACE_SEP = f"""BASIS 1 BBL
STAGE               TEMP                PRES                GOR                 CUMGOR              LVOL         K-C1
                    F                   PSIA                SCF/BBL             SCF/BBL             BBL
1                   {SEP_T:.0f}                  {SEP_P1:.0f}
2                   {SEP_T:.0f}                  {SEP_P2:.3f}
END"""


def header():
    sibs = "\n".join(
        f"MIX {sib_name(fr)} {1.0 - fr:.2f} MOLE OILREF {fr:.2f} MOLE GASREF"
        for fr in LADDER_FRACS if fr > 1e-9
    )
    return f"""TITLE "Permian band-validation: recombination saturated locus up to p_c + EOS critical point"

; Same EOS as the GOR2000 deck (Milan's Permian model). OILREF/GASREF are the
; equilibrium phases of GOR2000 flashed at {REF_P:.0f} psia / {TRES_F:.0f} F;
; recombined to a spread of saturation pressures running oil-like -> near-critical
; to trace the saturated locus up toward the critical point (the supercritical band).
; Generated by scripts/gen_phazecomp_band_validation.py -- edit the generator, not this file.

STAB  ON

INCLUDE eos.inc
INCLUDE compositions.inc

; ---- reference split: flash GOR2000 at {REF_P:.0f} psia / {TRES_F:.0f} F (two-phase) ----
MIX FEED GOR2000 1 MOLE
TEMP {TRES_F:.0f} F
PRES {REF_P:.0f} PSIA
FLASH
MIX OILREF EQL 1 MOLE
MIX GASREF EQV 1 MOLE

; ---- recombined sibling fluids (mole basis: <amt> MOLE <src>) ----
{sibs}
"""


def satp_block(name, src):
    """Saturation-pressure calc for one sibling. The .Out 'One Phase ... Pressure
    = <psat> psia:' header gives p_sat; we tag it so the harvester keys FLUID=."""
    return f"""
NOTE: "===SATP=== FLUID={name}  T={TRES_F:.0f} F  (upper saturation pressure)"
MIX FEED {src} 1 MOLE
TEMP {TRES_F:.0f} F
SATP
"""


def locus_flash_block(name, src, p, tag="LOCUS"):
    """Two-phase flash at p (< psat) with both phases surface-separated, so the
    harvester recovers Rs/Bo (oil side) and rs (gas side) on a locus point."""
    pstr = f"{p:.2f}"
    return f"""
NOTE: "===FLASHPOINT=== FLUID={name} P={pstr} PSIA  T={TRES_F:.0f} F  ({tag})"
MIX FEED {src} 1 MOLE
TEMP {TRES_F:.0f} F
PRES {pstr} PSIA
FLASH
MIX FEED EQL 1 MOLE
DLE ID "OIL SEP FLUID={name} P={pstr} PSIA: {SEP_P1:.0f}/{SEP_T:.0f}F | {SEP_P2:.3f}/{SEP_T:.0f}F -> Rs=CumGOR, Bo=1/LiqVol"
{SURFACE_SEP}
MIX FEED {src} 1 MOLE
TEMP {TRES_F:.0f} F
PRES {pstr} PSIA
FLASH
MIX FEED EQV 1 MOLE
DLE ID "GAS SEP FLUID={name} P={pstr} PSIA: {SEP_P1:.0f}/{SEP_T:.0f}F | {SEP_P2:.3f}/{SEP_T:.0f}F -> rs=1e6/CumGOR"
{SURFACE_SEP}
"""


def envelope_block(name, src):
    """P-T phase envelope for a near-critical sibling: prints the EOS-true
    critical point, cricondenbar, cricondentherm, and the full P-T locus."""
    return f"""
NOTE: "===ENVELOPE=== FLUID={name}  (P-T phase envelope -> Critical Point, Cricondenbar)"
MIX FEED {src} 1 MOLE
TEMP {TRES_F:.0f} F
PRES psia
P-T ID "P-T envelope FLUID={name}" P_vs_T
"""


def build():
    parts = [header()]

    # --- saturated locus: psat + one locus flash per sibling ---
    parts.append("\n; ============== SATURATED LOCUS (psat + locus flash per sibling) ==============\n")
    n_flash = 0
    for fr in LADDER_FRACS:
        name = sib_name(fr)
        src = name
        parts.append(f"\n; ----- sibling {name} (GASREF fraction {fr:.2f}) -----\n")
        parts.append(satp_block(name, src))
        # Locus flash(es) at pressures we KNOW are below this sibling's psat, so
        # the flash is two-phase. psat isn't known at generate time, so we place
        # the flash at SAT_FRAC times a monotone psat(frac) estimate calibrated
        # off the recomb-deck anchors (_psat_guess). The EXACT psat comes from
        # SATP in the .Out; here we only need a safe sub-psat flash pressure.
        for p in locus_pressures(fr):
            tag = "LOCUS-NEARCRIT" if fr in NEAR_CRIT_FRACS else "LOCUS"
            parts.append(locus_flash_block(name, src, p, tag))
            n_flash += 1

    # --- near-critical envelopes (EOS-true critical point); PhazeComp >= 2.0.0 only ---
    n_env = 0
    if EMIT_ENVELOPES:
        parts.append("\n; ============== EOS CRITICAL POINTS (P-T envelopes, near-critical siblings) ==============\n")
        for fr in NEAR_CRIT_FRACS:
            name = sib_name(fr)
            parts.append(envelope_block(name, name))
            n_env += 1
        # Also envelope the original feed GOR2000 as a cross-check reference point.
        parts.append(envelope_block("GOR2000", "GOR2000"))
        n_env += 1
    else:
        parts.append("\n; ====== EOS CRITICAL POINTS (P-T envelopes) OMITTED: require PhazeComp >= 2.0.0 ======\n"
                     "; 1.8.1 has no critical-point calc; set EMIT_ENVELOPES=True and regenerate on 2.0.0+.\n")

    parts.append("\nEND\n")
    return "".join(parts), n_flash, n_env


# Saturation pressures rise monotonically with GASREF fraction. Calibrated from
# the recomb deck (.Out): OILREF~2000, SIB20(0.20)~2720, SIB45(0.45)~3838,
# GOR2000 feed~4110. We extrapolate a smooth psat(frac) to place each sibling's
# locus flashes BELOW its own psat (so the flash is two-phase). The exact psat
# comes from SATP in the .Out; here we only need a safe sub-psat flash pressure.
def _psat_guess(frac):
    """Monotone-increasing psat estimate (psia) from the recomb-deck anchors.
    Piecewise-linear in frac through (0.00,2000),(0.20,2720),(0.45,3838),
    then a steeper near-critical rise toward ~p_c for high fractions."""
    anchors = [(0.00, 2000.0), (0.20, 2720.0), (0.45, 3838.0),
               (0.58, 4600.0), (0.66, 5100.0), (0.74, 5400.0)]
    if frac <= anchors[0][0]:
        return anchors[0][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if frac <= x1 + 1e-9:
            return y0 + (y1 - y0) * (frac - x0) / (x1 - x0)
    return anchors[-1][1]


def locus_pressures(frac):
    """Flash pressures for a sibling's locus point(s), all safely below its psat
    so the flash is two-phase. Near-critical siblings also get sub-psat flashes
    for phase properties approaching criticality. SAT_FRAC keeps the top flash
    a hair below psat (the locus point); SUBPSAT_FRACS add deeper interior points."""
    psat = _psat_guess(frac)
    ps = [round(SAT_FRAC * psat, 2)]
    if frac in NEAR_CRIT_FRACS:
        ps += [round(fr * psat, 2) for fr in SUBPSAT_FRACS]
    # de-dup, descending
    return sorted(set(ps), reverse=True)


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(here, "phz", "band-validation")
    os.makedirs(out_dir, exist_ok=True)
    deck, n_flash, n_env = build()
    out = os.path.join(out_dir, "Permian-band-validation.phz")
    with open(out, "w") as f:
        f.write(deck)
    n_sib = len(LADDER_FRACS)
    print(f"wrote {out}")
    print(f"  siblings (recombination ladder): {n_sib} "
          f"(GASREF frac {LADDER_FRACS[0]:.2f} .. {LADDER_FRACS[-1]:.2f})")
    print(f"  SATP saturation-pressure calcs : {n_sib}")
    print(f"  locus flashpoints (2-phase)    : {n_flash}")
    print(f"  P-T envelopes (critical points): {n_env} "
          f"(near-critical siblings {', '.join(sib_name(fr) for fr in NEAR_CRIT_FRACS)} + GOR2000)")
    print(f"  reference split: GOR2000 flashed at {REF_P:.0f} psia -> OILREF (EQL) + GASREF (EQV)")
    print(f"  includes: eos.inc, compositions.inc (copied into {out_dir})")
    print("  run in PhazeComp 1.8.1 off-machine; bring back Permian-band-validation.Out")
