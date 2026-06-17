"""Locate the EOS convergence/critical pressure at reservoir T from SATP K-values.

WHY (the 1.8.1 route). PhazeComp 1.8.1 has no critical-point calculation -- the
P-T phase envelope that prints "Critical Point: <T>, <P>" was added in 2.0.0.
But the critical point is simply the saturation state at which every component's
equilibrium K-value collapses to 1 (the incipient phase becomes identical to the
bulk), and `SATP` already prints every component's K-value. So the EOS p_c at
reservoir temperature is reachable on 1.8.1 with SATP alone.

Aaron Zick (PhazeComp's author, email 13 Dec 2023 in phz/band-validation/) uses
exactly this mechanism the other way round -- to *tune* an EOS to a *known*
critical point he matches the SATP saturation pressure to p_c while driving the
heavy-component K-value to 1. We invert it to *locate* the EOS's own p_c:

  SATP a recombination ladder spanning oil-like -> near-critical compositions at
  T_res; for each sibling read the K-value spread S = max_i |ln K_i| (zero at
  criticality); extrapolate S -> 0 against saturation pressure to get p_c at
  T_res -- the same convergence pressure the molar-volume crossing and Singh
  App. B (SPE 109596) estimate, now from an EOS truth, no 2.0.0 envelope needed.

Cross-check signal: SATP also tells us whether the bulk is liquid (bubble point)
or vapour (dew point). The critical composition is the bubble->dew flip; p_c is
the saturation-locus apex there. We report the flip alongside the S->0 root.

This reads a completed PhazeComp .Out (run the band-validation deck first). Usage:

    python3 scripts/locate_pc_from_satp.py <run.Out> [--nodes N] \
        [--crossing PK] [--singh PK]

--crossing / --singh print botkit's estimates next to the EOS p_c for scoring.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# "One Phase at Temperature = 200 F, Pressure = 3838.37 psia:"
_PHASE_HDR = re.compile(
    r"One Phase at Temperature\s*=\s*([-\d.eE+]+)\s*F,\s*Pressure\s*=\s*([-\d.eE+]+)\s*psia",
    re.IGNORECASE,
)
# component row: name  overall  liquid  vapor  kvalue
_ROW = re.compile(
    r"^\s*(\S+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*$"
)
_SATP_NOTE = re.compile(r"===SATP===\s*FLUID=(\S+)", re.IGNORECASE)


@dataclass
class SatNode:
    psat: float
    temp: float
    fluid: Optional[str]
    sat_type: str                       # "bubble" | "dew" | "?"
    k: Dict[str, float] = field(default_factory=dict)

    @property
    def spread(self) -> float:
        """K-value spread S = max|ln K|; -> 0 at the critical point."""
        lnk = [abs(math.log(v)) for v in self.k.values() if v > 0]
        return max(lnk) if lnk else float("nan")

    @property
    def k_extremes(self):
        ks = list(self.k.values())
        return (min(ks), max(ks)) if ks else (float("nan"), float("nan"))


def _classify(overall: float, liquid: float, vapor: float) -> Optional[str]:
    """Per-row phase vote: bubble if the bulk matches the liquid column, dew if it
    matches the vapour column. Aggregated across rows by the caller."""
    if abs(overall - liquid) < abs(overall - vapor):
        return "bubble"
    if abs(overall - vapor) < abs(overall - liquid):
        return "dew"
    return None


def parse_satp(text: str) -> List[SatNode]:
    """Pull every SATP block out of a PhazeComp .Out as SatNodes (psat ascending)."""
    lines = text.splitlines()
    nodes: List[SatNode] = []
    # index of every saturation-calc header so we can scope each block
    calc_idx = [i for i, ln in enumerate(lines)
                if ln.startswith("Saturation Pressure Calculation")]
    bounds = calc_idx + [len(lines)]

    for start, end in zip(bounds, bounds[1:]):
        block = lines[start:end]
        # the most recent ===SATP=== FLUID= tag at or above this calc
        fluid = None
        for j in range(start, -1, -1):
            m = _SATP_NOTE.search(lines[j])
            if m:
                fluid = m.group(1)
                break
            if lines[j].startswith("Saturation Pressure Calculation") and j != start:
                break

        # Find the "One Phase at ... Pressure = <psat> psia:" header, then read
        # ONLY the single component table immediately below it (the saturation
        # K-values). Stop at the blank line that ends the table -- the rest of the
        # block holds unrelated flash/DLE tables whose K-values are not the
        # saturation state and must not pollute this node.
        temp = psat = None
        k: Dict[str, float] = {}
        votes = {"bubble": 0, "dew": 0}
        hdr_at = next((i for i, ln in enumerate(block) if _PHASE_HDR.search(ln)), None)
        if hdr_at is not None:
            h = _PHASE_HDR.search(block[hdr_at])
            temp, psat = float(h.group(1)), float(h.group(2))
            started = False
            for ln in block[hdr_at + 1:]:
                r = _ROW.match(ln)
                if not r:
                    if started:           # blank/non-row line ends the table
                        break
                    continue
                name = r.group(1)
                if not re.search(r"[A-Za-z0-9]", name):   # the "----" separator row
                    continue
                try:
                    overall, liquid, vapor, kv = (float(r.group(i)) for i in range(2, 6))
                except ValueError:                         # dashed/non-numeric field
                    continue
                if name.lower() in ("component", "overall") or kv <= 0:
                    continue
                started = True
                k[name] = kv
                v = _classify(overall, liquid, vapor)
                if v:
                    votes[v] += 1
        if psat is None or not k:
            continue
        sat_type = ("bubble" if votes["bubble"] >= votes["dew"] else "dew") \
            if (votes["bubble"] or votes["dew"]) else "?"
        nodes.append(SatNode(psat=psat, temp=temp, fluid=fluid,
                             sat_type=sat_type, k=k))

    nodes.sort(key=lambda n: n.psat)
    return nodes


def locate_pc(nodes: List[SatNode], n_nodes: int = 4):
    """EOS p_c at T_res from the K-value spread S(psat) = max|ln K| (zero at the
    critical point). Returns ``(pc, method)``.

    The spread is V-shaped in saturation pressure: it falls as the ladder climbs
    toward criticality and rises again past it (the dew side). Two regimes:

    * **Bracketed** -- the ladder crosses the minimum (an interior S-minimum, i.e.
      the deck reached and passed the critical mix). p_c is the vertex of a
      parabola through the minimum node and its two neighbours; this is the
      precise estimate and is what a near-critical ladder should hit.
    * **One-sided** -- S still decreasing at the top node (ladder has not yet
      reached critical). Fall back to a linear S->0 extrapolation over the top
      ``n_nodes`` (approximate; densify the ladder for a real bracket).

    ``(nan, reason)`` if neither applies.
    """
    if len(nodes) < 3:
        return float("nan"), "need >=3 nodes"
    p = np.array([n.psat for n in nodes])
    s = np.array([n.spread for n in nodes])
    if not np.all(np.isfinite(s)):
        return float("nan"), "non-finite spread"

    imin = int(np.argmin(s))
    if 0 < imin < len(s) - 1:                       # interior minimum -> bracketed
        x0, x1, x2 = p[imin - 1], p[imin], p[imin + 1]
        y0, y1, y2 = s[imin - 1], s[imin], s[imin + 1]
        # vertex of the parabola through the three points
        d = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if d != 0:
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
            b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
            if a > 0:                                # upward parabola -> real min
                return float(-b / (2 * a)), "K-spread minimum (bracketed)"
        return float(p[imin]), "K-spread minimum node (bracketed)"

    # one-sided: linear S->0 over the top nodes
    top = slice(-min(n_nodes, len(nodes)), None)
    slope, intercept = np.polyfit(p[top], s[top], 1)
    if slope >= 0:
        return float("nan"), "spread not decreasing"
    pc = -intercept / slope
    if not np.isfinite(pc) or pc <= p[-1]:
        return float("nan"), "root below top node"
    return float(pc), "S->0 extrapolation (one-sided, not yet bracketed)"


def bubble_dew_flip(nodes: List[SatNode]) -> Optional[float]:
    """Saturation pressure bracketed by the last bubble and first dew node, i.e.
    the locus apex ~ p_c. ``None`` if the ladder never crosses to the dew side."""
    last_bubble = first_dew = None
    for n in nodes:
        if n.sat_type == "bubble":
            last_bubble = n.psat
        elif n.sat_type == "dew" and first_dew is None and last_bubble is not None:
            first_dew = n.psat
            break
    if last_bubble is not None and first_dew is not None:
        return 0.5 * (last_bubble + first_dew)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="completed PhazeComp .Out file")
    ap.add_argument("--nodes", type=int, default=4,
                    help="top-N ladder nodes for the S->0 extrapolation (default 4)")
    ap.add_argument("--crossing", type=float, default=None,
                    help="botkit molar-volume crossing Pk, for scoring")
    ap.add_argument("--singh", type=float, default=None,
                    help="botkit Singh App. B Pk, for scoring")
    args = ap.parse_args(argv)

    with open(args.out) as f:
        nodes = parse_satp(f.read())
    if not nodes:
        print("No SATP blocks found in", args.out)
        return 1

    print(f"SATP ladder from {args.out} ({len(nodes)} nodes):\n")
    print(f"  {'fluid':>8}  {'psat':>9}  {'type':>6}  {'S=max|lnK|':>10}  "
          f"{'Kmin':>8}  {'Kmax':>8}")
    for n in nodes:
        kmin, kmax = n.k_extremes
        print(f"  {(n.fluid or '-'):>8}  {n.psat:9.1f}  {n.sat_type:>6}  "
              f"{n.spread:10.4f}  {kmin:8.4f}  {kmax:8.3f}")

    pc, method = locate_pc(nodes, n_nodes=args.nodes)
    flip = bubble_dew_flip(nodes)
    print()
    if math.isfinite(pc):
        print(f"EOS p_c at T_res [{method}]: {pc:8.1f} psia")
    else:
        print(f"EOS p_c: unavailable -- {method}.")
    if flip is not None:
        print(f"EOS p_c cross-check (bubble->dew flip apex):            {flip:8.1f} psia")
    else:
        print("EOS p_c cross-check: ladder stays on the bubble side (no dew flip) "
              "-- extend the GASREF fraction higher to bracket the critical mix.")

    if args.crossing is not None or args.singh is not None:
        print("\nScoring botkit Pk estimates vs EOS p_c"
              + (f" = {pc:.1f} psia:" if math.isfinite(pc) else " (unavailable):"))
        for label, val in (("molar-volume crossing", args.crossing),
                           ("Singh App. B average", args.singh)):
            if val is None:
                continue
            if math.isfinite(pc):
                err = 100.0 * (val - pc) / pc
                print(f"  {label:<22}: {val:8.1f} psia  ({err:+5.1f}% vs EOS p_c)")
            else:
                print(f"  {label:<22}: {val:8.1f} psia")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
