"""Read and write Black-Oil Tables.

Supports the project's Excel workbook layout (saturated PVTO/PVTG sheets) and
Eclipse-style ``PVTO`` / ``PVTG`` keyword decks (with grouped undersaturated
rows).  The Eclipse writer follows the deck layout used by the original
SPE 109596 notebook.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd

from .model import BlackOilTable, PVTGTable, PVTOTable, SurfaceFluids


# --- Excel ---------------------------------------------------------------

def read_excel(path: str, pvto_sheet: str = "PVTO", pvtg_sheet: str = "PVTG",
               surface: Optional[SurfaceFluids] = None) -> BlackOilTable:
    """Read saturated PVTO/PVTG sheets from an Excel workbook.

    Expected columns: PVTO -> [Rs, Pb, Bo, uo]; PVTG -> [P, Rv, Bg, ug].
    The workbook carries saturated rows only; undersaturated branches start
    empty and are populated downstream.
    """
    dfo = pd.read_excel(path, sheet_name=pvto_sheet)
    dfg = pd.read_excel(path, sheet_name=pvtg_sheet)

    pvto = PVTOTable(rs=dfo["Rs"], p=dfo["Pb"], bo=dfo["Bo"], uo=dfo["uo"])
    pvtg = PVTGTable(p=dfg["P"], rv=dfg["Rv"], bg=dfg["Bg"], ug=dfg["ug"])
    return BlackOilTable(pvto=pvto, pvtg=pvtg, surface=surface)


# --- Eclipse parsing -----------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove Eclipse ``--`` line comments."""
    out = []
    for line in text.splitlines():
        idx = line.find("--")
        out.append(line if idx < 0 else line[:idx])
    return "\n".join(out)


def _records(block: str) -> List[List[float]]:
    """Split a keyword body into slash-terminated records of float tokens.

    Each record is the run of numeric tokens up to the next ``/``.  An empty
    record (a lone slash) marks the end of the keyword and is dropped.
    """
    records: List[List[float]] = []
    current: List[float] = []
    for token in re.split(r"\s+", block.strip()):
        if token == "":
            continue
        if token == "/":
            records.append(current)
            current = []
            continue
        # token may be like "0.771/" without surrounding whitespace
        if token.endswith("/"):
            num = token[:-1]
            if num:
                current.append(float("nan") if num == "*" else float(num))
            records.append(current)
            current = []
            continue
        # '*' is the Eclipse default marker (e.g. water density placeholder)
        current.append(float("nan") if token == "*" else float(token))
    if current:
        records.append(current)
    # the keyword terminator is an empty record; drop trailing empties
    return [r for r in records if r]


def _keyword_body(text: str, keyword: str) -> Optional[str]:
    """Return the text following ``keyword`` up to its closing lone slash."""
    text = _strip_comments(text)
    m = re.search(rf"(?mi)^\s*{keyword}\b", text)
    if not m:
        return None
    rest = text[m.end():]
    # the keyword's data ends at the first standalone '/' on its own line
    end = re.search(r"(?m)^\s*/\s*$", rest)
    return rest[: end.start()] if end else rest


def _parse_pvt_records(records: List[List[float]], n_props: int = 3):
    """Split records into (key, saturated_triple, undersaturated_rows).

    Each record is ``[key, c0, c1, c2, c0', c1', c2', ...]``: the key followed
    by groups of ``n_props`` columns.  The first group is the saturated row,
    the remainder are undersaturated rows.
    """
    keys, sats, usats = [], [], []
    for rec in records:
        key = rec[0]
        cols = np.asarray(rec[1:], dtype=float)
        if cols.size % n_props != 0:
            raise ValueError(f"PVT record has {cols.size} property values, "
                             f"not a multiple of {n_props}: {rec}")
        rows = cols.reshape(-1, n_props)
        keys.append(key)
        sats.append(rows[0])
        usats.append(rows[1:].copy() if len(rows) > 1 else np.empty((0, n_props)))
    return keys, sats, usats


def read_eclipse(path: str, surface: Optional[SurfaceFluids] = None) -> BlackOilTable:
    """Read a Black-Oil Table from an Eclipse PVTO/PVTG deck."""
    with open(path, "r") as fh:
        text = fh.read()

    # surface densities from a DENSITY keyword, if present. DENSITY is a single
    # record terminated by an inline slash, so read only up to the first '/'.
    if surface is None:
        clean = _strip_comments(text)
        m = re.search(r"(?mi)^\s*DENSITY\b", clean)
        if m:
            after = clean[m.end():]
            first_slash = after.find("/")
            body = after[:first_slash] if first_slash >= 0 else after
            vals = _records(body + " /")
            if vals and len(vals[0]) >= 3:
                deno, _denw, deng = vals[0][0], vals[0][1], vals[0][2]
                surface = SurfaceFluids(st_oil_density=deno, st_gas_density=deng)

    obody = _keyword_body(text, "PVTO")
    gbody = _keyword_body(text, "PVTG")
    if obody is None or gbody is None:
        raise ValueError("Deck must contain both PVTO and PVTG keywords")

    # PVTO record: [Rs, P, Bo, uo, P', Bo', uo', ...]
    rs, osat, ousat = _parse_pvt_records(_records(obody))
    pvto = PVTOTable(
        rs=np.array(rs),
        p=np.array([s[0] for s in osat]),
        bo=np.array([s[1] for s in osat]),
        uo=np.array([s[2] for s in osat]),
        usat=ousat,
    )

    # PVTG record: [P, Rv, Bg, ug, Rv', Bg', ug', ...]
    pdew, gsat, gusat = _parse_pvt_records(_records(gbody))
    pvtg = PVTGTable(
        p=np.array(pdew),
        rv=np.array([s[0] for s in gsat]),
        bg=np.array([s[1] for s in gsat]),
        ug=np.array([s[2] for s in gsat]),
        usat=gusat,
    )
    return BlackOilTable(pvto=pvto, pvtg=pvtg, surface=surface)


# --- Eclipse writing -----------------------------------------------------

def _fmt(value: float, width: int = 16, prec: int = 6) -> str:
    return f"{value:>{width}.{prec}g}"


def write_eclipse(table: BlackOilTable, path: Optional[str] = None,
                  header: str = "") -> str:
    """Render a Black-Oil Table as an Eclipse PVTO/PVTG deck.

    If ``path`` is given the deck is also written to disk.  Returns the deck
    text.  Undersaturated rows attached to each saturated node are emitted
    beneath it, with one slash terminating each saturated record.
    """
    lines: List[str] = []
    if header:
        for line in header.splitlines():
            lines.append(f"-- {line}")

    s = table.surface
    if s is not None:
        lines.append("DENSITY")
        lines.append("--  oil (lbm/ft3)    water (lbm/ft3)   gas (lbm/ft3)")
        lines.append(f"{_fmt(s.st_oil_density)} {'*':>16} "
                     f"{_fmt(s.st_gas_density)} /")
        lines.append("")

    o = table.pvto
    lines.append("PVTO")
    lines.append("-- Rs(Mscf/bbl)      P(psia)         Bo(rb/stb)      uo(cP)")
    for i in range(o.n):
        rows = [(o.p[i], o.bo[i], o.uo[i])] + [tuple(r) for r in o.usat[i]]
        for j, (p, bo, uo) in enumerate(rows):
            key = _fmt(o.rs[i]) if j == 0 else " " * 16
            term = " /" if j == len(rows) - 1 else ""
            lines.append(f"{key} {_fmt(p)} {_fmt(bo)} {_fmt(uo)}{term}")
    lines.append("/")
    lines.append("")

    g = table.pvtg
    lines.append("PVTG")
    lines.append("-- Pdew(psia)        Rv(bbl/Mscf)    Bg(rb/Mscf)     ug(cP)")
    for i in range(g.n):
        rows = [(g.rv[i], g.bg[i], g.ug[i])] + [tuple(r) for r in g.usat[i]]
        for j, (rv, bg, ug) in enumerate(rows):
            key = _fmt(g.p[i]) if j == 0 else " " * 16
            term = " /" if j == len(rows) - 1 else ""
            lines.append(f"{key} {_fmt(rv)} {_fmt(bg)} {_fmt(ug)}{term}")
    lines.append("/")
    lines.append("")

    deck = "\n".join(lines)
    if path is not None:
        with open(path, "w") as fh:
            fh.write(deck)
    return deck
