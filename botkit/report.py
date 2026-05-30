"""Render QC diagnostics as markdown / JSON and plot Black-Oil Tables."""

from __future__ import annotations

import json
from typing import Optional

from .model import BlackOilTable, ChangeLog, Diagnostics, Severity

_ORDER = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
_ICON = {Severity.ERROR: "[ERROR]", Severity.WARN: "[WARN]", Severity.INFO: "[INFO]"}


def diagnostics_to_markdown(diag: Diagnostics, suggestions: Optional[dict] = None) -> str:
    """Severity-ranked markdown summary of the QC anomalies."""
    lines = ["# Black-Oil Table QC report", ""]
    counts = {s: len(diag.by_severity(s)) for s in Severity}
    lines.append(f"**{counts[Severity.ERROR]} error(s), "
                 f"{counts[Severity.WARN]} warning(s), "
                 f"{counts[Severity.INFO]} note(s).**")
    lines.append("")

    if suggestions:
        lines.append("## Suggested configuration")
        for key, val in suggestions.items():
            lines.append(f"- `{key}` = {val:g}" if isinstance(val, (int, float))
                         else f"- `{key}` = {val}")
        lines.append("")

    if not diag.anomalies:
        lines.append("No anomalies detected.")
        return "\n".join(lines)

    lines.append("## Anomalies")
    for a in sorted(diag.anomalies, key=lambda x: _ORDER[x.severity]):
        lines.append(f"### {_ICON[a.severity]} {a.kind} - {a.location}")
        lines.append(a.message)
        if a.suggested_fix:
            lines.append(f"*Suggested fix:* {a.suggested_fix}")
        lines.append("")
    return "\n".join(lines)


def diagnostics_to_json(diag: Diagnostics, suggestions: Optional[dict] = None,
                        path: Optional[str] = None) -> str:
    """Machine-readable JSON of the diagnostics and suggestions."""
    payload = diag.to_dict()
    if suggestions:
        payload["suggestions"] = suggestions
    text = json.dumps(payload, indent=2)
    if path is not None:
        with open(path, "w") as fh:
            fh.write(text)
    return text


def changes_to_markdown(changes: ChangeLog) -> str:
    """Markdown summary of the fixes the pipeline applied, with their reasons."""
    lines = ["# Black-Oil Table change summary", ""]
    if len(changes) == 0:
        lines.append("No changes were applied; the table was extended without "
                     "trimming or correction.")
        return "\n".join(lines)
    lines.append(f"{len(changes)} change(s) were applied:")
    lines.append("")
    for i, c in enumerate(changes.changes, 1):
        lines.append(f"{i}. **{c.action}**")
        lines.append(f"   _Why:_ {c.reason}")
        if c.detail:
            lines.append(f"   {c.detail}")
        lines.append("")
    return "\n".join(lines)


def changes_to_text(changes: ChangeLog) -> str:
    """Plain-text change summary (for an Eclipse deck header)."""
    if len(changes) == 0:
        return "No corrections applied; table extended only."
    out = [f"Change summary ({len(changes)} applied):"]
    for i, c in enumerate(changes.changes, 1):
        out.append(f"{i}. {c.action}")
        out.append(f"   Why: {c.reason}")
    return "\n".join(out)


def plot_table(table: BlackOilTable, extended: Optional[BlackOilTable] = None,
               path: Optional[str] = None):
    """Plot Bo, Bg, Rs, Rv, uo, ug; overlay an extended table if supplied."""
    import matplotlib.pyplot as plt

    o, g = table.pvto, table.pvtg
    fig, ax = plt.subplots(3, 2, figsize=(14, 14))
    panels = [
        (ax[0, 0], o.p, o.bo, "Bo (rb/stb)", "green"),
        (ax[0, 1], g.p, g.bg, "Bg (rb/Mscf)", "green"),
        (ax[1, 0], o.p, o.rs, "Rs (Mscf/bbl)", "red"),
        (ax[1, 1], g.p, g.rv, "Rv (bbl/Mscf)", "red"),
        (ax[2, 0], o.p, o.uo, "uo (cP)", "blue"),
        (ax[2, 1], g.p, g.ug, "ug (cP)", "blue"),
    ]
    ext_map = {}
    if extended is not None:
        eo, eg = extended.pvto, extended.pvtg
        ext_map = {
            "Bo (rb/stb)": (eo.p, eo.bo), "Bg (rb/Mscf)": (eg.p, eg.bg),
            "Rs (Mscf/bbl)": (eo.p, eo.rs), "Rv (bbl/Mscf)": (eg.p, eg.rv),
            "uo (cP)": (eo.p, eo.uo), "ug (cP)": (eg.p, eg.ug),
        }
    for a, x, y, label, color in panels:
        a.plot(x, y, "o-", color=color, ms=4, label="table")
        if label in ext_map:
            ex, ey = ext_map[label]
            a.plot(ex, ey, "--", color=color, lw=1.5, label="extended")
        a.set_xlabel("Pressure (psia)")
        a.set_ylabel(label)
        a.grid(True, which="both")
        a.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110)
    return fig
