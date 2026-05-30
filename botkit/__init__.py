"""botkit -- Black-Oil Table quality-control and extension toolkit.

Implements the Black-Oil Table extension method of Singh & Whitson,
SPE 109596 (2007), with automated quality-control detectors, analytical
convergence-pressure estimation, and an EOS fallback gate.
"""

from .model import (
    AUTO,
    Anomaly,
    BlackOilTable,
    Change,
    ChangeLog,
    Config,
    Diagnostics,
    PVTGTable,
    PVTOTable,
    Severity,
    SurfaceFluids,
)
from .io import read_excel, read_eclipse, write_eclipse
from . import pipeline
from .qc import run_qc
from .report import (
    changes_to_markdown,
    changes_to_text,
    diagnostics_to_json,
    diagnostics_to_markdown,
    plot_table,
)

__all__ = [
    "AUTO",
    "Anomaly",
    "BlackOilTable",
    "Config",
    "Diagnostics",
    "PVTGTable",
    "PVTOTable",
    "Severity",
    "SurfaceFluids",
    "read_excel",
    "read_eclipse",
    "write_eclipse",
    "pipeline",
    "run_qc",
    "Change",
    "ChangeLog",
    "changes_to_markdown",
    "changes_to_text",
    "diagnostics_to_json",
    "diagnostics_to_markdown",
    "plot_table",
]

__version__ = "0.1.0"
