"""botkit -- Black-Oil Table quality-control and extension toolkit.

The underlying two-pseudocomponent (modified-black-oil) K-value formulation is
Whitson & Torp (JPT 1983, SPE 10067) and Coats (SPE 50990).  The consistent
table modification this toolkit automates -- negative-compressibility QC,
extension to the convergence pressure, near-critical consistency -- is Singh,
Fevang & Whitson, SPE 109596 (2007).  Added here: automated quality-control
detectors, analytical convergence-pressure estimation, and an EOS fallback gate.
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
