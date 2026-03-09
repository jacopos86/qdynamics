#!/usr/bin/env python3
"""Shared PDF rendering utilities for all pipeline report generators.

This module centralises matplotlib setup, page-rendering helpers, and the
parameter-manifest builder so that pipeline files can import them instead
of duplicating ~80 lines each.

Usage (from any pipeline)::

    from docs.reports.pdf_utils import (
        require_matplotlib,
        get_plt,
        get_PdfPages,
        render_text_page,
        render_command_page,
        render_info_page,
        render_compact_table,
        render_parameter_manifest,
        current_command_string,
        HAS_MATPLOTLIB,
    )

Design:
- matplotlib is imported lazily via the try/except block at module level.
- ``require_matplotlib()`` raises a clean RuntimeError if absent.
- ``get_plt()`` / ``get_PdfPages()`` return the real objects (or raise).
- All ``render_*`` functions accept a ``PdfPages`` handle and produce
  one or more pages.  They call ``require_matplotlib()`` internally.
"""

from __future__ import annotations

import os
import shlex
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lazy matplotlib import
# ---------------------------------------------------------------------------

_MPL_IMPORT_ERROR: str | None = None
_plt = None
_PdfPages = None

try:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt_mod
    from matplotlib.backends.backend_pdf import PdfPages as _PdfPagesReal

    _plt = _plt_mod
    _PdfPages = _PdfPagesReal
    HAS_MATPLOTLIB: bool = True
except Exception as exc:  # pragma: no cover
    HAS_MATPLOTLIB = False
    _MPL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def require_matplotlib() -> None:
    """Raise ``RuntimeError`` if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        err = _MPL_IMPORT_ERROR or "not installed"
        raise RuntimeError(
            f"matplotlib is required for PDF output. "
            f"Install matplotlib or run with --skip-pdf. Original error: {err}"
        )


def get_plt():  # noqa: ANN201 – returns matplotlib.pyplot
    """Return ``matplotlib.pyplot``, raising if unavailable."""
    require_matplotlib()
    return _plt


def get_PdfPages():  # noqa: ANN201, N802
    """Return the ``PdfPages`` class, raising if unavailable."""
    require_matplotlib()
    return _PdfPages


# ---------------------------------------------------------------------------
# Text / command / info page helpers
# ---------------------------------------------------------------------------

_DEFAULT_FIGSIZE = (11.0, 8.5)


def render_text_page(
    pdf: Any,
    lines: list[str],
    *,
    fontsize: int = 9,
    line_spacing: float = 0.028,
    max_line_width: int = 115,
) -> None:
    """Render *lines* onto one or more text-only PDF pages.

    Long lines are wrapped at *max_line_width*.  When the page fills up
    a new page is started automatically.
    """
    require_matplotlib()
    plt = get_plt()

    expanded: list[str] = []
    for raw in lines:
        if len(raw) <= max_line_width:
            expanded.append(raw)
        else:
            expanded.extend(
                textwrap.wrap(raw, width=max_line_width, subsequent_indent="    ")
            )

    fig = plt.figure(figsize=_DEFAULT_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")

    x0, y = 0.05, 0.95
    for line in expanded:
        ax.text(
            x0, y, line,
            transform=ax.transAxes,
            va="top", ha="left",
            family="monospace",
            fontsize=fontsize,
        )
        y -= line_spacing
        if y < 0.02:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=_DEFAULT_FIGSIZE)
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.95

    pdf.savefig(fig)
    plt.close(fig)


def render_command_page(
    pdf: Any,
    command: str,
    *,
    script_name: str = "",
    extra_header_lines: list[str] | None = None,
) -> None:
    """Render a "command executed" page with optional header metadata.

    Parameters
    ----------
    pdf : PdfPages
    command : str
        The full CLI invocation string.
    script_name : str, optional
        Displayed as ``Script: <script_name>`` in the header.
    extra_header_lines : list[str], optional
        Additional lines inserted between the header and the command.
    """
    lines = [
        "Executed Command",
        "",
        "Reference: pipelines/run_guide.md",
    ]
    if script_name:
        lines.append(f"Script: {script_name}")
    if extra_header_lines:
        lines.extend(extra_header_lines)
    lines += ["", command]
    render_text_page(pdf, lines, fontsize=10, line_spacing=0.03, max_line_width=110)


_INFO_BBOX = dict(
    boxstyle="round,pad=0.6",
    facecolor="#f7f7f7",
    edgecolor="#cccccc",
    alpha=0.92,
)


def render_info_page(
    pdf: Any,
    info_text: str,
    title: str = "",
) -> None:
    """Render a dedicated info / metrics page with a grey rounded box."""
    require_matplotlib()
    plt = get_plt()

    fig = plt.figure(figsize=_DEFAULT_FIGSIZE)
    ax = fig.add_subplot(111)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    ax.text(
        0.05, 0.92, info_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        family="monospace",
        bbox=_INFO_BBOX,
    )
    pdf.savefig(fig)
    plt.close(fig)


def render_compact_table(
    ax: Any,
    *,
    title: str,
    col_labels: list[str],
    rows: list[list[str]],
    fontsize: int = 7,
) -> None:
    """Render a compact table on an existing *ax*.

    Unlike other ``render_*`` functions this operates on a subplot axis,
    not a full PdfPages handle — call it inside a figure you manage.
    """
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=6)
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.3)


# ---------------------------------------------------------------------------
# Parameter manifest (AGENTS.md §1 requirement)
# ---------------------------------------------------------------------------

def render_parameter_manifest(
    pdf: Any,
    *,
    model: str = "Hubbard",
    ansatz: str = "",
    drive_enabled: bool = False,
    t: float = 0.0,
    U: float = 0.0,
    dv: float = 0.0,
    extra: dict[str, Any] | None = None,
    command: str = "",
) -> None:
    """First-page parameter manifest as required by AGENTS.md.

    This produces a clean, list-style summary of every run-defining
    parameter so that the physics of the PDF can be reproduced.
    """
    lines = [
        "=" * 60,
        "  PARAMETER MANIFEST",
        "=" * 60,
        "",
        f"  Model           : {model}",
        f"  Ansatz          : {ansatz or '(not specified)'}",
        f"  Drive enabled   : {drive_enabled}",
        "",
        f"  t (hopping)     : {t}",
        f"  U (interaction) : {U}",
        f"  dv (disorder)   : {dv}",
    ]
    if extra:
        lines.append("")
        for k, v in extra.items():
            lines.append(f"  {k:<16}: {v}")
    if command:
        lines += ["", "  Command:", f"    {command}"]
    lines += ["", "=" * 60]
    render_text_page(pdf, lines, fontsize=10, line_spacing=0.028, max_line_width=110)


# ---------------------------------------------------------------------------
# Misc shared helpers
# ---------------------------------------------------------------------------

def current_command_string() -> str:
    """Return the full CLI invocation as a shell-safe string."""
    return " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv])
