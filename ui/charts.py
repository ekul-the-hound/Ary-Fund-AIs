"""
report/charts.py
================
Chart artifact handling for the PDF report.

The report generator does not produce charts on its own. It consumes
artifacts the analytics layer already created. This module is the
adapter: turn whatever the caller hands us — a file path, a PNG bytes
blob, a matplotlib Figure, or a chart-ready data dict — into a
ReportLab ``Image`` flowable sized to the report column.

If a chart cannot be loaded (missing file, unsupported format, render
error), we return a labeled placeholder rather than raising. The report
must keep rendering even if one chart is broken.

Supported inputs
----------------
* ``str`` / ``pathlib.Path`` pointing to a ``.png``, ``.jpg``, ``.jpeg``,
  ``.gif``, or ``.svg`` file.
* ``bytes`` containing PNG/JPG image data.
* ``matplotlib.figure.Figure`` object (we ``savefig`` to PNG in memory).
* ``dict`` with shape ``{"type": "line", "x": [...], "y": [...], ...}``
  for the small set of chart types the analytics layer emits when no
  pre-rendered artifact exists. This is the last-resort fallback path.

SVG support requires ``svglib`` (which uses reportlab.graphics). If it
isn't installed, SVG artifacts produce a placeholder.
"""
from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, Spacer, Flowable, KeepTogether
from reportlab.platypus.flowables import HRFlowable

from .template import DocTheme, DEFAULT_THEME

logger = logging.getLogger(__name__)


ChartLike = Union[str, "Path", bytes, "Figure", dict, None]


@dataclass(frozen=True)
class ChartSpec:
    """Normalized chart input. The caller passes one of these in
    ``context["charts"]`` or in the form returned by analytics modules.

    ``source`` is whatever the caller had — we let charts.py decide how
    to coerce it to PNG bytes. ``caption`` becomes the figure caption.
    ``label`` is the figure number string (e.g. "Figure 1").
    """
    title: str
    source: Any
    caption: Optional[str] = None
    label: Optional[str] = None


# =============================================================================
# Public entry: turn a list of ChartSpecs into a Story of flowables
# =============================================================================


def build_chart_flowables(
    specs: list[ChartSpec],
    theme: DocTheme = DEFAULT_THEME,
) -> list[Flowable]:
    """Convert a list of ChartSpecs into a Story segment.

    Output structure per chart:

        [optional caption above],
        Image,
        Caption (Figure N: …),
        Spacer

    Captions are placed BELOW the image (standard academic / financial
    convention). Each ``Image`` is wrapped in ``KeepTogether`` with its
    caption so the page-breaker doesn't orphan one from the other.

    Missing or failed charts produce a placeholder + caption pair so the
    figure numbering stays consistent across runs even when an artifact
    is unavailable.
    """
    styles = theme.styles()
    out: list[Flowable] = []

    for i, spec in enumerate(specs, start=1):
        label = spec.label or f"Figure {i}"
        caption_text = f"<b>{_escape(label)}.</b> {_escape(spec.title)}"
        if spec.caption:
            caption_text += f"  <i>{_escape(spec.caption)}</i>"
        caption_flow = Paragraph(caption_text, styles["Caption"])

        try:
            img = _load_image_or_placeholder(spec, theme, styles)
        except Exception as e:  # noqa: BLE001 — last-resort guard
            logger.error("charts | %s | load failed: %s", label, e)
            img = _placeholder(
                f"Chart artifact failed to render: {e}",
                theme, styles,
            )

        # KeepTogether so caption stays glued to the image across page breaks
        out.append(KeepTogether([img, caption_flow]))
        out.append(Spacer(1, 8))

    return out


# =============================================================================
# Loader: dispatch on type
# =============================================================================


def _load_image_or_placeholder(
    spec: ChartSpec,
    theme: DocTheme,
    styles: dict,
) -> Flowable:
    """Resolve a ChartSpec.source to a ReportLab Image or placeholder."""
    src = spec.source

    if src is None:
        return _placeholder(
            f"No chart artifact available for '{spec.title}'.",
            theme, styles,
        )

    # ---- File path -----------------------------------------------------
    if isinstance(src, (str, os.PathLike)):
        p = Path(src)
        if not p.exists():
            return _placeholder(
                f"Chart file not found: {p}",
                theme, styles,
            )
        suffix = p.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".gif"):
            return _image_from_path(str(p), theme)
        if suffix == ".svg":
            return _image_from_svg(p, theme, styles)
        return _placeholder(
            f"Unsupported chart format '{suffix}' for {p.name}",
            theme, styles,
        )

    # ---- Raw bytes -----------------------------------------------------
    if isinstance(src, (bytes, bytearray)):
        return _image_from_bytes(bytes(src), theme)

    # ---- matplotlib Figure --------------------------------------------
    if _is_matplotlib_figure(src):
        return _image_from_matplotlib(src, theme)

    # ---- Chart-ready data dict (last resort) --------------------------
    if isinstance(src, dict):
        return _render_data_dict(src, theme, styles)

    return _placeholder(
        f"Unrecognized chart source type: {type(src).__name__}",
        theme, styles,
    )


# =============================================================================
# Loaders
# =============================================================================


def _image_from_path(path: str, theme: DocTheme) -> Image:
    """ReportLab Image from a PNG/JPG/GIF file, scaled to chart_max_*."""
    img = Image(path)
    img._restrictSize(theme.chart_max_width, theme.chart_max_height)
    img.hAlign = "CENTER"
    return img


def _image_from_bytes(data: bytes, theme: DocTheme) -> Image:
    """ReportLab Image from raw PNG/JPG bytes."""
    buf = io.BytesIO(data)
    img = Image(buf)
    img._restrictSize(theme.chart_max_width, theme.chart_max_height)
    img.hAlign = "CENTER"
    return img


def _image_from_matplotlib(fig: Any, theme: DocTheme) -> Image:
    """ReportLab Image from a matplotlib Figure.

    We render at 150 DPI — enough for clean print at the report size
    without blowing PDF file size out. The Figure is left intact for
    the caller; we just savefig() it to an in-memory buffer.
    """
    buf = io.BytesIO()
    # bbox_inches='tight' trims whitespace; matters because the report
    # column is narrow.
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return _image_from_bytes(buf.getvalue(), theme)


def _image_from_svg(path: Path, theme: DocTheme, styles: dict) -> Flowable:
    """SVG → ReportLab Drawing via svglib, if available.

    svglib is an optional dependency. If it's not installed we surface a
    placeholder rather than crashing the build.
    """
    try:
        from svglib.svglib import svg2rlg  # type: ignore
        from reportlab.graphics import renderPDF  # noqa: F401  — proves install
    except ImportError:
        return _placeholder(
            f"SVG support requires svglib (pip install svglib). "
            f"Chart at {path} skipped.",
            theme, styles,
        )
    try:
        drawing = svg2rlg(str(path))
        if drawing is None:
            return _placeholder(f"Failed to parse SVG: {path}", theme, styles)
        # Scale drawing to fit chart_max_* — svglib produces sized RLG drawings
        scale = min(
            theme.chart_max_width / max(drawing.width, 1.0),
            theme.chart_max_height / max(drawing.height, 1.0),
            1.0,
        )
        drawing.width *= scale
        drawing.height *= scale
        drawing.scale(scale, scale)
        return drawing
    except Exception as e:  # noqa: BLE001
        return _placeholder(f"SVG render error for {path}: {e}", theme, styles)


def _is_matplotlib_figure(obj: Any) -> bool:
    """Duck-type check that avoids importing matplotlib at module load."""
    return (
        type(obj).__module__.startswith("matplotlib")
        and hasattr(obj, "savefig")
    )


# =============================================================================
# Last-resort: render a chart-ready data dict via matplotlib
# =============================================================================


def _render_data_dict(d: dict, theme: DocTheme, styles: dict) -> Flowable:
    """Render the small set of chart 'shapes' that analytics may emit
    when no pre-rendered PNG is available.

    Supported shapes (kept intentionally minimal):

      * ``{"type": "line", "x": [...], "y": [...], "y_label": "..."}``
      * ``{"type": "bar",  "labels": [...], "values": [...]}``
      * ``{"type": "table", "columns": [...], "rows": [[...], ...]}``

    Anything else returns a placeholder. This branch exists so report
    generation can produce *something* useful even when the analytics
    layer hands us only structured data; it is NOT meant to be a
    full-featured charting library.
    """
    kind = (d.get("type") or "").lower()
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend for PDF builds
        import matplotlib.pyplot as plt
    except ImportError:
        return _placeholder(
            "matplotlib not installed; cannot render chart-ready data dict.",
            theme, styles,
        )

    try:
        if kind == "line":
            x = d.get("x") or []
            y = d.get("y") or []
            if not x or not y or len(x) != len(y):
                return _placeholder("Line chart data malformed.", theme, styles)
            fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=150)
            ax.plot(x, y, linewidth=1.2)
            if d.get("y_label"):
                ax.set_ylabel(d["y_label"])
            if d.get("x_label"):
                ax.set_xlabel(d["x_label"])
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            try:
                return _image_from_matplotlib(fig, theme)
            finally:
                plt.close(fig)

        if kind == "bar":
            labels = d.get("labels") or []
            values = d.get("values") or []
            if not labels or len(labels) != len(values):
                return _placeholder("Bar chart data malformed.", theme, styles)
            fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=150)
            ax.bar(labels, values)
            ax.tick_params(axis="x", rotation=30)
            ax.grid(True, alpha=0.3, axis="y")
            fig.tight_layout()
            try:
                return _image_from_matplotlib(fig, theme)
            finally:
                plt.close(fig)

        return _placeholder(
            f"Unsupported chart-ready data type: '{kind}'",
            theme, styles,
        )
    except Exception as e:  # noqa: BLE001
        return _placeholder(f"Chart render error: {e}", theme, styles)


# =============================================================================
# Helpers
# =============================================================================


def _placeholder(text: str, theme: DocTheme, styles: dict) -> Flowable:
    """A muted, framed paragraph used wherever a chart can't be embedded."""
    return Paragraph(_escape(text), styles["Placeholder"])


def _escape(s: str) -> str:
    """Escape characters that ReportLab Paragraph treats as markup."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
