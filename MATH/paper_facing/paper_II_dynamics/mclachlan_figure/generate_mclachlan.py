#!/usr/bin/env python3
"""Rebuild the McLachlan schematic as native SVG, with an optional PNG preview.

Run from any directory:
    python generate_mclachlan.py
    python generate_mclachlan.py --png --scale 3

Edit GEOMETRY, COLORS, and LABELS below. Projected blue arrow tips are derived
from the black segments, and projection guides reuse those exact coordinates.
Labels use Matplotlib MathText (STIX) and are exported as vector outlines.
No input image, pre-existing SVG, LaTeX installation, or font file is required.

This generator was reconstructed from the saved mclachlan_drift_corrected.svg;
it is not a recovered copy of the earlier temporary generation scripts.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import math
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

Point = tuple[float, float]
SVG_NS = "http://www.w3.org/2000/svg"
ROOT = Path(__file__).resolve().parent

# SVG screen coordinates: x increases rightward; y increases downward.
WIDTH, HEIGHT = 1448, 1086
# The lower plane reaches x = -43, outside the nominal canvas, so the view box is
# computed from drawn content rather than pinned to (0, 0, WIDTH, HEIGHT); pinning it
# clipped the left edge of every export.
VIEW_MARGIN = 6.0
COLORS = {
    "black": "#080808", "blue": "#007aff", "orange": "#ff8900",
    "red": "#ef1f27", "border": "#494c4f",
}
GEOMETRY = {
    "origin": (459.0, 705.0),
    "state_tip": (540.0, 230.0),
    "drift_tip": (972.0, 137.0),
    "projected_drift_tip": (972.0, 705.0),
    "black_vertices": ((459.0, 705.0), (636.0, 890.0),
                       (852.0, 790.0), (972.0, 705.0)),
    # Order throughout: 1, j, 3. Keep these positive for same-direction arrows.
    "indices": ("1", "j", "3"),
    "blue_scales": (0.78, 0.86, 0.90),
    "upper_blue_tips": ((660.0, 252.0), (700.0, 181.0), (604.0, 96.0)),
    "upper_plane": ((407.0, 14.0), (956.0, 77.0),
                    (710.0, 332.0), (127.0, 225.0)),
    "lower_plane": ((351.0, 442.0), (1434.0, 590.0),
                    (1101.0, 1069.0), (-43.0, 805.0)),
    "orange_plane": ((459.0, 530.0), (987.0, 681.0),
                     (723.0, 944.0), (201.0, 774.0)),
}


@dataclass(frozen=True)
class Label:
    """A MathText label at an explicit left-baseline position in SVG units."""
    id: str
    tex: str
    size: float
    x: float
    y: float
    color: str


# These are readable mathematical sources, not SVG path literals.
# Baselines preserve the saved SVG layout. Starred velocities are the default.
LABELS = (
    Label("tangent-map", r"T=\nabla_\theta|\psi\rangle", 55,
          316.75156250, 139.60000000, "black"),
    Label("projected-tangent-map", r"\bar T=\Pi\nabla_\theta|\psi\rangle", 57,
          44.63343750, 801.88000000, "black"),
    Label("state", r"|\psi\rangle", 47,
          558.84953125, 286.84000000, "orange"),
    Label("schrodinger-drift", r"-iH|\psi\rangle", 74,
          985.25937500, 163.28000000, "orange"),
    Label("projected-schrodinger-drift", r"-i\Pi H|\psi\rangle=\bar d", 65,
          1003.83593750, 703.55000000, "red"),
    Label("phase-projector", r"\Pi=I-|\psi\rangle\langle\psi|", 36,
          312.35312500, 779.92000000, "orange"),
    Label("upper-tangent-3", r"\partial_{\theta_3}|\psi\rangle", 46,
          608.16000000, 90.12000000, "blue"),
    Label("upper-tangent-j", r"\partial_{\theta_j}|\psi\rangle", 46,
          704.16000000, 183.12000000, "blue"),
    Label("upper-tangent-1", r"\partial_{\theta_1}|\psi\rangle", 46,
          665.16000000, 261.12000000, "blue"),
    Label("projected-tangent-3", r"\Pi\partial_{\theta_3}|\psi\rangle", 44,
          603.20937500, 593.68000000, "blue"),
    Label("projected-tangent-j", r"\Pi\partial_{\theta_j}|\psi\rangle", 44,
          665.20937500, 647.68000000, "blue"),
    Label("projected-tangent-1", r"\Pi\partial_{\theta_1}|\psi\rangle", 44,
          616.20937500, 831.68000000, "blue"),
    Label("velocity-1", r"\dot\theta_1^\star\Pi\partial_{\theta_1}|\psi\rangle", 43,
          678.83765625, 937.13000000, "black"),
    Label("velocity-j", r"\dot\theta_j^\star\Pi\partial_{\theta_j}|\psi\rangle", 43,
          862.83765625, 849.13000000, "black"),
    Label("velocity-3", r"\dot\theta_3^\star\Pi\partial_{\theta_3}|\psi\rangle", 43,
          995.83765625, 766.13000000, "black"),
)


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def mul(s: float, a: Point) -> Point:
    return s * a[0], s * a[1]


def norm(a: Point) -> float:
    return math.hypot(*a)


def fmt(x: float) -> str:
    return f"{x:.8f}"


def xy(p: Point) -> str:
    return f"{fmt(p[0])},{fmt(p[1])}"


def projected_blue_tips() -> tuple[Point, ...]:
    """Each lower blue vector = positive scale * matching black direction."""
    vertices = GEOMETRY["black_vertices"]
    origin = GEOMETRY["origin"]
    return tuple(add(origin, mul(scale, sub(end, start)))
                 for start, end, scale in zip(vertices, vertices[1:],
                                               GEOMETRY["blue_scales"]))


def guide_segments(start: Point, end: Point,
                   nominal_dash: float = 10.0) -> list[tuple[Point, Point]]:
    """Alternate equal-length dashes/gaps, with a dash touching each endpoint."""
    if nominal_dash <= 0 or norm(sub(end, start)) == 0:
        raise ValueError("Guide length and nominal dash length must be positive.")
    count = max(2, round(norm(sub(end, start)) / (2 * nominal_dash)))
    intervals = 2 * count - 1
    delta = sub(end, start)
    return [(add(start, mul(2 * k / intervals, delta)),
             add(start, mul((2 * k + 1) / intervals, delta)))
            for k in range(count)]


def validate_geometry() -> None:
    """Fail before export if the intended correspondence has been broken."""
    vertices = GEOMETRY["black_vertices"]
    n = len(vertices) - 1
    if any(len(GEOMETRY[key]) != n for key in
           ("indices", "blue_scales", "upper_blue_tips")):
        raise ValueError("Black segments, indices, blue scales, and tips must agree.")
    if vertices[0] != GEOMETRY["origin"]:
        raise ValueError("The black chain must start at the common origin.")
    if vertices[-1] != GEOMETRY["projected_drift_tip"]:
        raise ValueError("This zero-residual schematic closes at the red drift tip.")
    for i, (a, b, tip, scale, upper) in enumerate(zip(
            vertices, vertices[1:], projected_blue_tips(),
            GEOMETRY["blue_scales"], GEOMETRY["upper_blue_tips"])):
        u, v = sub(b, a), sub(tip, GEOMETRY["origin"])
        if not math.isfinite(scale) or scale <= 0 or norm(u) == 0:
            raise ValueError(f"Invalid segment/scale at index {i}.")
        cross = u[0] * v[1] - u[1] * v[0]
        dot = u[0] * v[0] + u[1] * v[1]
        if abs(cross) > 1e-12 * norm(u) * norm(v) or dot <= 0:
            raise ValueError(f"Projected arrow {i} lost its parallel direction.")
        dashes = guide_segments(upper, tip)
        if (norm(sub(dashes[0][0], upper)) > 1e-9 or
                norm(sub(dashes[-1][1], tip)) > 1e-9):
            raise ValueError(f"Projection guide {i} missed an arrow tip.")


def element(parent: ET.Element, tag: str, **attrs: Any) -> ET.Element:
    return ET.SubElement(parent, tag, {k.replace("_", "-"): str(v)
                                       for k, v in attrs.items()})


def polygon_path(points: tuple[Point, ...]) -> str:
    return "M " + " L ".join(xy(p) for p in points) + " Z"


def arrow(parent: ET.Element, name: str, start: Point, tip: Point,
          color: str, width: float, head_length: float,
          head_half_width: float) -> None:
    """Draw an arrow whose triangle apex is exactly at `tip`."""
    delta = sub(tip, start)
    length = norm(delta)
    if length <= head_length + 1:
        raise ValueError(f"Arrow {name!r} is shorter than its arrowhead.")
    unit = mul(1 / length, delta)
    normal = (-unit[1], unit[0])
    shaft_end = sub(tip, mul(head_length, unit))
    base = sub(tip, mul(head_length + 1, unit))
    group = element(parent, "g", id=name, data_start=xy(start), data_tip=xy(tip))
    element(group, "path", d=f"M {xy(start)} L {xy(shaft_end)}", fill="none",
            stroke=color, stroke_width=width, stroke_linecap="round")
    element(group, "path", fill=color,
            d=polygon_path((tip, add(base, mul(head_half_width, normal)),
                            sub(base, mul(head_half_width, normal)))))


def math_path_data(path: Any) -> str:
    """Convert Matplotlib's path commands to native SVG path data."""
    parts = []
    commands = {1: "M", 2: "L", 3: "Q", 4: "C"}
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        if code == 79:  # Matplotlib Path.CLOSEPOLY
            parts.append("Z")
        elif code in commands:
            coords = " ".join(xy((vertices[k], vertices[k + 1]))
                              for k in range(0, len(vertices), 2))
            parts.append(commands[code] + coords)
        elif code != 0:
            raise ValueError(f"Unexpected glyph path code {code}.")
    return " ".join(parts)


def render_math(labels: tuple[Label, ...]) -> list[tuple[Label, str, tuple]]:
    try:
        from matplotlib import rc_context
        from matplotlib.textpath import TextPath
    except ImportError as exc:
        raise RuntimeError("Install the local requirements.txt to render math labels.") from exc
    rendered = []
    with rc_context({"mathtext.fontset": "stix", "font.family": "STIXGeneral",
                     "text.usetex": False}):
        for label in labels:
            path = TextPath((0, 0), "$" + label.tex + "$", size=label.size)
            bounds = path.get_extents()
            # Flip y to SVG coordinates, retaining the explicit baseline.
            box = (label.x + bounds.xmin, label.y - bounds.ymax,
                   bounds.width, bounds.height)
            rendered.append((label, math_path_data(path), box))
    return rendered


def content_box(rendered: list[tuple[Label, str, tuple]]) -> tuple[float, float, float, float]:
    """View box over every drawn surface, arrow endpoint and label, plus a margin."""
    xs: list[float] = [0.0, float(WIDTH)]
    ys: list[float] = [0.0, float(HEIGHT)]
    for key in ("upper_plane", "lower_plane", "orange_plane"):
        for x, y in GEOMETRY[key]:
            xs.append(x); ys.append(y)
    for point in (GEOMETRY["origin"], GEOMETRY["state_tip"], GEOMETRY["drift_tip"],
                  GEOMETRY["projected_drift_tip"], *GEOMETRY["black_vertices"],
                  *GEOMETRY["upper_blue_tips"], *projected_blue_tips()):
        xs.append(point[0]); ys.append(point[1])
    for _label, _d, (x, y, width, height) in rendered:
        xs.extend((x, x + width)); ys.extend((y, y + height))
    x0, y0 = min(xs) - VIEW_MARGIN, min(ys) - VIEW_MARGIN
    return x0, y0, max(xs) + VIEW_MARGIN - x0, max(ys) + VIEW_MARGIN - y0


def build_svg(*, starred: bool = True) -> str:
    validate_geometry()
    labels = LABELS if starred else tuple(
        replace(label, tex=label.tex.replace(r"^\star", "")) for label in LABELS)
    rendered = render_math(labels)
    vx, vy, vw, vh = content_box(rendered)
    root = ET.Element("svg", {"xmlns": SVG_NS, "width": fmt(vw), "height": fmt(vh),
                             "viewBox": f"{fmt(vx)} {fmt(vy)} {fmt(vw)} {fmt(vh)}",
                             "role": "img"})
    element(root, "title").text = "McLachlan projected tangent geometry"
    element(root, "desc").text = (
        "Native vector schematic. Orange target: -i H |psi>. Red target: "
        "-i Pi H |psi> = d-bar. Each projected blue vector is parallel to its "
        "black velocity component. Blue guides join corresponding arrow tips. "
        "The black chain closes at the red target; no residual arrow is shown.")
    defs = element(root, "defs")
    for name, stops in (
            ("gray-top", ((0, "#eeeeee"), (0.48, "#e9edf0"), (1, "#e1e5e8"))),
            ("gray-lower", ((0, "#e4e8eb"), (0.48, "#ebeff2"), (1, "#dce1e5"))),
            ("orange-plane", ((0, "#fae8d7"), (0.55, "#f8e9dc"), (1, "#f9e5d3")))):
        gradient = element(defs, "linearGradient", id=name, x1="0", y1="0", x2="1", y2="1")
        for offset, color in stops:
            element(gradient, "stop", offset=offset, stop_color=color)

    # Prevent dashed guides from running through labels without moving the guides.
    clip = element(defs, "clipPath", id="guides-clip")
    clip_d = (f"M {fmt(vx)},{fmt(vy)} H {fmt(vx + vw)} "
              f"V {fmt(vy + vh)} H {fmt(vx)} Z ")
    for _, _, (x, y, width, height) in rendered:
        p = 4.0
        clip_d += polygon_path(((x-p, y-p), (x-p, y+height+p),
                                (x+width+p, y+height+p), (x+width+p, y-p))) + " "
    element(clip, "path", d=clip_d, fill_rule="evenodd", clip_rule="evenodd")

    element(root, "rect", x=fmt(vx), y=fmt(vy), width=fmt(vw),
            height=fmt(vh), fill="white")
    for key, gradient, stroke in (("upper_plane", "gray-top", 3.3),
                                  ("lower_plane", "gray-lower", 3.7),
                                  ("orange_plane", "orange-plane", 0)):
        attrs = {"id": "surface-" + key.replace("_", "-"), "d": polygon_path(GEOMETRY[key]),
                 "fill": f"url(#{gradient})"}
        if stroke:
            attrs.update(stroke=COLORS["border"], stroke_width=stroke,
                         stroke_linejoin="miter")
        element(root, "path", **attrs)

    blue_tips = projected_blue_tips()
    guides = element(root, "g", id="projection-guides", clip_path="url(#guides-clip)")
    for index, upper, lower in zip(GEOMETRY["indices"], GEOMETRY["upper_blue_tips"], blue_tips):
        group = element(guides, "g", id="guide-" + index,
                        data_start=xy(upper), data_end=xy(lower), fill="none",
                        stroke=COLORS["blue"], stroke_width=2.15,
                        stroke_linecap="round", opacity=0.84)
        for a, b in guide_segments(upper, lower):
            element(group, "path", d=f"M {xy(a)} L {xy(b)}")

    origin, state = GEOMETRY["origin"], GEOMETRY["state_tip"]
    drift, projected = GEOMETRY["drift_tip"], GEOMETRY["projected_drift_tip"]
    element(root, "path", id="drift-projection-guide",
            d=f"M {xy(drift)} L {xy(projected)}", fill="none", stroke=COLORS["orange"],
            stroke_width=7.5, stroke_dasharray="15 15", stroke_linecap="round")
    # The previously removed gray dashed state guide is deliberately absent.
    arrow(root, "state-arrow", origin, state, COLORS["orange"], 10.5, 33, 14)
    arrow(root, "drift-arrow", origin, drift, COLORS["orange"], 12, 39, 21)
    arrow(root, "projected-drift-arrow", origin, projected, COLORS["red"], 6, 30, 14)
    vertices = GEOMETRY["black_vertices"]
    for index, a, b in zip(GEOMETRY["indices"], vertices, vertices[1:]):
        arrow(root, "black-arrow-" + index, a, b, COLORS["black"], 4.6, 23, 11.5)
    for index, tip in zip(GEOMETRY["indices"], GEOMETRY["upper_blue_tips"]):
        arrow(root, "upper-blue-arrow-" + index, state, tip, COLORS["blue"], 5.1, 21, 11)
    for index, tip in zip(GEOMETRY["indices"], blue_tips):
        arrow(root, "lower-blue-arrow-" + index, origin, tip, COLORS["blue"], 5.2, 22, 11.5)
    for center, radius, color in ((state, 4.9, "blue"), (origin, 9.2, "black"),
                                  (projected, 9.0, "red")):
        element(root, "circle", cx=center[0], cy=center[1], r=radius, fill=COLORS[color])
    for label, path_d, _ in rendered:
        group = element(root, "g", id="label-" + label.id, aria_label=label.tex)
        element(group, "title").text = label.tex
        element(group, "path", d=path_d, fill=COLORS[label.color],
                transform=f"translate({fmt(label.x)},{fmt(label.y)}) scale(1,-1)")
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def build(output: Path, *, png: bool = False, scale: float = 3.0,
          starred: bool = True) -> tuple[Path, Path | None]:
    """Write SVG and, when requested, a PNG at `scale` pixels per SVG unit."""
    output = Path(output)
    if output.suffix.lower() != ".svg":
        raise ValueError("The output filename must have an .svg extension.")
    if not math.isfinite(scale) or not 0.1 <= scale <= 8:
        raise ValueError("PNG scale must lie between 0.1 and 8.")
    renderer = None
    if png:
        try:
            import cairosvg as renderer
        except (ImportError, OSError) as exc:
            raise RuntimeError("PNG export requires CairoSVG and its Cairo runtime. "
                               "Omit --png to export native SVG alone.") from exc
    svg = build_svg(starred=starred)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    png_path = None
    if renderer is not None:
        png_path = output.with_suffix(".png")
        renderer.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path),
                         output_width=round(WIDTH * scale), output_height=round(HEIGHT * scale))
    return output, png_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mclachlan_geometry.svg")
    parser.add_argument("--png", action="store_true", help="Also export a PNG preview.")
    parser.add_argument("--scale", type=float, default=3.0, help="PNG scale, default 3.")
    parser.add_argument("--unstarred", action="store_true", help="Use earlier unstarred velocity labels.")
    args = parser.parse_args()
    try:
        paths = build(args.output, png=args.png, scale=args.scale, starred=not args.unstarred)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    for path in paths:
        if path is not None:
            print(path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
