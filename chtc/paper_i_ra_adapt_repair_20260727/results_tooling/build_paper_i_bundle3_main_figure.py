#!/usr/bin/env python3
"""Build the Paper-I-format six-panel Bundle-3 convergence figure.

The weak-Holstein row is display-cropped at iteration 45.  The strong-Holstein
row extends to iteration 75; incomplete insertion trajectories stop at their
latest locally evidenced point and are never extrapolated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent
KSTAR_TABLES = RESULTS_DIR / "kstar_tables.json"
WEAK_B3_TRACE_SHA256 = (
    "27c9250cac91ab18a9fcb35291e885c930b553634fe10f7b6007056c03402357"
)

REGIMES = [
    ("weak_weak", "Weak–weak", 3, 45),
    ("intermediate_weak", "Intermediate–weak", 3, 45),
    ("strong_weak_u8", "Strong–weak", 3, 45),
    ("weak_strong", "Weak–strong", 7, 75),
    ("intermediate_strong", "Intermediate–strong", 7, 75),
    ("strong_strong_u8", "Strong–strong", 7, 75),
]
WEAK_REGIMES = {row[0] for row in REGIMES[:3]}
STRONG_REGIMES = {row[0] for row in REGIMES[3:]}
STRONG_APPEND_TRACE_SHA256 = {
    "weak_strong": "e496ec6c0ae0802d3691e7f454ed770a9a386ab064baab3d1035e8e102fe0ab1",
    "intermediate_strong": "6d552a0379e5c14bca5f49829da3b2fe278e2689e90c3c1f812999f48ac8fbf2",
    "strong_strong_u8": "9ceb3e387c9c91d0133fff2efe232be04c2482cf52df209f7feb6fcb9a9fe515",
}

SERIES = [
    ("AVQE", "AVQE", "#0072B2", "--", "o"),
    ("b_depth_append_ra", "RA-Append", "#8B0000", "-", "o"),
    (
        "b_depth_costexp_plateau_position",
        "RA-Plateau",
        "#E69F00",
        "-",
        "s",
    ),
    (
        "b_depth_costexp_always_open",
        "RA-Insert",
        "#009E73",
        "-",
        "D",
    ),
]

ERR_FLOOR = 1e-16
ROW_SCALES = {
    3: ((1e-15, 4e0), [1e0, 1e-5, 1e-10, 1e-15]),
    7: ((1e-9, 4e0), [1e0, 1e-3, 1e-6, 1e-9]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def weak_b3_trace_sha256(kstar: dict) -> str:
    subset = {
        key: entry["trace"]
        for key, entry in kstar["b3"].items()
        if key.rsplit("|", 1)[-1] in WEAK_REGIMES
    }
    encoded = json.dumps(
        subset, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trace_sha256(points: list[tuple[int, float]]) -> str:
    encoded = json.dumps(points, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_manifest_input(manifest: dict, label: str, path: Path) -> None:
    expected = manifest["inputs"][label]["sha256"]
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"manifest hash mismatch for {label}: expected {expected}, got {actual}"
        )


def parse_ai_log(path: Path, exact_energy: float) -> list[tuple[int, float]]:
    points: dict[int, float] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"hardcoded_adapt_iter"' not in line:
                continue
            start = line.find("{")
            if start < 0:
                continue
            try:
                row = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            k = int(row["depth"])
            points[k] = max(abs(float(row["energy"]) - exact_energy), ERR_FLOOR)
    return sorted(points.items())


def queue_map(queue_path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with queue_path.open(encoding="utf-8") as handle:
        for proc, line in enumerate(handle):
            mapping[proc] = line.split("\t", 1)[0].strip()
    return mapping


def parse_live_snapshot(
    path: Path,
    proc_to_eid: dict[int, str],
    exact: dict[str, float],
) -> dict[tuple[str, str], list[tuple[int, float]]]:
    points: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    pattern = re.compile(
        r"sixregime__(.+?)__((?:weak|intermediate|strong)_\w+?)__nph\d"
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(sample.get("cluster", -1)) != 9664325:
                continue
            eid = sample.get("eid") or proc_to_eid.get(int(sample["proc"]), "")
            match = pattern.match(eid)
            if not match:
                continue
            arm, regime = match.groups()
            if regime not in STRONG_REGIMES or arm not in {
                "b_depth_costexp_always_open",
                "b_depth_costexp_plateau_position",
            }:
                continue
            row = sample["row"]
            k = int(row["depth"])
            if k < 1:
                continue
            points[(arm, regime)][k] = max(
                abs(float(row["energy"]) - exact[regime]), ERR_FLOOR
            )
    return {key: sorted(rows.items()) for key, rows in points.items()}


def find_manifest_stream(manifest: dict, arm: str, regime: str) -> Path:
    needle = f"__{arm}__{regime}__"
    matches = [
        Path(entry["path"])
        for label, entry in manifest["inputs"].items()
        if label.startswith("stream:") and needle in label
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one manifest stream for {arm}|{regime}, found {len(matches)}"
        )
    return matches[0]


def crop(points: list[tuple[int, float]], horizon: int) -> list[tuple[int, float]]:
    return [(k, err) for k, err in points if k <= horizon]


def load_plot_style(output_dir: Path):
    style_path = output_dir.parent / "paper_i_main_plot_style.py"
    spec = importlib.util.spec_from_file_location("paper_i_main_plot_style", style_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shared plot style {style_path}")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    return style, style_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/Users/jakestrobel/local_repos/ADAPT---Paper-I/"
            "Paper_I_overleaf_20260818/figures/"
            "paper_i_ra_adaptive_append_main_20260818"
        ),
    )
    args = parser.parse_args()

    bundle_dir = (
        args.base_root
        / "output/pdf/paper_i_ra_allphase_adaptive_20260817"
    )
    bundle_pdf = bundle_dir / "bundle3_final_results.pdf"
    bundle_manifest_path = bundle_dir / "bundle3_final_results_manifest.json"
    manifest = load_json(bundle_manifest_path)
    if sha256(bundle_pdf) != manifest["artifact_sha256"]:
        raise RuntimeError("Bundle-3 visual-reference PDF does not match its manifest")

    # The results table also receives unrelated campaign-row updates.  The
    # figure uses only the nine completed weak-sector Bundle-3 traces, so bind
    # precisely that projection while separately recording the complete file
    # hash supplied by the current Bundle-3 manifest.
    kstar = load_json(KSTAR_TABLES)
    current_weak_trace_sha = weak_b3_trace_sha256(kstar)
    if current_weak_trace_sha != WEAK_B3_TRACE_SHA256:
        raise RuntimeError(
            "weak Bundle-3 trajectories do not match the manuscript handoff: "
            f"expected {WEAK_B3_TRACE_SHA256}, got {current_weak_trace_sha}"
        )

    aavqe_path = Path(
        manifest["inputs"]["aavqe_full_hist_tolmatch_9665004.json"]["path"]
    )
    validate_manifest_input(
        manifest, "aavqe_full_hist_tolmatch_9665004.json", aavqe_path
    )
    aavqe_energy = load_json(aavqe_path)

    comparison_module = (
        args.base_root
        / "chtc/paper_i_ra_adapt_repair_20260727/"
        "build_paper_i_ra_phase123_qiskit_comparison_pdf_20260817.py"
    )
    spec = importlib.util.spec_from_file_location("paper_i_comparison", comparison_module)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {comparison_module}")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    exact = {
        regime: float(row["exact_energy"])
        for regime, row in comparison.load_hist_plateau().items()
    }

    traces: dict[tuple[str, str], list[tuple[int, float]]] = {}
    sources: dict[tuple[str, str], dict] = {}

    for regime, *_ in REGIMES:
        traces[("AVQE", regime)] = [
            (int(k), max(abs(float(energy) - exact[regime]), ERR_FLOOR))
            for k, energy in aavqe_energy[regime]
        ]
        sources[("AVQE", regime)] = {
            "kind": "authenticated_r75_history",
            "path": str(aavqe_path),
            "sha256": sha256(aavqe_path),
        }

    for regime in WEAK_REGIMES:
        for arm, *_ in SERIES[1:]:
            entry = kstar["b3"].get(f"{arm}|{regime}")
            if entry is None or not entry.get("trace"):
                raise RuntimeError(f"missing completed weak trace for {arm}|{regime}")
            traces[(arm, regime)] = [
                (int(k), max(float(err), ERR_FLOOR)) for k, err in entry["trace"]
            ]
            sources[(arm, regime)] = {
                "kind": "completed_bundle3_trace",
                "path": str(KSTAR_TABLES),
                "sha256": sha256(KSTAR_TABLES),
                "weak_bundle3_trace_projection_sha256": current_weak_trace_sha,
                "json_key": f"b3.{arm}|{regime}.trace",
            }

    for regime in STRONG_REGIMES:
        stream = find_manifest_stream(manifest, "b_depth_append_ra", regime)
        points = parse_ai_log(stream, exact[regime])
        if trace_sha256(points) != STRONG_APPEND_TRACE_SHA256[regime]:
            raise RuntimeError(f"strong append trajectory changed for {regime}")
        if not points or points[-1][0] != 75:
            raise RuntimeError(f"append trace for {regime} does not reach iteration 75")
        traces[("b_depth_append_ra", regime)] = points
        sources[("b_depth_append_ra", regime)] = {
            "kind": "completed_bundle3_stream",
            "path": str(stream),
            "sha256": sha256(stream),
        }

    live_path = Path(manifest["inputs"]["live_samples_fresh.jsonl"]["path"])
    validate_manifest_input(manifest, "live_samples_fresh.jsonl", live_path)
    queue_path = (
        ROOT
        / "chtc/paper_i_ra_adapt_repair_20260727/"
        "ra_adapt_costexp_depthfix_sixregime_20260820_v2_chtc/queue_fresh.tsv"
    )
    live = parse_live_snapshot(live_path, queue_map(queue_path), exact)
    for regime in STRONG_REGIMES:
        for arm in (
            "b_depth_costexp_always_open",
            "b_depth_costexp_plateau_position",
        ):
            points = live.get((arm, regime))
            if not points:
                raise RuntimeError(f"missing strong partial trace for {arm}|{regime}")
            traces[(arm, regime)] = points
            sources[(arm, regime)] = {
                "kind": "partial_live_snapshot",
                "path": str(live_path),
                "sha256": sha256(live_path),
                "queue_map": str(queue_path),
            }

    style, style_path = load_plot_style(args.output_dir)
    style.configure()
    fig = plt.figure(figsize=(7.10, 3.32))
    axes = style.add_axes_grid(
        fig,
        nrows=2,
        ncols=3,
        left=0.082,
        right=0.995,
        bottom=0.10,
        top=0.85,
        hgap=0.035,
        vgap=0.12,
    )
    stats: dict[str, dict] = {}

    for index, (ax, (regime, title, cutoff, horizon)) in enumerate(
        zip((axis for row in axes for axis in row), REGIMES)
    ):
        regime_kstar = int(kstar["aavqe"][regime]["k_star"])
        stats[regime] = {
            "display_horizon": horizon,
            "k_star": regime_kstar,
            "series": {},
        }
        for arm, label, color, linestyle, marker in SERIES:
            displayed = crop(traces[(arm, regime)], horizon)
            if not displayed:
                raise RuntimeError(f"no displayed points for {arm}|{regime}")
            ks = [k for k, _ in displayed]
            errors = [err for _, err in displayed]
            ax.plot(
                ks,
                errors,
                color=color,
                lw=style.PRIMARY_LINEWIDTH,
                ls=linestyle,
                solid_capstyle="round",
            )
            marker_error = dict(displayed).get(regime_kstar)
            if marker is not None and marker_error is not None:
                ax.plot(
                    [regime_kstar],
                    [marker_error],
                    linestyle="none",
                    marker=marker,
                    markersize=style.MARKER_SIZE,
                    markerfacecolor=("none" if arm == "AVQE" else color),
                    markeredgecolor=color,
                    markeredgewidth=style.MARKER_EDGEWIDTH,
                )
            source = sources[(arm, regime)]
            stats[regime]["series"][arm] = {
                "label": label,
                "history_point_count": len(traces[(arm, regime)]),
                "displayed_point_count": len(displayed),
                "marker_k": (
                    regime_kstar
                    if marker is not None and marker_error is not None
                    else None
                ),
                "marker_error": (
                    marker_error
                    if marker is not None and marker_error is not None
                    else None
                ),
                "latest_displayed_k": ks[-1],
                "latest_displayed_error": errors[-1],
                "status": source["kind"],
                "source": source,
            }

        if horizon == 45:
            xticks = [0, 10, 20, 30, 40, 45]
        else:
            xticks = [0, 15, 30, 45, 60, 75]
        ylimits, yticks = ROW_SCALES[cutoff]
        style.style_log_axis(
            ax,
            title=title,
            xlim=(0, horizon + (1 if horizon == 75 else 0)),
            xticks=xticks,
            ylim=ylimits,
            yticks=yticks,
            show_ylabels=(index % 3 == 0),
        )

    legend = [
        Line2D(
            [0], [0], color="#8B0000", lw=style.PRIMARY_LINEWIDTH,
            marker="o", markersize=style.MARKER_SIZE,
            markerfacecolor="#8B0000", label="RA-Append"
        ),
        Line2D(
            [0], [0], color="#E69F00", lw=style.PRIMARY_LINEWIDTH,
            marker="s", markersize=style.MARKER_SIZE,
            markerfacecolor="#E69F00", label="RA-Plateau"
        ),
        Line2D(
            [0], [0], color="#009E73", lw=style.PRIMARY_LINEWIDTH,
            marker="D", markersize=style.MARKER_SIZE,
            markerfacecolor="#009E73", label="RA-Insert"
        ),
        Line2D(
            [0], [0], color="#0072B2", lw=style.PRIMARY_LINEWIDTH, ls="--",
            marker="o", markerfacecolor="none",
            markeredgewidth=style.MARKER_EDGEWIDTH,
            markersize=style.MARKER_SIZE, label="AVQE"
        ),
    ]
    style.add_top_legend(fig, legend, ncol=4)
    fig.text(
        0.54, 0.018, "ADAPT iteration", ha="center", va="center",
        fontsize=style.AXIS_LABEL_SIZE,
    )
    fig.text(
        0.010, 0.475, "Absolute energy-error", rotation=90,
        ha="center", va="center", fontsize=style.AXIS_LABEL_SIZE,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "paper_i_ra_adaptive_append_main.pdf"
    png_path = args.output_dir / "paper_i_ra_adaptive_append_main.png"
    provenance_path = (
        args.output_dir / "paper_i_ra_adaptive_append_main_provenance.json"
    )
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    provenance = {
        "schema": "paper_i_ra_adaptive_append_main_bundle3_v9",
        "evidence_class": (
            "weak_complete_report_crop_strong_append_complete_"
            "strong_insertion_partial_snapshot_20260820"
        ),
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "png": {"path": str(png_path), "sha256": sha256(png_path)},
        },
        "display": {
            "error_metric": "same-cutoff |E_k-E_exact|",
            "weak_holstein_x_crop": 45,
            "strong_holstein_x_crop": 75,
            "weak_holstein_y_limits": [1e-15, 4.0],
            "strong_holstein_y_limits": [1e-9, 4.0],
            "y_axis_policy": "shared within each Holstein row; labels on the left panel only",
            "shared_style": {"path": str(style_path), "sha256": sha256(style_path)},
            "crop_policy": "display only; no history was rerun or truncated",
            "marker_policy": (
                "one method-specific marker at the regime's AVQE reference "
                "round k_star when that series contains the round; shorter "
                "partial insertion traces receive no marker"
            ),
            "strong_insertion_policy": (
                "partial locally evidenced traces stop at their latest point; "
                "no extrapolation"
            ),
        },
        "per_regime": stats,
        "bindings": {
            "bundle3_visual_reference": {
                "path": str(bundle_pdf),
                "sha256": sha256(bundle_pdf),
            },
            "bundle3_manifest": {
                "path": str(bundle_manifest_path),
                "sha256": sha256(bundle_manifest_path),
            },
            "kstar_tables": {
                "path": str(KSTAR_TABLES),
                "sha256": sha256(KSTAR_TABLES),
                "weak_bundle3_trace_projection_sha256": current_weak_trace_sha,
                "weak_result_source_commits": ["9c64e72e", "faa6b63c"],
                "bundle3_manifest_input_sha256": manifest["inputs"]
                ["kstar_tables.json"]["sha256"],
            },
            "comparison_exact_energy_loader": {
                "path": str(comparison_module),
                "sha256": sha256(comparison_module),
            },
            "generator": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
    }
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    for regime, *_ in REGIMES:
        fields = []
        for arm, *_ in SERIES:
            row = stats[regime]["series"][arm]
            fields.append(
                f"{arm}:n={row['displayed_point_count']},"
                f"k={row['latest_displayed_k']},"
                f"dE={row['latest_displayed_error']:.3e},"
                f"status={row['status']}"
            )
        print(regime, " | ".join(fields))
    print("wrote", pdf_path)
    print("wrote", png_path)
    print("wrote", provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
