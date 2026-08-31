#!/usr/bin/env python3
"""Build the Paper-I-format six-regime Bundle-5 convergence figure.

The weak-Holstein row is display-cropped at iteration 45.  The completed
strong-Holstein trajectories are displayed through iteration 75.  Trace
sources are selected in the order promoted k-star table, transferred
checkpoint archive, then live tail.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


RESULTS_DIR = Path(__file__).resolve().parent
KSTAR_TABLES = RESULTS_DIR / "kstar_tables.json"
HANDOFF = RESULTS_DIR.parent / "HANDOFF_PAPER_I_B5_SINGLE_EVIDENCE_20260822.md"
FINDINGS = RESULTS_DIR.parent / "FINDINGS_ESTIMATOR_LEDGER_DUPLICATION_20260821.md"
WEAK_B5_PROJECTION_SHA256 = (
    "84f46d0ba64bd268038beb33b52dbaaa32591d8c65933d807f6ef60d478b7bdd"
)
AAVQE_HISTORY = Path(
    "/Users/jakestrobel/local_repos/Holstein_test_fullclone_3/output/"
    "local_runs/paper_i_aavqe_r50_to_r75_tolmatch_20260820_v2/"
    "aavqe_full_hist_tolmatch_9665004.json"
)
AAVQE_HISTORY_SHA256 = (
    "a05789300cd95f5d3d564bdd53e84f7e465d9bb9dc10570b36aa23594674e271"
)
B5_RUN_RELATIVE = Path(
    "output/local_runs/paper_i_hh_b3_measured_residual_20260821"
)

REGIMES = [
    ("weak_weak", "Weak-weak", 3, 45),
    ("intermediate_weak", "Intermediate-weak", 3, 45),
    ("strong_weak_u8", "Strong-weak", 3, 45),
    ("weak_strong", "Weak-strong", 7, 75),
    ("intermediate_strong", "Intermediate-strong", 7, 75),
    ("strong_strong_u8", "Strong-strong", 7, 75),
]
WEAK_REGIMES = {regime for regime, _, cutoff, _ in REGIMES if cutoff == 3}
SERIES = [
    ("AVQE", "AVQE", "#0072B2", "--", "o"),
    ("mr_append_ra", "RA-Append", "#8B0000", "-", "o"),
    ("mr_plateau_insertion", "RA-Plateau", "#E69F00", "-", "s"),
    ("mr_always_insertion", "RA-Insert", "#009E73", "-", "D"),
]
ERR_FLOOR = 1e-16
ROW_SCALES = {
    3: ((1e-15, 4.0), [1e0, 1e-5, 1e-10, 1e-15]),
    7: ((1e-9, 4.0), [1e0, 1e-3, 1e-6, 1e-9]),
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


def weak_b5_projection_sha256(kstar: dict) -> str:
    subset = {
        key: entry
        for key, entry in kstar["b5"].items()
        if key.rsplit("|", 1)[-1] in WEAK_REGIMES
    }
    encoded = json.dumps(
        subset, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_plot_style(output_dir: Path):
    style_path = output_dir.parent / "paper_i_main_plot_style.py"
    spec = importlib.util.spec_from_file_location("paper_i_main_plot_style", style_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load shared plot style {style_path}")
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    return style, style_path


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
            # The streaming event is emitted after accepting a depth-d state;
            # that state belongs to controller iteration k=d-1.
            iteration = int(row["depth"]) - 1
            if iteration < 1:
                continue
            points[iteration] = max(
                abs(float(row["energy"]) - exact_energy), ERR_FLOOR
            )
    return sorted(points.items())


def archive_member_bytes(archive: tarfile.TarFile, suffix: str) -> bytes | None:
    match = None
    for member in archive:
        if member.name.endswith(suffix):
            if match is not None:
                raise RuntimeError(
                    f"expected one archive member ending in {suffix!r}"
                )
            match = member
    if match is None:
        return None
    handle = archive.extractfile(match)
    if handle is None:
        raise RuntimeError(f"could not read archive member {match.name}")
    return handle.read()


def archive_worker_exit_status(path: Path) -> str | None:
    """Read the early worker-status member without indexing the full archive."""
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if not member.name.endswith("worker_exit_status.txt"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError(f"could not read archive member {member.name}")
            return handle.read().decode("utf-8", errors="replace").strip()
    return None


def parse_checkpoint_archive(
    path: Path, exact_energy: float
) -> tuple[list[tuple[int, float]], dict]:
    worker_status = archive_worker_exit_status(path)
    with tarfile.open(path, "r:gz") as archive:
        checkpoint_bytes = archive_member_bytes(
            archive, "run/checkpoints/current.json"
        )
        if checkpoint_bytes is None:
            raise RuntimeError(f"checkpoint missing from {path}")
        checkpoint = json.loads(checkpoint_bytes)

    history = checkpoint.get("adapt_vqe", {}).get("history") or []
    points: dict[int, float] = {}
    for row in history:
        if row.get("depth") is None or row.get("energy_after_opt") is None:
            continue
        depth = int(row["depth"])
        points[depth] = max(
            abs(float(row["energy_after_opt"]) - exact_energy), ERR_FLOOR
        )
    trace = sorted(points.items())
    if not trace:
        raise RuntimeError(f"no accepted energy history in {path}")

    checkpoint_meta = checkpoint.get("checkpoint", {})
    adapt_meta = checkpoint.get("adapt_vqe", {})
    source = {
        "kind": "recorded_bundle5_checkpoint_archive",
        "path": str(path),
        "sha256": sha256(path),
        "member": "run/checkpoints/current.json",
        "worker_exit_status": worker_status,
        "checkpoint_complete": checkpoint_meta.get("complete"),
        "checkpoint_reason": checkpoint_meta.get("reason"),
        "checkpoint_depth": checkpoint_meta.get("depth"),
        "history_count": adapt_meta.get("history_count", len(trace)),
    }
    return trace, source


def unique_archive(transfer_dir: Path, arm: str, regime: str) -> Path | None:
    pattern = f"b3mr__{arm}__{regime}__nph7__*.tar.gz"
    matches = sorted(transfer_dir.glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple transferred archives match {arm}|{regime}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0] if matches else None


def load_ra_trace(
    *,
    kstar: dict,
    arm: str,
    regime: str,
    exact_energy: float,
    b5_run_root: Path,
) -> tuple[list[tuple[int, float]], dict]:
    key = f"{arm}|{regime}"
    entry = kstar.get("b5", {}).get(key)
    if entry is not None and entry.get("trace"):
        if regime in WEAK_REGIMES and entry.get("salg_star_corrected") is None:
            raise RuntimeError(f"missing corrected estimator prefix for {key}")
        entry_source = str(entry.get("source", ""))
        trace_uses_checkpoint_depth = (
            regime not in WEAK_REGIMES
            and "completed checkpoint archive" in entry_source
        )
        iteration_offset = -1 if trace_uses_checkpoint_depth else 0
        promoted_trace = {
            int(depth) + iteration_offset: max(float(error), ERR_FLOOR)
            for depth, error in entry["trace"]
        }
        terminal_k = entry.get("k_term")
        terminal_error = entry.get("dE_term")
        if terminal_k is not None and terminal_error is not None:
            promoted_trace[int(terminal_k)] = max(float(terminal_error), ERR_FLOOR)
        return (
            sorted(promoted_trace.items()),
            {
                "kind": "promoted_bundle5_measured_residual_trace",
                "path": str(KSTAR_TABLES),
                "sha256": sha256(KSTAR_TABLES),
                "json_key": f"b5.{key}.trace",
                "iteration_normalization": (
                    "controller iteration k = stored accepted depth d - 1"
                    if trace_uses_checkpoint_depth
                    else "stored controller iteration k"
                ),
                "terminal_point_policy": (
                    "replace the streaming terminal point by dE_term from the "
                    "accepted checkpoint state"
                ),
            },
        )

    archive_path = unique_archive(b5_run_root / "transfer", arm, regime)
    rejected_archive = None
    if archive_path is not None:
        # Append archives were promoted before the live-prefix monitor and are
        # outside its six-proc scope; preserve their established figure source.
        if arm == "mr_append_ra":
            return parse_checkpoint_archive(archive_path, exact_energy)
        worker_status = archive_worker_exit_status(archive_path)
        if worker_status == "0":
            archive_trace, archive_source = parse_checkpoint_archive(
                archive_path, exact_energy
            )
            if archive_source.get("checkpoint_complete") is True:
                return archive_trace, archive_source
            rejected_archive = archive_source
        else:
            rejected_archive = {
                "kind": "recorded_bundle5_checkpoint_archive",
                "path": str(archive_path),
                "sha256": sha256(archive_path),
                "worker_exit_status": worker_status,
            }

    tail_path = b5_run_root / "live_tails" / f"b3mr__{arm}__{regime}__nph7.tail"
    if tail_path.is_file():
        trace = parse_ai_log(tail_path, exact_energy)
        if not trace:
            raise RuntimeError(f"no accepted iterations in {tail_path}")
        source = {
            "kind": "recorded_bundle5_live_tail",
            "path": str(tail_path),
            "sha256": sha256(tail_path),
            "iteration_normalization": "controller k = raw accepted depth - 1",
        }
        if rejected_archive is not None:
            source["rejected_archive"] = {
                **rejected_archive,
                "promotion_status": "rejected_failed_or_incomplete",
            }
        return trace, source

    raise RuntimeError(f"no Bundle-5 trace source found for {key}")


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

    kstar = load_json(KSTAR_TABLES)
    projection_sha = weak_b5_projection_sha256(kstar)
    if projection_sha != WEAK_B5_PROJECTION_SHA256:
        raise RuntimeError(
            "weak Bundle-5 evidence changed: "
            f"expected {WEAK_B5_PROJECTION_SHA256}, got {projection_sha}"
        )
    if sha256(AAVQE_HISTORY) != AAVQE_HISTORY_SHA256:
        raise RuntimeError("AVQE history does not match the authenticated comparator")
    if not HANDOFF.is_file() or not FINDINGS.is_file():
        raise RuntimeError("Bundle-5 handoff or estimator-accounting findings missing")

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
    aavqe_energy = load_json(AAVQE_HISTORY)
    b5_run_root = args.base_root / B5_RUN_RELATIVE

    traces: dict[tuple[str, str], list[tuple[int, float]]] = {}
    sources: dict[tuple[str, str], dict] = {}
    for regime, *_ in REGIMES:
        traces[("AVQE", regime)] = [
            (int(depth), max(abs(float(energy) - exact[regime]), ERR_FLOOR))
            for depth, energy in aavqe_energy[regime]
        ]
        sources[("AVQE", regime)] = {
            "kind": "authenticated_r75_history",
            "path": str(AAVQE_HISTORY),
            "sha256": sha256(AAVQE_HISTORY),
        }
        for arm, *_ in SERIES[1:]:
            trace, source = load_ra_trace(
                kstar=kstar,
                arm=arm,
                regime=regime,
                exact_energy=exact[regime],
                b5_run_root=b5_run_root,
            )
            traces[(arm, regime)] = trace
            sources[(arm, regime)] = source

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

    flat_axes = [axis for row in axes for axis in row]
    for index, (axis, (regime, title, cutoff, horizon)) in enumerate(
        zip(flat_axes, REGIMES)
    ):
        regime_kstar = int(kstar["aavqe"][regime]["k_star"])
        stats[regime] = {
            "display_horizon": horizon,
            "local_phonon_cutoff": cutoff,
            "k_star": regime_kstar,
            "series": {},
        }
        for arm, label, color, linestyle, marker in SERIES:
            displayed = [
                (depth, error)
                for depth, error in traces[(arm, regime)]
                if depth <= horizon
            ]
            if not displayed:
                raise RuntimeError(f"no displayed points for {arm}|{regime}")
            depths = [depth for depth, _ in displayed]
            errors = [error for _, error in displayed]
            axis.plot(
                depths,
                errors,
                color=color,
                lw=style.PRIMARY_LINEWIDTH,
                ls=linestyle,
                solid_capstyle="round",
            )
            marker_error = dict(displayed).get(regime_kstar)
            if marker_error is not None:
                axis.plot(
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
                "displayed_trace": [[depth, error] for depth, error in displayed],
                "marker_k": regime_kstar if marker_error is not None else None,
                "marker_error": marker_error,
                "latest_displayed_k": depths[-1],
                "latest_displayed_error": errors[-1],
                "status": source["kind"],
                "source": source,
            }

        xticks = (
            [0, 10, 20, 30, 40, 45]
            if horizon == 45
            else [0, 15, 30, 45, 60, 75]
        )
        ylimits, yticks = ROW_SCALES[cutoff]
        style.style_log_axis(
            axis,
            title=title,
            xlim=(0, horizon + 1),
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
        0.54, 0.035, "ADAPT iteration", ha="center", va="center",
        fontsize=style.AXIS_LABEL_SIZE,
    )
    fig.text(
        0.010, 0.47, "Absolute energy-error", rotation=90,
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
        "schema": "paper_i_ra_adaptive_append_main_bundle5_sixregime_v1",
        "evidence_class": "bundle5_measured_residual_all_regimes_complete_20260822",
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "png": {"path": str(png_path), "sha256": sha256(png_path)},
        },
        "display": {
            "error_metric": "same-cutoff |E_k-E_exact|",
            "weak_holstein_x_crop": 45,
            "strong_holstein_x_extent": 75,
            "weak_holstein_local_phonon_cutoff": 3,
            "strong_holstein_local_phonon_cutoff": 7,
            "weak_holstein_y_limits": list(ROW_SCALES[3][0]),
            "strong_holstein_y_limits": list(ROW_SCALES[7][0]),
            "crop_policy": "display only; stored histories are not truncated",
            "strong_trace_policy": (
                "display every completed Bundle-5 strong-Holstein trajectory "
                "through iteration 75"
            ),
            "source_precedence": [
                "promoted kstar_tables b5 trace",
                "transferred checkpoint archive",
                "live tail",
            ],
            "marker_policy": (
                "one method-specific marker at AVQE k_star only when the "
                "displayed trajectory contains that round"
            ),
            "shared_style": {"path": str(style_path), "sha256": sha256(style_path)},
        },
        "per_regime": stats,
        "bindings": {
            "kstar_tables": {
                "path": str(KSTAR_TABLES),
                "sha256": sha256(KSTAR_TABLES),
                "weak_bundle5_projection_sha256": projection_sha,
                "json_keys": ["aavqe", "b5"],
            },
            "bundle5_run_root": str(b5_run_root),
            "bundle5_handoff": {"path": str(HANDOFF), "sha256": sha256(HANDOFF)},
            "estimator_findings": {
                "path": str(FINDINGS),
                "sha256": sha256(FINDINGS),
                "reporting_policy": "staged_candidate_primitive_reuse_v1",
            },
            "aavqe_history": {
                "path": str(AAVQE_HISTORY),
                "sha256": sha256(AAVQE_HISTORY),
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
                f"source={row['status']}"
            )
        print(regime, " | ".join(fields))
    print("wrote", pdf_path)
    print("wrote", png_path)
    print("wrote", provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
