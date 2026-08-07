"""Plot AP-McLachlan energy and cached-reference energy errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLOT_RUNNER_SCHEMA_V1 = "ap_energy_diagnostics_plot_v1"


def build_energy_diagnostics_plot(
    *,
    input_jsons: Sequence[str | Path],
    output_png: str | Path,
    title: str | None = None,
    allow_unmatched_reference: bool = False,
) -> dict[str, Any]:
    runs = [_load_run(Path(path)) for path in input_jsons]
    if not runs:
        raise ValueError("At least one input JSON is required.")
    _validate_runs(runs, allow_unmatched_reference=bool(allow_unmatched_reference))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_energy, ax_error) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    for run in runs:
        label = str(run["label"])
        rows = run["rows"]
        times = np.asarray([float(row["time"]) for row in rows], dtype=float)
        energy = np.asarray([float(row["energy_expectation"]) for row in rows], dtype=float)
        reference = np.asarray([float(row["reference_energy"]) for row in rows], dtype=float)
        abs_error = np.asarray([float(row["abs_energy_error"]) for row in rows], dtype=float)
        ax_energy.plot(times, energy, linewidth=1.8, label=f"{label} AP")
        ax_energy.plot(times, reference, linewidth=1.4, linestyle="--", label=f"{label} exact")
        ax_error.semilogy(times, np.maximum(abs_error, 1.0e-16), linewidth=1.8, label=label)

    ax_energy.set_ylabel("total energy")
    ax_error.set_ylabel("abs energy error")
    ax_error.set_xlabel("time")
    ax_energy.grid(True, alpha=0.25)
    ax_error.grid(True, alpha=0.25, which="both")
    ax_energy.legend(fontsize=8)
    ax_error.legend(fontsize=8)
    if title:
        fig.suptitle(str(title))
    output_path = Path(output_png)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return {
        "schema": PLOT_RUNNER_SCHEMA_V1,
        "output_png": str(output_path),
        "input_jsons": [str(path) for path in input_jsons],
        "run_count": int(len(runs)),
        "labels": [str(run["label"]) for run in runs],
        "allow_unmatched_reference": bool(allow_unmatched_reference),
    }


def _load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Runner JSON must be an object: {path}")
    rows = payload.get("plot_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"Runner JSON is missing sequence `plot_rows`: {path}")
    label = _label_for_payload(path=path, payload=payload)
    return {
        "path": str(path),
        "label": label,
        "rows": [dict(row) for row in rows],
        "summary": dict(payload.get("summary", {}) or {}),
    }


def _label_for_payload(*, path: Path, payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary", {}) or {}
    if isinstance(summary, Mapping):
        seed_label = summary.get("seed_kind") or summary.get("label")
        if seed_label not in {None, ""}:
            return str(seed_label)
    stem = path.stem
    for prefix in ("weak_weak_",):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def _validate_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    allow_unmatched_reference: bool,
) -> None:
    for run in runs:
        rows = list(run["rows"])
        if not rows:
            raise ValueError(f"Run has no plot rows: {run['path']}")
        summary = dict(run.get("summary", {}) or {})
        unmatched = int(summary.get("reference_energy_unmatched_count", 0) or 0)
        if unmatched and not bool(allow_unmatched_reference):
            raise ValueError(
                f"Run has {unmatched} unmatched reference energy rows: {run['path']}"
            )
        for index, row in enumerate(rows):
            for key in ("time", "energy_expectation", "reference_energy", "abs_energy_error"):
                if row.get(key) is None:
                    raise ValueError(
                        f"Run {run['path']} row {index} is missing required plot field {key!r}."
                    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot AP-McLachlan energy and cached-reference energy error."
    )
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--allow-unmatched-reference", action="store_true")
    parser.add_argument("--output-manifest", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_energy_diagnostics_plot(
            input_jsons=tuple(args.input_json),
            output_png=args.output_png,
            title=args.title,
            allow_unmatched_reference=bool(args.allow_unmatched_reference),
        )
        if args.output_manifest not in {None, ""}:
            manifest_path = Path(args.output_manifest)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PLOT_RUNNER_SCHEMA_V1",
    "build_energy_diagnostics_plot",
    "main",
]
