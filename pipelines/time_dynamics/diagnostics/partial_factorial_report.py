"""Interim matched-pair report for a partially completed Paper-II campaign.

Exact propagation is constructed only after the trajectories have completed.
It supplies physical-error diagnostics and never enters either online method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from pipelines.time_dynamics.diagnostics.exact_driven_reference import exact_rows
from pipelines.time_dynamics.run_lock import assert_comparable


REPORT_SCHEMA_V1 = "paper_ii_partial_factorial_report_v1"
METHOD_APM = "exchange"
METHOD_AVQDS = "avqds"
GRAPH_KEYS = (
    "c_hat_2q",
    "c_hat_d",
    "c_hat_1q",
    "theta_count",
)
ERROR_KEYS = (
    "mean_abs_energy_error",
    "mean_abs_doublon_error",
    "mean_max_abs_site_occupation_error",
    "mean_abs_primary_density_error",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(cell_id: str) -> dict[str, str]:
    parts = str(cell_id).split("__")
    if len(parts) != 7:
        raise ValueError(f"unrecognized campaign cell id: {cell_id!r}")
    return {
        "seed_id": parts[0],
        "method": parts[1],
        "controller": parts[2],
        "drive": parts[3],
        "horizon": parts[4],
        "search_profile": parts[5],
        "cut": parts[6],
    }


def _mean_abs(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("observable-error series must be nonempty and finite")
    return float(np.mean(np.abs(array)))


def _cell_record(
    run_path: Path,
    *,
    seed_path: Path,
    reference_cache: dict[str, list[dict[str, Any]]],
    reference_substeps: int,
) -> dict[str, Any]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    cost_path = run_path.with_name("terminal_qiskit_cost.json")
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    cost_rows = list(cost.get("rows") or ())
    if len(cost_rows) != 1:
        raise ValueError(f"{cost_path} must contain exactly one cost row")
    cost_row = dict(cost_rows[0])
    parity = dict(cost_row.get("terminal_reconstruction_parity") or {})
    if parity.get("passed") is not True:
        raise ValueError(f"{cost_path} failed terminal reconstruction parity")

    cell_id = run_path.parent.name
    identity = _identity(cell_id)
    lock = dict(run.get("run_lock") or {})
    fingerprint = str(lock.get("physics_fingerprint") or "")
    if not fingerprint:
        raise ValueError(f"{run_path} has no physics fingerprint")
    if fingerprint not in reference_cache:
        reference_cache[fingerprint] = exact_rows(
            seed_path,
            run_path,
            substeps=int(reference_substeps),
        )
    reference = reference_cache[fingerprint]
    plot_rows = list(run.get("plot_rows") or ())
    if len(reference) != len(plot_rows):
        raise ValueError(f"reference grid mismatch for {cell_id}")

    energy_errors: list[float] = []
    doublon_errors: list[float] = []
    site_errors: list[float] = []
    primary_errors: list[float] = []
    for observed, exact in zip(plot_rows, reference):
        if not np.isclose(
            float(observed["time"]),
            float(exact["time"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"reference time mismatch for {cell_id}")
        energy_errors.append(
            float(observed["energy_expectation"]) - float(exact["energy"])
        )
        doublon_errors.append(float(observed["doublon"]) - float(exact["doublon"]))
        observed_sites = np.asarray(observed["site_occupations"], dtype=float)
        exact_sites = np.asarray(exact["site_occupations"], dtype=float)
        site_errors.append(float(np.max(np.abs(observed_sites - exact_sites))))
        primary_errors.append(
            float(observed["primary_density"]) - float(exact["primary_density"])
        )

    summary = dict(run.get("summary") or {})
    uncured = int(summary.get("uncured_local_subdivision_count") or 0)
    if uncured:
        raise ValueError(f"successful-row archive contains {uncured} uncured steps: {cell_id}")
    graph = {
        key: float(cost_row[f"graph_proxy_terminal_{key}"])
        for key in GRAPH_KEYS
    }
    return {
        "cell_id": cell_id,
        **identity,
        "run_json": str(run_path),
        "run_json_sha256": _sha256(run_path),
        "cost_json": str(cost_path),
        "cost_json_sha256": _sha256(cost_path),
        "code_revision": str(lock.get("code_revision") or ""),
        "physics_fingerprint": fingerprint,
        "comparison_fingerprint": str(lock.get("comparison_fingerprint") or ""),
        "initial_graph_support_sha256": str(
            cost_row.get("graph_proxy_initial_support_sha256") or ""
        ),
        "physical_patch_sha256": str(
            dict(cost.get("compile_defaults") or {}).get("physical_patch_sha256") or ""
        ),
        "mean_abs_energy_error": _mean_abs(energy_errors),
        "mean_abs_doublon_error": _mean_abs(doublon_errors),
        "mean_max_abs_site_occupation_error": _mean_abs(site_errors),
        "mean_abs_primary_density_error": _mean_abs(primary_errors),
        "final_runtime_parameter_count": int(
            summary.get("runtime_parameter_count_final") or 0
        ),
        "accepted_deletion_containing_patch_count": int(
            summary.get("accepted_delete_count") or 0
        )
        + int(summary.get("accepted_exchange_count") or 0),
        "graph_proxy": graph,
        "qiskit_secondary": {
            key: int(cost_row[key]) for key in ("N2q", "D2q", "Dc")
        },
    }


def _dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    keys: Sequence[str],
) -> bool:
    return bool(
        all(float(left[key]) <= float(right[key]) for key in keys)
        and any(float(left[key]) < float(right[key]) for key in keys)
    )


def _error_no_worse_and_c2q_lower(
    apm: Mapping[str, Any],
    avqds: Mapping[str, Any],
    metric: str,
) -> bool:
    """Return the declared per-observable accuracy--cost comparison."""

    return bool(
        float(apm[metric]) <= float(avqds[metric])
        and float(apm["graph_proxy"]["c_hat_2q"])
        < float(avqds["graph_proxy"]["c_hat_2q"])
    )


def build_report(
    *,
    root: Path,
    seed_path: Path,
    reference_substeps: int = 40,
) -> dict[str, Any]:
    run_paths = sorted(root.glob("*/run.json"))
    if not run_paths:
        raise ValueError(f"no completed run.json files below {root}")
    reference_cache: dict[str, list[dict[str, Any]]] = {}
    cells = [
        _cell_record(
            run_path,
            seed_path=seed_path,
            reference_cache=reference_cache,
            reference_substeps=int(reference_substeps),
        )
        for run_path in run_paths
    ]
    by_key: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for cell in cells:
        key = (cell["controller"], cell["drive"], cell["horizon"])
        by_key.setdefault(key, {})[str(cell["method"])] = cell

    pairs: list[dict[str, Any]] = []
    for (controller, drive, horizon), methods in sorted(by_key.items()):
        if METHOD_APM not in methods or METHOD_AVQDS not in methods:
            continue
        apm = methods[METHOD_APM]
        avqds = methods[METHOD_AVQDS]
        assert_comparable((_load_lock(apm), _load_lock(avqds)))
        if apm["initial_graph_support_sha256"] != avqds["initial_graph_support_sha256"]:
            raise ValueError(f"initial graph-support mismatch for {controller}/{drive}")
        if apm["physical_patch_sha256"] != avqds["physical_patch_sha256"]:
            raise ValueError(f"physical compile-patch mismatch for {controller}/{drive}")
        error_ratios = {
            key: float(apm[key]) / max(float(avqds[key]), np.finfo(float).tiny)
            for key in ERROR_KEYS
        }
        graph_ratios = {
            key: float(apm["graph_proxy"][key])
            / max(float(avqds["graph_proxy"][key]), np.finfo(float).tiny)
            for key in GRAPH_KEYS
        }
        per_observable = {
            key: _error_no_worse_and_c2q_lower(apm, avqds, key)
            for key in ERROR_KEYS
        }
        pairs.append(
            {
                "controller": controller,
                "drive": drive,
                "horizon": horizon,
                "apm": apm,
                "avqds": avqds,
                "error_ratios_apm_over_avqds": error_ratios,
                "graph_ratios_apm_over_avqds": graph_ratios,
                "per_observable_error_no_worse_and_c2q_lower": per_observable,
                "energy_error_no_worse_and_c2q_lower": per_observable[
                    "mean_abs_energy_error"
                ],
                "all_observable_errors_no_worse_and_graph_pareto_lower": bool(
                    all(float(apm[key]) <= float(avqds[key]) for key in ERROR_KEYS)
                    and _dominates(apm["graph_proxy"], avqds["graph_proxy"], GRAPH_KEYS)
                ),
            }
        )

    paired_cell_ids = {
        pair[method]["cell_id"]
        for pair in pairs
        for method in ("apm", "avqds")
    }
    per_observable_counts = {
        key: sum(
            bool(pair["per_observable_error_no_worse_and_c2q_lower"][key])
            for pair in pairs
        )
        for key in ERROR_KEYS
    }
    return {
        "schema": REPORT_SCHEMA_V1,
        "status": "INTERIM_PARTIAL_RESULTS",
        "decision_data_scope": "exact_reference_reporting_only",
        "seed_json": str(seed_path),
        "seed_json_sha256": _sha256(seed_path),
        "reference_substeps_per_reporting_interval": int(reference_substeps),
        "successful_cell_count": len(cells),
        "complete_pair_count": len(pairs),
        "per_observable_qualifying_pair_counts": per_observable_counts,
        "strict_all_observable_qualifying_pair_count": sum(
            bool(pair["all_observable_errors_no_worse_and_graph_pareto_lower"])
            for pair in pairs
        ),
        "unpaired_successful_cell_ids": [
            cell["cell_id"] for cell in cells if cell["cell_id"] not in paired_cell_ids
        ],
        "cells": cells,
        "pairs": pairs,
    }


def _load_lock(cell: Mapping[str, Any]) -> dict[str, Any]:
    run = json.loads(Path(str(cell["run_json"])).read_text(encoding="utf-8"))
    return dict(run.get("run_lock") or {})


def write_csv(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "controller",
        "drive",
        "method",
        *ERROR_KEYS,
        *(f"graph_{key}" for key in GRAPH_KEYS),
        "final_runtime_parameter_count",
        "accepted_deletion_containing_patch_count",
        "energy_error_no_worse_and_c2q_lower",
        "all_observable_errors_no_worse_and_graph_pareto_lower",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in report["pairs"]:
            for method in ("apm", "avqds"):
                cell = pair[method]
                writer.writerow(
                    {
                        "controller": pair["controller"],
                        "drive": pair["drive"],
                        "method": method,
                        **{key: cell[key] for key in ERROR_KEYS},
                        **{
                            f"graph_{key}": cell["graph_proxy"][key]
                            for key in GRAPH_KEYS
                        },
                        "final_runtime_parameter_count": cell[
                            "final_runtime_parameter_count"
                        ],
                        "accepted_deletion_containing_patch_count": cell[
                            "accepted_deletion_containing_patch_count"
                        ],
                        "energy_error_no_worse_and_c2q_lower": pair[
                            "energy_error_no_worse_and_c2q_lower"
                        ],
                        "all_observable_errors_no_worse_and_graph_pareto_lower": pair[
                            "all_observable_errors_no_worse_and_graph_pareto_lower"
                        ],
                    }
                )


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Paper II interim matched-pair diagnostic",
        "",
        (
            f"This report contains {report['complete_pair_count']} complete method pairs "
            f"from {report['successful_cell_count']} successful cells. Exact propagation "
            "is reporting-only. Running and safeguarded-failure cells are excluded."
        ),
        "",
        (
            "A qualifying observable has APM mean absolute error no greater than "
            "AVQDS and strictly lower terminal graph proxy $\\widehat C_{2q}$. "
            "This is tested separately for each observable; `all` requires every "
            "reported observable and every graph-proxy coordinate to be no worse."
        ),
        "",
        "| controller | drive | mean |ΔE| APM / AVQDS | mean |ΔD| APM / AVQDS | "
        "mean max |Δn| APM / AVQDS | mean |Δn_primary| APM / AVQDS | "
        "(Ĉ₂q, ĈD, Ĉ₁q) APM / AVQDS | qualifying | all |",
        "|---|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for pair in report["pairs"]:
        apm = pair["apm"]
        avqds = pair["avqds"]
        lines.append(
            "| {controller} | {drive} | {ae:.3e} / {ve:.3e} | "
            "{ad:.3e} / {vd:.3e} | {an:.3e} / {vn:.3e} | "
            "{ap:.3e} / {vp:.3e} | ({ac:.0f}, {acd:.0f}, {ac1:.0f}) / "
            "({vc:.0f}, {vcd:.0f}, {vc1:.0f}) | {qualifying} | {strict} |".format(
                controller=pair["controller"],
                drive=pair["drive"],
                ae=apm["mean_abs_energy_error"],
                ve=avqds["mean_abs_energy_error"],
                ad=apm["mean_abs_doublon_error"],
                vd=avqds["mean_abs_doublon_error"],
                an=apm["mean_max_abs_site_occupation_error"],
                vn=avqds["mean_max_abs_site_occupation_error"],
                ap=apm["mean_abs_primary_density_error"],
                vp=avqds["mean_abs_primary_density_error"],
                ac=apm["graph_proxy"]["c_hat_2q"],
                vc=avqds["graph_proxy"]["c_hat_2q"],
                acd=apm["graph_proxy"]["c_hat_d"],
                vcd=avqds["graph_proxy"]["c_hat_d"],
                ac1=apm["graph_proxy"]["c_hat_1q"],
                vc1=avqds["graph_proxy"]["c_hat_1q"],
                qualifying=", ".join(
                    label
                    for key, label in (
                        ("mean_abs_energy_error", "E"),
                        ("mean_abs_doublon_error", "D"),
                        ("mean_max_abs_site_occupation_error", "n"),
                        ("mean_abs_primary_density_error", "n_primary"),
                    )
                    if pair["per_observable_error_no_worse_and_c2q_lower"][key]
                )
                or "none",
                strict=(
                    "yes"
                    if pair["all_observable_errors_no_worse_and_graph_pareto_lower"]
                    else "no"
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pairs(report: Mapping[str, Any], path: Path) -> None:
    panels = (
        ("mean_abs_energy_error", r"mean $|\Delta E|$"),
        ("mean_abs_doublon_error", r"mean $|\Delta D|$"),
        ("mean_max_abs_site_occupation_error", r"mean max$_i|\Delta n_i|$"),
        ("mean_abs_primary_density_error", r"mean $|\Delta n_{\rm primary}|$"),
    )
    controller_colors = {
        "state_motion_5e-5": "#277da1",
        "delta_theta_5e-5": "#f8961e",
        "state_motion_1e-3_plus_parameter_5e-5": "#43aa8b",
    }
    figure, axes = plt.subplots(2, 2, figsize=(7.1, 5.7), constrained_layout=True)
    for axis, (metric, label) in zip(axes.flat, panels):
        for pair in report["pairs"]:
            apm = pair["apm"]
            avqds = pair["avqds"]
            color = controller_colors.get(pair["controller"], "#666666")
            xs = (
                avqds["graph_proxy"]["c_hat_2q"],
                apm["graph_proxy"]["c_hat_2q"],
            )
            ys = (avqds[metric], apm[metric])
            axis.plot(xs, ys, color=color, alpha=0.5, linewidth=1.0)
            axis.scatter(xs[0], ys[0], marker="s", color=color, s=30, zorder=3)
            axis.scatter(xs[1], ys[1], marker="o", color=color, s=30, zorder=3)
            if pair["per_observable_error_no_worse_and_c2q_lower"][metric]:
                axis.scatter(
                    xs[1], ys[1], marker="*", facecolors="none", edgecolors="black",
                    linewidths=0.8, s=105, zorder=4,
                )
        axis.set_yscale("log")
        axis.set_xlabel(r"terminal graph proxy $\widehat C_{2q}$")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.2)
    figure.suptitle(
        "Paper II interim t=10 matched pairs\n"
        "circle: APM; square: AVQDS; star: APM no-worse panel error at lower Ĉ₂q",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.08)
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seed-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-substeps", type=int, default=40)
    args = parser.parse_args(argv)
    report = build_report(
        root=args.root,
        seed_path=args.seed_json,
        reference_substeps=int(args.reference_substeps),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "interim_matched_pair_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(report, output_dir / "interim_matched_pair_report.csv")
    write_markdown(report, output_dir / "interim_matched_pair_report.md")
    plot_pairs(report, output_dir / "interim_matched_pair_tradeoff.png")
    print(json.dumps({
        "status": report["status"],
        "successful_cell_count": report["successful_cell_count"],
        "complete_pair_count": report["complete_pair_count"],
        "output_dir": str(output_dir),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ERROR_KEYS",
    "GRAPH_KEYS",
    "REPORT_SCHEMA_V1",
    "_dominates",
    "_error_no_worse_and_c2q_lower",
    "build_report",
    "main",
    "plot_pairs",
    "write_csv",
    "write_markdown",
]
