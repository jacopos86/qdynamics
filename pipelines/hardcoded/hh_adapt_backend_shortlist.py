#!/usr/bin/env python3
"""Run HH backend-aware Phase 3 ADAPT separately for a fixed backend shortlist."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded import adapt_pipeline as adapt_mod


_DEFAULT_SHORTLIST = "ibm_boston,ibm_miami"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_")
    return token or "backend"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the heavy HH Phase 3 ADAPT search once per backend in a fixed shortlist, "
            "then write a comparison summary. Remaining arguments are forwarded to adapt_pipeline.py."
        )
    )
    p.add_argument("--backend-shortlist", type=str, default=_DEFAULT_SHORTLIST)
    p.add_argument("--backend-transpile-seed", type=int, default=7)
    p.add_argument("--backend-optimization-level", type=int, default=1)
    p.add_argument("--summary-json", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--run-prefix", type=str, default=None)
    p.set_defaults(include_proxy_baseline=True)
    p.add_argument("--include-proxy-baseline", dest="include_proxy_baseline", action="store_true")
    p.add_argument("--no-proxy-baseline", dest="include_proxy_baseline", action="store_false")
    return p


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = build_parser()
    return parser.parse_known_args(argv)


def _extract_backend_summary(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    adapt = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    summary = adapt.get("backend_compile_cost_summary", {}) if isinstance(adapt, Mapping) else {}
    return summary if isinstance(summary, Mapping) else {}


def _summarize_run(
    *,
    label: str,
    requested_backend: str | None,
    output_json: Path,
) -> dict[str, Any]:
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    adapt = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    backend_summary = _extract_backend_summary(payload)
    selected = backend_summary.get("selected_backend", {}) if isinstance(backend_summary, Mapping) else {}
    if not isinstance(selected, Mapping):
        selected = {}
    op_counts = selected.get("compiled_op_counts", {}) if isinstance(selected.get("compiled_op_counts", {}), Mapping) else {}
    row = {
        "label": str(label),
        "requested_backend": (None if requested_backend is None else str(requested_backend)),
        "resolved_backend": (None if selected.get("transpile_backend") is None else str(selected.get("transpile_backend"))),
        "resolution_kind": (None if selected.get("resolution_kind") is None else str(selected.get("resolution_kind"))),
        "using_fake_backend": bool(selected.get("using_fake_backend", False)) if selected else None,
        "compile_cost_mode": (
            str(adapt.get("compile_cost_mode"))
            if isinstance(adapt, Mapping) and adapt.get("compile_cost_mode") is not None
            else str(payload.get("settings", {}).get("phase3_backend_cost_mode", "proxy"))
        ),
        "energy": adapt.get("energy"),
        "abs_delta_e": adapt.get("abs_delta_e"),
        "ansatz_depth": adapt.get("ansatz_depth"),
        "logical_num_parameters": adapt.get("logical_num_parameters", adapt.get("ansatz_depth")),
        "compiled_count_2q": selected.get("compiled_count_2q"),
        "compiled_depth": selected.get("compiled_depth"),
        "compiled_size": selected.get("compiled_size"),
        "compiled_cx_count": selected.get("compiled_cx_count"),
        "compiled_ecr_count": selected.get("compiled_ecr_count"),
        "swap_count": op_counts.get("swap"),
        "absolute_burden_score_v1": selected.get("absolute_burden_score_v1"),
        "stop_reason": adapt.get("stop_reason"),
        "output_json": str(output_json),
    }
    return row


def _write_summary_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "requested_backend",
        "resolved_backend",
        "resolution_kind",
        "using_fake_backend",
        "compile_cost_mode",
        "energy",
        "abs_delta_e",
        "ansatz_depth",
        "logical_num_parameters",
        "compiled_count_2q",
        "compiled_depth",
        "compiled_size",
        "compiled_cx_count",
        "compiled_ecr_count",
        "swap_count",
        "absolute_burden_score_v1",
        "stop_reason",
        "output_json",
        "delta_energy_vs_proxy",
        "delta_abs_delta_e_vs_proxy",
        "delta_ansatz_depth_vs_proxy",
        "delta_logical_num_parameters_vs_proxy",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main(argv: list[str] | None = None) -> None:
    args, forwarded = parse_args(argv)
    forwarded_args = [str(x) for x in forwarded]
    forbidden = {
        "--output-json",
        "--output-pdf",
        "--phase3-backend-cost-mode",
        "--phase3-backend-name",
        "--phase3-backend-shortlist",
        "--phase3-backend-transpile-seed",
        "--phase3-backend-optimization-level",
    }
    overlap = sorted(flag for flag in forbidden if flag in forwarded_args)
    if overlap:
        raise ValueError(
            "hh_adapt_backend_shortlist.py manages per-run outputs and backend flags itself; remove forwarded flags: "
            + ", ".join(overlap)
        )

    shortlist = [str(tok).strip() for tok in str(args.backend_shortlist).split(",") if str(tok).strip() != ""]
    if not shortlist:
        raise ValueError("Expected at least one backend in --backend-shortlist.")

    skip_pdf = "--skip-pdf" in forwarded_args
    run_prefix = str(args.run_prefix or f"hh_backend_adapt_{_utc_stamp()}")
    json_dir = REPO_ROOT / "artifacts" / "json"
    pdf_dir = REPO_ROOT / "artifacts" / "pdf"
    csv_dir = REPO_ROOT / "artifacts" / "csv"
    json_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    summary_json = args.summary_json or (json_dir / f"{run_prefix}_summary.json")
    summary_csv = args.summary_csv or (csv_dir / f"{run_prefix}_summary.csv")

    rows: list[dict[str, Any]] = []

    def _run_case(*, label: str, backend_name: str | None, mode: str) -> None:
        slug = _slugify(label)
        output_json = json_dir / f"{run_prefix}_{slug}.json"
        output_pdf = pdf_dir / f"{run_prefix}_{slug}.pdf"
        run_args = list(forwarded_args)
        run_args.extend(["--problem", "hh", "--adapt-continuation-mode", "phase3_v1", "--phase3-backend-cost-mode", str(mode)])
        if backend_name is not None:
            run_args.extend(["--phase3-backend-name", str(backend_name)])
        run_args.extend(
            [
                "--phase3-backend-transpile-seed",
                str(int(args.backend_transpile_seed)),
                "--phase3-backend-optimization-level",
                str(int(args.backend_optimization_level)),
                "--output-json",
                str(output_json),
            ]
        )
        if not skip_pdf:
            run_args.extend(["--output-pdf", str(output_pdf)])
        adapt_mod.main(run_args)
        rows.append(_summarize_run(label=str(label), requested_backend=backend_name, output_json=output_json))

    if bool(args.include_proxy_baseline):
        _run_case(label="proxy_baseline", backend_name=None, mode="proxy")
    for backend_name in shortlist:
        _run_case(label=backend_name, backend_name=str(backend_name), mode="transpile_single_v1")

    proxy_row = next((row for row in rows if str(row.get("label")) == "proxy_baseline"), None)
    for row in rows:
        if proxy_row is None or row is proxy_row:
            row["delta_energy_vs_proxy"] = None
            row["delta_abs_delta_e_vs_proxy"] = None
            row["delta_ansatz_depth_vs_proxy"] = None
            row["delta_logical_num_parameters_vs_proxy"] = None
            continue
        row["delta_energy_vs_proxy"] = (
            None
            if row.get("energy") is None or proxy_row.get("energy") is None
            else float(row["energy"]) - float(proxy_row["energy"])
        )
        row["delta_abs_delta_e_vs_proxy"] = (
            None
            if row.get("abs_delta_e") is None or proxy_row.get("abs_delta_e") is None
            else float(row["abs_delta_e"]) - float(proxy_row["abs_delta_e"])
        )
        row["delta_ansatz_depth_vs_proxy"] = (
            None
            if row.get("ansatz_depth") is None or proxy_row.get("ansatz_depth") is None
            else int(row["ansatz_depth"]) - int(proxy_row["ansatz_depth"])
        )
        row["delta_logical_num_parameters_vs_proxy"] = (
            None
            if row.get("logical_num_parameters") is None or proxy_row.get("logical_num_parameters") is None
            else int(row["logical_num_parameters"]) - int(proxy_row["logical_num_parameters"])
        )

    backend_rows = [row for row in rows if str(row.get("compile_cost_mode")) != "proxy"]
    best_compile = None
    if backend_rows:
        best_compile = min(
            backend_rows,
            key=lambda row: (
                float("inf") if row.get("compiled_count_2q") is None else float(row.get("compiled_count_2q", 0.0)),
                float("inf") if row.get("compiled_depth") is None else float(row.get("compiled_depth", 0.0)),
                float("inf") if row.get("compiled_size") is None else float(row.get("compiled_size", 0.0)),
                str(row.get("resolved_backend", row.get("requested_backend", ""))),
            ),
        )
    best_energy = None
    if backend_rows:
        valid_energy_rows = [row for row in backend_rows if row.get("abs_delta_e") is not None]
        if valid_energy_rows:
            best_energy = min(valid_energy_rows, key=lambda row: (float(row.get("abs_delta_e", float("inf"))), str(row.get("resolved_backend", row.get("requested_backend", "")))))

    summary_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "hh_adapt_backend_shortlist",
        "run_prefix": str(run_prefix),
        "backend_shortlist": [str(x) for x in shortlist],
        "backend_transpile_seed": int(args.backend_transpile_seed),
        "backend_optimization_level": int(args.backend_optimization_level),
        "include_proxy_baseline": bool(args.include_proxy_baseline),
        "selection_rules": {
            "compile_rank": ["compiled_count_2q", "compiled_depth", "compiled_size"],
            "energy_rank": ["abs_delta_e"],
        },
        "best_compile_backend": (None if best_compile is None else dict(best_compile)),
        "best_energy_backend": (None if best_energy is None else dict(best_energy)),
        "rows": rows,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _write_summary_csv(summary_csv, rows)

    print(f"summary_json={summary_json}")
    print(f"summary_csv={summary_csv}")
    if best_compile is not None:
        print(f"best_compile_backend={best_compile.get('resolved_backend') or best_compile.get('requested_backend')}")
    if best_energy is not None:
        print(f"best_energy_backend={best_energy.get('resolved_backend') or best_energy.get('requested_backend')}")


if __name__ == "__main__":
    main()
