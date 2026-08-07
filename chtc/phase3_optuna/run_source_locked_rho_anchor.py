#!/usr/bin/env python3
"""Run one source-locked rho source-value anchor.

The runner replays the staged visible-source command and rewrites only
environment/output paths plus the explicitly tested trust-region rho value.
It never invokes Optuna, oracle-grid, or a settings-search wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


IMPORTANT_SETTINGS = [
    "phase1_score_mode",
    "phase1_shortlist_size",
    "phase2_shortlist_size",
    "phase2_enable_batching",
    "phase2_batch_target_size",
    "phase2_batch_size_cap",
    "phase3_enable_batching",
    "adapt_pool",
    "adapt_inner_optimizer",
    "adapt_insertion_mode",
    "adapt_reopt_policy",
    "adapt_window_size",
    "adapt_window_topk",
    "adapt_beam_live_branches",
    "adapt_beam_children_per_parent",
    "adapt_beam_terminated_keep",
    "adapt_max_depth",
    "adapt_maxiter",
    "static_route_id",
    "static_meta_feature_profile",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _read_command(path: Path) -> list[str]:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("set "):
            continue
        if "pipelines.static_adapt.adapt_pipeline" in stripped:
            return shlex.split(stripped)
    raise ValueError(f"no adapt_pipeline command found in {path}")


def _remove_flag(tokens: list[str], flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == flag:
            i += 2
            continue
        out.append(tokens[i])
        i += 1
    return out


def _set_flag(tokens: list[str], flag: str, value: str) -> list[str]:
    tokens = _remove_flag(tokens, flag)
    return [*tokens, flag, value]


def _get_flag(tokens: list[str], flag: str) -> str | None:
    try:
        idx = tokens.index(flag)
    except ValueError:
        return None
    return tokens[idx + 1] if idx + 1 < len(tokens) else None


def _normalize_command(tokens: list[str], spec: dict[str, Any], out_root: Path) -> tuple[list[str], list[dict[str, str]]]:
    mutations: list[dict[str, str]] = []
    if tokens and (tokens[0].endswith("/python") or tokens[0].endswith("/python3")):
        old = tokens[0]
        tokens[0] = "python"
        mutations.append({"field": "python_executable", "old": old, "new": "python"})

    result_json = out_root / "json" / "result.json"
    current_json = out_root / "json" / "current.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)

    old_output = _get_flag(tokens, "--output-json")
    tokens = _set_flag(tokens, "--output-json", str(result_json))
    mutations.append({"field": "--output-json", "old": old_output or "", "new": str(result_json)})

    old_current = _get_flag(tokens, "--adapt-current-json")
    if old_current is not None:
        tokens = _set_flag(tokens, "--adapt-current-json", str(current_json))
        mutations.append({"field": "--adapt-current-json", "old": old_current, "new": str(current_json)})

    old_rho = _get_flag(tokens, "--phase2-rho")
    rho_value = str(spec["rho_source_value"])
    tokens = _set_flag(tokens, "--phase2-rho", rho_value)
    mutations.append({"field": "--phase2-rho", "old": old_rho or "<code default>", "new": rho_value})

    selected_logical = spec.get("staged_selected_logical")
    old_selected = _get_flag(tokens, "--adapt-selected-logical-source-json")
    if old_selected and selected_logical:
        staged = selected_logical["staged_path"]
        tokens = _set_flag(tokens, "--adapt-selected-logical-source-json", staged)
        mutations.append({"field": "--adapt-selected-logical-source-json", "old": old_selected, "new": staged})

    return tokens, mutations


def _extract_abs_delta_e(doc: dict[str, Any]) -> float | None:
    for container in (doc.get("adapt_vqe", {}), doc):
        for key in ("abs_delta_e", "raw_external_abs_delta_e", "external_abs_energy_error", "same_cutoff_abs_delta_e"):
            val = container.get(key) if isinstance(container, dict) else None
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _extract_energy(doc: dict[str, Any]) -> float | None:
    for container in (doc.get("adapt_vqe", {}), doc):
        val = container.get("energy") if isinstance(container, dict) else None
        if isinstance(val, (int, float)):
            return float(val)
    return None


def _extract_depth(doc: dict[str, Any]) -> int | None:
    for container in (doc.get("adapt_vqe", {}), doc):
        for key in ("ansatz_depth", "adapt_depth_reached"):
            val = container.get(key) if isinstance(container, dict) else None
            if isinstance(val, int):
                return val
    return None


def _extract_ops(doc: dict[str, Any]) -> list[Any] | None:
    av = doc.get("adapt_vqe", {})
    if not isinstance(av, dict):
        return None
    for key in ("operators", "operator_sequence", "selected_operators"):
        val = av.get(key)
        if isinstance(val, list):
            return val
    return None


def _settings_diff(source_doc: dict[str, Any], run_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_settings = source_doc.get("settings", {})
    run_settings = run_doc.get("settings", {})
    source_av = source_doc.get("adapt_vqe", {})
    run_av = run_doc.get("adapt_vqe", {})
    diffs: dict[str, dict[str, Any]] = {}
    for key in IMPORTANT_SETTINGS:
        source_val = source_settings.get(key, source_av.get(key) if isinstance(source_av, dict) else None)
        run_val = run_settings.get(key, run_av.get(key) if isinstance(run_av, dict) else None)
        if source_val is not None and run_val is not None and source_val != run_val:
            diffs[key] = {"source": source_val, "anchor": run_val}
    return diffs


def _compare(spec: dict[str, Any], result_json: Path) -> dict[str, Any]:
    run_doc = _load_json(result_json) if result_json.exists() else {}
    comparison: dict[str, Any] = {
        "result_json_exists": result_json.exists(),
        "source_class": spec["source_class"],
        "pass": False,
    }
    source_result = spec.get("staged_result")
    visible_source = spec.get("staged_visible_source")
    if source_result:
        source_doc = _load_json(Path(source_result["staged_path"]))
        source_ops = _extract_ops(source_doc)
        run_ops = _extract_ops(run_doc)
        source_abs = _extract_abs_delta_e(source_doc)
        run_abs = _extract_abs_delta_e(run_doc)
        source_energy = _extract_energy(source_doc)
        run_energy = _extract_energy(run_doc)
        source_depth = _extract_depth(source_doc)
        run_depth = _extract_depth(run_doc)
        comparison.update(
            {
                "comparison_kind": "result_json",
                "source_abs_delta_e": source_abs,
                "anchor_abs_delta_e": run_abs,
                "abs_delta_e_absdiff": None if source_abs is None or run_abs is None else abs(source_abs - run_abs),
                "source_energy": source_energy,
                "anchor_energy": run_energy,
                "energy_absdiff": None if source_energy is None or run_energy is None else abs(source_energy - run_energy),
                "source_depth": source_depth,
                "anchor_depth": run_depth,
                "operator_sequence_available": source_ops is not None and run_ops is not None,
                "operator_sequence_equal": None if source_ops is None or run_ops is None else source_ops == run_ops,
                "non_swept_settings_diff": _settings_diff(source_doc, run_doc),
            }
        )
        comparison["pass"] = (
            result_json.exists()
            and comparison["abs_delta_e_absdiff"] is not None
            and comparison["abs_delta_e_absdiff"] <= 1e-9
            and comparison["energy_absdiff"] is not None
            and comparison["energy_absdiff"] <= 1e-9
            and comparison["source_depth"] == comparison["anchor_depth"]
            and (comparison["operator_sequence_equal"] is not False)
            and not comparison["non_swept_settings_diff"]
        )
        return comparison

    if visible_source:
        source_doc = _load_json(Path(visible_source["staged_path"]))
        source_av = source_doc.get("adapt_vqe", {}) if isinstance(source_doc, dict) else {}
        source_abs = _extract_abs_delta_e(source_doc)
        run_abs = _extract_abs_delta_e(run_doc)
        source_energy = _extract_energy(source_doc)
        run_energy = _extract_energy(run_doc)
        source_depth = _extract_depth(source_doc)
        run_depth = _extract_depth(run_doc)
        comparison.update(
            {
                "comparison_kind": "stdout_history_scalar",
                "source_abs_delta_e": source_abs,
                "anchor_abs_delta_e": run_abs,
                "abs_delta_e_absdiff": None if source_abs is None or run_abs is None else abs(source_abs - run_abs),
                "source_energy": source_energy,
                "anchor_energy": run_energy,
                "energy_absdiff": None if source_energy is None or run_energy is None else abs(source_energy - run_energy),
                "source_depth": source_depth,
                "anchor_depth": run_depth,
                "source_adapt_vqe_keys": sorted(source_av.keys()) if isinstance(source_av, dict) else [],
                "operator_sequence_available": False,
                "operator_sequence_equal": None,
            }
        )
        comparison["pass"] = (
            result_json.exists()
            and comparison["energy_absdiff"] is not None
            and comparison["energy_absdiff"] <= 1e-8
            and comparison["source_depth"] == comparison["anchor_depth"]
        )
        return comparison

    comparison["comparison_kind"] = "none"
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    spec_doc = _load_json(Path(args.spec_json))
    specs = {row["case_id"]: row for row in spec_doc["anchors"]}
    if args.case_id not in specs:
        raise SystemExit(f"unknown case_id {args.case_id}")
    spec = specs[args.case_id]

    out_root = Path(args.output_root) / spec["case_id"]
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    command_path = Path(spec["staged_command"]["staged_path"])
    source_tokens = _read_command(command_path)
    tokens, mutations = _normalize_command(source_tokens, spec, out_root)

    command_audit = {
        "schema": "source_locked_rho_anchor_command_v1",
        "case_id": spec["case_id"],
        "batch_id": spec["batch_id"],
        "run_class": spec["run_class"],
        "forbidden_wrappers_absent": all("phase3_policy_optuna" not in token and "oracle-grid" not in token for token in tokens),
        "source_command": spec["staged_command"],
        "mutations": mutations,
        "command_tokens": tokens,
        "command_line": shlex.join(tokens),
    }
    (logs_dir / "source_locked_command.json").write_text(json.dumps(command_audit, indent=2) + "\n")
    (logs_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(tokens) + "\n")

    with (logs_dir / "stdout.txt").open("w") as stdout, (logs_dir / "stderr.txt").open("w") as stderr:
        proc = subprocess.run(tokens, stdout=stdout, stderr=stderr, text=True)

    result_json = out_root / "json" / "result.json"
    comparison = _compare(spec, result_json)
    audit = {
        "schema": "source_locked_rho_anchor_result_v1",
        "case_id": spec["case_id"],
        "batch_id": spec["batch_id"],
        "returncode": proc.returncode,
        "command_audit": command_audit,
        "comparison": comparison,
        "anchor_pass": proc.returncode == 0 and comparison["pass"],
    }
    (logs_dir / "anchor_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    print(json.dumps(audit, indent=2, default=str))
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
