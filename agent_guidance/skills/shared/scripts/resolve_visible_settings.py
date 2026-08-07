#!/usr/bin/env python3
"""Resolve paper-visible table settings to a reusable run baseline.

This is intentionally conservative. It starts from a visible table/source-map
entry, follows that entry to the source JSON, verifies provenance when possible,
and emits the settings/contracts an agent must reuse before changing only the
user-requested variable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

TARGET_AXES = (
    "regimes",
    "cases",
    "rows",
    "benchmarks",
    "targets",
    "entries",
)

ROOT_METADATA_KEYS = (
    "table_label",
    "figure_label",
    "manuscript_tex",
    "method_order",
    "output_dir",
    "output_prefix",
    "plot_script",
    "updated_date",
    "schema",
)

PAYLOAD_SETTINGS_KEYS = (
    "visible_cells",
    "route_contract",
    "cutoff_contract",
    "pool_contract",
    "regime_parameters",
    "case_parameters",
    "parameters",
    "n_ph_work",
    "n_ph_ref",
    "n_ph_eval",
    "n_ph_ed",
    "reference_energy",
    "source_current_json",
    "source_access",
    "run_continuation",
    "trial_number",
    "trial_params",
    "settings",
    "class_settings",
    "algorithm_settings",
    "controller_settings",
    "policy",
    "spec",
    "runner",
    "guardrails",
    "manifest",
    "physical_target_manifest",
    "pipeline",
    "hamiltonian",
    "method_id",
    "algorithm_id",
    "case_id",
    "family",
    "table_i",
)

VISIBLE_VALUE_KEYS = (
    "visible_value",
    "display_value",
    "table_value",
    "value",
    "status",
    "last_final_abs_delta_e",
    "last_abs_delta_e",
    "final_abs_delta_e",
    "abs_delta_e",
    "display_delta_e",
)


def die(message: str, *, detail: Any | None = None) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    if detail is not None:
        print(json.dumps(detail, indent=2, sort_keys=True), file=sys.stderr)
    raise SystemExit(2)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        die(f"JSON file not found: {path}")
    except json.JSONDecodeError as exc:
        die(f"Invalid JSON in {path}: {exc}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def mapping_lookup(mapping: dict[str, Any], requested: str) -> tuple[str, Any] | None:
    if requested in mapping:
        return requested, mapping[requested]
    wanted = normalize_key(requested)
    matches = [(k, v) for k, v in mapping.items() if normalize_key(k) == wanted]
    if len(matches) == 1:
        return matches[0]
    return None


def mapping_get(mapping: dict[str, Any], requested: str, label: str) -> tuple[str, Any]:
    match = mapping_lookup(mapping, requested)
    if match is not None:
        return match
    die(
        f"Could not resolve {label} '{requested}'",
        detail={"available": sorted(mapping.keys())},
    )


def is_probably_uri(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value))


def as_repo_path(value: str, repo_root: Path) -> Path | None:
    expanded = value.replace("~", str(Path.home()), 1) if value.startswith("~") else value
    if is_probably_uri(expanded):
        return None
    path = Path(expanded)
    if not path.is_absolute():
        path = repo_root / path
    return path


def trim_node_context(node: dict[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for key, value in node.items():
        if key == "methods":
            continue
        context[key] = value
    return context


def resolve_target(
    data: dict[str, Any],
    *,
    target_key: str | None,
    target_axis: str | None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    if not target_key:
        return None, None, data

    axes = (target_axis,) if target_axis else TARGET_AXES
    checked: dict[str, list[str]] = {}
    for axis in axes:
        container = data.get(axis)
        if isinstance(container, dict):
            checked[axis] = sorted(container.keys())
            match = mapping_lookup(container, target_key)
            if match is None:
                continue
            actual_key, target = match
            if not isinstance(target, dict):
                die(f"Resolved {axis}.{actual_key} is not an object")
            return axis, actual_key, target
    die(f"Could not resolve target key '{target_key}'", detail={"checked_axes": checked})


def resolve_method(target: dict[str, Any], method: str | None) -> tuple[str | None, dict[str, Any]]:
    methods = target.get("methods")
    if isinstance(methods, dict):
        if method is None:
            die("Target contains a methods map; pass --method", detail={"available_methods": sorted(methods.keys())})
        actual_method, entry = mapping_get(methods, method, "method")
        if not isinstance(entry, dict):
            die(f"Resolved method {actual_method} is not an object")
        return actual_method, entry
    if method is not None:
        die("Target has no methods map; omit --method or choose a different --target-axis")
    return None, target


def first_visible_value(*objects: dict[str, Any]) -> Any | None:
    for obj in objects:
        for key in VISIBLE_VALUE_KEYS:
            if key in obj:
                return obj[key]
    return None


def extract_settings(payload: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    for key in PAYLOAD_SETTINGS_KEYS:
        if key in payload:
            extracted[key] = payload[key]
    return extracted


def merged_reusable_settings(*named_payloads: tuple[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    merged: dict[str, Any] = {}
    sources: dict[str, list[str]] = {}
    for name, payload in named_payloads:
        extracted = extract_settings(payload)
        if extracted:
            sources[name] = sorted(extracted.keys())
            merged.update(extracted)
    return merged, sources


def build_trace(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    repo_root = Path.cwd()
    source_map_path = as_repo_path(args.source_map, repo_root)
    if source_map_path is None:
        die("--source-map must be a local path, not a URI")
    data = load_json(source_map_path)
    if not isinstance(data, dict):
        die("Source map root must be a JSON object")

    target_key = args.target_key or args.regime or args.case
    target_axis, actual_target_key, target = resolve_target(
        data,
        target_key=target_key,
        target_axis=args.target_axis,
    )
    actual_method, entry = resolve_method(target, args.method)

    root_metadata = {k: data[k] for k in ROOT_METADATA_KEYS if k in data}
    target_context = trim_node_context(target)

    trace: dict[str, Any] = {
        "source_map": str(source_map_path.relative_to(repo_root) if source_map_path.is_relative_to(repo_root) else source_map_path),
        "target_axis": target_axis,
        "regime_or_case": actual_target_key,
        "method": actual_method,
        "root_metadata": root_metadata,
        "target_context": target_context,
        "source_entry": entry,
        "visible_value": first_visible_value(entry),
    }

    source_json_value = entry.get("source_json") or entry.get("source_manifest") or entry.get("manifest")
    trace["source_json"] = source_json_value
    problems: list[str] = []

    if not isinstance(source_json_value, str) or not source_json_value:
        problems.append("source entry does not contain source_json/source_manifest/manifest")
        trace["source_json_exists_locally"] = False
        settings_reused, settings_sources = merged_reusable_settings(
            ("target_context", target_context),
            ("source_entry", entry),
        )
        trace["settings_reused"] = settings_reused
        trace["settings_reused_sources"] = settings_sources
        return trace, problems

    source_path = as_repo_path(source_json_value, repo_root)
    trace["source_json_is_local_path"] = source_path is not None
    if source_path is None:
        problems.append("source JSON is not a local path; fetch or materialize it before reusing settings")
        trace["source_json_exists_locally"] = False
        settings_reused, settings_sources = merged_reusable_settings(
            ("target_context", target_context),
            ("source_entry", entry),
        )
        trace["settings_reused"] = settings_reused
        trace["settings_reused_sources"] = settings_sources
        return trace, problems

    trace["source_json_resolved_path"] = str(source_path)
    if not source_path.exists():
        problems.append("source JSON is missing locally")
        trace["source_json_exists_locally"] = False
        settings_reused, settings_sources = merged_reusable_settings(
            ("target_context", target_context),
            ("source_entry", entry),
        )
        trace["settings_reused"] = settings_reused
        trace["settings_reused_sources"] = settings_sources
        return trace, problems

    trace["source_json_exists_locally"] = True
    actual_sha = sha256_file(source_path)
    expected_sha = entry.get("source_sha256") or entry.get("sha256")
    trace["source_sha256_expected"] = expected_sha
    trace["source_sha256_actual"] = actual_sha
    if expected_sha:
        trace["source_sha256_match"] = actual_sha == expected_sha
        if actual_sha != expected_sha:
            problems.append("source JSON SHA-256 does not match source map")
    else:
        trace["source_sha256_match"] = None

    payload = load_json(source_path)
    if not isinstance(payload, dict):
        problems.append("source JSON root is not an object")
        settings_reused, settings_sources = merged_reusable_settings(
            ("target_context", target_context),
            ("source_entry", entry),
        )
        trace["settings_reused"] = settings_reused
        trace["settings_reused_sources"] = settings_sources
        return trace, problems

    settings_reused, settings_sources = merged_reusable_settings(
        ("target_context", target_context),
        ("source_entry", entry),
        ("source_payload", payload),
    )
    trace["source_payload_keys"] = sorted(payload.keys())
    trace["settings_reused"] = settings_reused
    trace["settings_reused_sources"] = settings_sources
    trace["settings_changed"] = []
    trace["visible_value"] = trace["visible_value"] if trace["visible_value"] is not None else first_visible_value(payload)

    if not settings_reused:
        problems.append("no reusable settings/contracts found in source JSON")

    trace["run_note_template"] = {
        "table_label": root_metadata.get("table_label") or root_metadata.get("figure_label"),
        "regime_or_case": actual_target_key,
        "method": actual_method,
        "visible_value": trace.get("visible_value"),
        "source_map": trace["source_map"],
        "source_json": source_json_value,
        "source_sha256": expected_sha or actual_sha,
        "settings_reused": "copy from settings_reused in this resolver output",
        "settings_changed": "list only the user-requested changes",
    }
    return trace, problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a visible table/source-map entry to source settings for a paper-facing rerun."
    )
    parser.add_argument("--source-map", required=True, help="Path to the visible table/source-map JSON.")
    parser.add_argument("--target-axis", choices=TARGET_AXES, help="Container to use, e.g. regimes or cases. Defaults to auto-search.")
    parser.add_argument("--target-key", help="Visible row/case/regime key. Equivalent to --regime/--case.")
    parser.add_argument("--regime", help="Visible regime key alias for --target-key.")
    parser.add_argument("--case", help="Visible case key alias for --target-key.")
    parser.add_argument("--method", help="Visible method cell key. Required when the target contains a methods map.")
    parser.add_argument("--output-json", help="Optional path to write the resolver trace JSON.")
    parser.add_argument("--allow-missing-source-json", action="store_true", help="Audit mode only: do not fail if the source JSON is missing/non-local.")
    parser.add_argument("--allow-sha-mismatch", action="store_true", help="Audit mode only: do not fail on source SHA mismatch.")
    parser.add_argument("--allow-empty-settings", action="store_true", help="Audit mode only: do not fail when no reusable settings/contracts are found.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if sum(bool(x) for x in (args.target_key, args.regime, args.case)) > 1:
        die("Use only one of --target-key, --regime, or --case")

    trace, problems = build_trace(args)

    fatal: list[str] = []
    for problem in problems:
        if "missing locally" in problem or "not a local path" in problem:
            if not args.allow_missing_source_json:
                fatal.append(problem)
        elif "SHA-256" in problem:
            if not args.allow_sha_mismatch:
                fatal.append(problem)
        elif "no reusable settings" in problem:
            if not args.allow_empty_settings:
                fatal.append(problem)
        else:
            fatal.append(problem)

    trace["status"] = "ok" if not fatal else "blocked"
    trace["problems"] = problems

    output = json.dumps(trace, indent=2, sort_keys=True)
    if args.output_json:
        out_path = Path(args.output_json)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n")
    print(output)

    if fatal:
        print("\nFAIL CLOSED:", file=sys.stderr)
        for problem in fatal:
            print(f"- {problem}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
