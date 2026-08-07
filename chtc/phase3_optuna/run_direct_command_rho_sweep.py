#!/usr/bin/env python3
"""Run one row of the direct-command Paper-I rho sweep."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


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


def _get_flag(tokens: list[str], flag: str) -> str | None:
    try:
        idx = tokens.index(flag)
    except ValueError:
        return None
    return tokens[idx + 1] if idx + 1 < len(tokens) else None


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
    return [*_remove_flag(tokens, flag), flag, value]


def _rho_slug(rho: float) -> str:
    return str(rho).replace(".", "p")


def _normalize(tokens: list[str], row: dict[str, Any], out_root: Path) -> tuple[list[str], list[dict[str, str]]]:
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
    rho_value = str(row["rho"])
    tokens = _set_flag(tokens, "--phase2-rho", rho_value)
    mutations.append({"field": "--phase2-rho", "old": old_rho or "<code default>", "new": rho_value})

    selected = (row.get("template") or {}).get("selected_logical")
    old_selected = _get_flag(tokens, "--adapt-selected-logical-source-json")
    if old_selected and selected:
        staged = selected["staged_path"]
        tokens = _set_flag(tokens, "--adapt-selected-logical-source-json", staged)
        mutations.append({"field": "--adapt-selected-logical-source-json", "old": old_selected, "new": staged})

    return tokens, mutations


def _summarize_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"result_json_exists": False}
    doc = _load_json(path)
    av = doc.get("adapt_vqe", {}) if isinstance(doc, dict) else {}
    return {
        "result_json_exists": True,
        "energy": av.get("energy"),
        "abs_delta_e": av.get("abs_delta_e"),
        "ansatz_depth": av.get("ansatz_depth"),
        "operators": av.get("operators"),
        "settings_phase2_rho": (doc.get("settings") or {}).get("phase2_rho") if isinstance(doc, dict) else None,
        "settings_phase1_score_mode": (doc.get("settings") or {}).get("phase1_score_mode") if isinstance(doc, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    spec = _load_json(Path(args.spec_json))
    rows = {row["row_id"]: row for row in spec["rows"]}
    if args.row_id not in rows:
        raise SystemExit(f"unknown row_id {args.row_id}")
    row = rows[args.row_id]

    out_root = Path(args.output_root) / row["row_id"]
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    command_path = Path(row["template"]["staged_path"])
    tokens = _read_command(command_path)
    tokens, mutations = _normalize(tokens, row, out_root)
    if any("phase3_policy_optuna" in token or "oracle-grid" in token for token in tokens):
        raise SystemExit("forbidden wrapper token in normalized command")

    command_audit = {
        "schema": "direct_command_rho_sweep_command_v1",
        "row_id": row["row_id"],
        "batch_id": row["batch_id"],
        "case": row["case"],
        "rho": row["rho"],
        "template": row["template"],
        "mutations": mutations,
        "command_tokens": tokens,
        "command_line": shlex.join(tokens),
    }
    (logs_dir / "direct_command_audit.json").write_text(json.dumps(command_audit, indent=2) + "\n")
    (logs_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(tokens) + "\n")

    start = time.time()
    with (logs_dir / "stdout.txt").open("w") as stdout, (logs_dir / "stderr.txt").open("w") as stderr:
        proc = subprocess.run(tokens, stdout=stdout, stderr=stderr, text=True)
    elapsed = time.time() - start

    result_json = out_root / "json" / "result.json"
    result_audit = {
        "schema": "direct_command_rho_sweep_result_v1",
        "row_id": row["row_id"],
        "batch_id": row["batch_id"],
        "case": row["case"],
        "rho": row["rho"],
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "result_summary": _summarize_result(result_json),
    }
    (logs_dir / "result_audit.json").write_text(json.dumps(result_audit, indent=2, default=str) + "\n")
    print(json.dumps(result_audit, indent=2, default=str))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
