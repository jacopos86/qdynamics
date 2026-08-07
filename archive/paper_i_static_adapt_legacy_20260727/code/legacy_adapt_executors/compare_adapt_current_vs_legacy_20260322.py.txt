#!/usr/bin/env python3
"""Run current vs 2026-03-22 legacy ADAPT and summarize first divergence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_ENTRY = REPO_ROOT / "pipelines/hardcoded/adapt_pipeline_legacy_20260322.py"
DEFAULT_CURRENT_ENTRY = REPO_ROOT / "pipelines/hardcoded/adapt_pipeline.py"


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _coerce_path(value: str | None) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(value).expanduser().resolve()


def _run_pipeline(
    *,
    name: str,
    entrypoint: Path,
    python_bin: str,
    pipeline_args: Sequence[str],
    out_dir: Path,
) -> dict[str, Any]:
    case_dir = out_dir / name
    logs_dir = case_dir / "logs"
    json_dir = case_dir / "json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    output_json = json_dir / "result.json"
    output_pdf = case_dir / "pdf" / "result.pdf"
    command = [
        python_bin,
        str(entrypoint),
        *pipeline_args,
        "--output-json",
        str(output_json),
        "--output-pdf",
        str(output_pdf),
        "--skip-pdf",
    ]
    (logs_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + subprocess.list2cmdline(command)
        + "\n",
        encoding="utf-8",
    )
    with (logs_dir / "stdout.log").open("w", encoding="utf-8") as stdout_fh, (
        logs_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_fh:
        proc = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            check=False,
            text=True,
            stdout=stdout_fh,
            stderr=stderr_fh,
        )
    return {
        "case_dir": case_dir,
        "output_json": output_json,
        "stdout_log": logs_dir / "stdout.log",
        "stderr_log": logs_dir / "stderr.log",
        "returncode": int(proc.returncode),
    }


def _discover_supported_long_options(*, entrypoint: Path, python_bin: str) -> set[str]:
    result = subprocess.run(
        [python_bin, str(entrypoint), "--help"],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = (result.stdout or "") + "\n" + (result.stderr or "")
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", help_text))


def _filter_pipeline_args_for_entry(
    *,
    entrypoint: Path,
    python_bin: str,
    pipeline_args: Sequence[str],
) -> tuple[list[str], list[str]]:
    supported = _discover_supported_long_options(entrypoint=entrypoint, python_bin=python_bin)
    filtered: list[str] = []
    dropped: list[str] = []
    idx = 0
    while idx < len(pipeline_args):
        token = str(pipeline_args[idx])
        if token.startswith("--"):
            option_name = token.split("=", 1)[0]
            if option_name not in supported:
                dropped.append(token)
                if "=" not in token and idx + 1 < len(pipeline_args) and not str(pipeline_args[idx + 1]).startswith("--"):
                    dropped.append(str(pipeline_args[idx + 1]))
                    idx += 2
                    continue
                idx += 1
                continue
            filtered.append(token)
            if "=" not in token and idx + 1 < len(pipeline_args) and not str(pipeline_args[idx + 1]).startswith("--"):
                filtered.append(str(pipeline_args[idx + 1]))
                idx += 2
                continue
            idx += 1
            continue
        filtered.append(token)
        idx += 1
    return filtered, dropped


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_selected_ops(payload: dict[str, Any]) -> list[str]:
    history = payload.get("adapt_vqe", {}).get("history", [])
    selected: list[str] = []
    for row in history:
        ops = row.get("selected_ops")
        if isinstance(ops, list) and ops:
            selected.append(str(ops[0]))
        elif row.get("selected_op") is not None:
            selected.append(str(row.get("selected_op")))
    return selected


def _first_divergence(left: Sequence[str], right: Sequence[str]) -> dict[str, Any]:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if str(left[idx]) != str(right[idx]):
            return {
                "depth_1based": int(idx + 1),
                "legacy_op": str(left[idx]),
                "current_op": str(right[idx]),
            }
    if len(left) != len(right):
        longer_name = "legacy" if len(left) > len(right) else "current"
        longer_seq = left if len(left) > len(right) else right
        return {
            "depth_1based": int(limit + 1),
            "legacy_op": (str(left[limit]) if len(left) > limit else None),
            "current_op": (str(right[limit]) if len(right) > limit else None),
            "reason": f"{longer_name}_has_extra_depth",
            "extra_op": str(longer_seq[limit]),
        }
    return {
        "depth_1based": None,
        "legacy_op": None,
        "current_op": None,
        "reason": "none",
    }


def _compile_summary(
    *,
    legacy_payload: dict[str, Any],
    current_payload: dict[str, Any],
) -> dict[str, Any]:
    legacy_ops = _extract_selected_ops(legacy_payload)
    current_ops = _extract_selected_ops(current_payload)
    return {
        "legacy": {
            "energy": legacy_payload.get("adapt_vqe", {}).get("energy"),
            "abs_delta_e": legacy_payload.get("adapt_vqe", {}).get("abs_delta_e"),
            "ansatz_depth": legacy_payload.get("adapt_vqe", {}).get("ansatz_depth"),
            "logical_num_parameters": legacy_payload.get("adapt_vqe", {}).get("logical_num_parameters"),
            "num_parameters": legacy_payload.get("adapt_vqe", {}).get("num_parameters"),
            "selected_ops": legacy_ops,
            "final_ops": legacy_payload.get("adapt_vqe", {}).get("operators"),
        },
        "current": {
            "energy": current_payload.get("adapt_vqe", {}).get("energy"),
            "abs_delta_e": current_payload.get("adapt_vqe", {}).get("abs_delta_e"),
            "ansatz_depth": current_payload.get("adapt_vqe", {}).get("ansatz_depth"),
            "logical_num_parameters": current_payload.get("adapt_vqe", {}).get("logical_num_parameters"),
            "num_parameters": current_payload.get("adapt_vqe", {}).get("num_parameters"),
            "selected_ops": current_ops,
            "final_ops": current_payload.get("adapt_vqe", {}).get("operators"),
        },
        "first_divergence": _first_divergence(legacy_ops, current_ops),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run current vs legacy 2026-03-22 ADAPT entrypoints on the same CLI and summarize the first divergence.",
    )
    parser.add_argument("--label", type=str, default=None, help="Run label for the output artifact directory.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional explicit output directory.")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--legacy-entry", type=str, default=str(DEFAULT_LEGACY_ENTRY))
    parser.add_argument("--current-entry", type=str, default=str(DEFAULT_CURRENT_ENTRY))
    parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Common pipeline args for both entrypoints. Use `--` before the forwarded args.",
    )
    args = parser.parse_args(argv)
    if args.pipeline_args and args.pipeline_args[0] == "--":
        args.pipeline_args = args.pipeline_args[1:]
    if not args.pipeline_args:
        parser.error("Provide forwarded pipeline args after `--`.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    label = str(args.label or f"{_timestamp_slug()}_adapt_legacy_current_compare")
    output_dir = _coerce_path(args.output_dir)
    if output_dir is None:
        output_dir = (REPO_ROOT / "artifacts/agent_runs" / label).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_pipeline_args, legacy_dropped_args = _filter_pipeline_args_for_entry(
        entrypoint=Path(args.legacy_entry).resolve(),
        python_bin=str(args.python_bin),
        pipeline_args=list(args.pipeline_args),
    )
    current_pipeline_args, current_dropped_args = _filter_pipeline_args_for_entry(
        entrypoint=Path(args.current_entry).resolve(),
        python_bin=str(args.python_bin),
        pipeline_args=list(args.pipeline_args),
    )

    legacy_run = _run_pipeline(
        name="legacy_20260322",
        entrypoint=Path(args.legacy_entry).resolve(),
        python_bin=str(args.python_bin),
        pipeline_args=legacy_pipeline_args,
        out_dir=output_dir,
    )
    current_run = _run_pipeline(
        name="current_head",
        entrypoint=Path(args.current_entry).resolve(),
        python_bin=str(args.python_bin),
        pipeline_args=current_pipeline_args,
        out_dir=output_dir,
    )

    legacy_payload = _load_payload(legacy_run["output_json"])
    current_payload = _load_payload(current_run["output_json"])
    summary = _compile_summary(
        legacy_payload=legacy_payload,
        current_payload=current_payload,
    )
    summary["meta"] = {
        "legacy_entry": str(Path(args.legacy_entry).resolve()),
        "current_entry": str(Path(args.current_entry).resolve()),
        "pipeline_args": [str(x) for x in args.pipeline_args],
        "legacy_pipeline_args": [str(x) for x in legacy_pipeline_args],
        "legacy_dropped_args": [str(x) for x in legacy_dropped_args],
        "current_pipeline_args": [str(x) for x in current_pipeline_args],
        "current_dropped_args": [str(x) for x in current_dropped_args],
        "output_dir": str(output_dir),
    }

    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    summary_path = json_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote comparison summary: {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
