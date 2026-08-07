#!/usr/bin/env python3
"""Path-local tmux wrapper for focused HH Path A Optuna launches.

This wrapper keeps Path A automation non-invasive:
- refreshes the neutral Path A ledger,
- checks whether the tagged tmux session is already alive,
- launches a focused `hh_cost_energy_optuna.py` run only when idle,
- writes controller logs under `artifacts/agent_runs/<tag>/logs/`.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench.hh_path_a_ledger import DEFAULT_OUTPUT_PATH as DEFAULT_LEDGER_PATH
from pipelines.exact_bench.hh_path_a_ledger import main as refresh_path_a_ledger

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("hh_path_a_tmux_manifest.json")
_DEFAULT_TMUX_SESSION_PREFIX = "hh_path_a"
_OPTUNA_MODULE = "pipelines.exact_bench.hh_cost_energy_optuna"
_WRAPPER_PIPELINE_NAME = "hh_path_a_tmux_wrapper_v1"


@dataclass(frozen=True)
class WrapperManifest:
    tmux_session_prefix: str
    python_bin: str
    optuna_args: tuple[str, ...]


@dataclass(frozen=True)
class LaunchResult:
    requested_tag: str
    resolved_tag: str
    session_name: str
    status: str
    output_dir: str
    logs_dir: str
    ledger_path: str
    manifest_path: str
    command_path: str | None = None
    optuna_command: tuple[str, ...] = ()


# Built Math: tag_safe = sanitize(tag) where sanitize drops non [a-zA-Z0-9._-] symbols.
def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"


# Built Math: session(tag) = prefix ⊕ "_" ⊕ safe(tag).
def _session_name(prefix: str, tag: str) -> str:
    return f"{_safe_slug(prefix)}_{_safe_slug(tag)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> WrapperManifest:
    payload = _load_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Manifest must decode to a JSON object: {path}")
    raw_args = payload.get("optuna_args", [])
    if not isinstance(raw_args, list):
        raise ValueError(f"Manifest field `optuna_args` must be a JSON list: {path}")
    return WrapperManifest(
        tmux_session_prefix=str(payload.get("tmux_session_prefix") or _DEFAULT_TMUX_SESSION_PREFIX),
        python_bin=str(payload.get("python_bin") or sys.executable),
        optuna_args=tuple(str(token) for token in raw_args),
    )


def _remove_option(args: Sequence[str], flag: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(args):
        token = str(args[idx])
        if token == str(flag):
            idx += 1
            if idx < len(args) and not str(args[idx]).startswith("--"):
                idx += 1
            continue
        out.append(token)
        idx += 1
    return out


# Built Math: cmd = [python, -m, optuna_module, --tag, tag, --output-dir, out] ⊕ manifest_args.
def _build_optuna_command(manifest: WrapperManifest, resolved_tag: str, output_dir: Path) -> list[str]:
    args = list(manifest.optuna_args)
    for reserved in ("--tag", "--output-dir", "--python-bin"):
        args = _remove_option(args, reserved)
    return [
        str(manifest.python_bin),
        "-m",
        _OPTUNA_MODULE,
        "--tag",
        str(resolved_tag),
        "--output-dir",
        str(output_dir),
        *args,
    ]


def _render_command_script(command: Sequence[str], stdout_path: Path, stderr_path: Path) -> str:
    quoted_command = " ".join(shlex.quote(str(token)) for token in command)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"exec {quoted_command} > {shlex.quote(str(stdout_path))} 2> {shlex.quote(str(stderr_path))}\n"
    )


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _tmux_bin() -> str:
    tmux = shutil.which("tmux")
    if tmux is None:
        raise RuntimeError("tmux is required for Path A wrapper launches but was not found on PATH.")
    return str(tmux)


def _tmux_session_alive(session_name: str) -> bool:
    proc = subprocess.run(
        [_tmux_bin(), "has-session", "-t", str(session_name)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if int(proc.returncode) == 0:
        return True
    joined = "\n".join(x for x in (proc.stdout, proc.stderr) if x).lower()
    if int(proc.returncode) == 1 or "can't find session" in joined or "no server running" in joined:
        return False
    raise RuntimeError(
        f"tmux session probe failed for {session_name!r} with code {proc.returncode}: {joined.strip()}"
    )


# Built Math: complete(tag) = 1[summary.json exists ∨ progress.done = true].
def _run_marked_complete(output_dir: Path) -> bool:
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        return True
    progress_path = Path(output_dir) / "progress.json"
    if not progress_path.exists():
        return False
    try:
        payload = _load_json(progress_path)
    except Exception:
        return False
    return bool(payload.get("done"))


def _write_launch_manifest(
    *,
    path: Path,
    requested_tag: str,
    resolved_tag: str,
    session_name: str,
    manifest_path: Path,
    ledger_path: Path,
    output_dir: Path,
    command_path: Path,
    optuna_command: Sequence[str],
) -> None:
    payload = {
        "generated_utc": _utc_now(),
        "pipeline": _WRAPPER_PIPELINE_NAME,
        "requested_tag": str(requested_tag),
        "resolved_tag": str(resolved_tag),
        "session_name": str(session_name),
        "manifest_path": str(manifest_path),
        "ledger_path": str(ledger_path),
        "output_dir": str(output_dir),
        "command_path": str(command_path),
        "optuna_command": [str(token) for token in optuna_command],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _launch_tmux_session(session_name: str, command_path: Path) -> None:
    subprocess.run(
        [_tmux_bin(), "new-session", "-d", "-s", str(session_name), str(command_path)],
        check=True,
        cwd=str(REPO_ROOT),
    )


def ensure_path_a_run(tag: str, manifest_path: Path = DEFAULT_MANIFEST_PATH) -> LaunchResult:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    resolved_tag = _safe_slug(str(tag))
    output_dir = REPO_ROOT / "artifacts" / "agent_runs" / str(resolved_tag)
    logs_dir = output_dir / "logs"
    session_name = _session_name(manifest.tmux_session_prefix, resolved_tag)
    ledger_path = Path(refresh_path_a_ledger(["--output-path", str(DEFAULT_LEDGER_PATH)]))
    optuna_command = tuple(_build_optuna_command(manifest, resolved_tag, output_dir))

    base_result = LaunchResult(
        requested_tag=str(tag),
        resolved_tag=str(resolved_tag),
        session_name=str(session_name),
        status="pending",
        output_dir=str(output_dir),
        logs_dir=str(logs_dir),
        ledger_path=str(ledger_path),
        manifest_path=str(manifest_path),
        optuna_command=optuna_command,
    )
    if _tmux_session_alive(session_name):
        return LaunchResult(**{**asdict(base_result), "status": "already_running"})
    if _run_marked_complete(output_dir):
        return LaunchResult(**{**asdict(base_result), "status": "already_complete"})

    stdout_path = logs_dir / "stdout.log"
    stderr_path = logs_dir / "stderr.log"
    command_path = logs_dir / "command.sh"
    _write_executable(command_path, _render_command_script(optuna_command, stdout_path, stderr_path))
    _write_launch_manifest(
        path=logs_dir / "launch_manifest.json",
        requested_tag=str(tag),
        resolved_tag=str(resolved_tag),
        session_name=str(session_name),
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        output_dir=output_dir,
        command_path=command_path,
        optuna_command=optuna_command,
    )
    _launch_tmux_session(session_name, command_path)
    return LaunchResult(**{**asdict(base_result), "status": "launched", "command_path": str(command_path)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, type=str, help="Run tag used for artifacts/agent_runs/<tag> and the tmux session suffix.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> LaunchResult:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = ensure_path_a_run(tag=str(args.tag), manifest_path=Path(args.manifest_path))
    print(json.dumps(asdict(result), indent=2))
    return result


if __name__ == "__main__":
    main()
