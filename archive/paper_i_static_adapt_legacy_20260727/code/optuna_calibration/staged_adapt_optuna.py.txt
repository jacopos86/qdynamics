#!/usr/bin/env python3
"""Family-generic Optuna harness for staged ADAPT policy tuning.

This harness targets the canonical staged ADAPT CLI in
``pipelines.static_adapt.adapt_pipeline`` and optimizes production
``abs_delta_e`` outputs per problem family.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.static_adapt.adapt_pipeline as static_adapt_pipeline
from pipelines.static_adapt.builders.problem_registry import get_problem_family_spec
from pipelines.static_adapt.cli_config import _build_adapt_arg_parser
from pipelines.static_adapt.output_artifacts import AdaptEnergyMetrics, extract_adapt_energy_metrics

_PIPELINE_NAME = "staged_adapt_optuna_v1"
_LARGE_OBJECTIVE = float(10**18)
_CANONICAL_LAUNCHER = ("-u", "-m", "pipelines.static_adapt.adapt_pipeline")
_CONTROL_PREFIXES = ("--adapt-", "--phase1-", "--phase2-", "--phase3-")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


@dataclass(frozen=True)
class BaseRunConfig:
    family: str
    pipeline_args: tuple[str, ...]
    env_overrides: tuple[tuple[str, str], ...] = ()
    source_kind: str = "cli_remainder"
    source_command_sh: str | None = None
    source_artifact_dir: str | None = None
    dropped_base_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchSpaceConfig:
    adapt_max_depths: tuple[int, ...]
    adapt_maxiters: tuple[int, ...]
    adapt_drop_floors: tuple[float, ...]
    adapt_drop_patiences: tuple[int, ...]
    adapt_drop_min_depths: tuple[int, ...]
    phase1_shortlist_sizes: tuple[int, ...]
    phase2_shortlist_fractions: tuple[float, ...]
    phase2_shortlist_sizes: tuple[int, ...]
    phase2_frontier_ratios: tuple[float, ...]
    phase3_frontier_ratios: tuple[float, ...]
    phase3_tie_beam_score_ratios: tuple[float, ...]
    phase3_tie_beam_abs_tols: tuple[float, ...]
    phase3_tie_beam_max_branches: tuple[int, ...]
    adapt_beam_live_branches: tuple[int, ...]
    adapt_beam_children_per_parent: tuple[int, ...]
    adapt_reopt_policies: tuple[str, ...]
    adapt_window_sizes: tuple[int, ...]
    adapt_eps_grads: tuple[float, ...]
    adapt_eps_energies: tuple[float, ...]
    batching_modes: tuple[str, ...]
    repeats_modes: tuple[str, ...]
    selection_cost_modes: tuple[str, ...]
    phase1_prune_policies: tuple[str, ...]
    phase1_prune_modes: tuple[str, ...]
    inactive_dimensions: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrialParams:
    adapt_max_depth: int
    adapt_maxiter: int
    adapt_drop_floor: float
    adapt_drop_patience: int
    adapt_drop_min_depth: int
    phase1_shortlist_size: int
    phase2_shortlist_fraction: float
    phase2_shortlist_size: int
    phase2_frontier_ratio: float
    phase3_frontier_ratio: float
    phase3_tie_beam_score_ratio: float
    phase3_tie_beam_abs_tol: float
    phase3_tie_beam_max_branches: int
    adapt_beam_live_branches: int
    adapt_beam_children_per_parent: int
    adapt_reopt_policy: str
    adapt_window_size: int
    adapt_eps_grad: float
    adapt_eps_energy: float
    batching_mode: str = "base"
    repeats_mode: str = "base"
    selection_cost_mode: str = "base"
    phase1_prune_policy: str = "base"
    phase1_prune_mode: str = "base"


@dataclass(frozen=True)
class TrialObservation:
    trial_number: int | None
    source_kind: str
    family: str
    params: dict[str, Any]
    objective: float
    status: str
    abs_delta_e: float | None
    energy: float | None
    exact_gs_energy: float | None
    stop_reason: str | None
    ansatz_depth: int | None
    compiled_count_2q: int | None
    compiled_depth: int | None
    logical_parameter_count: int | None
    runtime_parameter_count: int | None
    compile_status: str
    invalid_reasons: list[str] = field(default_factory=list)
    compile_invalid_reasons: list[str] = field(default_factory=list)
    case_dir: str | None = None
    result_json: str | None = None
    compile_json: str | None = None
    returncode: int | None = None
    compile_returncode: int | None = None
    pipeline_elapsed_s: float | None = None
    compile_elapsed_s: float | None = None
    total_elapsed_s: float | None = None
    dropped_args: list[str] = field(default_factory=list)
    family_path_signature: list[str] = field(default_factory=list)
    selected_op_signature: list[str] = field(default_factory=list)
    source_artifact_dir: str | None = None
    error: str | None = None


def _import_optuna() -> Any:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Optuna is required. Install with `python -m pip install optuna`.") from exc
    return optuna


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip()).strip("_")
    return cleaned or "unnamed"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def _parse_csv(raw: str | None) -> list[str]:
    if raw in {None, ""}:
        return []
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_int_csv(raw: str | None) -> list[int]:
    return [int(tok) for tok in _parse_csv(raw)]


def _parse_float_csv(raw: str | None) -> list[float]:
    return [float(tok) for tok in _parse_csv(raw)]


def _dedupe_preserve_order(values: Sequence[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    for value in values:
        if value in out:
            continue
        out.append(value)
    return tuple(out)


def _strip_redirections(tokens: Sequence[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(tokens):
        tok = str(tokens[idx])
        if tok in {">", "1>", "2>", ">>", "1>>", "2>>"}:
            break
        out.append(tok)
        idx += 1
    return out


def _shell_expand_text(text: str, variables: Mapping[str, str]) -> str:
    env = {**{str(k): str(v) for k, v in variables.items()}, "PWD": str(REPO_ROOT)}

    def _replace(match: re.Match[str]) -> str:
        braced = match.group(1)
        plain = match.group(2)
        name = str(braced or plain or "")
        return str(env.get(name, match.group(0)))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", _replace, str(text))


def _is_python_token(token: str) -> bool:
    raw = Path(str(token)).name
    return bool(re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)*)?", raw))


def _extract_command_line_from_script(command_sh: Path) -> tuple[dict[str, str], list[str], list[str]]:
    text = command_sh.read_text(encoding="utf-8")
    logical_lines: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = str(raw).rstrip()
        if not line:
            if current:
                logical_lines.append(" ".join(current).strip())
                current = []
            continue
        if line.endswith("\\"):
            current.append(line[:-1].strip())
            continue
        if current:
            current.append(line.strip())
            logical_lines.append(" ".join(current).strip())
            current = []
        else:
            logical_lines.append(line.strip())
    if current:
        logical_lines.append(" ".join(current).strip())

    shell_vars: dict[str, str] = {"PWD": str(REPO_ROOT)}
    candidate_lines: list[str] = []
    for line in logical_lines:
        if not line or line.startswith("#") or line.startswith("set ") or line.startswith("cd "):
            continue
        assign_line = str(line)
        if assign_line.startswith("export "):
            assign_line = assign_line[len("export ") :].strip()
        assign_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", assign_line)
        if assign_match and "python" not in assign_line:
            key = str(assign_match.group(1))
            raw_value = _shell_expand_text(str(assign_match.group(2)), shell_vars)
            try:
                parsed = shlex.split(raw_value)
                value = parsed[0] if parsed else ""
            except Exception:
                value = raw_value.strip('"').strip("'")
            shell_vars[key] = str(value)
            continue
        if "python" not in line or "--" not in line:
            continue
        candidate_lines.append(_shell_expand_text(line, shell_vars))
    if not candidate_lines:
        raise ValueError(f"No pipeline command found in {command_sh}")
    cmd_tokens = _strip_redirections(shlex.split(candidate_lines[-1]))
    env_overrides: dict[str, str] = {}
    idx = 0
    while idx < len(cmd_tokens) and _ENV_ASSIGN_RE.match(str(cmd_tokens[idx])):
        key, value = str(cmd_tokens[idx]).split("=", 1)
        env_overrides[str(key)] = str(value)
        idx += 1
    if idx < len(cmd_tokens) and str(cmd_tokens[idx]) == "/usr/bin/env":
        idx += 1
        while idx < len(cmd_tokens) and _ENV_ASSIGN_RE.match(str(cmd_tokens[idx])):
            key, value = str(cmd_tokens[idx]).split("=", 1)
            env_overrides[str(key)] = str(value)
            idx += 1
    if idx >= len(cmd_tokens) or not _is_python_token(str(cmd_tokens[idx])):
        raise ValueError(f"Could not locate python launcher in {command_sh}")
    idx += 1
    first_flag_idx = next((pos for pos in range(idx, len(cmd_tokens)) if str(cmd_tokens[pos]).startswith("--")), None)
    if first_flag_idx is None:
        raise ValueError(f"Could not locate pipeline args in {command_sh}")
    launcher_tokens = list(cmd_tokens[idx:first_flag_idx])
    pipeline_args = list(cmd_tokens[first_flag_idx:])
    return env_overrides, launcher_tokens, pipeline_args


def _remove_option(args: Sequence[str], flag: str) -> list[str]:
    out: list[str] = []
    idx = 0
    while idx < len(args):
        tok = str(args[idx])
        if tok == flag:
            idx += 1
            if idx < len(args) and not str(args[idx]).startswith("--"):
                idx += 1
            continue
        if tok.startswith(flag + "="):
            idx += 1
            continue
        out.append(tok)
        idx += 1
    return out


def _set_option(args: Sequence[str], flag: str, value: str | int | float | None) -> list[str]:
    updated = _remove_option(args, flag)
    if value is None:
        return updated
    return [*updated, str(flag), str(value)]


def _get_option_value(args: Sequence[str], flag: str) -> str | None:
    idx = 0
    while idx < len(args):
        tok = str(args[idx])
        if tok == flag:
            if idx + 1 < len(args) and not str(args[idx + 1]).startswith("--"):
                return str(args[idx + 1])
            return None
        if tok.startswith(flag + "="):
            return str(tok.split("=", 1)[1])
        idx += 1
    return None


def _set_toggle_pair(args: Sequence[str], positive_flag: str, negative_flag: str, enabled: bool) -> list[str]:
    updated = _remove_option(_remove_option(args, positive_flag), negative_flag)
    return [*updated, str(positive_flag if enabled else negative_flag)]


@lru_cache(maxsize=1)
def _canonical_supported_long_options() -> frozenset[str]:
    parser = _build_adapt_arg_parser(adapt_gradient_parity_rtol=1e-8)
    out: set[str] = set()
    for action in getattr(parser, "_actions", []):
        for option in getattr(action, "option_strings", []):
            if str(option).startswith("--"):
                out.add(str(option))
    return frozenset(out)


def _filter_args_for_entrypoint(args: Sequence[str], supported_options: Sequence[str]) -> tuple[list[str], list[str]]:
    supported = {str(x) for x in supported_options}
    filtered: list[str] = []
    dropped: list[str] = []
    idx = 0
    while idx < len(args):
        tok = str(args[idx])
        if tok.startswith("--"):
            flag = tok.split("=", 1)[0]
            if flag not in supported:
                dropped.append(flag)
                if "=" not in tok and idx + 1 < len(args) and not str(args[idx + 1]).startswith("--"):
                    idx += 2
                else:
                    idx += 1
                continue
        filtered.append(tok)
        idx += 1
    return filtered, dropped


def _compile_backend_slug_candidates(compile_backend: str) -> tuple[str, ...]:
    raw = str(compile_backend).strip()
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", re.sub(r"[^A-Za-z0-9]+", "_", raw)).lower().strip("_")
    compact = _safe_slug(raw).lower()
    out: list[str] = []
    for value in (snake, compact):
        if value and value not in out:
            out.append(value)
    return tuple(out)


def _compile_scout_output_path_for_artifact_dir(artifact_dir: Path, compile_backend: str) -> Path:
    primary = _compile_backend_slug_candidates(compile_backend)[0]
    return artifact_dir / "json" / f"compile_scout_{primary}.json"


def _compile_scout_path_for_artifact_dir(artifact_dir: Path, compile_backend: str) -> Path | None:
    for slug in _compile_backend_slug_candidates(compile_backend):
        candidate = artifact_dir / "json" / f"compile_scout_{slug}.json"
        if candidate.exists():
            return candidate
    return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_compile_metrics(payload: Mapping[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    selected_backend = payload.get("selected_backend", {}) if isinstance(payload, Mapping) else {}
    logical_circuit = payload.get("logical_circuit", {}) if isinstance(payload, Mapping) else {}

    def _maybe_int(raw: Any) -> int | None:
        try:
            value = int(raw)
        except Exception:
            return None
        return value

    return (
        _maybe_int(selected_backend.get("compiled_count_2q")),
        _maybe_int(selected_backend.get("compiled_depth")),
        _maybe_int(logical_circuit.get("logical_parameter_count")),
        _maybe_int(logical_circuit.get("runtime_parameter_count")),
    )


def _extract_history_signature(payload: Mapping[str, Any], limit: int = 6) -> tuple[list[str], list[str]]:
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    history = adapt_vqe.get("history", []) if isinstance(adapt_vqe, Mapping) else []
    families: list[str] = []
    ops: list[str] = []
    if not isinstance(history, Sequence):
        return families, ops
    for row in history[: int(max(0, limit))]:
        if not isinstance(row, Mapping):
            continue
        op = row.get("selected_op")
        family = row.get("candidate_family")
        if family in {None, ""} and op not in {None, ""}:
            family = str(op).split(":", 1)[0]
        if family not in {None, ""}:
            families.append(str(family))
        if op not in {None, ""}:
            ops.append(str(op))
    return families, ops


def _parse_base_pipeline_args_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError("--pipeline-args-json must point to a JSON array of CLI tokens.")
    return [str(x) for x in payload]


def _resolved_source_tokens(argv_remainder: Sequence[str]) -> list[str]:
    out = [str(x) for x in argv_remainder]
    if out[:1] == ["--"]:
        out = out[1:]
    return out


def _validate_base_source_choice(args: argparse.Namespace) -> tuple[str, Path | None, Path | None, list[str]]:
    remainder = _resolved_source_tokens(getattr(args, "pipeline_args", []))
    chosen = []
    if args.base_artifact_dir is not None:
        chosen.append("artifact_dir")
    if args.base_command_sh is not None:
        chosen.append("command_sh")
    if args.pipeline_args_json is not None:
        chosen.append("pipeline_args_json")
    if remainder:
        chosen.append("cli_remainder")
    if len(chosen) != 1:
        raise ValueError("Provide exactly one base source: --base-artifact-dir, --base-command-sh, --pipeline-args-json, or trailing pipeline args after --.")
    return chosen[0], args.base_artifact_dir, args.base_command_sh, remainder


def _normalized_base_args(pipeline_args: Sequence[str]) -> tuple[list[str], list[str], argparse.Namespace]:
    filtered, dropped = _filter_args_for_entrypoint(pipeline_args, _canonical_supported_long_options())
    normalized = _remove_option(_remove_option(_remove_option(filtered, "--output-json"), "--output-pdf"), "--skip-pdf")
    try:
        parsed = static_adapt_pipeline.parse_args(normalized)
    except SystemExit as exc:  # pragma: no cover - parse error bridge
        raise ValueError(f"Base pipeline args are incompatible with canonical static ADAPT CLI: {normalized}") from exc
    bad_control_drops = [flag for flag in dropped if str(flag).startswith(_CONTROL_PREFIXES)]
    if bad_control_drops:
        raise ValueError(f"Base source dropped staged-ADAPT control flags against the canonical CLI: {bad_control_drops}")
    return normalized, dropped, parsed


def _build_base_run_config(args: argparse.Namespace) -> BaseRunConfig:
    source_kind, artifact_dir, command_sh, remainder = _validate_base_source_choice(args)
    env_overrides: dict[str, str] = {}
    pipeline_args: list[str]
    source_artifact_dir: str | None = None
    source_command_sh: str | None = None
    if source_kind == "artifact_dir":
        assert artifact_dir is not None
        command_path = Path(artifact_dir) / "logs" / "command.sh"
        env_overrides, _launcher_tokens, pipeline_args = _extract_command_line_from_script(command_path)
        source_artifact_dir = str(Path(artifact_dir).resolve())
        source_command_sh = str(command_path.resolve())
    elif source_kind == "command_sh":
        assert command_sh is not None
        env_overrides, _launcher_tokens, pipeline_args = _extract_command_line_from_script(Path(command_sh))
        source_artifact_dir = str(Path(command_sh).resolve().parents[1])
        source_command_sh = str(Path(command_sh).resolve())
    elif source_kind == "pipeline_args_json":
        assert args.pipeline_args_json is not None
        pipeline_args = _parse_base_pipeline_args_json(Path(args.pipeline_args_json))
    else:
        pipeline_args = list(remainder)

    normalized_args, dropped_args, parsed = _normalized_base_args(pipeline_args)
    parsed_problem = str(getattr(parsed, "problem", "")).strip().lower()
    family = str(args.family or parsed_problem).strip().lower()
    if family != parsed_problem:
        raise ValueError(f"Explicit --family={family!r} does not match base command problem={parsed_problem!r}.")
    if family == "hh":
        raise ValueError("Use pipelines/exact_bench/hh_cost_energy_optuna.py for HH; this staged family-generic harness rejects hh.")
    get_problem_family_spec(family)
    return BaseRunConfig(
        family=family,
        pipeline_args=tuple(normalized_args),
        env_overrides=tuple(sorted((str(k), str(v)) for k, v in env_overrides.items())),
        source_kind=str(source_kind),
        source_command_sh=source_command_sh,
        source_artifact_dir=source_artifact_dir,
        dropped_base_args=tuple(sorted(set(str(x) for x in dropped_args))),
    )


def _baseline_trial_params(base_cfg: BaseRunConfig) -> TrialParams:
    args = list(base_cfg.pipeline_args)
    return TrialParams(
        adapt_max_depth=int(_get_option_value(args, "--adapt-max-depth") or 32),
        adapt_maxiter=int(_get_option_value(args, "--adapt-maxiter") or 1200),
        adapt_drop_floor=float(_get_option_value(args, "--adapt-drop-floor") or 1e-6),
        adapt_drop_patience=int(_get_option_value(args, "--adapt-drop-patience") or 5),
        adapt_drop_min_depth=int(_get_option_value(args, "--adapt-drop-min-depth") or 12),
        phase1_shortlist_size=int(_get_option_value(args, "--phase1-shortlist-size") or 64),
        phase2_shortlist_fraction=float(_get_option_value(args, "--phase2-shortlist-fraction") or 1.0),
        phase2_shortlist_size=int(_get_option_value(args, "--phase2-shortlist-size") or 64),
        phase2_frontier_ratio=float(_get_option_value(args, "--phase2-frontier-ratio") or 0.9),
        phase3_frontier_ratio=float(_get_option_value(args, "--phase3-frontier-ratio") or 0.9),
        phase3_tie_beam_score_ratio=float(_get_option_value(args, "--phase3-tie-beam-score-ratio") or 1.0),
        phase3_tie_beam_abs_tol=float(_get_option_value(args, "--phase3-tie-beam-abs-tol") or 0.0),
        phase3_tie_beam_max_branches=int(_get_option_value(args, "--phase3-tie-beam-max-branches") or 1),
        adapt_beam_live_branches=int(_get_option_value(args, "--adapt-beam-live-branches") or 1),
        adapt_beam_children_per_parent=int(_get_option_value(args, "--adapt-beam-children-per-parent") or 1),
        adapt_reopt_policy=str(_get_option_value(args, "--adapt-reopt-policy") or "windowed"),
        adapt_window_size=int(_get_option_value(args, "--adapt-window-size") or 3),
        adapt_eps_grad=float(_get_option_value(args, "--adapt-eps-grad") or 1e-9),
        adapt_eps_energy=float(_get_option_value(args, "--adapt-eps-energy") or 1e-12),
        batching_mode=("off" if "--phase2-no-batching" in args else "on" if "--phase2-enable-batching" in args else "on"),
        repeats_mode=("disable" if "--adapt-no-repeats" in args else "allow"),
        selection_cost_mode=str(_get_option_value(args, "--phase3-backend-cost-mode") or "proxy"),
        phase1_prune_policy=str(_get_option_value(args, "--phase1-prune-policy") or "base"),
        phase1_prune_mode=(
            "off"
            if "--phase1-no-prune" in args
            else str(_get_option_value(args, "--phase1-prune-mode") or "live")
            if ("--phase1-prune-enabled" in args or _get_option_value(args, "--phase1-prune-mode") is not None)
            else "base"
        ),
    )


def _space_with_baseline(values: Sequence[Any], baseline: Any) -> tuple[Any, ...]:
    return _dedupe_preserve_order([baseline, *values])


def _search_space(args: argparse.Namespace, base_params: TrialParams) -> SearchSpaceConfig:
    return SearchSpaceConfig(
        adapt_max_depths=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_adapt_max_depths), base_params.adapt_max_depth)),
        adapt_maxiters=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_adapt_maxiters), base_params.adapt_maxiter)),
        adapt_drop_floors=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_adapt_drop_floors), base_params.adapt_drop_floor)),
        adapt_drop_patiences=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_adapt_drop_patiences), base_params.adapt_drop_patience)),
        adapt_drop_min_depths=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_adapt_drop_min_depths), base_params.adapt_drop_min_depth)),
        phase1_shortlist_sizes=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_phase1_shortlist_sizes), base_params.phase1_shortlist_size)),
        phase2_shortlist_fractions=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_phase2_shortlist_fractions), base_params.phase2_shortlist_fraction)),
        phase2_shortlist_sizes=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_phase2_shortlist_sizes), base_params.phase2_shortlist_size)),
        phase2_frontier_ratios=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_phase2_frontier_ratios), base_params.phase2_frontier_ratio)),
        phase3_frontier_ratios=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_phase3_frontier_ratios), base_params.phase3_frontier_ratio)),
        phase3_tie_beam_score_ratios=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_phase3_tie_beam_score_ratios), base_params.phase3_tie_beam_score_ratio)),
        phase3_tie_beam_abs_tols=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_phase3_tie_beam_abs_tols), base_params.phase3_tie_beam_abs_tol)),
        phase3_tie_beam_max_branches=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_phase3_tie_beam_max_branches), base_params.phase3_tie_beam_max_branches)),
        adapt_beam_live_branches=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_adapt_beam_live_branches), base_params.adapt_beam_live_branches)),
        adapt_beam_children_per_parent=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_adapt_beam_children_per_parent), base_params.adapt_beam_children_per_parent)),
        adapt_reopt_policies=tuple(str(x) for x in _space_with_baseline(_parse_csv(args.search_adapt_reopt_policies), base_params.adapt_reopt_policy)),
        adapt_window_sizes=tuple(int(x) for x in _space_with_baseline(_parse_int_csv(args.search_adapt_window_sizes), base_params.adapt_window_size)),
        adapt_eps_grads=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_adapt_eps_grads), base_params.adapt_eps_grad)),
        adapt_eps_energies=tuple(float(x) for x in _space_with_baseline(_parse_float_csv(args.search_adapt_eps_energies), base_params.adapt_eps_energy)),
        batching_modes=tuple(str(x) for x in _space_with_baseline(_parse_csv(args.search_batching_modes), base_params.batching_mode)),
        repeats_modes=tuple(str(x) for x in _space_with_baseline(_parse_csv(args.search_repeats_modes), base_params.repeats_mode)),
        selection_cost_modes=tuple(str(x) for x in _space_with_baseline(_parse_csv(args.search_selection_cost_modes), base_params.selection_cost_mode)),
        phase1_prune_policies=tuple(str(x) for x in _space_with_baseline(_parse_csv(args.search_phase1_prune_policies), base_params.phase1_prune_policy)),
        phase1_prune_modes=tuple(str(x) for x in _space_with_baseline(_parse_csv(args.search_phase1_prune_modes), base_params.phase1_prune_mode)),
        inactive_dimensions=(),
    )


def _params_fit_search_space(params: TrialParams, space: SearchSpaceConfig) -> bool:
    return (
        params.adapt_max_depth in space.adapt_max_depths
        and params.adapt_maxiter in space.adapt_maxiters
        and params.adapt_drop_floor in space.adapt_drop_floors
        and params.adapt_drop_patience in space.adapt_drop_patiences
        and params.adapt_drop_min_depth in space.adapt_drop_min_depths
        and params.phase1_shortlist_size in space.phase1_shortlist_sizes
        and params.phase2_shortlist_fraction in space.phase2_shortlist_fractions
        and params.phase2_shortlist_size in space.phase2_shortlist_sizes
        and params.phase2_frontier_ratio in space.phase2_frontier_ratios
        and params.phase3_frontier_ratio in space.phase3_frontier_ratios
        and params.phase3_tie_beam_score_ratio in space.phase3_tie_beam_score_ratios
        and params.phase3_tie_beam_abs_tol in space.phase3_tie_beam_abs_tols
        and params.phase3_tie_beam_max_branches in space.phase3_tie_beam_max_branches
        and params.adapt_beam_live_branches in space.adapt_beam_live_branches
        and params.adapt_beam_children_per_parent in space.adapt_beam_children_per_parent
        and params.adapt_reopt_policy in space.adapt_reopt_policies
        and params.adapt_window_size in space.adapt_window_sizes
        and params.adapt_eps_grad in space.adapt_eps_grads
        and params.adapt_eps_energy in space.adapt_eps_energies
        and params.batching_mode in space.batching_modes
        and params.repeats_mode in space.repeats_modes
        and params.selection_cost_mode in space.selection_cost_modes
        and params.phase1_prune_policy in space.phase1_prune_policies
        and params.phase1_prune_mode in space.phase1_prune_modes
    )


def _build_distributions(space: SearchSpaceConfig) -> dict[str, Any]:
    optuna = _import_optuna()
    return {
        "adapt_max_depth": optuna.distributions.CategoricalDistribution(list(space.adapt_max_depths)),
        "adapt_maxiter": optuna.distributions.CategoricalDistribution(list(space.adapt_maxiters)),
        "adapt_drop_floor": optuna.distributions.CategoricalDistribution(list(space.adapt_drop_floors)),
        "adapt_drop_patience": optuna.distributions.CategoricalDistribution(list(space.adapt_drop_patiences)),
        "adapt_drop_min_depth": optuna.distributions.CategoricalDistribution(list(space.adapt_drop_min_depths)),
        "phase1_shortlist_size": optuna.distributions.CategoricalDistribution(list(space.phase1_shortlist_sizes)),
        "phase2_shortlist_fraction": optuna.distributions.CategoricalDistribution(list(space.phase2_shortlist_fractions)),
        "phase2_shortlist_size": optuna.distributions.CategoricalDistribution(list(space.phase2_shortlist_sizes)),
        "phase2_frontier_ratio": optuna.distributions.CategoricalDistribution(list(space.phase2_frontier_ratios)),
        "phase3_frontier_ratio": optuna.distributions.CategoricalDistribution(list(space.phase3_frontier_ratios)),
        "phase3_tie_beam_score_ratio": optuna.distributions.CategoricalDistribution(list(space.phase3_tie_beam_score_ratios)),
        "phase3_tie_beam_abs_tol": optuna.distributions.CategoricalDistribution(list(space.phase3_tie_beam_abs_tols)),
        "phase3_tie_beam_max_branches": optuna.distributions.CategoricalDistribution(list(space.phase3_tie_beam_max_branches)),
        "adapt_beam_live_branches": optuna.distributions.CategoricalDistribution(list(space.adapt_beam_live_branches)),
        "adapt_beam_children_per_parent": optuna.distributions.CategoricalDistribution(list(space.adapt_beam_children_per_parent)),
        "adapt_reopt_policy": optuna.distributions.CategoricalDistribution(list(space.adapt_reopt_policies)),
        "adapt_window_size": optuna.distributions.CategoricalDistribution(list(space.adapt_window_sizes)),
        "adapt_eps_grad": optuna.distributions.CategoricalDistribution(list(space.adapt_eps_grads)),
        "adapt_eps_energy": optuna.distributions.CategoricalDistribution(list(space.adapt_eps_energies)),
        "batching_mode": optuna.distributions.CategoricalDistribution(list(space.batching_modes)),
        "repeats_mode": optuna.distributions.CategoricalDistribution(list(space.repeats_modes)),
        "selection_cost_mode": optuna.distributions.CategoricalDistribution(list(space.selection_cost_modes)),
        "phase1_prune_policy": optuna.distributions.CategoricalDistribution(list(space.phase1_prune_policies)),
        "phase1_prune_mode": optuna.distributions.CategoricalDistribution(list(space.phase1_prune_modes)),
    }


def _suggest_trial_params(trial: Any, space: SearchSpaceConfig) -> TrialParams:
    return TrialParams(
        adapt_max_depth=int(trial.suggest_categorical("adapt_max_depth", list(space.adapt_max_depths))),
        adapt_maxiter=int(trial.suggest_categorical("adapt_maxiter", list(space.adapt_maxiters))),
        adapt_drop_floor=float(trial.suggest_categorical("adapt_drop_floor", list(space.adapt_drop_floors))),
        adapt_drop_patience=int(trial.suggest_categorical("adapt_drop_patience", list(space.adapt_drop_patiences))),
        adapt_drop_min_depth=int(trial.suggest_categorical("adapt_drop_min_depth", list(space.adapt_drop_min_depths))),
        phase1_shortlist_size=int(trial.suggest_categorical("phase1_shortlist_size", list(space.phase1_shortlist_sizes))),
        phase2_shortlist_fraction=float(trial.suggest_categorical("phase2_shortlist_fraction", list(space.phase2_shortlist_fractions))),
        phase2_shortlist_size=int(trial.suggest_categorical("phase2_shortlist_size", list(space.phase2_shortlist_sizes))),
        phase2_frontier_ratio=float(trial.suggest_categorical("phase2_frontier_ratio", list(space.phase2_frontier_ratios))),
        phase3_frontier_ratio=float(trial.suggest_categorical("phase3_frontier_ratio", list(space.phase3_frontier_ratios))),
        phase3_tie_beam_score_ratio=float(trial.suggest_categorical("phase3_tie_beam_score_ratio", list(space.phase3_tie_beam_score_ratios))),
        phase3_tie_beam_abs_tol=float(trial.suggest_categorical("phase3_tie_beam_abs_tol", list(space.phase3_tie_beam_abs_tols))),
        phase3_tie_beam_max_branches=int(trial.suggest_categorical("phase3_tie_beam_max_branches", list(space.phase3_tie_beam_max_branches))),
        adapt_beam_live_branches=int(trial.suggest_categorical("adapt_beam_live_branches", list(space.adapt_beam_live_branches))),
        adapt_beam_children_per_parent=int(trial.suggest_categorical("adapt_beam_children_per_parent", list(space.adapt_beam_children_per_parent))),
        adapt_reopt_policy=str(trial.suggest_categorical("adapt_reopt_policy", list(space.adapt_reopt_policies))),
        adapt_window_size=int(trial.suggest_categorical("adapt_window_size", list(space.adapt_window_sizes))),
        adapt_eps_grad=float(trial.suggest_categorical("adapt_eps_grad", list(space.adapt_eps_grads))),
        adapt_eps_energy=float(trial.suggest_categorical("adapt_eps_energy", list(space.adapt_eps_energies))),
        batching_mode=str(trial.suggest_categorical("batching_mode", list(space.batching_modes))),
        repeats_mode=str(trial.suggest_categorical("repeats_mode", list(space.repeats_modes))),
        selection_cost_mode=str(trial.suggest_categorical("selection_cost_mode", list(space.selection_cost_modes))),
        phase1_prune_policy=str(trial.suggest_categorical("phase1_prune_policy", list(space.phase1_prune_policies))),
        phase1_prune_mode=str(trial.suggest_categorical("phase1_prune_mode", list(space.phase1_prune_modes))),
    )


def _apply_trial_overrides(params: TrialParams, pipeline_args: Sequence[str]) -> list[str]:
    args = list(str(x) for x in pipeline_args)
    args = _set_option(args, "--adapt-max-depth", params.adapt_max_depth)
    args = _set_option(args, "--adapt-maxiter", params.adapt_maxiter)
    args = _set_option(args, "--adapt-drop-floor", params.adapt_drop_floor)
    args = _set_option(args, "--adapt-drop-patience", params.adapt_drop_patience)
    args = _set_option(args, "--adapt-drop-min-depth", params.adapt_drop_min_depth)
    args = _set_option(args, "--phase1-shortlist-size", params.phase1_shortlist_size)
    args = _set_option(args, "--phase2-shortlist-fraction", params.phase2_shortlist_fraction)
    args = _set_option(args, "--phase2-shortlist-size", params.phase2_shortlist_size)
    args = _set_option(args, "--phase2-frontier-ratio", params.phase2_frontier_ratio)
    args = _set_option(args, "--phase3-frontier-ratio", params.phase3_frontier_ratio)
    args = _set_option(args, "--phase3-tie-beam-score-ratio", params.phase3_tie_beam_score_ratio)
    args = _set_option(args, "--phase3-tie-beam-abs-tol", params.phase3_tie_beam_abs_tol)
    args = _set_option(args, "--phase3-tie-beam-max-branches", params.phase3_tie_beam_max_branches)
    args = _set_option(args, "--adapt-beam-live-branches", params.adapt_beam_live_branches)
    args = _set_option(args, "--adapt-beam-children-per-parent", params.adapt_beam_children_per_parent)
    args = _set_option(args, "--adapt-reopt-policy", params.adapt_reopt_policy)
    args = _set_option(args, "--adapt-window-size", params.adapt_window_size)
    args = _set_option(args, "--adapt-eps-grad", params.adapt_eps_grad)
    args = _set_option(args, "--adapt-eps-energy", params.adapt_eps_energy)

    if params.batching_mode == "on":
        args = _set_toggle_pair(args, "--phase2-enable-batching", "--phase2-no-batching", True)
    elif params.batching_mode == "off":
        args = _set_toggle_pair(args, "--phase2-enable-batching", "--phase2-no-batching", False)

    if params.repeats_mode == "disable":
        args = [*_remove_option(args, "--adapt-no-repeats"), "--adapt-no-repeats"]
    elif params.repeats_mode == "allow":
        args = _remove_option(args, "--adapt-no-repeats")

    if params.selection_cost_mode == "proxy":
        args = _set_option(args, "--phase3-backend-cost-mode", "proxy")
        args = _remove_option(args, "--phase3-backend-name")
        args = _remove_option(args, "--phase3-backend-shortlist")
        args = _remove_option(args, "--phase3-backend-transpile-seed")
        args = _remove_option(args, "--phase3-backend-optimization-level")
    elif params.selection_cost_mode == "transpile_single_v1":
        args = _set_option(args, "--phase3-backend-cost-mode", "transpile_single_v1")
        args = _set_option(args, "--phase3-backend-name", "FakeMarrakesh")
        args = _set_option(args, "--phase3-backend-transpile-seed", 7)
        args = _set_option(args, "--phase3-backend-optimization-level", 1)
    else:
        args = _set_option(args, "--phase3-backend-cost-mode", params.selection_cost_mode)

    if params.phase1_prune_policy != "base":
        args = _set_option(args, "--phase1-prune-policy", params.phase1_prune_policy)

    if params.phase1_prune_mode == "off":
        args = _set_toggle_pair(args, "--phase1-prune-enabled", "--phase1-no-prune", False)
        args = _remove_option(args, "--phase1-prune-mode")
    elif params.phase1_prune_mode == "live":
        args = _set_toggle_pair(args, "--phase1-prune-enabled", "--phase1-no-prune", True)
        if "--phase1-prune-mode" in _canonical_supported_long_options():
            args = _set_option(args, "--phase1-prune-mode", "live")
    return args


def _run_subprocess_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env_overrides: Mapping[str, str] | None = None,
) -> tuple[int, float]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_fh, stderr_path.open("w", encoding="utf-8") as stderr_fh:
        env = dict(os.environ)
        if env_overrides:
            env.update({str(k): str(v) for k, v in env_overrides.items()})
        proc = subprocess.run(
            [str(x) for x in command],
            cwd=str(cwd),
            stdout=stdout_fh,
            stderr=stderr_fh,
            env=env,
            check=False,
        )
    return int(proc.returncode), float(time.perf_counter() - started)


def _write_command_log(path: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join([str(x) for x in command]) + "\n", encoding="utf-8")


def _trial_case_dir(output_dir: Path, trial_number: int) -> Path:
    return output_dir / "trials" / f"trial_{int(trial_number):04d}"


def _build_trial_command(
    *,
    python_bin: str,
    base_cfg: BaseRunConfig,
    params: TrialParams,
    case_dir: Path,
    exact_gs_override: float | None = None,
    exact_gs_reference_json: Path | None = None,
) -> tuple[list[str], list[str], tuple[tuple[str, str], ...], dict[str, Any]]:
    pipeline_args = _apply_trial_overrides(params, base_cfg.pipeline_args)
    if exact_gs_override is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-exact-gs-override", repr(float(exact_gs_override)))
    if exact_gs_reference_json is not None:
        pipeline_args = _set_option(pipeline_args, "--adapt-exact-gs-reference-json", str(Path(exact_gs_reference_json)))
    pipeline_args = _remove_option(_remove_option(_remove_option(pipeline_args, "--output-json"), "--output-pdf"), "--skip-pdf")
    pipeline_args = [*pipeline_args, "--output-json", str(case_dir / "json" / "result.json"), "--skip-pdf"]
    filtered_args, dropped_args = _filter_args_for_entrypoint(pipeline_args, _canonical_supported_long_options())
    effective_params = asdict(params)
    if exact_gs_override is not None:
        effective_params["adapt_exact_gs_override"] = float(exact_gs_override)
    if exact_gs_reference_json is not None:
        effective_params["adapt_exact_gs_reference_json"] = str(Path(exact_gs_reference_json))
    return (
        [str(python_bin), *list(_CANONICAL_LAUNCHER), *filtered_args],
        dropped_args,
        tuple(base_cfg.env_overrides),
        effective_params,
    )


def _compile_command(*, python_bin: str, artifact_json: Path, compile_json: Path, compile_backend: str, compile_opt_level: int, compile_seed: int) -> list[str]:
    return [
        str(python_bin),
        "-u",
        "-m",
        "pipelines.scaffold.adapt_circuit_cost",
        "--artifact-json",
        str(artifact_json),
        "--backend-name",
        str(compile_backend),
        "--optimization-level",
        str(int(compile_opt_level)),
        "--seed-transpiler",
        str(int(compile_seed)),
        "--output-json",
        str(compile_json),
    ]


def _trial_metrics_from_payload(payload: Mapping[str, Any]) -> tuple[AdaptEnergyMetrics, str | None, int | None]:
    metrics = extract_adapt_energy_metrics(payload)
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    stop_reason = str(adapt_vqe.get("stop_reason")) if isinstance(adapt_vqe, Mapping) and adapt_vqe.get("stop_reason") not in {None, ""} else None
    ansatz_depth: int | None = None
    if isinstance(adapt_vqe, Mapping) and adapt_vqe.get("ansatz_depth") is not None:
        try:
            ansatz_depth = int(adapt_vqe.get("ansatz_depth"))
        except Exception:
            ansatz_depth = None
    return metrics, stop_reason, ansatz_depth


def _complete_status(metrics: AdaptEnergyMetrics, *, result_exists: bool, returncode: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not result_exists:
        reasons.append("missing_result_json")
        return "failed", reasons
    if returncode != 0:
        reasons.append("pipeline_nonzero_returncode")
    if metrics.abs_delta_e is None:
        reasons.append("missing_abs_delta_e")
        return "invalid", reasons
    return "completed", reasons


def _evaluate_trial(
    *,
    python_bin: str,
    base_cfg: BaseRunConfig,
    params: TrialParams,
    output_dir: Path,
    trial_number: int,
    compile_backend: str,
    compile_opt_level: int,
    compile_seed: int,
    exact_gs_override: float | None = None,
    exact_gs_reference_json: Path | None = None,
) -> TrialObservation:
    case_dir = _trial_case_dir(output_dir, trial_number)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)
    (case_dir / "json").mkdir(parents=True, exist_ok=True)
    command, dropped_args, env_overrides, effective_params = _build_trial_command(
        python_bin=python_bin,
        base_cfg=base_cfg,
        params=params,
        case_dir=case_dir,
        exact_gs_override=exact_gs_override,
        exact_gs_reference_json=exact_gs_reference_json,
    )
    _write_command_log(case_dir / "logs" / "command.sh", command)
    if dropped_args:
        (case_dir / "logs" / "dropped_args.json").write_text(json.dumps(dropped_args, indent=2), encoding="utf-8")
    returncode, pipeline_elapsed_s = _run_subprocess_logged(
        command,
        cwd=REPO_ROOT,
        stdout_path=case_dir / "logs" / "stdout.log",
        stderr_path=case_dir / "logs" / "stderr.log",
        env_overrides=dict(env_overrides),
    )

    result_json = case_dir / "json" / "result.json"
    compile_json = _compile_scout_output_path_for_artifact_dir(case_dir, compile_backend)
    compile_status = "skipped"
    compile_returncode: int | None = None
    compile_elapsed_s: float | None = None
    if result_json.exists():
        compile_cmd = _compile_command(
            python_bin=python_bin,
            artifact_json=result_json,
            compile_json=compile_json,
            compile_backend=compile_backend,
            compile_opt_level=compile_opt_level,
            compile_seed=compile_seed,
        )
        _write_command_log(case_dir / "logs" / "compile_command.sh", compile_cmd)
        compile_returncode, compile_elapsed_s = _run_subprocess_logged(
            compile_cmd,
            cwd=REPO_ROOT,
            stdout_path=case_dir / "logs" / "compile_stdout.log",
            stderr_path=case_dir / "logs" / "compile_stderr.log",
        )
        compile_status = "completed" if compile_returncode == 0 and compile_json.exists() else "failed"

    result_payload = _load_json(result_json) if result_json.exists() else {}
    metrics, stop_reason, ansatz_depth = _trial_metrics_from_payload(result_payload)
    family_signature, op_signature = _extract_history_signature(result_payload)
    status, invalid_reasons = _complete_status(metrics, result_exists=result_json.exists(), returncode=returncode)
    if dropped_args:
        invalid_reasons.append("dropped_search_flags")
        status = "invalid"
    compiled_count_2q = compiled_depth = logical_parameter_count = runtime_parameter_count = None
    compile_invalid_reasons: list[str] = []
    if compile_json.exists() and compile_status == "completed":
        compile_payload = _load_json(compile_json)
        compiled_count_2q, compiled_depth, logical_parameter_count, runtime_parameter_count = _extract_compile_metrics(compile_payload)
    else:
        if compile_status == "failed":
            compile_invalid_reasons.append("compile_failed")
    return TrialObservation(
        trial_number=int(trial_number),
        source_kind="trial",
        family=str(base_cfg.family),
        params=dict(effective_params),
        objective=float(metrics.abs_delta_e if metrics.abs_delta_e is not None else _LARGE_OBJECTIVE),
        status=str(status),
        abs_delta_e=metrics.abs_delta_e,
        energy=metrics.energy,
        exact_gs_energy=metrics.exact_gs_energy,
        stop_reason=stop_reason,
        ansatz_depth=ansatz_depth,
        compiled_count_2q=compiled_count_2q,
        compiled_depth=compiled_depth,
        logical_parameter_count=logical_parameter_count,
        runtime_parameter_count=runtime_parameter_count,
        compile_status=compile_status,
        invalid_reasons=list(invalid_reasons),
        compile_invalid_reasons=compile_invalid_reasons,
        case_dir=str(case_dir),
        result_json=str(result_json) if result_json.exists() else None,
        compile_json=str(compile_json) if compile_json.exists() else None,
        returncode=int(returncode),
        compile_returncode=(None if compile_returncode is None else int(compile_returncode)),
        pipeline_elapsed_s=float(pipeline_elapsed_s),
        compile_elapsed_s=(None if compile_elapsed_s is None else float(compile_elapsed_s)),
        total_elapsed_s=float(pipeline_elapsed_s + float(compile_elapsed_s or 0.0)),
        dropped_args=list(dropped_args),
        family_path_signature=family_signature,
        selected_op_signature=op_signature,
        source_artifact_dir=base_cfg.source_artifact_dir,
        error=None,
    )


def _artifact_trial_params_from_command(command_sh: Path) -> tuple[str, TrialParams]:
    _env, _launcher, pipeline_args = _extract_command_line_from_script(command_sh)
    normalized_args, _dropped, parsed = _normalized_base_args(pipeline_args)
    family = str(getattr(parsed, "problem", "")).strip().lower()
    base_cfg = BaseRunConfig(family=family, pipeline_args=tuple(normalized_args))
    return family, _baseline_trial_params(base_cfg)


def _load_observation_from_artifact_dir(*, artifact_dir: Path, family: str, compile_backend: str) -> tuple[TrialObservation | None, TrialParams | None, str | None]:
    command_sh = artifact_dir / "logs" / "command.sh"
    result_json = artifact_dir / "json" / "result.json"
    compile_json = _compile_scout_path_for_artifact_dir(artifact_dir, compile_backend)
    if not command_sh.exists():
        return None, None, "missing_command_sh"
    if not result_json.exists():
        return None, None, "missing_result_json"
    if compile_json is None:
        return None, None, "missing_compile_scout"
    try:
        seed_family, params = _artifact_trial_params_from_command(command_sh)
    except Exception:
        return None, None, "command_parse_failed"
    if str(seed_family) != str(family):
        return None, None, f"family_mismatch:{seed_family}"
    result_payload = _load_json(result_json)
    compile_payload = _load_json(compile_json)
    metrics, stop_reason, ansatz_depth = _trial_metrics_from_payload(result_payload)
    compiled_count_2q, compiled_depth, logical_parameter_count, runtime_parameter_count = _extract_compile_metrics(compile_payload)
    history_signature, op_signature = _extract_history_signature(result_payload)
    status = "completed" if metrics.abs_delta_e is not None else "invalid"
    invalid_reasons = [] if status == "completed" else ["missing_abs_delta_e"]
    observation = TrialObservation(
        trial_number=None,
        source_kind="reference",
        family=str(family),
        params=asdict(params) if params is not None else {},
        objective=float(metrics.abs_delta_e if metrics.abs_delta_e is not None else _LARGE_OBJECTIVE),
        status=status,
        abs_delta_e=metrics.abs_delta_e,
        energy=metrics.energy,
        exact_gs_energy=metrics.exact_gs_energy,
        stop_reason=stop_reason,
        ansatz_depth=ansatz_depth,
        compiled_count_2q=compiled_count_2q,
        compiled_depth=compiled_depth,
        logical_parameter_count=logical_parameter_count,
        runtime_parameter_count=runtime_parameter_count,
        compile_status="completed",
        invalid_reasons=invalid_reasons,
        compile_invalid_reasons=[],
        case_dir=str(artifact_dir),
        result_json=str(result_json),
        compile_json=str(compile_json),
        returncode=0,
        compile_returncode=0,
        pipeline_elapsed_s=None,
        compile_elapsed_s=None,
        total_elapsed_s=None,
        dropped_args=[],
        family_path_signature=history_signature,
        selected_op_signature=op_signature,
        source_artifact_dir=str(artifact_dir),
        error=None,
    )
    return observation, params, None


def _observation_to_user_attrs(observation: TrialObservation) -> dict[str, Any]:
    return {
        "family": str(observation.family),
        "status": str(observation.status),
        "abs_delta_e": observation.abs_delta_e,
        "compiled_count_2q": observation.compiled_count_2q,
        "compiled_depth": observation.compiled_depth,
        "logical_parameter_count": observation.logical_parameter_count,
        "runtime_parameter_count": observation.runtime_parameter_count,
        "stop_reason": observation.stop_reason,
        "ansatz_depth": observation.ansatz_depth,
    }


def _observation_to_optuna_params(params: TrialParams) -> dict[str, Any]:
    return asdict(params)


def _observation_rows(observations: Sequence[TrialObservation], *, completed_only: bool = False, compile_required: bool = False) -> list[TrialObservation]:
    rows = [obs for obs in observations if (not completed_only or obs.status == "completed")]
    if compile_required:
        rows = [obs for obs in rows if obs.compiled_count_2q is not None and obs.compiled_depth is not None]
    return rows


def _best_objective_row(observations: Sequence[TrialObservation], *, compile_required: bool = False, metric: str = "objective") -> TrialObservation | None:
    rows = _observation_rows(observations, completed_only=True, compile_required=compile_required)
    if not rows:
        return None
    if metric == "compiled_count_2q":
        key = lambda obs: (int(obs.compiled_count_2q or 10**9), float(obs.abs_delta_e or _LARGE_OBJECTIVE), int(obs.compiled_depth or 10**9))
    elif metric == "compiled_depth":
        key = lambda obs: (int(obs.compiled_depth or 10**9), float(obs.abs_delta_e or _LARGE_OBJECTIVE), int(obs.compiled_count_2q or 10**9))
    else:
        key = lambda obs: (
            float(obs.abs_delta_e or _LARGE_OBJECTIVE),
            int(obs.compiled_count_2q or 10**9),
            int(obs.compiled_depth or 10**9),
            int(obs.runtime_parameter_count or 10**9),
        )
    return min(rows, key=key)


def _pareto_front(observations: Sequence[TrialObservation], *, x_field: str, y_field: str) -> list[dict[str, Any]]:
    rows = []
    for obs in observations:
        if obs.status != "completed":
            continue
        x = getattr(obs, x_field)
        y = getattr(obs, y_field)
        if x is None or y is None:
            continue
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            continue
        rows.append(obs)
    front: list[TrialObservation] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            x_other = float(getattr(other, x_field))
            y_other = float(getattr(other, y_field))
            x_cand = float(getattr(candidate, x_field))
            y_cand = float(getattr(candidate, y_field))
            if (x_other <= x_cand and y_other <= y_cand) and (x_other < x_cand or y_other < y_cand):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    front.sort(key=lambda obs: (float(getattr(obs, x_field)), float(getattr(obs, y_field))))
    return [asdict(obs) for obs in front]


def _study_progress_snapshot(*, tag: str, family: str, output_dir: Path, observations: Sequence[TrialObservation], warm_start_count: int, done: bool) -> dict[str, Any]:
    completed = [obs for obs in observations if obs.source_kind == "trial" and obs.status == "completed"]
    invalid = [obs for obs in observations if obs.source_kind == "trial" and obs.status == "invalid"]
    failed = [obs for obs in observations if obs.source_kind == "trial" and obs.status == "failed"]
    compile_success = [obs for obs in observations if obs.source_kind == "trial" and obs.compile_status == "completed"]
    best = _best_objective_row(observations)
    return {
        "generated_utc": _now_utc(),
        "pipeline": _PIPELINE_NAME,
        "tag": str(tag),
        "family": str(family),
        "output_dir": str(output_dir),
        "warm_start_count": int(warm_start_count),
        "completed_trial_count": int(len(completed)),
        "invalid_trial_count": int(len(invalid)),
        "failed_trial_count": int(len(failed)),
        "compile_success_count": int(len(compile_success)),
        "best_objective": (None if best is None else best.abs_delta_e),
        "best_trial_number": (None if best is None else best.trial_number),
        "done": bool(done),
    }


def _write_progress(path: Path, **kwargs: Any) -> None:
    _write_json(path, _study_progress_snapshot(**kwargs))


def _seed_artifact_dirs(base_cfg: BaseRunConfig, args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    warm_paths: list[Path] = []
    ref_paths: list[Path] = []
    if base_cfg.source_artifact_dir and not bool(args.no_base_warm_start):
        warm_paths.append(Path(base_cfg.source_artifact_dir))
    warm_paths.extend(Path(x) for x in _parse_csv(args.warm_start_artifacts))
    ref_paths.extend(Path(x) for x in _parse_csv(args.reference_artifacts))
    seen: set[Path] = set()
    warm_deduped: list[Path] = []
    for path in warm_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        warm_deduped.append(resolved)
    ref_deduped: list[Path] = []
    for path in ref_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ref_deduped.append(resolved)
    return warm_deduped, ref_deduped


def _load_seed_observations(*, family: str, compile_backend: str, search_space: SearchSpaceConfig, warm_start_dirs: Sequence[Path], reference_dirs: Sequence[Path]) -> tuple[list[TrialObservation], list[TrialObservation], list[dict[str, Any]], list[TrialParams]]:
    study_obs: list[TrialObservation] = []
    reference_obs: list[TrialObservation] = []
    skipped: list[dict[str, Any]] = []
    warm_params: list[TrialParams] = []
    for path in [*warm_start_dirs, *reference_dirs]:
        observation, params, reason = _load_observation_from_artifact_dir(artifact_dir=Path(path), family=family, compile_backend=compile_backend)
        if observation is None:
            skipped.append({"artifact_dir": str(path), "reason": str(reason)})
            continue
        if Path(path) in warm_start_dirs and params is not None and _params_fit_search_space(params, search_space) and observation.status == "completed":
            study_obs.append(TrialObservation(**{**asdict(observation), "source_kind": "warm_start"}))
            warm_params.append(params)
        else:
            reference_obs.append(TrialObservation(**{**asdict(observation), "source_kind": "reference"}))
    return study_obs, reference_obs, skipped, warm_params


def _run_study(
    *,
    tag: str,
    output_dir: Path,
    python_bin: str,
    base_cfg: BaseRunConfig,
    search_space: SearchSpaceConfig,
    baseline_params: TrialParams,
    n_trials: int,
    n_startup_trials: int,
    sampler_seed: int,
    enqueue_baseline: bool,
    compile_backend: str,
    compile_opt_level: int,
    compile_seed: int,
    warm_start_observations: Sequence[TrialObservation],
    warm_start_params: Sequence[TrialParams],
    reference_observations: Sequence[TrialObservation],
    skipped_artifacts: Sequence[Mapping[str, Any]],
    exact_gs_override: float | None = None,
    exact_gs_reference_json: Path | None = None,
) -> dict[str, Any]:
    optuna = _import_optuna()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    observations: list[TrialObservation] = list(warm_start_observations)
    _write_progress(progress_path, tag=tag, family=base_cfg.family, output_dir=output_dir, observations=observations, warm_start_count=len(warm_start_observations), done=False)

    sampler = optuna.samplers.TPESampler(
        seed=int(sampler_seed),
        n_startup_trials=int(max(0, n_startup_trials)),
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler)
    distributions = _build_distributions(search_space)
    for obs, params in zip(warm_start_observations, warm_start_params):
        study.add_trial(
            optuna.trial.create_trial(
                params=_observation_to_optuna_params(params),
                distributions=distributions,
                value=float(obs.objective),
                user_attrs=_observation_to_user_attrs(obs),
            )
        )
    if enqueue_baseline:
        study.enqueue_trial(_observation_to_optuna_params(baseline_params))

    for _ in range(int(max(0, n_trials))):
        trial = study.ask(distributions)
        params = _suggest_trial_params(trial, search_space)
        observation = _evaluate_trial(
            python_bin=python_bin,
            base_cfg=base_cfg,
            params=params,
            output_dir=output_dir,
            trial_number=int(trial.number),
            compile_backend=compile_backend,
            compile_opt_level=compile_opt_level,
            compile_seed=compile_seed,
            exact_gs_override=exact_gs_override,
            exact_gs_reference_json=exact_gs_reference_json,
        )
        observations.append(observation)
        if observation.status == "completed":
            study.tell(trial, float(observation.objective), state=optuna.trial.TrialState.COMPLETE)
        else:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        _write_progress(progress_path, tag=tag, family=base_cfg.family, output_dir=output_dir, observations=observations, warm_start_count=len(warm_start_observations), done=False)

    _write_progress(progress_path, tag=tag, family=base_cfg.family, output_dir=output_dir, observations=observations, warm_start_count=len(warm_start_observations), done=True)
    summary = {
        "generated_utc": _now_utc(),
        "pipeline": _PIPELINE_NAME,
        "tag": str(tag),
        "family": str(base_cfg.family),
        "output_dir": str(output_dir),
        "base_run_config": asdict(base_cfg),
        "search_space": asdict(search_space),
        "compile_backend": str(compile_backend),
        "compile_opt_level": int(compile_opt_level),
        "compile_seed": int(compile_seed),
        "adapt_exact_gs_override": (None if exact_gs_override is None else float(exact_gs_override)),
        "adapt_exact_gs_reference_json": (None if exact_gs_reference_json is None else str(Path(exact_gs_reference_json))),
        "n_trials_requested": int(n_trials),
        "n_startup_trials": int(n_startup_trials),
        "enqueue_baseline": bool(enqueue_baseline),
        "warm_start_count": int(len(warm_start_observations)),
        "feasible_trial_count": int(len([obs for obs in observations if obs.status == "completed"])),
        "compile_success_count": int(len([obs for obs in observations if obs.compile_status == "completed"])),
        "observations": [asdict(obs) for obs in observations],
        "reference_observations": [asdict(obs) for obs in reference_observations],
        "skipped_artifacts": [_jsonable(dict(item)) for item in skipped_artifacts],
        "best_objective_trial": (_jsonable(asdict(best)) if (best := _best_objective_row(observations)) is not None else None),
        "best_compile_aware_trial": (_jsonable(asdict(best)) if (best := _best_objective_row(observations, compile_required=True)) is not None else None),
        "best_compiled_count_trial": (_jsonable(asdict(best)) if (best := _best_objective_row(observations, compile_required=True, metric="compiled_count_2q")) is not None else None),
        "best_compiled_depth_trial": (_jsonable(asdict(best)) if (best := _best_objective_row(observations, compile_required=True, metric="compiled_depth")) is not None else None),
        "study_energy_compile_frontier": _pareto_front(observations, x_field="abs_delta_e", y_field="compiled_count_2q"),
        "combined_energy_compile_frontier": _pareto_front([*observations, *reference_observations], x_field="abs_delta_e", y_field="compiled_count_2q"),
        "study_energy_depth_frontier": _pareto_front(observations, x_field="abs_delta_e", y_field="compiled_depth"),
        "combined_energy_depth_frontier": _pareto_front([*observations, *reference_observations], x_field="abs_delta_e", y_field="compiled_depth"),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Family-generic Optuna harness for canonical staged ADAPT policy tuning.")
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--n-trials", type=int, default=24)
    p.add_argument("--n-startup-trials", type=int, default=6)
    p.add_argument("--sampler-seed", type=int, default=7)
    p.add_argument("--no-baseline-trial", action="store_true")
    p.add_argument("--compile-backend", type=str, default="FakeMarrakesh")
    p.add_argument("--compile-opt-level", type=int, default=1)
    p.add_argument("--compile-seed", type=int, default=7)
    p.add_argument("--adapt-exact-gs-override", type=float, default=None, help="Precomputed working-cutoff exact ground-state energy passed to every trial subprocess.")
    p.add_argument("--adapt-exact-gs-reference-json", type=Path, default=None, help="Precomputed exact-reference manifest passed to every trial subprocess for strict per-run lookup.")
    p.add_argument("--family", type=str, default=None)
    p.add_argument("--base-artifact-dir", type=Path, default=None)
    p.add_argument("--base-command-sh", type=Path, default=None)
    p.add_argument("--pipeline-args-json", type=Path, default=None)
    p.add_argument("--warm-start-artifacts", type=str, default="")
    p.add_argument("--reference-artifacts", type=str, default="")
    p.add_argument("--no-base-warm-start", action="store_true")
    p.add_argument("--search-adapt-max-depths", type=str, default="16,24,32,48,64")
    p.add_argument("--search-adapt-maxiters", type=str, default="800,1200,2400")
    p.add_argument("--search-adapt-drop-floors", type=str, default="1e-6,1e-5,1e-4")
    p.add_argument("--search-adapt-drop-patiences", type=str, default="3,5,8")
    p.add_argument("--search-adapt-drop-min-depths", type=str, default="8,12,16")
    p.add_argument("--search-phase1-shortlist-sizes", type=str, default="32,64,128")
    p.add_argument("--search-phase2-shortlist-fractions", type=str, default="0.5,0.75,1.0")
    p.add_argument("--search-phase2-shortlist-sizes", type=str, default="16,32,64,128")
    p.add_argument("--search-phase2-frontier-ratios", type=str, default="0.8,0.9,1.0")
    p.add_argument("--search-phase3-frontier-ratios", type=str, default="0.8,0.9,1.0")
    p.add_argument("--search-phase3-tie-beam-score-ratios", type=str, default="1.0,1.01,1.05")
    p.add_argument("--search-phase3-tie-beam-abs-tols", type=str, default="0.0,1e-6,1e-4")
    p.add_argument("--search-phase3-tie-beam-max-branches", type=str, default="1,2,3")
    p.add_argument("--search-adapt-beam-live-branches", type=str, default="1,2,3")
    p.add_argument("--search-adapt-beam-children-per-parent", type=str, default="1,2,3")
    p.add_argument("--search-adapt-reopt-policies", type=str, default="append_only,full,windowed")
    p.add_argument("--search-adapt-window-sizes", type=str, default="8,16,32")
    p.add_argument("--search-adapt-eps-grads", type=str, default="1e-9,5e-9,1e-8")
    p.add_argument("--search-adapt-eps-energies", type=str, default="1e-12,1e-10,1e-9")
    p.add_argument("--search-batching-modes", type=str, default="on,off")
    p.add_argument("--search-repeats-modes", type=str, default="allow,disable")
    p.add_argument("--search-selection-cost-modes", type=str, default="proxy,transpile_single_v1")
    p.add_argument("--search-phase1-prune-policies", type=str, default="base,legacy_small_angle_v1,recoverability_ladder_v1")
    p.add_argument("--search-phase1-prune-modes", type=str, default="base,off,live")
    p.add_argument("pipeline_args", nargs=argparse.REMAINDER)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_cfg = _build_base_run_config(args)
    baseline_params = _baseline_trial_params(base_cfg)
    search_space = _search_space(args, baseline_params)
    warm_dirs, ref_dirs = _seed_artifact_dirs(base_cfg, args)
    warm_obs, reference_obs, skipped_artifacts, warm_params = _load_seed_observations(
        family=base_cfg.family,
        compile_backend=str(args.compile_backend),
        search_space=search_space,
        warm_start_dirs=warm_dirs,
        reference_dirs=ref_dirs,
    )
    tag = str(args.tag or f"{_safe_slug(base_cfg.family)}_staged_adapt_optuna_{_timestamp_slug()}")
    output_dir = Path(args.output_dir) if args.output_dir is not None else (REPO_ROOT / "artifacts" / "agent_runs" / tag)
    _run_study(
        tag=tag,
        output_dir=output_dir,
        python_bin=str(args.python_bin),
        base_cfg=base_cfg,
        search_space=search_space,
        baseline_params=baseline_params,
        n_trials=int(args.n_trials),
        n_startup_trials=int(args.n_startup_trials),
        sampler_seed=int(args.sampler_seed),
        enqueue_baseline=not bool(args.no_baseline_trial),
        compile_backend=str(args.compile_backend),
        compile_opt_level=int(args.compile_opt_level),
        compile_seed=int(args.compile_seed),
        warm_start_observations=warm_obs,
        warm_start_params=warm_params,
        reference_observations=reference_obs,
        skipped_artifacts=skipped_artifacts,
        exact_gs_override=args.adapt_exact_gs_override,
        exact_gs_reference_json=args.adapt_exact_gs_reference_json,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
