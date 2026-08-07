#!/usr/bin/env python3
"""Monitor live HH search artifacts and refresh Math front-page top-2 standings."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CURRENT_SUMMARIES = (
    REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_current_focus_spsa_optuna_v1/summary.json",
)
DEFAULT_LEGACY_TRIAL_ROOTS = (
    REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_legacy_focus_spsa_optuna_v1/legacy/eps_6.200em05",
)
DEFAULT_LEGACY_SUMMARIES = (
    REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_legacy_focus_spsa_optuna_v1/summary.json",
)
DEFAULT_MATH_PATH = REPO_ROOT / "MATH/Math.md"
DEFAULT_STATE_JSON = REPO_ROOT / "artifacts/agent_runs/20260414_hh_l2_math_top2_monitor_v3/state.json"
DEFAULT_LEGACY_EPSILON = 6.2e-5

SUMMARY_START = "<!-- AUTO_HH_TOP2_SUMMARY_START -->"
SUMMARY_END = "<!-- AUTO_HH_TOP2_SUMMARY_END -->"
ROWS_START = "<!-- AUTO_HH_TOP2_ROWS_START -->"
ROWS_END = "<!-- AUTO_HH_TOP2_ROWS_END -->"
STATUS_START = "<!-- AUTO_HH_TOP2_STATUS_START -->"
STATUS_END = "<!-- AUTO_HH_TOP2_STATUS_END -->"


@dataclass(frozen=True)
class CandidateRecord:
    case_dir: str
    delta_e_abs: float
    compiled_count_2q: int
    compiled_depth: int
    logical_operator_count: int
    runtime_parameter_count: int
    surface_note: str
    source_kind: str


POWELL_LEGACY_TOP2 = (
    CandidateRecord(
        case_dir="artifacts/agent_runs/20260414_hh_l2_legacy_focus_optuna_v1/legacy/eps_6.200em05/trial_0016",
        delta_e_abs=5.617823464446059e-05,
        compiled_count_2q=75,
        compiled_depth=178,
        logical_operator_count=16,
        runtime_parameter_count=29,
        surface_note="legacy Powell Optuna lane (`runtime_split` on, no batching, transpile-single burden, no prune)",
        source_kind="legacy_powell",
    ),
    CandidateRecord(
        case_dir="artifacts/agent_runs/20260414_hh_l2_legacy_focus_optuna_v1/legacy/eps_6.200em05/trial_0001",
        delta_e_abs=5.617823464482141e-05,
        compiled_count_2q=81,
        compiled_depth=151,
        logical_operator_count=16,
        runtime_parameter_count=27,
        surface_note="legacy Powell Optuna lane (`runtime_split` on, batching on, transpile-single burden, prune on)",
        source_kind="legacy_powell",
    ),
)

POWELL_CURRENT_TOP2 = (
    CandidateRecord(
        case_dir="artifacts/agent_runs/20260414_hh_l2_bridge_diag_focus_optuna_v1/global/eps_6.200em05/trial_0024",
        delta_e_abs=5.617823464465488e-05,
        compiled_count_2q=98,
        compiled_depth=267,
        logical_operator_count=13,
        runtime_parameter_count=28,
        surface_note="focused `bridge_diag` Powell Optuna global lane (`proxy_reduced`, repeats off, shortlist split on, no prune)",
        source_kind="current_powell",
    ),
    CandidateRecord(
        case_dir="artifacts/agent_runs/20260414_hh_l2_bridge_diag_focus_optuna_v1/global/eps_6.200em05/trial_0005",
        delta_e_abs=5.617823464487692e-05,
        compiled_count_2q=118,
        compiled_depth=351,
        logical_operator_count=13,
        runtime_parameter_count=30,
        surface_note="focused `bridge_diag` Powell Optuna global lane (`proxy_reduced`, repeats off, shortlist split on, no prune)",
        source_kind="current_powell",
    ),
)


# Built Math: x* = argmin_x (N_2Q(x), |ΔE(x)|, D(x), N_logical(x), N_runtime(x)).
def _candidate_sort_key(record: CandidateRecord) -> tuple[int, float, int, int, int]:
    return (
        int(record.compiled_count_2q),
        float(record.delta_e_abs),
        int(record.compiled_depth),
        int(record.logical_operator_count),
        int(record.runtime_parameter_count),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_delta(value: float) -> str:
    return f"{float(value):.10e}"


def _replace_managed_block(text: str, start_marker: str, end_marker: str, body: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    replacement = f"{start_marker}\n{body.rstrip()}\n{end_marker}"
    return text[:start] + replacement + text[end + len(end_marker):]


# Built Math: keep one representative per transpiled 2Q burden and report the two smallest burdens.
def _unique_top2_by_2q(records: Sequence[CandidateRecord]) -> list[CandidateRecord]:
    ordered = sorted(list(records), key=_candidate_sort_key)
    seen_q2: set[int] = set()
    out: list[CandidateRecord] = []
    for record in ordered:
        q2 = int(record.compiled_count_2q)
        if q2 in seen_q2:
            continue
        seen_q2.add(q2)
        out.append(record)
        if len(out) >= 2:
            break
    return out


def _current_surface_note(params: dict[str, Any]) -> str:
    base_preset = str(params.get("base_preset", "search"))
    selector = str(params.get("selector_geometry_mode", "base"))
    repeats = str(params.get("repeats_mode", "base"))
    split = str(params.get("runtime_split_mode", "base"))
    prune = str(params.get("phase1_prune_mode", "base"))

    repeats_note = "repeats off" if repeats == "disable" else "repeats on/base"
    split_note = "shortlist split on" if split == "shortlist_pauli_children_v1" else f"split={split}"
    prune_note = "no prune" if prune == "off" else ("prune on" if prune == "live" else f"prune={prune}")
    return (
        f"focused `{base_preset}` SPSA Optuna global lane "
        f"(`{selector}`, {repeats_note}, {split_note}, {prune_note})"
    )


def _legacy_surface_note(command_text: str) -> str:
    split_note = "`runtime_split` on" if "--phase3-runtime-split-mode shortlist_pauli_children_v1" in command_text else "`runtime_split` base/off"
    batching_note = "no batching" if "--phase2-no-batching" in command_text else ("batching on" if "--phase2-enable-batching" in command_text else "batching base")
    if "--phase3-backend-cost-mode transpile_single_v1" in command_text:
        burden_note = "transpile-single burden"
    elif "--phase3-backend-cost-mode proxy" in command_text:
        burden_note = "proxy burden"
    else:
        burden_note = "base burden"
    if "--phase1-no-prune" in command_text:
        prune_note = "no prune"
    elif "--phase1-prune-enabled" in command_text:
        prune_note = "prune on"
    else:
        prune_note = "prune base"
    return f"legacy SPSA Optuna lane ({split_note}, {batching_note}, {burden_note}, {prune_note})"


def _extract_current_top2(summary_paths: Sequence[Path]) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for summary_path in summary_paths:
        if not Path(summary_path).exists():
            continue
        payload = _read_json(Path(summary_path))
        for study in payload.get("studies", []):
            for obs in study.get("observations", []):
                if not bool(obs.get("feasible", False)):
                    continue
                if bool(obs.get("warm_start", False)):
                    continue
                q2 = obs.get("compiled_count_2q")
                depth = obs.get("compiled_depth")
                delta = obs.get("abs_delta_e")
                if q2 is None or depth is None or delta is None:
                    continue
                logical = obs.get("logical_operator_count")
                runtime = obs.get("runtime_parameter_count")
                if logical is None or runtime is None:
                    continue
                records.append(
                    CandidateRecord(
                        case_dir=str(obs.get("case_dir")),
                        delta_e_abs=float(delta),
                        compiled_count_2q=int(q2),
                        compiled_depth=int(depth),
                        logical_operator_count=int(logical),
                        runtime_parameter_count=int(runtime),
                        surface_note=_current_surface_note(dict(obs.get("params", {}))),
                        source_kind="current",
                    )
                )
    return _unique_top2_by_2q(records)


def _extract_legacy_top2(trial_roots: Sequence[Path], epsilon_abs_delta_e: float) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for trial_root in trial_roots:
        if not Path(trial_root).exists():
            continue
        for trial_dir in sorted(Path(trial_root).glob("trial_*")):
            result_json = trial_dir / "json" / "result.json"
            compile_json = trial_dir / "json" / "compile_scout_fake_marrakesh.json"
            command_sh = trial_dir / "logs" / "command.sh"
            if not result_json.exists() or not compile_json.exists() or not command_sh.exists():
                continue
            result = _read_json(result_json)
            compile_payload = _read_json(compile_json)
            rows = list(compile_payload.get("rows", []))
            if not rows:
                continue
            best_row = min(rows, key=lambda row: (int(row.get("compiled_count_2q", 10**9)), int(row.get("compiled_depth", 10**9))))
            delta = abs(float(result.get("adapt_vqe", {}).get("abs_delta_e", 1.0)))
            if delta > float(epsilon_abs_delta_e):
                continue
            records.append(
                CandidateRecord(
                    case_dir=str(trial_dir.relative_to(REPO_ROOT)),
                    delta_e_abs=float(delta),
                    compiled_count_2q=int(best_row.get("compiled_count_2q", 10**9)),
                    compiled_depth=int(best_row.get("compiled_depth", 10**9)),
                    logical_operator_count=int(result.get("adapt_vqe", {}).get("ansatz_depth", 0)),
                    runtime_parameter_count=int(result.get("adapt_vqe", {}).get("num_parameters", 0)),
                    surface_note=_legacy_surface_note(command_sh.read_text(encoding="utf-8")),
                    source_kind="legacy",
                )
            )
    return _unique_top2_by_2q(records)


def _describe_pair(label: str, records: Sequence[CandidateRecord]) -> str:
    if len(records) >= 2:
        return (
            f"- {label}: best `{records[0].compiled_count_2q}` 2Q at `|\\Delta E|={_format_delta(records[0].delta_e_abs)}`; "
            f"runner-up `{records[1].compiled_count_2q}` 2Q at `|\\Delta E|={_format_delta(records[1].delta_e_abs)}`."
        )
    if len(records) == 1:
        return f"- {label}: best so far `{records[0].compiled_count_2q}` 2Q at `|\\Delta E|={_format_delta(records[0].delta_e_abs)}`; runner-up pending."
    return f"- {label}: no in-band feasible candidate yet; search still running."


def _render_summary_block(
    powell_legacy_top2: Sequence[CandidateRecord],
    powell_current_top2: Sequence[CandidateRecord],
    spsa_legacy_top2: Sequence[CandidateRecord],
    spsa_current_top2: Sequence[CandidateRecord],
) -> str:
    return "\n".join(
        [
            _describe_pair("Powell legacy incumbents", powell_legacy_top2),
            _describe_pair("Powell current-route incumbents", powell_current_top2),
            _describe_pair("SPSA legacy incumbents", spsa_legacy_top2),
            _describe_pair("SPSA current-route incumbents", spsa_current_top2),
            "- The validated public/deployment anchor remains the April 5 SPSA route at `218` 2Q and `|\\Delta E|=1.0822209459e-04`.",
        ]
    )


def _render_row(role: str, record: CandidateRecord) -> str:
    return (
        f"| {role} | `{record.case_dir}` | {record.surface_note} | `{_format_delta(record.delta_e_abs)}` | "
        f"{int(record.logical_operator_count)} | {int(record.runtime_parameter_count)} | {int(record.compiled_count_2q)} | {int(record.compiled_depth)} |"
    )


def _render_pending_row(role: str, note: str) -> str:
    return f"| {role} | `pending` | {note} | `--` | -- | -- | -- | -- |"


def _render_pair_rows(prefix: str, records: Sequence[CandidateRecord], pending_note: str) -> list[str]:
    rows: list[str] = []
    if len(records) >= 1:
        rows.append(_render_row(f"Best {prefix}", records[0]))
    else:
        rows.append(_render_pending_row(f"Best {prefix}", pending_note))
    if len(records) >= 2:
        rows.append(_render_row(f"Runner-up {prefix}", records[1]))
    else:
        rows.append(_render_pending_row(f"Runner-up {prefix}", pending_note))
    return rows


def _render_rows_block(
    powell_legacy_top2: Sequence[CandidateRecord],
    powell_current_top2: Sequence[CandidateRecord],
    spsa_legacy_top2: Sequence[CandidateRecord],
    spsa_current_top2: Sequence[CandidateRecord],
) -> str:
    return "\n".join(
        [
            *_render_pair_rows("Powell legacy-focused compatibility-search line", powell_legacy_top2, "fixed Powell legacy reference rows"),
            *_render_pair_rows("Powell current-code focused-bridge search line", powell_current_top2, "fixed Powell current-route reference rows"),
            *_render_pair_rows("SPSA legacy-focused compatibility-search line", spsa_legacy_top2, "live SPSA legacy search in progress"),
            *_render_pair_rows("SPSA current-code focused search line", spsa_current_top2, "live SPSA current-route search in progress"),
        ]
    )


def _render_status_block(
    powell_legacy_top2: Sequence[CandidateRecord],
    powell_current_top2: Sequence[CandidateRecord],
    spsa_legacy_top2: Sequence[CandidateRecord],
    spsa_current_top2: Sequence[CandidateRecord],
) -> str:
    def _label(record: CandidateRecord | None) -> str:
        return str(record.compiled_count_2q) if record is not None else "pending"

    powell_legacy_best = powell_legacy_top2[0] if powell_legacy_top2 else None
    powell_current_best = powell_current_top2[0] if powell_current_top2 else None
    spsa_legacy_best = spsa_legacy_top2[0] if spsa_legacy_top2 else None
    spsa_current_best = spsa_current_top2[0] if spsa_current_top2 else None
    return (
        "The repo now tracks Powell and SPSA separately: "
        f"Powell incumbents are `{_label(powell_legacy_best)}` 2Q (legacy) and `{_label(powell_current_best)}` 2Q (current), "
        f"while SPSA incumbents are `{_label(spsa_legacy_best)}` 2Q (legacy) and `{_label(spsa_current_best)}` 2Q (current)."
    )


def _render_payload(spsa_legacy_top2: Sequence[CandidateRecord], spsa_current_top2: Sequence[CandidateRecord]) -> dict[str, Any]:
    powell_legacy_top2 = list(POWELL_LEGACY_TOP2)
    powell_current_top2 = list(POWELL_CURRENT_TOP2)
    return {
        "powell_legacy_top2": [asdict(x) for x in powell_legacy_top2],
        "powell_current_top2": [asdict(x) for x in powell_current_top2],
        "spsa_legacy_top2": [asdict(x) for x in spsa_legacy_top2],
        "spsa_current_top2": [asdict(x) for x in spsa_current_top2],
        "summary_block": _render_summary_block(powell_legacy_top2, powell_current_top2, spsa_legacy_top2, spsa_current_top2),
        "rows_block": _render_rows_block(powell_legacy_top2, powell_current_top2, spsa_legacy_top2, spsa_current_top2),
        "status_block": _render_status_block(powell_legacy_top2, powell_current_top2, spsa_legacy_top2, spsa_current_top2),
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# Built Math: refresh Math only when the leaderboard state changes, i.e. S_t != S_{t-1}.
def _update_math_from_payload(math_path: Path, payload: dict[str, Any]) -> bool:
    original = math_path.read_text(encoding="utf-8")
    updated = _replace_managed_block(original, SUMMARY_START, SUMMARY_END, str(payload["summary_block"]))
    updated = _replace_managed_block(updated, ROWS_START, ROWS_END, str(payload["rows_block"]))
    updated = _replace_managed_block(updated, STATUS_START, STATUS_END, str(payload["status_block"]))
    if updated == original:
        return False
    math_path.write_text(updated, encoding="utf-8")
    return True


def _rebuild_math() -> None:
    subprocess.run([sys.executable, str(REPO_ROOT / "MATH" / "build_math_from_md.py")], check=True, cwd=str(REPO_ROOT))


def _legacy_completed(legacy_summary_path: Path) -> bool:
    return legacy_summary_path.exists()


def _write_state(state_json: Path, state: dict[str, Any]) -> None:
    state_json.parent.mkdir(parents=True, exist_ok=True)
    state_json.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_path_csv(values: str | None) -> list[Path]:
    if values in {None, ""}:
        return []
    return [Path(str(raw).strip()) for raw in str(values).split(",") if str(raw).strip()]


def _all_completed(summary_paths: Sequence[Path]) -> bool:
    return bool(summary_paths) and all(Path(path).exists() for path in summary_paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--math-path", type=Path, default=DEFAULT_MATH_PATH)
    parser.add_argument("--current-summaries", type=str, default=",".join(str(path) for path in DEFAULT_CURRENT_SUMMARIES))
    parser.add_argument("--legacy-trial-roots", type=str, default=",".join(str(path) for path in DEFAULT_LEGACY_TRIAL_ROOTS))
    parser.add_argument("--legacy-summaries", type=str, default=",".join(str(path) for path in DEFAULT_LEGACY_SUMMARIES))
    parser.add_argument("--legacy-epsilon", type=float, default=DEFAULT_LEGACY_EPSILON)
    parser.add_argument("--state-json", type=Path, default=DEFAULT_STATE_JSON)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-polls", type=int, default=0, help="0 means unlimited until completion.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-when-legacy-complete", action="store_true", default=True)
    parser.add_argument("--no-stop-when-legacy-complete", dest="stop_when_legacy_complete", action="store_false")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    current_summaries = _parse_path_csv(args.current_summaries)
    legacy_trial_roots = _parse_path_csv(args.legacy_trial_roots)
    legacy_summaries = _parse_path_csv(args.legacy_summaries)
    poll_count = 0
    last_hash = None
    if args.state_json.exists():
        try:
            last_hash = json.loads(args.state_json.read_text(encoding="utf-8")).get("payload_hash")
        except Exception:
            last_hash = None

    while True:
        poll_count += 1
        spsa_legacy_top2 = _extract_legacy_top2(legacy_trial_roots, float(args.legacy_epsilon))
        spsa_current_top2 = _extract_current_top2(current_summaries)
        payload = _render_payload(spsa_legacy_top2, spsa_current_top2)
        payload_hash = _payload_hash(payload)
        changed = payload_hash != last_hash
        math_changed = False
        rebuild_ran = False
        if changed:
            math_changed = _update_math_from_payload(Path(args.math_path), payload)
            if math_changed:
                _rebuild_math()
                rebuild_ran = True
            last_hash = payload_hash
        legacy_completed = _all_completed(legacy_summaries)
        state = {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "poll_count": int(poll_count),
            "payload_hash": payload_hash,
            "payload_changed": bool(changed),
            "math_changed": bool(math_changed),
            "rebuild_ran": bool(rebuild_ran),
            "legacy_completed": bool(legacy_completed),
            "legacy_summaries": [str(path) for path in legacy_summaries],
            "current_summaries": [str(path) for path in current_summaries],
            "powell_legacy_top2": payload["powell_legacy_top2"],
            "powell_current_top2": payload["powell_current_top2"],
            "spsa_legacy_top2": payload["spsa_legacy_top2"],
            "spsa_current_top2": payload["spsa_current_top2"],
        }
        _write_state(Path(args.state_json), state)
        print(json.dumps(state, indent=2))

        if args.once:
            return
        if args.max_polls > 0 and poll_count >= int(args.max_polls):
            return
        if args.stop_when_legacy_complete and legacy_completed:
            return
        time.sleep(float(args.poll_seconds))


if __name__ == "__main__":
    main()
