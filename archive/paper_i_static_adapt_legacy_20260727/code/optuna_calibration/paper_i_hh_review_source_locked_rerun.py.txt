#!/usr/bin/env python3
"""Source-locked reruns for reviewed Paper-I HH Optuna candidates.

This is not an Optuna sampler.  It resolves the candidate rows from the review
JSON back to their Optuna SQLite rows, reads the recorded effective ``params``
user attribute, and reuses the same low-level HH trial runner that originally
created the reviewed artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sqlite3
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.exact_bench import hh_cost_energy_optuna as optuna_hh
from pipelines.exact_bench import paper_i_hh_live_optuna_overlay_refresh as live_overlay


DEFAULT_REVIEW_JSON = ROOT / "output/pdf/paper_i_hh_snake_optuna_overlay_review_20260616.json"
DEFAULT_OUTPUT_BASE = ROOT / "raw_outputs/local_hh_source_locked_review_reruns"
DEFAULT_MANIFEST_DIR = ROOT / "output/pdf"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repo_rel(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(value)


def maybe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def maybe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def maybe_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def maybe_path(value: Any) -> Path | None:
    if value in {None, ""}:
        return None
    return Path(str(value))


def force_bool(value: Any, *, default: bool = False) -> bool:
    parsed = maybe_bool(value)
    return bool(default) if parsed is None else bool(parsed)


def lower_bool_string(value: Any) -> str | None:
    parsed = maybe_bool(value)
    if parsed is not None:
        return "true" if parsed else "false"
    if value in {None, ""}:
        return None
    return str(value).strip().lower()


def canonical_hash(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def trial_attrs(con: sqlite3.Connection, trial_id: int) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value_json in con.execute("select key,value_json from trial_user_attributes where trial_id=?", (int(trial_id),)):
        try:
            attrs[str(key)] = json.loads(value_json)
        except Exception:
            attrs[str(key)] = value_json
    return attrs


def trial_row(con: sqlite3.Connection, trial_id: int) -> dict[str, Any] | None:
    row = con.execute(
        """
        select t.trial_id,t.number,t.state,t.study_id,s.study_name,t.datetime_start,t.datetime_complete
        from trials t
        left join studies s on s.study_id=t.study_id
        where t.trial_id=?
        """,
        (int(trial_id),),
    ).fetchone()
    if row is None:
        return None
    return {
        "trial_id": int(row[0]),
        "trial_number": int(row[1]),
        "state": row[2],
        "study_id": int(row[3]) if row[3] is not None else None,
        "study_name": row[4],
        "datetime_start": row[5],
        "datetime_complete": row[6],
    }


def value_for_trial(con: sqlite3.Connection, trial_id: int) -> float | None:
    try:
        row = con.execute(
            "select value from trial_values where trial_id=? order by objective limit 1",
            (int(trial_id),),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else maybe_float(row[0])


def db_candidates_for_regime(regime: str) -> list[Path]:
    out: list[Path] = []
    cfg = live_overlay.ACTIVE_LANES.get(str(regime))
    if isinstance(cfg, Mapping):
        db = live_overlay.optuna_db(str(cfg["slug"]), str(cfg["db_name"]))
        out.append(db)
    root = ROOT / "raw_outputs/optuna_studies/local_hh_optuna_supervisor"
    if root.exists():
        for db in sorted(root.glob(f"*/{regime}.sqlite3")):
            if db not in out:
                out.append(db)
    return out


def find_db_row(regime: str, candidate: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any], float | None]:
    trial_id = maybe_int(candidate.get("trial_id"))
    trial_number = maybe_int(candidate.get("trial_number", candidate.get("trial")))
    expected_study_id = maybe_int(candidate.get("study_id"))
    expected_study_name = candidate.get("study_name")
    source_result_json = candidate.get("result_json")
    source_case_dir = candidate.get("case_dir")
    failures: list[str] = []
    for db in db_candidates_for_regime(regime):
        if not db.exists():
            failures.append(f"missing_db:{repo_rel(db)}")
            continue
        con = sqlite3.connect(db)
        try:
            matched_rows: list[dict[str, Any]] = []
            if trial_id is not None:
                row = trial_row(con, trial_id)
                if row is not None:
                    matched_rows.append(row)
            if not matched_rows and trial_number is not None:
                for raw in con.execute(
                    """
                    select t.trial_id,t.number,t.state,t.study_id,s.study_name,t.datetime_start,t.datetime_complete
                    from trials t
                    left join studies s on s.study_id=t.study_id
                    where t.number=?
                    order by t.study_id,t.trial_id
                    """,
                    (int(trial_number),),
                ):
                    matched_rows.append(
                        {
                            "trial_id": int(raw[0]),
                            "trial_number": int(raw[1]),
                            "state": raw[2],
                            "study_id": int(raw[3]) if raw[3] is not None else None,
                            "study_name": raw[4],
                            "datetime_start": raw[5],
                            "datetime_complete": raw[6],
                        }
                    )
            ranked: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
            for row in matched_rows:
                attrs = trial_attrs(con, int(row["trial_id"]))
                score = 0
                if expected_study_id is not None and row.get("study_id") == expected_study_id:
                    score += 8
                if expected_study_name not in {None, ""} and str(row.get("study_name")) == str(expected_study_name):
                    score += 8
                if source_result_json not in {None, ""} and str(attrs.get("result_json")) == str(source_result_json):
                    score += 4
                if source_case_dir not in {None, ""} and str(attrs.get("case_dir")) == str(source_case_dir):
                    score += 2
                if str(row.get("state")) == "COMPLETE":
                    score += 1
                ranked.append((score, row, attrs))
            if ranked:
                ranked.sort(key=lambda item: item[0], reverse=True)
                score, row, attrs = ranked[0]
                if not isinstance(attrs.get("params"), Mapping):
                    raise RuntimeError(
                        f"Resolved {regime} trial_id={row['trial_id']} in {repo_rel(db)} but it has no params user attr."
                    )
                return db, row, attrs, value_for_trial(con, int(row["trial_id"]))
        finally:
            con.close()
    raise RuntimeError(
        f"Could not resolve reviewed candidate for regime={regime!r}, "
        f"trial_id={trial_id!r}, trial_number={trial_number!r}. Tried: {failures}"
    )


def candidate_key(row: Mapping[str, Any]) -> str:
    if row.get("trial_id") not in {None, ""}:
        return f"trial_id:{row['trial_id']}"
    if row.get("result_json") not in {None, ""}:
        return f"result_json:{row['result_json']}"
    return f"trial_number:{row.get('trial_number', row.get('trial'))}"


def selected_candidates(
    payload: Mapping[str, Any],
    *,
    regimes: Sequence[str] | None,
    labels: Sequence[str] | None,
    limit_per_regime: int,
    pool_policy_filter: str | None = None,
) -> list[dict[str, Any]]:
    lane_map = payload.get("lanes") if isinstance(payload.get("lanes"), Mapping) else None
    regime_blocks = payload.get("regimes") if isinstance(payload.get("regimes"), list) else None
    if lane_map is None and regime_blocks is None:
        raise RuntimeError("Input JSON does not contain review 'lanes' or report 'regimes' candidate rows.")
    report_regime_map: dict[str, list[Mapping[str, Any]]] = {}
    if regime_blocks is not None:
        for block in regime_blocks:
            if not isinstance(block, Mapping):
                continue
            regime = block.get("regime")
            if regime in {None, ""}:
                continue
            rows = block.get("candidates") or block.get("candidate_rows") or []
            if isinstance(rows, list):
                report_regime_map[str(regime)] = [row for row in rows if isinstance(row, Mapping)]
    source_regimes = set(str(x) for x in (lane_map or {}).keys()) | set(report_regime_map)
    wanted_regimes = [str(x) for x in regimes] if regimes else sorted(source_regimes)
    wanted_labels = {str(x) for x in labels or ()}
    selected: list[dict[str, Any]] = []
    for regime in wanted_regimes:
        if report_regime_map:
            rows = report_regime_map.get(regime) or []
        else:
            lane = (lane_map or {}).get(regime) or {}
            rows = lane.get("candidate_rows") or []
        seen: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            if raw.get("missing"):
                continue
            if pool_policy_filter not in {None, ""} and str(raw.get("pool_policy_observed")) != str(pool_policy_filter):
                continue
            label = str(raw.get("label") or "")
            if wanted_labels and label not in wanted_labels:
                continue
            key = candidate_key(raw)
            if key not in seen:
                item = dict(raw)
                item["regime"] = regime
                item["source_labels"] = [label] if label else []
                seen[key] = item
            elif label:
                labels_out = set(seen[key].get("source_labels") or [])
                labels_out.add(label)
                seen[key]["source_labels"] = sorted(labels_out)
        selected.extend(list(seen.values())[: int(max(0, limit_per_regime))])
    return selected


def trial_params_from_effective(params_payload: Mapping[str, Any]) -> optuna_hh.TrialParams:
    fields = set(optuna_hh.TrialParams.__dataclass_fields__)
    kwargs = {
        str(key): value
        for key, value in params_payload.items()
        if str(key) in fields and value is not None
    }
    if "base_preset" not in kwargs or "adapt_max_depth" not in kwargs:
        raise RuntimeError("Recorded params are missing base_preset/adapt_max_depth.")
    return optuna_hh.TrialParams(**kwargs)


def hamiltonian_overrides_from_effective(params_payload: Mapping[str, Any]) -> optuna_hh.HhHamiltonianOverrides | None:
    raw = params_payload.get("hamiltonian_overrides")
    if not isinstance(raw, Mapping) or raw.get("active") is False:
        return None
    overrides = optuna_hh.HhHamiltonianOverrides(
        L=maybe_int(raw.get("L")),
        t=maybe_float(raw.get("t")),
        u=maybe_float(raw.get("u")),
        omega0=maybe_float(raw.get("omega0")),
        lambda_value=maybe_float(raw.get("lambda_value", raw.get("lambda"))),
        g_ep=maybe_float(raw.get("g_ep")),
        n_ph_work=maybe_int(raw.get("n_ph_work")),
        n_ph_ref=maybe_int(raw.get("n_ph_ref")),
        adapt_pool=(None if raw.get("adapt_pool") in {None, ""} else str(raw.get("adapt_pool"))),
    )
    return overrides if overrides.active else None


def eval_kwargs_from_effective(params_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "phase2_w_shot_override": maybe_float(params_payload.get("phase2_w_shot_override")),
        "runtime_split_mode_override": (
            None
            if params_payload.get("runtime_split_mode_override") in {None, ""}
            else str(params_payload.get("runtime_split_mode_override"))
        ),
        "exact_gs_override": maybe_float(params_payload.get("adapt_exact_gs_override")),
        "exact_gs_reference_json": maybe_path(params_payload.get("adapt_exact_gs_reference_json")),
        "force_adapt_max_depth": maybe_int(params_payload.get("force_adapt_max_depth")),
        "force_adapt_maxiter": maybe_int(params_payload.get("force_adapt_maxiter")),
        "force_adapt_final_refit_maxiter": maybe_int(params_payload.get("force_adapt_final_refit_maxiter")),
        "force_adapt_full_refit_every": maybe_int(params_payload.get("force_adapt_full_refit_every")),
        "force_adapt_final_full_refit": lower_bool_string(params_payload.get("force_adapt_final_full_refit")),
        "force_adapt_allow_repeats": maybe_bool(params_payload.get("force_adapt_allow_repeats")),
        "force_phase0_pilot_max_records": maybe_int(params_payload.get("force_phase0_pilot_max_records")),
        "force_phase1_shortlist_size": maybe_int(params_payload.get("force_phase1_shortlist_size")),
        "force_phase2_shortlist_fraction": maybe_float(params_payload.get("force_phase2_shortlist_fraction")),
        "force_phase2_shortlist_size": maybe_int(params_payload.get("force_phase2_shortlist_size")),
        "force_adapt_parallel_gradient_workers": maybe_int(params_payload.get("force_adapt_parallel_gradient_workers")),
        "force_adapt_beam_parent_workers": maybe_int(params_payload.get("force_adapt_beam_parent_workers")),
        "force_adapt_spsa_parallel_evaluations": maybe_int(params_payload.get("force_adapt_spsa_parallel_evaluations")),
        "force_adapt_pool_class_filter_json": maybe_path(params_payload.get("force_adapt_pool_class_filter_json")),
        "force_adapt_resume_scaffold_json": maybe_path(params_payload.get("force_adapt_resume_scaffold_json")),
        "force_adapt_resume_mode": (
            None if params_payload.get("force_adapt_resume_mode") in {None, ""} else str(params_payload.get("force_adapt_resume_mode"))
        ),
        "force_adapt_segment_id": (
            None if params_payload.get("force_adapt_segment_id") in {None, ""} else str(params_payload.get("force_adapt_segment_id"))
        ),
        "force_adapt_segment_target_depth": maybe_int(params_payload.get("force_adapt_segment_target_depth")),
        "force_adapt_segment_max_new_admissions": maybe_int(params_payload.get("force_adapt_segment_max_new_admissions")),
        "force_adapt_segment_wallclock_cap_s": maybe_float(params_payload.get("force_adapt_segment_wallclock_cap_s")),
        "force_adapt_resume_compile_smoke": (
            None
            if params_payload.get("force_adapt_resume_compile_smoke") in {None, ""}
            else str(params_payload.get("force_adapt_resume_compile_smoke"))
        ),
        "force_adapt_resume_smoke_backend": (
            None
            if params_payload.get("force_adapt_resume_smoke_backend") in {None, ""}
            else str(params_payload.get("force_adapt_resume_smoke_backend"))
        ),
        "force_static_route_id": (
            None if params_payload.get("force_static_route_id") in {None, ""} else str(params_payload.get("force_static_route_id"))
        ),
        "force_static_meta_feature_profile": (
            None
            if params_payload.get("force_static_meta_feature_profile") in {None, ""}
            else str(params_payload.get("force_static_meta_feature_profile"))
        ),
        "force_phase3_symmetry_mitigation_mode": (
            None
            if params_payload.get("force_phase3_symmetry_mitigation_mode") in {None, ""}
            else str(params_payload.get("force_phase3_symmetry_mitigation_mode"))
        ),
        "force_route_a_paper_i_production": force_bool(params_payload.get("force_route_a_paper_i_production")),
    }


def dominance_kwargs_from_attrs(attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dominance_target_abs_delta_e": maybe_float(attrs.get("dominance_target_abs_delta_e")),
        "dominance_target_iteration": maybe_int(attrs.get("dominance_target_iteration")),
        "dominance_target_graph_count_2q": maybe_float(attrs.get("dominance_target_graph_count_2q")),
        "dominance_target_graph_depth": maybe_float(attrs.get("dominance_target_graph_depth")),
        "dominance_target_s_alg": maybe_float(attrs.get("dominance_target_s_alg")),
    }


def parse_resolved_trial_specs(values: Sequence[str] | None) -> set[tuple[str | None, int]]:
    specs: set[tuple[str | None, int]] = set()
    for raw in values or ():
        text = str(raw).strip()
        if not text:
            continue
        if ":" in text:
            regime, trial_id = text.split(":", 1)
            specs.add((regime.strip() or None, int(trial_id)))
        else:
            specs.add((None, int(text)))
    return specs


def resolved_trial_matches(specs: set[tuple[str | None, int]], *, regime: str, trial_id: int) -> bool:
    if not specs:
        return False
    return (None, int(trial_id)) in specs or (str(regime), int(trial_id)) in specs


def apply_promotion_overrides(eval_kwargs: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(eval_kwargs)
    if args.override_adapt_maxiter is not None:
        out["force_adapt_maxiter"] = int(args.override_adapt_maxiter)
    if args.override_final_refit_maxiter is not None:
        out["force_adapt_final_refit_maxiter"] = int(args.override_final_refit_maxiter)
    if bool(args.promotion_enable_hva):
        out["force_adapt_pool_class_filter_json"] = None
    if bool(args.promotion_aggressive_screening):
        if not bool(args.promotion_enable_hva):
            raise RuntimeError("--promotion-aggressive-screening requires --promotion-enable-hva.")
        out["force_phase0_pilot_max_records"] = 96
        out["force_phase1_shortlist_size"] = 24
        out["force_phase2_shortlist_fraction"] = 0.25
        out["force_phase2_shortlist_size"] = 12
    if args.override_phase0_pilot_max_records is not None:
        out["force_phase0_pilot_max_records"] = int(args.override_phase0_pilot_max_records)
    if args.override_phase1_shortlist_size is not None:
        out["force_phase1_shortlist_size"] = int(args.override_phase1_shortlist_size)
    if args.override_phase2_shortlist_fraction is not None:
        out["force_phase2_shortlist_fraction"] = float(args.override_phase2_shortlist_fraction)
    if args.override_phase2_shortlist_size is not None:
        out["force_phase2_shortlist_size"] = int(args.override_phase2_shortlist_size)
    if bool(args.write_trajectory):
        out["force_skip_trajectory"] = False
    return out


def semantic_overrides_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "override_adapt_maxiter": args.override_adapt_maxiter,
        "override_final_refit_maxiter": args.override_final_refit_maxiter,
        "promotion_enable_hva": bool(args.promotion_enable_hva),
        "promotion_aggressive_screening": bool(args.promotion_aggressive_screening),
        "override_phase0_pilot_max_records": args.override_phase0_pilot_max_records,
        "override_phase1_shortlist_size": args.override_phase1_shortlist_size,
        "override_phase2_shortlist_fraction": args.override_phase2_shortlist_fraction,
        "override_phase2_shortlist_size": args.override_phase2_shortlist_size,
        "write_trajectory": bool(args.write_trajectory),
        "pool_policy": (
            "full_meta_hva_enabled"
            if bool(args.promotion_enable_hva)
            else "source_locked_reviewed_pool_surface"
        ),
        "screening_policy": (
            "phase0_96_phase1_24_phase2_0p25_12"
            if bool(args.promotion_aggressive_screening)
            else "source_locked_or_explicit_overrides"
        ),
    }


def command_only_row(
    *,
    python_bin: str,
    trial_params: optuna_hh.TrialParams,
    regime: str,
    epsilon_abs_delta_e: float,
    output_root: Path,
    rerun_trial_index: int,
    hamiltonian_overrides: optuna_hh.HhHamiltonianOverrides | None,
    eval_kwargs: Mapping[str, Any],
) -> tuple[Path, list[str], list[str], dict[str, Any]]:
    case_dir = optuna_hh._trial_case_dir(output_root, regime, epsilon_abs_delta_e, rerun_trial_index)
    command, dropped_args, _env, effective_params = optuna_hh._build_trial_command(
        python_bin=str(python_bin),
        params=trial_params,
        case_dir=case_dir,
        hamiltonian_overrides=hamiltonian_overrides,
        **dict(eval_kwargs),
    )
    return case_dir, command, list(dropped_args), dict(effective_params)


def observation_payload(obs: optuna_hh.TrialObservation) -> dict[str, Any]:
    payload = asdict(obs)
    return payload


def replay_row_for_report(
    *,
    candidate: Mapping[str, Any],
    db: Path,
    trial_db_row: Mapping[str, Any],
    attrs: Mapping[str, Any],
    params_hash: str,
    observation: Mapping[str, Any] | None,
    command_case_dir: Path,
    command: Sequence[str] | None,
    status: str,
    semantic_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    trial_number = maybe_int(candidate.get("trial_number", candidate.get("trial"))) or maybe_int(trial_db_row.get("trial_number"))
    obs = observation or {}
    return {
        "regime": candidate.get("regime"),
        "trial": trial_number,
        "trial_number": trial_number,
        "trial_id": trial_db_row.get("trial_id"),
        "study_id": trial_db_row.get("study_id"),
        "study_name": trial_db_row.get("study_name"),
        "label": "source-locked-rerun",
        "source_labels": candidate.get("source_labels") or ([candidate.get("label")] if candidate.get("label") else []),
        "candidate_source": "source_locked_review_rerun_v1",
        "semantic_overrides": dict(semantic_overrides),
        "source_db": repo_rel(db),
        "source_params_sha256": params_hash,
        "result_json": candidate.get("result_json"),
        "case_dir": candidate.get("case_dir"),
        "replay_case_dir": repo_rel(obs.get("case_dir") or command_case_dir),
        "replay_result_json": repo_rel(obs.get("result_json")),
        "compile_json": repo_rel(obs.get("compile_json")),
        "returncode": obs.get("returncode"),
        "compile_returncode": obs.get("compile_returncode"),
        "status": status,
        "dE": obs.get("abs_delta_e", attrs.get("abs_delta_e")),
        "objective_value": attrs.get("objective_lexicographic"),
        "k": obs.get("adapt_iteration_count", attrs.get("adapt_iteration_count")),
        "N2Q_proxy": obs.get("graph_count_2q", attrs.get("graph_count_2q")),
        "D2Q_proxy": obs.get("graph_depth", attrs.get("graph_depth")),
        "N1Q_proxy": obs.get("graph_count_1q", attrs.get("graph_count_1q")),
        "S_alg": obs.get("paper_i_table_s_alg", attrs.get("paper_i_table_s_alg")),
        "Salg": obs.get("paper_i_table_s_alg", attrs.get("paper_i_table_s_alg")),
        "Salg_status": obs.get("paper_i_table_shots_status", attrs.get("paper_i_table_shots_status")),
        "source_result_json_exists_locally": bool(candidate.get("result_json") and Path(str(candidate.get("result_json"))).exists()),
        "command": shlex.join([str(x) for x in command]) if command is not None else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW_JSON)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--merged-review-json", type=Path, default=None)
    parser.add_argument("--regime", action="append", default=[])
    parser.add_argument("--candidate-label", action="append", default=[])
    parser.add_argument(
        "--pool-policy-filter",
        default=None,
        help=(
            "When the input is a rebuilt all-candidate report, rerun only rows whose "
            "pool_policy_observed equals this value, e.g. full_meta_minus_hva."
        ),
    )
    parser.add_argument("--limit-per-regime", type=int, default=6)
    parser.add_argument("--epsilon-abs-delta-e", type=float, default=1.0e9)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--compile-backend", default="FakeMarrakesh")
    parser.add_argument("--compile-optimization-level", type=int, default=1)
    parser.add_argument("--compile-seed", type=int, default=7)
    parser.add_argument("--skip-qiskit-cost", action="store_true")
    parser.add_argument(
        "--promotion-enable-hva",
        action="store_true",
        help=(
            "Remove any reviewed force_adapt_pool_class_filter_json override so full_meta includes HVA. "
            "This is a promotion override, not a bit-for-bit reviewed-settings rerun."
        ),
    )
    parser.add_argument(
        "--promotion-aggressive-screening",
        action="store_true",
        help=(
            "With --promotion-enable-hva, tighten early screening defaults to keep HVA in the pool "
            "but prune weak candidates earlier."
        ),
    )
    parser.add_argument("--override-phase0-pilot-max-records", type=int, default=None)
    parser.add_argument("--override-phase1-shortlist-size", type=int, default=None)
    parser.add_argument("--override-phase2-shortlist-fraction", type=float, default=None)
    parser.add_argument("--override-phase2-shortlist-size", type=int, default=None)
    parser.add_argument(
        "--override-adapt-maxiter",
        type=int,
        default=None,
        help="Override the reviewed trial's --adapt-maxiter for this source-locked rerun.",
    )
    parser.add_argument(
        "--override-final-refit-maxiter",
        type=int,
        default=None,
        help="Override the reviewed trial's --adapt-final-refit-maxiter for this source-locked rerun.",
    )
    parser.add_argument(
        "--write-trajectory",
        action="store_true",
        help=(
            "Do not append --skip-trajectory to delegated adapt_pipeline commands. "
            "Use for final overlay replays that need error-vs-iteration histories."
        ),
    )
    parser.add_argument("--print-commands-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--only-resolved-trial",
        action="append",
        default=[],
        metavar="REGIME:TRIAL_ID",
        help="After resolving DB rows, run only this resolved trial id. May be repeated.",
    )
    parser.add_argument(
        "--exclude-resolved-trial",
        action="append",
        default=[],
        metavar="REGIME:TRIAL_ID",
        help="After resolving DB rows, skip this resolved trial id. May be repeated.",
    )
    parser.add_argument(
        "--write-incremental-manifest",
        action="store_true",
        help="During --execute, update the manifest and merged review JSON before and after each candidate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = utc_stamp()
    review_json = args.review_json.resolve()
    output_root = args.output_root or (DEFAULT_OUTPUT_BASE / f"paper_i_hh_review_source_locked_{stamp}")
    manifest_json = args.manifest_json or (DEFAULT_MANIFEST_DIR / f"paper_i_hh_review_source_locked_reruns_{stamp}.json")
    merged_review_json = args.merged_review_json or (
        DEFAULT_MANIFEST_DIR / f"paper_i_hh_snake_optuna_overlay_review_20260616_source_locked_replays_{stamp}.json"
    )
    payload = load_json(review_json)
    candidates = selected_candidates(
        payload,
        regimes=args.regime or None,
        labels=args.candidate_label or None,
        limit_per_regime=int(args.limit_per_regime),
        pool_policy_filter=args.pool_policy_filter,
    )
    if not candidates:
        raise SystemExit("No reviewed candidates matched the requested filters.")

    run_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped_duplicates: list[dict[str, Any]] = []
    filtered_out: list[dict[str, Any]] = []
    seen_resolved_trials: set[tuple[str, int | None, int]] = set()
    only_resolved_trials = parse_resolved_trial_specs(args.only_resolved_trial)
    exclude_resolved_trials = parse_resolved_trial_specs(args.exclude_resolved_trial)

    def emit_manifest(*, finalized: bool) -> None:
        manifest = {
            "schema": "paper_i_hh_review_source_locked_rerun_manifest_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "review_json": repo_rel(review_json),
            "output_root": repo_rel(output_root),
            "mode": "execute" if args.execute and not args.print_commands_only else "print_commands_only" if args.print_commands_only else "dry_run",
            "finalized": bool(finalized),
            "selected_candidate_count": len(candidates),
            "candidate_count": len(run_rows) + len(errors),
            "completed_count": sum(1 for row in report_rows if row.get("status") == "complete"),
            "running_count": sum(1 for row in report_rows if row.get("status") == "running"),
            "blocked_count": len(errors),
            "filtered_out_count": len(filtered_out),
            "skipped_duplicate_count": len(skipped_duplicates),
            "selection": {
                "regimes": list(args.regime or []),
                "candidate_labels": list(args.candidate_label or []),
                "limit_per_regime": int(args.limit_per_regime),
                "pool_policy_filter": args.pool_policy_filter,
                "only_resolved_trial": list(args.only_resolved_trial or []),
                "exclude_resolved_trial": list(args.exclude_resolved_trial or []),
            },
            "qiskit_cost": {
                "enabled": not bool(args.skip_qiskit_cost),
                "backend": args.compile_backend,
                "optimization_level": int(args.compile_optimization_level),
                "seed": int(args.compile_seed),
            },
            "semantic_overrides": semantic_overrides_payload(args),
            "runs": run_rows,
            "replayed_candidate_rows": report_rows,
            "errors": errors,
            "filtered_out": filtered_out,
            "skipped_duplicates": skipped_duplicates,
        }
        write_json(manifest_json, manifest)

        merged = dict(payload)
        merged["replayed_candidate_rows"] = report_rows
        merged["source_locked_rerun_manifest"] = repo_rel(manifest_json)
        merged["source_locked_rerun_policy"] = {
            "source": "Optuna trial_user_attributes.params",
            "wrapper": "pipelines.exact_bench.hh_cost_energy_optuna._evaluate_trial",
            "is_fresh_optuna_sampling": False,
            "semantic_overrides": semantic_overrides_payload(args),
            "finalized": bool(finalized),
        }
        write_json(merged_review_json, merged)

    for ordinal, candidate in enumerate(candidates, start=1):
        regime = str(candidate["regime"])
        try:
            db, row, attrs, objective_value = find_db_row(regime, candidate)
            row_trial_id = int(row["trial_id"])
            if only_resolved_trials and not resolved_trial_matches(
                only_resolved_trials,
                regime=regime,
                trial_id=row_trial_id,
            ):
                filtered_out.append(
                    {
                        "regime": regime,
                        "trial_id": row.get("trial_id"),
                        "trial_number": row.get("trial_number"),
                        "study_id": row.get("study_id"),
                        "reason": "not_in_only_resolved_trial_filter",
                    }
                )
                continue
            if resolved_trial_matches(exclude_resolved_trials, regime=regime, trial_id=row_trial_id):
                filtered_out.append(
                    {
                        "regime": regime,
                        "trial_id": row.get("trial_id"),
                        "trial_number": row.get("trial_number"),
                        "study_id": row.get("study_id"),
                        "reason": "excluded_resolved_trial",
                    }
                )
                continue
            resolved_trial_key = (regime, row.get("study_id"), int(row["trial_id"]))
            if resolved_trial_key in seen_resolved_trials:
                skipped_duplicates.append(
                    {
                        "regime": regime,
                        "trial_id": row.get("trial_id"),
                        "trial_number": row.get("trial_number"),
                        "study_id": row.get("study_id"),
                        "reason": "duplicate_resolved_optuna_trial",
                    }
                )
                continue
            seen_resolved_trials.add(resolved_trial_key)
            params_payload = attrs.get("params")
            if not isinstance(params_payload, Mapping):
                raise RuntimeError("Resolved trial lacks params user attr.")
            trial_params = trial_params_from_effective(params_payload)
            hamiltonian_overrides = hamiltonian_overrides_from_effective(params_payload)
            source_eval_kwargs = eval_kwargs_from_effective(params_payload)
            eval_kwargs = apply_promotion_overrides(source_eval_kwargs, args)
            dominance_kwargs = dominance_kwargs_from_attrs(attrs)
            params_hash = canonical_hash(params_payload)
            rerun_trial_index = maybe_int(row.get("trial_id")) or ordinal
            case_dir, command, dropped_args, effective_params = command_only_row(
                python_bin=str(args.python_bin),
                trial_params=trial_params,
                regime=regime,
                epsilon_abs_delta_e=float(args.epsilon_abs_delta_e),
                output_root=output_root,
                rerun_trial_index=int(rerun_trial_index),
                hamiltonian_overrides=hamiltonian_overrides,
                eval_kwargs=eval_kwargs,
            )
            command_record = {
                "regime": regime,
                "source_trial_id": row.get("trial_id"),
                "source_trial_number": row.get("trial_number"),
                "source_labels": candidate.get("source_labels") or [],
                "source_db": repo_rel(db),
                "source_params_sha256": params_hash,
                "case_dir": repo_rel(case_dir),
                "command": shlex.join([str(x) for x in command]),
                "dropped_args": dropped_args,
                "effective_params": effective_params,
                "semantic_overrides": semantic_overrides_payload(args),
                "source_force_adapt_pool_class_filter_json": repo_rel(
                    source_eval_kwargs.get("force_adapt_pool_class_filter_json")
                ),
                "effective_force_adapt_pool_class_filter_json": repo_rel(
                    eval_kwargs.get("force_adapt_pool_class_filter_json")
                ),
                "source_objective_value": objective_value,
                "source_attrs": {
                    key: attrs.get(key)
                    for key in (
                        "abs_delta_e",
                        "adapt_iteration_count",
                        "graph_count_2q",
                        "graph_depth",
                        "graph_count_1q",
                        "paper_i_table_s_alg",
                        "paper_i_table_shots_status",
                    )
                },
            }
            observation: dict[str, Any] | None = None
            status = "planned"
            if args.print_commands_only:
                print(command_record["command"])
            if args.execute and not args.print_commands_only:
                status = "running"
                command_record["status"] = status
                run_rows.append(command_record)
                report_rows.append(
                    replay_row_for_report(
                        candidate=candidate,
                        db=db,
                        trial_db_row=row,
                        attrs=attrs,
                        params_hash=params_hash,
                        observation=observation,
                        command_case_dir=case_dir,
                        command=command,
                        status=status,
                        semantic_overrides=semantic_overrides_payload(args),
                    )
                )
                if args.write_incremental_manifest:
                    emit_manifest(finalized=False)
                obs = optuna_hh._evaluate_trial(
                    python_bin=str(args.python_bin),
                    params=trial_params,
                    lane=regime,
                    epsilon_abs_delta_e=float(args.epsilon_abs_delta_e),
                    output_dir=output_root,
                    trial_index=int(rerun_trial_index),
                    compile_backend=str(args.compile_backend),
                    compile_opt_level=int(args.compile_optimization_level),
                    compile_seed=int(args.compile_seed),
                    hamiltonian_overrides=hamiltonian_overrides,
                    compile_enabled=not bool(args.skip_qiskit_cost),
                    require_graph_cost=True,
                    **eval_kwargs,
                    **dominance_kwargs,
                )
                observation = observation_payload(obs)
                pipeline_ok = obs.returncode is not None and int(obs.returncode) == 0
                compile_ok = bool(args.skip_qiskit_cost) or (
                    obs.compile_returncode is not None and int(obs.compile_returncode) == 0
                )
                status = "complete" if pipeline_ok and compile_ok else "failed"
                command_record["observation"] = observation
                command_record["status"] = status
                report_rows[-1] = replay_row_for_report(
                    candidate=candidate,
                    db=db,
                    trial_db_row=row,
                    attrs=attrs,
                    params_hash=params_hash,
                    observation=observation,
                    command_case_dir=case_dir,
                    command=command,
                    status=status,
                    semantic_overrides=semantic_overrides_payload(args),
                )
                if args.write_incremental_manifest:
                    emit_manifest(finalized=False)
            elif args.print_commands_only:
                status = "print_commands_only"
                command_record["status"] = status
                run_rows.append(command_record)
                report_rows.append(
                    replay_row_for_report(
                        candidate=candidate,
                        db=db,
                        trial_db_row=row,
                        attrs=attrs,
                        params_hash=params_hash,
                        observation=observation,
                        command_case_dir=case_dir,
                        command=command,
                        status=status,
                        semantic_overrides=semantic_overrides_payload(args),
                    )
                )
            else:
                command_record["status"] = status
                run_rows.append(command_record)
                report_rows.append(
                    replay_row_for_report(
                        candidate=candidate,
                        db=db,
                        trial_db_row=row,
                        attrs=attrs,
                        params_hash=params_hash,
                        observation=observation,
                        command_case_dir=case_dir,
                        command=command,
                        status=status,
                        semantic_overrides=semantic_overrides_payload(args),
                    )
                )
        except Exception as exc:
            error = {
                "regime": candidate.get("regime"),
                "trial_id": candidate.get("trial_id"),
                "trial_number": candidate.get("trial_number", candidate.get("trial")),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            print(f"[blocked] {error}", file=sys.stderr)
            if args.write_incremental_manifest:
                emit_manifest(finalized=False)

    emit_manifest(finalized=True)
    print(f"manifest: {manifest_json}")
    print(f"merged_review_json: {merged_review_json}")
    if errors:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
