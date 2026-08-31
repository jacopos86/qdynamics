"""Shared Paper-I clean phonon cutoff-ladder row contract.

This module is intentionally small and data-oriented.  It lets generic static
rows and Route-A/SNAKE Optuna rows share the same clean Paper-I ladder semantics
without coupling their runners.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_CLEAN_NPH1_REF4_PROFILE,
    TABLE_I_CLEAN_NPH2_REF4_PROFILE,
    TABLE_I_CLEAN_NPH2_REF5_PROFILE,
    TABLE_I_CLEAN_NPH3_REF4_PROFILE,
    TABLE_I_CLEAN_NPH4_REF5_PROFILE,
    TABLE_I_CLEAN_NPH4_REF7_PROFILE,
    TABLE_I_CLEAN_NPH6_REF9_PROFILE,
)

# Paper-I clean fixed-accuracy target.  The updated run skill defines the
# headline target as 1e-4 energy density for L=2 unit-normalized rows:
# tau_phys = 1e-4 * N_phys * E0 = 2e-4.  Keep the legacy "tight" field equal
# to the primary target so generated rows cannot use the stale 1e-3 policy.
PAPER_I_CLEAN_TAU_PHYS = 2e-4
PAPER_I_CLEAN_TAU_TIGHT = PAPER_I_CLEAN_TAU_PHYS
PAPER_I_PHONON_FAMILIES = ("bose_hubbard", "harmonic_kerr_chain", "spin_boson", "hh")
PAPER_I_LADDER_SNAKE_POLICIES = ("included", "snake_only", "benchmarks_only_explicit")
PAPER_I_SELECTED_RECOVERY_INTENT = "selected_logical_current_evidence_recovery"
PAPER_I_SELECTED_RECOVERY_DEFAULT_SOURCE_MANIFEST_JSON = (
    "chtc/phase3_optuna/input/paper_i_reduced_pool_selected_logical_20260524_v1/manifest.json"
)
PAPER_I_SELECTED_RECOVERY_N_TRIALS = 8
PAPER_I_SELECTED_RECOVERY_MAX_TRIALS = 8
PAPER_I_SELECTED_RECOVERY_MAX_GRADIENT_WORKERS = 10
PAPER_I_SELECTED_RECOVERY_MAX_BEAM_PARENT_WORKERS = 4
PAPER_I_SELECTED_RECOVERY_MAX_REQUEST_CPUS = 10
PAPER_I_SELECTED_RECOVERY_SOURCE_NAME_BY_FAMILY = {
    "harmonic_kerr_chain": "harmonic_kerr_chain_L2_nph2",
    "hh": "hh_L2_nph2",
    "spin_boson": "spin_boson_problem",
}
PAPER_I_PRODUCTION_OBJECTIVE_WEIGHTS = {
    "objective_energy_weight": "8.0",
    "objective_2q_weight": "4.0",
    "objective_2q_depth_weight": "0.5",
    "objective_depth_weight": "0.05",
    "objective_parameter_weight": "0.10",
    "objective_shot_weight": "0.05",
}

PHONON_CUTOFF_TSV_FIELDS = (
    "n_ph_work",
    "n_ph_ref",
    "primary_energy_metric",
    "same_cutoff_error_role",
    "paper_i_cutoff_ladder_stage",
    "paper_i_ladder_acceptance_threshold",
    "paper_i_ladder_requires_prior_failure",
    "paper_i_ladder_escalation_reason",
    "paper_i_ladder_allow_ref5",
    "paper_i_ladder_snake_policy",
    "reference_energy_cache_json",
    "same_cutoff_reference_energy_key",
    "reference_cutoff_energy_key",
    "same_cutoff_exact_gs_energy",
    "exact_reference_energy",
    "exact_reference_n_ph_max",
    "reference_energy_status",
    "tau_phys",
    "tau_tight",
    "paper_i_ladder_candidate_manifest_json",
    "paper_i_ladder_candidate_key",
    "paper_i_ladder_source_audit_json",
    "paper_i_ladder_source_audit_sha256",
    "paper_i_ladder_source_case_id",
    "paper_i_ladder_source_stage",
    "paper_i_ladder_source_status",
    "paper_i_ladder_source_threshold_status",
    "paper_i_ladder_source_n_ph_work",
    "paper_i_ladder_source_n_ph_ref",
    "paper_i_ladder_source_threshold",
    "paper_i_ladder_source_target_profile",
    "paper_i_ladder_source_payload_path",
    "paper_i_ladder_source_payload_sha256",
    "paper_i_ladder_source_record_id",
)


@dataclass(frozen=True)
class PaperICleanLadderStageConfig:
    enabled: bool
    suite_profile: str | None
    n_ph_work: int | None
    n_ph_ref: int | None
    requires_prior_failure: bool = False
    requires_ref5_allowance: bool = False

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


PAPER_I_LADDER_STAGE_CONFIGS: dict[str, PaperICleanLadderStageConfig] = {
    "off": PaperICleanLadderStageConfig(False, None, None, None),
    "nph1_ref4_screen": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH1_REF4_PROFILE,
        1,
        4,
    ),
    "nph2_ref4_screen": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH2_REF4_PROFILE,
        2,
        4,
    ),
    "nph3_ref4_escalation": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        3,
        4,
        requires_prior_failure=True,
    ),
    "nph3_ref4_screen": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH3_REF4_PROFILE,
        3,
        4,
    ),
    "nph4_ref5_optional": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH4_REF5_PROFILE,
        4,
        5,
        requires_prior_failure=True,
        requires_ref5_allowance=True,
    ),
    "table_nph2_ref5": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH2_REF5_PROFILE,
        2,
        5,
    ),
    "table_nph4_ref7": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH4_REF7_PROFILE,
        4,
        7,
    ),
    "table_nph6_ref9": PaperICleanLadderStageConfig(
        True,
        TABLE_I_CLEAN_NPH6_REF9_PROFILE,
        6,
        9,
    ),
}


def normalize_paper_i_ladder_stage(value: str | None) -> tuple[str, PaperICleanLadderStageConfig]:
    stage = str(value or "off").strip().lower().replace("-", "_")
    if stage not in PAPER_I_LADDER_STAGE_CONFIGS:
        raise ValueError(f"paper_i_cutoff_ladder_stage must be one of {sorted(PAPER_I_LADDER_STAGE_CONFIGS)}")
    return stage, PAPER_I_LADDER_STAGE_CONFIGS[stage]


def blank_phonon_cutoff_fields() -> dict[str, str]:
    return {field: "" for field in PHONON_CUTOFF_TSV_FIELDS}


def is_paper_i_clean_ladder_row(row: Mapping[str, str]) -> bool:
    return bool(str(row.get("paper_i_cutoff_ladder_stage") or "").strip())


def _field(row: Mapping[str, str], name: str) -> str:
    return str(row.get(name) or "").strip()


def _falsey(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"0", "false", "no", "n", "off"}


def _route_key(value: str | None) -> str:
    return str(value or "standard").strip().lower().replace("-", "_")


def _parse_int_field(row: Mapping[str, Any], field: str, blockers: list[str]) -> int | None:
    raw = str(row.get(field) or "").strip().strip('"').strip("'")
    if not raw:
        blockers.append(f"paper_i_selected_recovery_{field}_missing")
        return None
    try:
        return int(raw)
    except Exception:
        blockers.append(f"paper_i_selected_recovery_{field}_invalid:{raw}")
        return None


def _resolve_repo_path(path_value: str, repo_root: Path | None) -> Path:
    path = Path(str(path_value))
    if path.is_absolute() or repo_root is None:
        return path
    return Path(repo_root) / path


def _float_is(value: str, expected: float) -> bool:
    try:
        parsed = float(value)
    except Exception:
        return False
    return math.isfinite(parsed) and math.isclose(parsed, float(expected), rel_tol=0.0, abs_tol=1e-15)


def is_routea_selected_recovery_row(row: Mapping[str, str]) -> bool:
    """Return true for Route-A clean ladder rows using selected logical recovery."""

    intent = _field(row, "paper_i_recovery_intent")
    if intent == PAPER_I_SELECTED_RECOVERY_INTENT:
        return True
    return is_paper_i_clean_ladder_row(row) and _route_key(row.get("selected_logical_route")) == "historical_selected"


def validate_routea_selected_recovery_contract(
    row: Mapping[str, str],
    *,
    submit_contract: Mapping[str, Any] | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    """Return fail-closed blockers for current-evidence selected-pool reruns.

    This contract deliberately applies only to clean Route-A/SNAKE rows that
    select the historical-logical route.  Ordinary full-search diagnostics keep
    their existing broader defaults, but a selected-pool recovery cannot launch
    with the old 36-trial/10-worker/no-historical-warm-start shape.
    """

    if not is_routea_selected_recovery_row(row):
        return []

    blockers: list[str] = []
    route = _route_key(row.get("selected_logical_route"))
    if route != "historical_selected":
        blockers.append(f"paper_i_selected_recovery_route_mismatch:{route}")
    transfer = str(row.get("selected_logical_transfer_mode") or "exact_match_v1").strip().lower()
    if transfer not in {"exact_match_v1", "boundary_v1"}:
        blockers.append(f"paper_i_selected_recovery_transfer_mode_invalid:{transfer}")

    selected_source = _field(row, "selected_logical_source_json")
    historical_ledger = _field(row, "historical_ledger")
    if not selected_source:
        blockers.append("selected_logical_source_missing_for_historical_selected")
    if not historical_ledger:
        blockers.append("paper_i_selected_recovery_historical_ledger_missing")
    if _falsey(row.get("enqueue_historical")):
        blockers.append("paper_i_selected_recovery_enqueue_historical_false")
    if not _falsey(row.get("enqueue_default")):
        blockers.append("paper_i_selected_recovery_enqueue_default_not_false")

    n_trials = _parse_int_field(row, "n_trials", blockers)
    if n_trials is not None:
        is_smoke = _field(row, "record_id").endswith("_smoke")
        max_trials = 1 if is_smoke else PAPER_I_SELECTED_RECOVERY_MAX_TRIALS
        if n_trials < 1 or n_trials > max_trials:
            blockers.append(f"paper_i_selected_recovery_n_trials_out_of_bounds:{n_trials}:{max_trials}")
    n_jobs = _parse_int_field(row, "n_jobs", blockers)
    if n_jobs is not None and n_jobs != 1:
        blockers.append(f"paper_i_selected_recovery_n_jobs_must_be_1:{n_jobs}")
    bench_jobs = _parse_int_field(row, "benchmarks_per_trial_jobs", blockers)
    if bench_jobs is not None and bench_jobs != 1:
        blockers.append(f"paper_i_selected_recovery_benchmarks_per_trial_jobs_must_be_1:{bench_jobs}")
    grad_workers = _parse_int_field(row, "phase3_adapt_parallel_gradient_workers", blockers)
    if grad_workers is not None and grad_workers > PAPER_I_SELECTED_RECOVERY_MAX_GRADIENT_WORKERS:
        blockers.append(
            "paper_i_selected_recovery_parallel_gradient_workers_too_high:"
            f"{grad_workers}:{PAPER_I_SELECTED_RECOVERY_MAX_GRADIENT_WORKERS}"
        )
    beam_workers = _parse_int_field(row, "phase3_adapt_beam_parent_workers", blockers)
    if beam_workers is not None and beam_workers > PAPER_I_SELECTED_RECOVERY_MAX_BEAM_PARENT_WORKERS:
        blockers.append(
            "paper_i_selected_recovery_beam_parent_workers_too_high:"
            f"{beam_workers}:{PAPER_I_SELECTED_RECOVERY_MAX_BEAM_PARENT_WORKERS}"
        )

    if submit_contract is not None:
        request_cpus_raw = str(submit_contract.get("request_cpus") or "").strip().strip('"').strip("'")
        if not request_cpus_raw:
            blockers.append("paper_i_selected_recovery_request_cpus_missing")
        else:
            try:
                request_cpus = int(request_cpus_raw)
            except Exception:
                blockers.append(f"paper_i_selected_recovery_request_cpus_invalid:{request_cpus_raw}")
            else:
                if request_cpus > PAPER_I_SELECTED_RECOVERY_MAX_REQUEST_CPUS:
                    blockers.append(
                        "paper_i_selected_recovery_request_cpus_too_high:"
                        f"{request_cpus}:{PAPER_I_SELECTED_RECOVERY_MAX_REQUEST_CPUS}"
                    )

    if repo_root is not None:
        if historical_ledger:
            ledger_path = _resolve_repo_path(historical_ledger, repo_root)
            if not ledger_path.exists():
                blockers.append(f"paper_i_selected_recovery_historical_ledger_missing_file:{historical_ledger}")
        if selected_source:
            source_path = _resolve_repo_path(selected_source, repo_root)
            if not source_path.exists():
                blockers.append(f"paper_i_selected_recovery_selected_source_missing_file:{selected_source}")
            else:
                try:
                    payload = json.loads(source_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    blockers.append(f"paper_i_selected_recovery_selected_source_unreadable:{type(exc).__name__}:{exc}")
                else:
                    schema = str(payload.get("schema") or payload.get("library_version") or "")
                    if schema != "selected_logical_library_v1":
                        blockers.append(f"paper_i_selected_recovery_selected_source_schema_mismatch:{schema}")
                    records = payload.get("records")
                    if not isinstance(records, list) or not records:
                        blockers.append("paper_i_selected_recovery_selected_source_records_missing_or_empty")
                    try:
                        record_count = int(payload.get("record_count"))
                    except Exception:
                        record_count = len(records) if isinstance(records, list) else 0
                    if record_count < 1:
                        blockers.append(f"paper_i_selected_recovery_selected_source_record_count_invalid:{record_count}")
    return blockers


def validate_clean_ladder_common_fields(
    row: Mapping[str, str],
    *,
    stage: str | None = None,
    stage_config: PaperICleanLadderStageConfig | None = None,
    row_kind: str = "generic_static",
) -> list[str]:
    """Return fail-closed blockers for common clean ladder row metadata."""

    blockers: list[str] = []
    row_stage = _field(row, "paper_i_cutoff_ladder_stage")
    if not row_stage:
        return blockers
    if stage_config is None:
        try:
            stage, stage_config = normalize_paper_i_ladder_stage(stage or row_stage)
        except ValueError:
            blockers.append(f"paper_i_ladder_unknown_stage:{row_stage}")
            return blockers
    stage = str(stage or row_stage)
    if row_stage != stage:
        blockers.append(f"paper_i_ladder_stage_mismatch:{row_stage}:{stage}")
    if not stage_config.enabled:
        blockers.append(f"paper_i_ladder_stage_not_enabled:{stage}")
    if stage_config.suite_profile is not None and _field(row, "suite_profile") != stage_config.suite_profile:
        blockers.append(f"paper_i_ladder_suite_profile_mismatch:{row.get('suite_profile')}:{stage_config.suite_profile}")
    if _field(row, "family") and _field(row, "family") not in PAPER_I_PHONON_FAMILIES:
        blockers.append(f"paper_i_ladder_non_phonon_family:{row.get('family')}")
    if _field(row, "families") and _field(row, "families") not in PAPER_I_PHONON_FAMILIES:
        blockers.append(f"paper_i_ladder_non_phonon_family:{row.get('families')}")
    if stage_config.n_ph_work is not None and _field(row, "n_ph_work") != str(stage_config.n_ph_work):
        blockers.append(f"paper_i_ladder_n_ph_work_mismatch:{row.get('n_ph_work')}:{stage_config.n_ph_work}")
    if stage_config.n_ph_ref is not None and _field(row, "n_ph_ref") != str(stage_config.n_ph_ref):
        blockers.append(f"paper_i_ladder_n_ph_ref_mismatch:{row.get('n_ph_ref')}:{stage_config.n_ph_ref}")
    if _field(row, "primary_energy_metric") != "higher_cutoff_reference_abs_delta_e":
        blockers.append("paper_i_ladder_primary_metric_mismatch")
    if _field(row, "same_cutoff_error_role") != "diagnostic_only":
        blockers.append("paper_i_ladder_same_cutoff_role_mismatch")
    if not _float_is(_field(row, "paper_i_ladder_acceptance_threshold"), PAPER_I_CLEAN_TAU_TIGHT):
        blockers.append("paper_i_ladder_acceptance_threshold_mismatch")
    if not _float_is(_field(row, "tau_phys"), PAPER_I_CLEAN_TAU_PHYS):
        blockers.append("paper_i_ladder_tau_phys_mismatch")
    if not _float_is(_field(row, "tau_tight"), PAPER_I_CLEAN_TAU_TIGHT):
        blockers.append("paper_i_ladder_tau_tight_mismatch")
    snake_policy = _field(row, "paper_i_ladder_snake_policy")
    if snake_policy and snake_policy not in PAPER_I_LADDER_SNAKE_POLICIES:
        blockers.append(f"paper_i_ladder_snake_policy_missing_or_invalid:{snake_policy}")
    if row_kind == "routea_phase0" and snake_policy != "snake_only":
        blockers.append(f"paper_i_ladder_routea_requires_snake_only_policy:{snake_policy}")
    if stage_config.requires_prior_failure and not _field(row, "paper_i_ladder_escalation_reason"):
        blockers.append("paper_i_ladder_missing_escalation_reason")
    if stage_config.requires_ref5_allowance and _field(row, "paper_i_ladder_allow_ref5").lower() != "true":
        blockers.append("paper_i_ladder_missing_ref5_allowance")
    return blockers


def _expected_source_case_id(source_case_id: str, target_stage: str) -> str | None:
    if not source_case_id:
        return None
    if target_stage == "nph3_ref4_escalation":
        return source_case_id.replace("_nph2_", "_nph3_")
    if target_stage == "nph4_ref5_optional":
        return source_case_id.replace("_nph3_", "_nph4_")
    return None


def validate_clean_ladder_source_metadata(
    row: Mapping[str, str],
    *,
    stage: str | None = None,
    stage_config: PaperICleanLadderStageConfig | None = None,
    target_case_id: str | None = None,
) -> list[str]:
    """Return blockers for prior-failure source metadata on escalation rows.

    Generation-time helpers still allow explicit case-id/reason dry-runs, but
    preflight for submit surfaces should require this audit-derived metadata
    before an escalation row can run.
    """

    blockers: list[str] = []
    row_stage = _field(row, "paper_i_cutoff_ladder_stage")
    if not row_stage:
        return blockers
    if stage_config is None:
        try:
            stage, stage_config = normalize_paper_i_ladder_stage(stage or row_stage)
        except ValueError:
            return [f"paper_i_ladder_unknown_stage:{row_stage}"]
    stage = str(stage or row_stage)
    if not stage_config.requires_prior_failure:
        return blockers

    required_fields = (
        "paper_i_ladder_candidate_manifest_json",
        "paper_i_ladder_candidate_key",
        "paper_i_ladder_source_audit_json",
        "paper_i_ladder_source_case_id",
        "paper_i_ladder_source_stage",
        "paper_i_ladder_source_status",
        "paper_i_ladder_source_n_ph_work",
        "paper_i_ladder_source_n_ph_ref",
        "paper_i_ladder_source_threshold",
        "paper_i_ladder_source_target_profile",
    )
    for field in required_fields:
        if not _field(row, field):
            blockers.append(f"paper_i_ladder_source_metadata_missing:{field}")

    if _field(row, "paper_i_ladder_source_target_profile") and _field(row, "paper_i_ladder_source_target_profile") != "paper_i_phys_v1":
        blockers.append(f"paper_i_ladder_source_target_profile_mismatch:{row.get('paper_i_ladder_source_target_profile')}:paper_i_phys_v1")
    if _field(row, "paper_i_ladder_source_threshold") and not _float_is(_field(row, "paper_i_ladder_source_threshold"), PAPER_I_CLEAN_TAU_PHYS):
        blockers.append("paper_i_ladder_source_threshold_mismatch")
    source_status = _field(row, "paper_i_ladder_source_status")
    if source_status and source_status not in {"not_reached", "failed"}:
        blockers.append(f"paper_i_ladder_source_status_not_terminal_failure:{source_status}")

    expected_source: tuple[str, int, int] | None = None
    if stage == "nph3_ref4_escalation":
        expected_source = ("nph2_ref4_screen", 2, 4)
    elif stage == "nph4_ref5_optional":
        expected_source = ("nph3_ref4_escalation", 3, 4)
    if expected_source is not None:
        expected_stage, expected_work, expected_ref = expected_source
        if _field(row, "paper_i_ladder_source_stage") and _field(row, "paper_i_ladder_source_stage") != expected_stage:
            blockers.append(f"paper_i_ladder_source_stage_mismatch:{row.get('paper_i_ladder_source_stage')}:{expected_stage}")
        if _field(row, "paper_i_ladder_source_n_ph_work") and _field(row, "paper_i_ladder_source_n_ph_work") != str(expected_work):
            blockers.append(f"paper_i_ladder_source_n_ph_work_mismatch:{row.get('paper_i_ladder_source_n_ph_work')}:{expected_work}")
        if _field(row, "paper_i_ladder_source_n_ph_ref") and _field(row, "paper_i_ladder_source_n_ph_ref") != str(expected_ref):
            blockers.append(f"paper_i_ladder_source_n_ph_ref_mismatch:{row.get('paper_i_ladder_source_n_ph_ref')}:{expected_ref}")

    source_case = _field(row, "paper_i_ladder_source_case_id")
    expected_target = _expected_source_case_id(source_case, stage)
    if target_case_id and expected_target and str(target_case_id) != expected_target:
        blockers.append(f"paper_i_ladder_source_case_target_mismatch:{source_case}:{target_case_id}:{expected_target}")
    return blockers
