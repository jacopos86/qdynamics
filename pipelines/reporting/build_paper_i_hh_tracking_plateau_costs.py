#!/usr/bin/env python3
"""Reconstruct and compile exact Paper-I HH plateau prefixes.

The selected prefix is reporting-only: it is the first completed history row
whose same-cutoff absolute error is within ten percent of the minimum observed
over that run's complete stored trajectory.  SNAKE prefixes are taken from
signed active-prefix checkpoints whenever available; comparator prefixes are
taken from their embedded active-prefix checkpoints.  The script fails closed
on checkpoint/hash/sector drift and never replays a pruned SNAKE prefix from
admission history.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import tarfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Large source-locked results are materially cheaper to parse with orjson.
    import orjson
except ImportError:  # pragma: no cover - the repository runtime currently provides it.
    orjson = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRACKER_DIR = REPO_ROOT / (
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715"
)
TRACKER_STEM = (
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715"
)
DEFAULT_TRACKER_JSON = TRACKER_DIR / f"{TRACKER_STEM}.json"
DEFAULT_OUTPUT_JSON = TRACKER_DIR / "plateau_prefix_costs.json"
LEGACY_OFF_SW_PREFIX_SOURCE = REPO_ROOT / (
    "raw_outputs/paper_i_hh_sr_snake_noprune_nobeam_no_ordinary_novelty_"
    "five_20260715_v1_chtc/strong_weak_u8/json/result.json"
)

SCHEMA = "paper_i_hh_tracking_plateau_prefix_costs_v1"
RULE_ID = "first_prefix_within_10pct_of_complete_trajectory_minimum_v1"
RELATIVE_TOLERANCE = 0.10
COMPARATOR_ROUTE_IDS = {
    "geo_adapt_macro_nph3_7",
    "append_adapt_macro_nph3_7",
    "geo_adapt_projected_singleton_nph3_7",
    "append_adapt_projected_singleton_nph3_7",
}
COST_ARM_ROUTE_IDS = {
    "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7",
    "sr_macro_beam3x2_fs_prune_one_sided_cost_nph3_7",
}
PAPER_I_CLEAN_S_ALG_SNAKE_REPRESENTATIONS = {
    "sr_macro_physical_lanes_nph3_7": "intact_macro",
    "sr_macro_commutation_reduced_insertion_nph3_7": "intact_macro",
    "no_overlap_trust_projected_phase3_nph3_7": "projected_singleton",
}


from pipelines.exact_bench.paper_i_s_alg_accounting import (  # noqa: E402
    PAPER_I_S_ALG_ACCOUNTING_SCHEMA,
    SNAKE_REPRESENTATION_INTACT_MACRO,
    SNAKE_REPRESENTATION_PROJECTED_SINGLETON,
)
from pipelines.reporting.paper_i_run_summary import (  # noqa: E402
    LOCKED_QISKIT_COMPILE_CONVENTION,
    PaperIErrorTracePoint,
    select_paper_i_effective_plateau,
)
from pipelines.reporting.paper_i_qiskit_cost_tuple import (  # noqa: E402
    qiskit_cost_fields,
)

# Preserve the historical report-builder export without importing Qiskit.
TABLE_I_QISKIT_COMPILE_CONVENTION = LOCKED_QISKIT_COMPILE_CONVENTION


@lru_cache(maxsize=1)
def _table_i_compile_convention() -> str:
    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        TABLE_I_QISKIT_COMPILE_CONVENTION as installed_convention,
    )

    if installed_convention != TABLE_I_QISKIT_COMPILE_CONVENTION:
        raise RuntimeError("The installed Paper-I compile convention drifted.")
    return str(installed_convention)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=None)
def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _append_redundant_post_refit_verification_count(
    cumulative_occurrences: Mapping[str, Any],
    *,
    accepted_prefix_length: int,
) -> int:
    """Derive whether an Append prefix is a pre- or post-endpoint-reuse run."""

    scopes = cumulative_occurrences.get("occurrence_count_by_consumer_scope")
    if not isinstance(scopes, Mapping):
        raise ValueError(
            "Append cumulative occurrence summary lacks consumer-scope counts."
        )
    redundant = 0
    for scope, raw_count in scopes.items():
        if not str(scope).endswith(":post_optimizer_exact_verification"):
            continue
        if isinstance(raw_count, bool):
            raise ValueError(
                "Append post-optimizer verifier count must be nonnegative."
            )
        count = int(raw_count)
        if count < 0 or (
            isinstance(raw_count, float) and float(raw_count) != float(count)
        ):
            raise ValueError(
                "Append post-optimizer verifier count must be nonnegative."
            )
        redundant += count
    k = int(accepted_prefix_length)
    if redundant not in {0, k}:
        raise ValueError(
            "Append prefix has a mixed or incomplete post-optimizer verifier "
            f"pattern: observed {redundant}, expected either 0 or {k}."
        )
    return int(redundant)


def _loads_json(raw: bytes) -> tuple[Any, dict[str, int]]:
    if orjson is None:
        return json.loads(raw), {"negative_infinity": 0, "positive_infinity": 0, "nan": 0}
    try:
        return orjson.loads(raw), {"negative_infinity": 0, "positive_infinity": 0, "nan": 0}
    except orjson.JSONDecodeError:
        # Python's legacy JSON reader accepted these nonstandard score
        # sentinels.  They are irrelevant to prefix/checkpoint reconstruction,
        # but strict orjson rejects them.  Normalize only for the in-memory
        # reporting view and retain the raw source/member SHA-256 below.
        counts = {
            "negative_infinity": raw.count(b"-Infinity"),
            "positive_infinity": raw.count(b"Infinity") - raw.count(b"-Infinity"),
            "nan": raw.count(b"NaN"),
        }
        normalized = raw.replace(b"-Infinity", b"null")
        normalized = normalized.replace(b"Infinity", b"null")
        normalized = normalized.replace(b"NaN", b"null")
        return orjson.loads(normalized), counts


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is nonfinite: {value!r}")
    return parsed


def _source_path(source: Mapping[str, Any]) -> Path:
    path = Path(str(source.get("path") or ""))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = str(source.get("sha256") or "")
    observed = _sha256_path(path)
    if expected and observed != expected:
        raise ValueError(
            f"source SHA-256 drift for {path}: expected={expected}, observed={observed}"
        )
    return path


def _read_source_result(
    source: Mapping[str, Any], *, need_runtime_seed: bool
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    path = _source_path(source)
    archive_sha256 = _sha256_path(path)
    member_name = source.get("member")
    if member_name is None:
        raw = path.read_bytes()
        payload, nonfinite_tokens = _loads_json(raw)
        if not isinstance(payload, Mapping):
            raise TypeError(f"result root is not an object: {path}")
        return dict(payload), None, {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": archive_sha256,
            "result_member": None,
            "result_member_sha256": _sha256_bytes(raw),
            "runtime_seed_member": None,
            "runtime_seed_member_sha256": None,
            "parser_nonfinite_tokens_normalized_to_null": nonfinite_tokens,
        }

    target = str(member_name)
    seed_suffix = target.rsplit("/", 1)[0] + "/runtime_seed.json"
    result_raw: bytes | None = None
    seed_raw: bytes | None = None
    with tarfile.open(path, "r|gz") as archive:
        for info in archive:
            if info.name == target or (need_runtime_seed and info.name == seed_suffix):
                handle = archive.extractfile(info)
                if handle is None:
                    raise RuntimeError(f"cannot extract {info.name} from {path}")
                if info.name == target:
                    if result_raw is not None:
                        raise RuntimeError(f"duplicate result member {target} in {path}")
                    result_raw = handle.read()
                else:
                    if seed_raw is not None:
                        raise RuntimeError(f"duplicate runtime seed {seed_suffix} in {path}")
                    seed_raw = handle.read()
            archive.members.clear()
            if result_raw is not None and (not need_runtime_seed or seed_raw is not None):
                break
    if result_raw is None:
        raise RuntimeError(f"missing result member {target} in {path}")
    if need_runtime_seed and seed_raw is None:
        raise RuntimeError(f"missing runtime seed member {seed_suffix} in {path}")
    payload, result_nonfinite_tokens = _loads_json(result_raw)
    if seed_raw is None:
        seed = None
        seed_nonfinite_tokens = {"negative_infinity": 0, "positive_infinity": 0, "nan": 0}
    else:
        seed, seed_nonfinite_tokens = _loads_json(seed_raw)
    if not isinstance(payload, Mapping):
        raise TypeError(f"result member is not an object: {path}:{target}")
    if seed is not None and not isinstance(seed, Mapping):
        raise TypeError(f"runtime seed is not an object: {path}:{seed_suffix}")
    return dict(payload), None if seed is None else dict(seed), {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": archive_sha256,
        "result_member": target,
        "result_member_sha256": _sha256_bytes(result_raw),
        "runtime_seed_member": seed_suffix if seed_raw is not None else None,
        "runtime_seed_member_sha256": (
            None if seed_raw is None else _sha256_bytes(seed_raw)
        ),
        "parser_nonfinite_tokens_normalized_to_null": result_nonfinite_tokens,
        "runtime_seed_parser_nonfinite_tokens_normalized_to_null": seed_nonfinite_tokens,
    }


def _snake_history(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], float | None]:
    adapt = payload.get("adapt_vqe")
    ground = payload.get("ground_state")
    if not isinstance(adapt, Mapping):
        raise ValueError("SNAKE payload has no adapt_vqe object")
    history = adapt.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("SNAKE payload has no completed history")
    if not all(isinstance(row, Mapping) for row in history):
        raise TypeError("SNAKE history contains a non-object row")
    exact_raw = adapt.get("exact_gs_energy")
    if exact_raw is None and isinstance(ground, Mapping):
        exact_raw = ground.get("exact_energy")
    exact = None if exact_raw is None else _finite(exact_raw, label="SNAKE exact energy")
    return [row for row in history if isinstance(row, Mapping)], exact


def _comparator_history(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = payload.get("result")
    if payload.get("status") != "completed" or not isinstance(result, Mapping):
        raise ValueError("comparator payload is not complete")
    history = result.get("adapt_history")
    if not isinstance(history, list) or not history:
        raise ValueError("comparator payload has no completed history")
    if not all(isinstance(row, Mapping) for row in history):
        raise TypeError("comparator history contains a non-object row")
    return [row for row in history if isinstance(row, Mapping)]


def _history_errors(
    payload: Mapping[str, Any], *, method: str
) -> tuple[list[Mapping[str, Any]], list[float]]:
    if method == "snake":
        history, exact = _snake_history(payload)
        errors: list[float] = []
        for index, row in enumerate(history, start=1):
            value = row.get("delta_abs_current")
            if value is None:
                energy = row.get("energy_after_opt")
                if energy is None or exact is None:
                    raise ValueError(f"SNAKE row {index} lacks same-cutoff error")
                value = abs(_finite(energy, label=f"SNAKE row {index} energy") - exact)
            errors.append(abs(_finite(value, label=f"SNAKE row {index} error")))
        return history, errors

    history = _comparator_history(payload)
    result = payload["result"]
    exact_raw = result.get("same_cutoff_exact_gs_energy")
    exact = None if exact_raw is None else _finite(exact_raw, label="comparator exact energy")
    errors = []
    for index, row in enumerate(history, start=1):
        value = row.get("abs_delta_e_same_cutoff_after")
        if value is None:
            value = row.get("abs_delta_e_after")
        if value is None:
            energy = row.get("energy_after")
            if energy is None or exact is None:
                raise ValueError(f"comparator row {index} lacks same-cutoff error")
            value = abs(_finite(energy, label=f"comparator row {index} energy") - exact)
        errors.append(abs(_finite(value, label=f"comparator row {index} error")))
    return history, errors


def select_plateau_prefix(
    payload: Mapping[str, Any], *, method: str, relative_tolerance: float = RELATIVE_TOLERANCE
) -> dict[str, Any]:
    history, errors = _history_errors(payload, method=method)
    selected = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=index,
                absolute_energy_error=error,
            )
            for index, error in enumerate(errors, start=1)
        ),
        relative_tolerance=relative_tolerance,
    )
    zero_index = selected.selected_trace_index
    row = history[zero_index]
    k_pl = selected.controller_round
    outer_iteration = row.get("outer_iteration")
    if outer_iteration is None:
        iteration = row.get("iteration")
        outer_iteration = k_pl if iteration is None else int(iteration) + 1
    return {
        "history_position": k_pl,
        "k_pl": k_pl,
        "outer_iteration": int(outer_iteration),
        "horizon": selected.horizon_controller_rounds,
        "error": selected.absolute_energy_error,
        "best_observed_error": selected.best_observed_error,
        "threshold": selected.selection_threshold,
    }


def _metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return qiskit_cost_fields(payload)


def _prune_accepted_before(history: Sequence[Mapping[str, Any]], k_pl: int) -> bool:
    for row in history[: int(k_pl)]:
        prune = row.get("post_admission_prune")
        if isinstance(prune, Mapping) and int(prune.get("accepted_count") or 0) > 0:
            return True
    return False


def _candidate_representation_from_route_id(route_id: str) -> str:
    route_key = str(route_id).strip().lower()
    if "macro" in route_key:
        return SNAKE_REPRESENTATION_INTACT_MACRO
    # Every non-macro Paper-I SNAKE route in this tracker uses the projected
    # child representation, including older route IDs that predate the
    # explicit ``projected``/``singleton`` spelling.
    return SNAKE_REPRESENTATION_PROJECTED_SINGLETON


def _signed_checkpoint_estimator_work(
    checkpoint: Mapping[str, Any],
    *,
    outer_iteration: int,
) -> dict[str, Any] | None:
    """Return the closed cumulative runtime ledger for one signed prefix.

    Older signed checkpoints may predate the estimator-ledger receipt.  Those
    artifacts retain the legacy deterministic reconstruction below.  A receipt
    that is present but malformed fails closed instead of silently falling back
    to reconstructed work.
    """

    if "estimator_ledger_receipt" not in checkpoint:
        return None
    receipt = checkpoint.get("estimator_ledger_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("signed SNAKE checkpoint has a malformed estimator ledger receipt")
    if (
        receipt.get("schema") != "paper_i_active_prefix_estimator_ledger_receipt_v1"
        or receipt.get("status") != "complete"
        or int(receipt.get("outer_iteration") or 0) != int(outer_iteration)
        or receipt.get("canonical_same_state_deduplication_active") is not True
    ):
        raise ValueError("signed SNAKE checkpoint estimator ledger identity drift")
    cumulative_raw = receipt.get("cumulative_raw_occurrences")
    if not isinstance(cumulative_raw, Mapping):
        raise ValueError("signed SNAKE checkpoint lacks cumulative raw occurrences")
    components = cumulative_raw.get("components")
    required = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    if not isinstance(components, Mapping) or any(name not in components for name in required):
        raise ValueError("signed SNAKE checkpoint ledger lacks component closure")
    raw_components = {name: int(components[name]) for name in required}
    if any(value < 0 for value in raw_components.values()):
        raise ValueError("signed SNAKE checkpoint ledger contains a negative component")
    raw_total = int(cumulative_raw.get("total", -1))
    if raw_total < 0 or sum(raw_components.values()) != raw_total:
        raise ValueError("signed SNAKE checkpoint raw estimator ledger does not close")
    cumulative_unique = receipt.get("cumulative_unique_primitives")
    if not isinstance(cumulative_unique, Mapping):
        raise ValueError("signed SNAKE checkpoint lacks unique-identity diagnostic")
    return {
        "raw_total": int(raw_total),
        "raw_components": dict(raw_components),
        "full_receipt": dict(receipt),
        "receipt": {
            "schema": receipt.get("schema"),
            "status": receipt.get("status"),
            "outer_iteration": int(receipt["outer_iteration"]),
            "checkpoint_kind": receipt.get("checkpoint_kind"),
            "checkpoint_sequence": receipt.get("checkpoint_sequence"),
            "branch_id": receipt.get("branch_id"),
            "parent_branch_id": receipt.get("parent_branch_id"),
            "canonical_same_state_deduplication_active": True,
            "cumulative_raw_occurrences": {
                "total": int(raw_total),
                "components": dict(raw_components),
            },
            "cumulative_unique_primitives": dict(cumulative_unique),
        },
    }


def _snake_prefix(
    payload: Mapping[str, Any],
    *,
    selection: Mapping[str, Any],
    source: Mapping[str, Any],
    route_id: str,
    fallback_source_kind: str = "paper_i_hh_nonpruned_history_plateau_prefix",
) -> dict[str, Any]:
    from pipelines.exact_bench.paper_i_hh_recovery_prefix_qiskit_sidecar import (
        CHECKPOINT_SCHEMA,
        compile_historical_displayed_convention,
        derive_execution_order_repaired_checkpoint,
        reconstruct_reference_state,
        resolve_active_prefix_checkpoint,
        validate_active_prefix_checkpoint,
    )
    from pipelines.exact_bench.paper_i_s_alg_accounting import (
        snake_clean_prefix_work,
    )
    from pipelines.exact_bench.snake_table_i_measurement_work import (
        snake_algorithmic_work_from_payload,
    )
    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        compile_table_i_pauli_label_groups,
        pauli_label_groups_from_ansatz_terms,
    )
    from pipelines.reporting.build_paper_i_selected_prefix_qiskit_sidecar import (
        reconstruct_prefix_ansatz,
    )

    k_pl = int(selection["k_pl"])
    outer_iteration = int(selection["outer_iteration"])
    history, _exact = _snake_history(payload)
    checkpoint_mode = "signed_checkpoint"
    repair: dict[str, Any] | None = None
    checkpoint_sha256: str | None = None
    strict_replay: dict[str, Any] | None = None
    signed_checkpoint: Mapping[str, Any] | None = None
    try:
        resolution = resolve_active_prefix_checkpoint(
            payload,
            outer_iteration=outer_iteration,
            checkpoint_kind="post_admission_prune",
        )
        checkpoint = resolution.checkpoint
        checkpoint_sha256 = str(checkpoint.get("checkpoint_sha256") or "")
        strict = checkpoint.get("strict_replay")
        if not isinstance(strict, Mapping) or strict.get("passed") is not True:
            raise ValueError("selected SNAKE checkpoint lacks a passing strict replay receipt")
        strict_replay = {
            "passed": True,
            "fidelity": _finite(strict.get("fidelity"), label="strict replay fidelity"),
            "phase_aligned_l2": _finite(
                strict.get("phase_aligned_l2"), label="strict replay phase-aligned L2"
            ),
        }
        try:
            validated = validate_active_prefix_checkpoint(
                checkpoint,
                expected_outer_iteration=outer_iteration,
            )
            repair = {"status": "not_required", "substantive_term_changes": False}
        except ValueError:
            repaired, repair = derive_execution_order_repaired_checkpoint(
                checkpoint,
                expected_outer_iteration=outer_iteration,
            )
            validated = validate_active_prefix_checkpoint(
                repaired,
                expected_outer_iteration=outer_iteration,
            )
        reference_state, reference_receipt = reconstruct_reference_state(
            payload,
            num_qubits=validated.num_qubits,
        )
        compiled = compile_historical_displayed_convention(
            validated,
            reference_state=reference_state,
        )
        active_depth = int(validated.checkpoint["active_ansatz_depth"])
        signed_checkpoint = validated.checkpoint
        prefix_receipt = {
            "mode": checkpoint_mode,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_locations": list(resolution.locations),
            "repair": repair,
            "strict_replay": strict_replay,
            "reference_state": reference_receipt,
        }
    except ValueError as exc:
        if _prune_accepted_before(history, k_pl):
            raise ValueError(
                "pruned SNAKE prefix cannot fall back to admission-history reconstruction"
            ) from exc
        checkpoint_mode = "nonpruned_history_reconstruction"
        labels, ops, replay = reconstruct_prefix_ansatz(
            payload,
            history_position=k_pl,
            source_path=None,
        )
        groups = pauli_label_groups_from_ansatz_terms(ops)
        if not groups:
            raise ValueError("reconstructed SNAKE prefix is empty") from exc
        num_qubits = len(groups[0][0])
        reference_state, reference_receipt = reconstruct_reference_state(
            payload,
            num_qubits=num_qubits,
        )
        compiled_raw = compile_table_i_pauli_label_groups(
            pauli_label_groups=groups,
            num_qubits=num_qubits,
            reference_state=reference_state,
            source_kind=fallback_source_kind,
        )
        compiled = {
            "identity": _table_i_compile_convention(),
            "status": "ok",
            "metrics": _metrics(compiled_raw),
            "raw_compile_payload": compiled_raw,
        }
        active_depth = len(labels)
        prefix_receipt = {
            "mode": checkpoint_mode,
            "checkpoint_schema": None,
            "checkpoint_sha256": None,
            "checkpoint_error": str(exc),
            "repair": None,
            "strict_replay": None,
            "history_reconstruction": replay,
            "reference_state": reference_receipt,
        }

    signed_work = (
        _signed_checkpoint_estimator_work(
            signed_checkpoint,
            outer_iteration=outer_iteration,
        )
        if signed_checkpoint is not None
        else None
    )
    clean_work: dict[str, Any] | None = None
    if (
        signed_work is not None
        and route_id in PAPER_I_CLEAN_S_ALG_SNAKE_REPRESENTATIONS
    ):
        clean_work = snake_clean_prefix_work(
            history=history,
            accepted_prefix_length=k_pl,
            representation=PAPER_I_CLEAN_S_ALG_SNAKE_REPRESENTATIONS[route_id],
            estimator_ledger_receipt=signed_work["full_receipt"],
        )
        s_alg = clean_work["S_alg"]
        components = clean_work["components"]
        s_alg_scope = clean_work["scope"]
        s_alg_status = "clean_algorithm_recount_closed_signed_prefix"
        prefix_receipt["estimator_ledger_receipt"] = signed_work["receipt"]
        prefix_receipt["S_alg_recount"] = clean_work
    elif signed_work is not None:
        cumulative_unique = signed_work["receipt"].get(
            "cumulative_unique_primitives"
        )
        if not isinstance(cumulative_unique, Mapping):
            raise ValueError(
                "out-of-scope signed SNAKE route lacks its historical "
                "unique-primitive receipt"
            )
        s_alg = cumulative_unique.get("S_alg")
        components = cumulative_unique.get("components")
        if s_alg is None or not isinstance(components, Mapping):
            raise ValueError(
                "out-of-scope signed SNAKE historical receipt is incomplete"
            )
        s_alg_scope = "historical unique-primitive support route"
        s_alg_status = "outside_clean_append_snake_main_result_scope"
        prefix_receipt["estimator_ledger_receipt"] = signed_work["receipt"]
    else:
        work, work_audit = snake_algorithmic_work_from_payload(
            payload,
            scope="display_prefix",
            history_position=k_pl,
            source_label=f"{source.get('path')}:{source.get('result_member')}",
            allow_terminal_scope_equivalence=False,
        )
        s_alg = work.get("S_alg")
        if s_alg is None:
            raise ValueError(f"SNAKE prefix S_alg reconstruction failed: {work_audit}")
        algorithmic = work.get("algorithmic_measurement_work")
        components = (
            algorithmic.get("components")
            if isinstance(algorithmic, Mapping)
            else None
        )
        s_alg_scope = "deterministic legacy display-prefix reconstruction"
        s_alg_status = work.get("S_alg_status")
    return {
        "active_depth": active_depth,
        "S_alg": int(s_alg),
        "S_alg_scope": s_alg_scope,
        "S_alg_components": components,
        "S_alg_receipt": clean_work,
        "S_alg_reconstruction_status": s_alg_status,
        "qiskit": _metrics(compiled),
        "qiskit_compile": {
            "identity": compiled.get(
                "identity",
                _table_i_compile_convention(),
            ),
            "optimization_level": 0,
            "seed_transpiler": 7,
            "backend": None,
            "reference_state_included": True,
            "source_kind": "SNAKE structural active prefix",
        },
        "prefix_receipt": prefix_receipt,
    }


def _reference_state_from_seed(
    seed: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    state_payload = seed.get("ansatz_input_state")
    if not isinstance(state_payload, Mapping):
        raise ValueError("comparator runtime seed lacks ansatz_input_state")
    nq = int(state_payload.get("nq_total") or 0)
    amplitudes = state_payload.get("amplitudes_qn_to_q0")
    if nq <= 0 or not isinstance(amplitudes, Mapping) or not amplitudes:
        raise ValueError("comparator runtime seed has invalid reference amplitudes")
    state = np.zeros(1 << nq, dtype=complex)
    for raw_bits, raw_amplitude in amplitudes.items():
        bits = str(raw_bits)
        if len(bits) != nq or set(bits) - {"0", "1"}:
            raise ValueError(f"invalid comparator reference bitstring {bits!r}")
        if isinstance(raw_amplitude, Mapping):
            amplitude = complex(
                float(raw_amplitude.get("re", raw_amplitude.get("real", 0.0))),
                float(raw_amplitude.get("im", raw_amplitude.get("imag", 0.0))),
            )
        elif isinstance(raw_amplitude, Sequence) and not isinstance(
            raw_amplitude, (str, bytes, bytearray)
        ):
            amplitude = complex(float(raw_amplitude[0]), float(raw_amplitude[1]))
        else:
            amplitude = complex(float(raw_amplitude), 0.0)
        state[int(bits, 2)] = amplitude
    norm = float(np.linalg.norm(state))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("comparator reference state has zero or nonfinite norm")
    return state / norm, {
        "num_qubits": nq,
        "amplitude_count": len(amplitudes),
        "input_norm": norm,
        "normalized_for_circuit": True,
        "bitstring_order": "q_(n-1)...q_0; q0 is rightmost",
    }


def _comparator_prefix(
    payload: Mapping[str, Any],
    *,
    runtime_seed: Mapping[str, Any],
    selection: Mapping[str, Any],
    representation: str,
    source_kind: str = "paper_i_hh_comparator_exact_active_plateau_prefix",
) -> dict[str, Any]:
    from pipelines.exact_bench.paper_i_s_alg_accounting import (
        append_clean_prefix_work,
    )
    from pipelines.exact_bench.table_i_qiskit_resource_compile import (
        compile_table_i_ansatz_terms,
    )
    from pipelines.reporting.build_paper_i_selected_prefix_qiskit_sidecar import (
        _ansatz_term_from_serialized,
    )

    k_pl = int(selection["k_pl"])
    history = _comparator_history(payload)
    row = history[k_pl - 1]
    checkpoint = row.get("active_prefix_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("selected comparator row lacks an active-prefix checkpoint")
    if checkpoint.get("schema") != "paper_i_comparator_active_prefix_checkpoint_v1":
        raise ValueError(f"comparator checkpoint schema drift: {checkpoint.get('schema')}")
    expected_sha = str(checkpoint.get("checkpoint_sha256") or "")
    hash_input = dict(checkpoint)
    hash_input.pop("checkpoint_sha256", None)
    observed_sha = _canonical_sha256(hash_input)
    if not expected_sha or observed_sha != expected_sha:
        raise ValueError(
            "comparator checkpoint SHA-256 mismatch: "
            f"embedded={expected_sha}, observed={observed_sha}"
        )
    if int(checkpoint.get("outer_iteration") or -1) != k_pl:
        raise ValueError("comparator checkpoint outer iteration disagrees with k_pl")
    audit = checkpoint.get("sector_padding_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("comparator checkpoint lacks sector/padding audit")
    if audit.get("sector_leak_flag") is True or audit.get("boson_truncation_leak_flag") is True:
        raise ValueError("comparator checkpoint failed sector/padding leakage audit")

    order = checkpoint.get("active_operator_order")
    operators = checkpoint.get("active_operators")
    if not isinstance(order, list) or not isinstance(operators, list):
        raise ValueError("comparator checkpoint lacks ordered active operators")
    if len(order) != len(operators) or len(order) != int(
        checkpoint.get("active_ansatz_depth") or -1
    ):
        raise ValueError("comparator checkpoint active depth/order mismatch")
    ops = []
    for position, (label, operator) in enumerate(zip(order, operators, strict=True)):
        if not isinstance(operator, Mapping):
            raise TypeError(f"comparator active operator {position} is not an object")
        if str(operator.get("label")) != str(label):
            raise ValueError(f"comparator operator-order mismatch at position {position}")
        ops.append(
            _ansatz_term_from_serialized(
                label=str(label),
                terms=operator.get("pauli_terms") or [],
                execution_mode=str(operator.get("execution_mode") or ""),
            )
        )
    reference_state, reference_receipt = _reference_state_from_seed(runtime_seed)
    compiled = compile_table_i_ansatz_terms(
        ops=ops,
        num_qubits=int(reference_receipt["num_qubits"]),
        reference_state=reference_state,
        source_kind=source_kind,
    )
    receipts = payload["result"].get("estimator_call_round_receipts")
    if not isinstance(receipts, list) or len(receipts) < k_pl:
        raise ValueError("comparator estimator round receipts do not cover k_pl")
    round_receipt = receipts[k_pl - 1]
    if not isinstance(round_receipt, Mapping) or round_receipt.get("prefix_closed") is not True:
        raise ValueError("comparator plateau estimator prefix is not closed")
    if int(round_receipt.get("iteration") or 0) != k_pl - 1:
        raise ValueError("comparator estimator receipt iteration disagrees with k_pl")
    cumulative_occurrences = round_receipt.get(
        "cumulative_occurrence_summary"
    )
    if not isinstance(cumulative_occurrences, Mapping):
        raise ValueError(
            "comparator plateau estimator receipt lacks cumulative occurrences"
        )
    algorithm_id = str(checkpoint.get("algorithm_id") or "")
    clean_work: dict[str, Any] | None
    if "append" in algorithm_id.lower():
        redundant_verifier_count = (
            _append_redundant_post_refit_verification_count(
                cumulative_occurrences,
                accepted_prefix_length=k_pl,
            )
        )
        clean_work = append_clean_prefix_work(
            accepted_prefix_length=k_pl,
            cumulative_occurrence_summary=cumulative_occurrences,
            redundant_post_refit_verification_count=redundant_verifier_count,
            representation=str(representation),
        )
        s_alg = int(clean_work["S_alg"])
        s_alg_scope = str(clean_work["scope"])
        s_alg_components = clean_work["components"]
        s_alg_status = "clean_algorithm_recount_closed_comparator_prefix"
    else:
        # Geo-ADAPT is retained as a non-active support comparator and is
        # outside this Append/SNAKE correction.  Preserve its historical
        # receipt rather than silently applying the Append formula.
        cumulative_unique = round_receipt.get("cumulative_unique_summary")
        if (
            not isinstance(cumulative_unique, Mapping)
            or cumulative_unique.get("S_alg") is None
        ):
            raise ValueError(
                "non-Append comparator lacks its historical cumulative receipt"
            )
        clean_work = None
        s_alg = int(cumulative_unique["S_alg"])
        s_alg_scope = "historical unique-primitive support comparator"
        s_alg_components = cumulative_unique.get("components")
        s_alg_status = "outside_append_snake_correction_scope"
    return {
        "active_depth": int(checkpoint["active_ansatz_depth"]),
        "S_alg": int(s_alg),
        "S_alg_scope": s_alg_scope,
        "S_alg_components": s_alg_components,
        "S_alg_receipt": clean_work,
        "S_alg_reconstruction_status": s_alg_status,
        "qiskit": _metrics(compiled),
        "qiskit_compile": {
            "identity": compiled.get("compile_convention"),
            "optimization_level": compiled.get("qiskit_transpile_optimization_level"),
            "seed_transpiler": compiled.get("qiskit_transpile_seed"),
            "backend": None,
            "reference_state_included": compiled.get("compiled_circuit_scope")
            == "ansatz_circuit_including_reference_state",
            "source_kind": compiled.get("compiled_resource_source_kind"),
            "execution_aware": True,
        },
        "prefix_receipt": {
            "mode": "embedded_comparator_active_prefix_checkpoint",
            "checkpoint_schema": checkpoint.get("schema"),
            "checkpoint_sha256": expected_sha,
            "sector_padding_audit": audit,
            "reference_state": reference_receipt,
            "estimator_round_receipt_schema": round_receipt.get("schema"),
            "estimator_prefix_closed": True,
            "S_alg_recount": clean_work,
        },
    }


def _cached_rows(output_json: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not output_json.is_file():
        return {}
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        return {}
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}
    return {
        (str(row.get("route_id")), str(row.get("regime"))): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }


def _comparator_plateau_prefix_streaming(
    *,
    source: Mapping[str, Any],
    result: Mapping[str, Any],
    route_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compile one comparator plateau prefix without materializing result.json."""

    from pipelines.reporting.build_paper_i_hh_comparator_tracking_summary import (
        _tar_array_item,
        _tar_json_member,
    )

    trajectory = result.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("completed comparator row lacks a trajectory")
    errors: list[float] = []
    rounds: list[int] = []
    for position, point in enumerate(trajectory, start=1):
        if not isinstance(point, Mapping):
            raise TypeError(f"comparator trajectory row {position} is not an object")
        error = _finite(point.get("error"), label=f"trajectory row {position} error")
        round_id = int(point.get("round") or position)
        if round_id != position:
            raise ValueError(
                "bounded comparator plateau reconstruction requires ordered "
                f"rounds 1..N; observed round {round_id} at position {position}"
            )
        errors.append(abs(error))
        rounds.append(round_id)

    selected = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=round_id,
                absolute_energy_error=error,
            )
            for round_id, error in zip(rounds, errors, strict=True)
        ),
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    zero_index = selected.selected_trace_index
    k_pl = selected.controller_round
    selected_error = selected.absolute_energy_error

    path = _source_path(source)
    observed_sha = _sha256_path(path)
    member_name = str(source.get("member") or "")
    if not member_name:
        raise ValueError("streaming comparator source lacks a result member")

    history_row = _tar_array_item(
        path,
        member_name=member_name,
        array_key="adapt_history",
        zero_index=zero_index,
    )
    receipt_row = _tar_array_item(
        path,
        member_name=member_name,
        array_key="estimator_call_round_receipts",
        zero_index=zero_index,
    )
    seed_member = member_name.rsplit("/", 1)[0] + "/runtime_seed.json"
    runtime_seed = _tar_json_member(path, member_name=seed_member)

    source_error = history_row.get("abs_delta_e_same_cutoff_after")
    if source_error is None:
        source_error = history_row.get("abs_delta_e_after")
    if source_error is None or not math.isclose(
        abs(_finite(source_error, label="streamed comparator plateau error")),
        selected_error,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("streamed comparator plateau row disagrees with tracker trajectory")

    minimal_payload = {
        "status": "completed",
        "result": {
            "adapt_history": [{} for _ in range(zero_index)] + [history_row],
            "estimator_call_round_receipts": [{} for _ in range(zero_index)]
            + [receipt_row],
        },
    }
    selection = {
        "history_position": k_pl,
        "k_pl": k_pl,
        "outer_iteration": rounds[zero_index],
        "horizon": selected.horizon_controller_rounds,
        "error": selected_error,
        "best_observed_error": selected.best_observed_error,
        "threshold": selected.selection_threshold,
    }
    prefix = _comparator_prefix(
        minimal_payload,
        runtime_seed=runtime_seed,
        selection=selection,
        representation=_candidate_representation_from_route_id(route_id),
    )
    source_receipt = {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": observed_sha,
        "result_member": member_name,
        "runtime_seed_member": seed_member,
        "streaming_bounded_memory": True,
    }
    return {**selection, **prefix}, source_receipt


def _cost_arm_plateau_selection(result: Mapping[str, Any]) -> dict[str, Any]:
    """Select the plateau directly from the validated compact trajectory."""

    trajectory_role = str(
        result.get("trajectory_role") or "selected_terminal_path_v1"
    )
    if trajectory_role == "controller_frontier_non_selected_v1":
        trajectory = result.get("selected_winner_history")
        expected_horizon = int(result.get("selected_terminal", {}).get("round") or 0)
    else:
        trajectory = result.get("trajectory")
        expected_horizon = 50
    if not isinstance(trajectory, list) or len(trajectory) != expected_horizon:
        raise ValueError("cost-arm tracker row lacks its validated selected history")
    errors = [
        _finite(point.get("error"), label=f"cost-arm round {index} error")
        for index, point in enumerate(trajectory, start=1)
        if isinstance(point, Mapping)
    ]
    if len(errors) != expected_horizon:
        raise ValueError("cost-arm compact trajectory contains malformed rows")
    selection = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=index,
                absolute_energy_error=error,
            )
            for index, error in enumerate(errors, start=1)
        ),
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    zero_index = selection.selected_trace_index
    point = trajectory[zero_index]
    assert isinstance(point, Mapping)
    round_id = int(point.get("round") or 0)
    if round_id != selection.controller_round:
        raise ValueError("cost-arm compact trajectory round order drift")
    return {
        "history_position": selection.controller_round,
        "k_pl": selection.controller_round,
        "outer_iteration": round_id,
        "horizon": selection.horizon_controller_rounds,
        "trajectory_scope": (
            "selected_terminal_winner_history"
            if trajectory_role == "controller_frontier_non_selected_v1"
            else "selected_terminal_path"
        ),
        "error": selection.absolute_energy_error,
        "best_observed_error": selection.best_observed_error,
        "threshold": selection.selection_threshold,
    }


def _cost_arm_terminal_prefix(
    *,
    source: Mapping[str, Any],
    result: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the small signed terminal checkpoint when it is the selected prefix.

    Earlier pruned prefixes cannot be reconstructed from the terminal ansatz.
    Such a selection fails closed instead of reopening the raw beam archive.
    """

    path = _source_path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("cost-arm executable checkpoint root is not an object")
    repair = payload.get("repair")
    source_identity = payload.get("source")
    checkpoint = payload.get("repaired_checkpoint")
    if not all(
        isinstance(record, Mapping)
        for record in (repair, source_identity, checkpoint)
    ):
        raise ValueError("cost-arm executable checkpoint is malformed")
    ledger = checkpoint.get("estimator_ledger_receipt")
    if not isinstance(ledger, Mapping):
        raise ValueError("cost-arm executable checkpoint lacks ledger identity")
    selected_round = int(selection["outer_iteration"])
    checkpoint_round = int(checkpoint.get("outer_iteration") or 0)
    if selected_round != checkpoint_round:
        raise ValueError(
            "cost-arm selected prefix is not the validated terminal checkpoint; "
            "raw beam archive remains closed"
        )
    active_depth = int(checkpoint.get("active_ansatz_depth") or -1)
    trajectory = (
        result.get("selected_winner_history")
        if result.get("trajectory_role") == "controller_frontier_non_selected_v1"
        else result.get("trajectory")
    )
    point = (
        trajectory[int(selection["history_position"]) - 1]
        if isinstance(trajectory, list)
        else None
    )
    qiskit = result.get("terminal_prefix_qiskit")
    expected_route = str(result.get("route_contract_sha256") or "")
    if (
        payload.get("schema") != "paper_i_checkpoint_execution_order_repair_v1"
        or repair.get("status") not in {"repaired_permutation_only", "not_required"}
        or repair.get("substantive_term_changes") is not False
        or checkpoint.get("schema") != "paper_i_signed_active_prefix_checkpoint_v1"
        or checkpoint.get("sr_route_profile_contract_sha256") != expected_route
        or ledger.get("status") != "complete"
        or int(ledger.get("outer_iteration") or 0) != checkpoint_round
        or not isinstance(point, Mapping)
        or int(point.get("active_depth") or -1) != active_depth
        or not isinstance(qiskit, Mapping)
    ):
        raise ValueError("cost-arm terminal checkpoint/history/ledger identity drift")
    for field in (
        "ordered_active_operator_labels",
        "ordered_active_operators",
        "signed_unwrapped_logical_parameters",
    ):
        values = checkpoint.get(field)
        if not isinstance(values, list) or len(values) != active_depth:
            raise ValueError(f"cost-arm terminal checkpoint lacks complete {field}")
    runtime = checkpoint.get("signed_unwrapped_runtime_parameters")
    if not isinstance(runtime, list) or not runtime:
        raise ValueError("cost-arm terminal checkpoint lacks signed runtime parameters")
    s_alg = (
        int(result.get("s_alg") or -1)
        if result.get("trajectory_role") == "controller_frontier_non_selected_v1"
        else int(point.get("S_alg") or -1)
    )
    if s_alg < 0:
        raise ValueError("cost-arm compact terminal row lacks cumulative S_alg")
    metrics = _metrics(qiskit)
    source_receipt = {
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": _sha256_path(path),
        "result_member": None,
        "source_kind": "validated execution-order-repaired signed terminal checkpoint",
        "raw_archive_reopened": False,
        "checkpoint_schema": checkpoint.get("schema"),
        "checkpoint_outer_iteration": checkpoint_round,
        "checkpoint_ledger_status": ledger.get("status"),
        "trajectory_receipt_sha256": source.get("trajectory_receipt_sha256"),
        "trajectory_receipt_path": source.get("trajectory_receipt_path"),
    }
    return {
        "active_depth": active_depth,
        "S_alg": s_alg,
        "S_alg_scope": "validated cumulative all-branch beam/prune work",
        "S_alg_components": None,
        "S_alg_reconstruction_status": "compact_all_branch_receipt",
        "qiskit": metrics,
        "qiskit_compile": {
            "identity": _table_i_compile_convention(),
            "optimization_level": 0,
            "seed_transpiler": 7,
            "backend": None,
            "reference_state_included": True,
            "source_kind": "validated terminal-prefix Qiskit sidecar",
        },
        "prefix_receipt": {
            "mode": "signed_terminal_checkpoint_validated",
            "checkpoint_schema": checkpoint.get("schema"),
            "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
            "repair": dict(repair),
            "estimator_ledger_receipt": {
                "status": ledger.get("status"),
                "outer_iteration": ledger.get("outer_iteration"),
                "branch_id": ledger.get("branch_id"),
                "parent_branch_id": ledger.get("parent_branch_id"),
            },
            "raw_archive_reopened": False,
        },
    }, source_receipt


def build_plateau_costs(*, tracker_json: Path, output_json: Path) -> dict[str, Any]:
    tracker_path = tracker_json.resolve()
    tracker = json.loads(tracker_path.read_text(encoding="utf-8"))
    routes = tracker.get("routes")
    if not isinstance(routes, list):
        raise TypeError("tracker JSON has no routes list")
    cache = _cached_rows(output_json)
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for route in routes:
        if not isinstance(route, Mapping):
            continue
        route_id = str(route.get("id"))
        results = route.get("results")
        if not isinstance(results, Mapping):
            raise TypeError(f"route {route_id} has no results mapping")
        for regime, result in results.items():
            if not isinstance(result, Mapping) or not result.get("trajectory"):
                unresolved.append(
                    {
                        "route_id": route_id,
                        "regime": str(regime),
                        "status": str(result.get("status") if isinstance(result, Mapping) else "missing"),
                        "reason": "no completed validated trajectory in tracker",
                    }
                )
                continue
            source = result.get("source")
            if not isinstance(source, Mapping):
                raise ValueError(f"completed row {route_id}/{regime} has no source")
            method = "comparator" if route_id in COMPARATOR_ROUTE_IDS else "snake"
            key = (route_id, str(regime))
            cached = cache.get(key)
            cached_source = cached.get("source") if isinstance(cached, Mapping) else None
            declared_source_sha = str(source.get("sha256") or "")
            declared_source_path = str(source.get("path") or "")
            declared_trajectory_sha = source.get("trajectory_receipt_sha256")
            clean_receipt_required = bool(
                route_id in PAPER_I_CLEAN_S_ALG_SNAKE_REPRESENTATIONS
                or route_id.startswith("append_adapt_")
            )
            cached_clean_receipt = (
                cached.get("S_alg_receipt")
                if isinstance(cached, Mapping)
                else None
            )
            if (
                isinstance(cached, Mapping)
                and cached.get("status") == "complete"
                and isinstance(cached_source, Mapping)
                and declared_source_sha
                and cached_source.get("sha256") == declared_source_sha
                and cached_source.get("path") == declared_source_path
                and cached_source.get("trajectory_receipt_sha256")
                == declared_trajectory_sha
                and cached.get("rule", {}).get("id") == RULE_ID
                and (
                    not clean_receipt_required
                    or (
                        isinstance(cached_clean_receipt, Mapping)
                        and cached_clean_receipt.get("schema")
                        == PAPER_I_S_ALG_ACCOUNTING_SCHEMA
                    )
                )
            ):
                rows.append(dict(cached))
                print(f"reuse {route_id}/{regime} k={cached.get('k_pl')}", flush=True)
                continue
            if method == "comparator":
                streamed, source_receipt = _comparator_plateau_prefix_streaming(
                    source=source,
                    result=result,
                    route_id=route_id,
                )
                rows.append(
                    {
                        "route_id": route_id,
                        "regime": str(regime),
                        "status": "complete",
                        "rule": {
                            "id": RULE_ID,
                            "relative_tolerance": RELATIVE_TOLERANCE,
                            "reporting_only": True,
                        },
                        **streamed,
                        "source": source_receipt,
                        "prefix_source": source_receipt,
                    }
                )
                print(
                    f"compile {route_id}/{regime} k={streamed['k_pl']} "
                    f"of {streamed['horizon']} (bounded-memory)",
                    flush=True,
                )
                _write_json_atomic(
                    output_json,
                    {
                        "schema": SCHEMA,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "rule": {"id": RULE_ID},
                        "rows": rows,
                        "unresolved": unresolved,
                        "summary": {
                            "status": "in_progress",
                            "complete_prefix_count": len(rows),
                        },
                    },
                )
                gc.collect()
                continue
            if route_id in COST_ARM_ROUTE_IDS:
                selection = _cost_arm_plateau_selection(result)
                try:
                    prefix, source_receipt = _cost_arm_terminal_prefix(
                        source=source,
                        result=result,
                        selection=selection,
                    )
                except ValueError as exc:
                    if "is not the validated terminal checkpoint" not in str(exc):
                        raise
                    unresolved.append(
                        {
                            "route_id": route_id,
                            "regime": str(regime),
                            "status": "exact_prefix_unavailable",
                            "reason": str(exc),
                            **selection,
                        }
                    )
                    print(
                        f"defer {route_id}/{regime} k={selection['k_pl']}: "
                        "only the terminal checkpoint is executable",
                        flush=True,
                    )
                    continue
                row = {
                    "route_id": route_id,
                    "regime": str(regime),
                    "status": "complete",
                    "rule": {
                        "id": RULE_ID,
                        "relative_tolerance": RELATIVE_TOLERANCE,
                        "reporting_only": True,
                    },
                    **selection,
                    **prefix,
                    "source": source_receipt,
                    "prefix_source": source_receipt,
                }
                rows.append(row)
                print(
                    f"compile {route_id}/{regime} k={selection['k_pl']} "
                    "from validated terminal checkpoint",
                    flush=True,
                )
                _write_json_atomic(
                    output_json,
                    {
                        "schema": SCHEMA,
                        "created_utc": datetime.now(timezone.utc).isoformat(),
                        "rule": {"id": RULE_ID},
                        "rows": rows,
                        "unresolved": unresolved,
                        "summary": {
                            "status": "in_progress",
                            "complete_prefix_count": len(rows),
                        },
                    },
                )
                gc.collect()
                continue
            payload, runtime_seed, source_receipt = _read_source_result(
                source,
                need_runtime_seed=False,
            )
            selection = select_plateau_prefix(payload, method=method)
            print(
                f"compile {route_id}/{regime} k={selection['k_pl']} "
                f"of {selection['horizon']}",
                flush=True,
            )
            prefix_source_receipt = source_receipt
            prefix_payload = payload
            if (
                route_id == "legacy_no_ordinary_novelty_nph2_4"
                and str(regime) == "strong_weak_u8"
                and int(selection["k_pl"]) <= 30
            ):
                if not LEGACY_OFF_SW_PREFIX_SOURCE.is_file():
                    raise FileNotFoundError(LEGACY_OFF_SW_PREFIX_SOURCE)
                prefix_payload, _seed, prefix_source_receipt = _read_source_result(
                    {
                        "path": str(LEGACY_OFF_SW_PREFIX_SOURCE.relative_to(REPO_ROOT)),
                        "sha256": _sha256_path(LEGACY_OFF_SW_PREFIX_SOURCE),
                    },
                    need_runtime_seed=False,
                )
                prefix_selection = select_plateau_prefix(
                    prefix_payload,
                    method="snake",
                )
                prefix_errors = _history_errors(prefix_payload, method="snake")[1]
                selected_prefix_error = prefix_errors[int(selection["k_pl"]) - 1]
                if not math.isclose(
                    selected_prefix_error,
                    float(selection["error"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        "legacy strong-weak authenticated prefix disagrees with "
                        "the displayed continuation trajectory"
                    )
                if int(prefix_selection["horizon"]) < int(selection["k_pl"]):
                    raise ValueError("legacy strong-weak prefix source is too short")
            prefix = _snake_prefix(
                prefix_payload,
                selection=selection,
                source=prefix_source_receipt,
                route_id=route_id,
            )
            row = {
                "route_id": route_id,
                "regime": str(regime),
                "status": "complete",
                "rule": {
                    "id": RULE_ID,
                    "relative_tolerance": RELATIVE_TOLERANCE,
                    "reporting_only": True,
                },
                **selection,
                **prefix,
                "source": source_receipt,
                "prefix_source": prefix_source_receipt,
            }
            rows.append(row)
            _write_json_atomic(
                output_json,
                {
                    "schema": SCHEMA,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "rule": {"id": RULE_ID},
                    "rows": rows,
                    "unresolved": unresolved,
                    "summary": {
                        "status": "in_progress",
                        "complete_prefix_count": len(rows),
                    },
                },
            )
            del payload, runtime_seed, prefix
            del prefix_payload
            gc.collect()

    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tracker": {
            "path": str(tracker_path.relative_to(REPO_ROOT)),
            "sha256": _sha256_path(tracker_path),
            "schema": tracker.get("schema"),
        },
        "rule": {
            "id": RULE_ID,
            "relative_tolerance": RELATIVE_TOLERANCE,
            "definition": (
                "first history prefix with same-cutoff absolute error <= "
                "1.10 * minimum error over the complete stored trajectory"
            ),
            "reporting_only": True,
        },
        "compile_policy": {
            "identity": _table_i_compile_convention(),
            "basis_gate_family": "Paper-I backend-free Table-I",
            "optimization_level": 0,
            "seed_transpiler": 7,
            "reference_state_included": True,
            "snake_synthesis": "historical structural Pauli-label-group convention",
            "comparator_synthesis": "execution-aware coefficient-bearing convention",
        },
        "rows": sorted(rows, key=lambda row: (row["route_id"], row["regime"])),
        "unresolved": sorted(
            unresolved,
            key=lambda row: (row["route_id"], row["regime"]),
        ),
        "summary": {
            "complete_prefix_count": len(rows),
            "unresolved_count": len(unresolved),
        },
    }
    _write_json_atomic(output_json, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-json", type=Path, default=DEFAULT_TRACKER_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()
    payload = build_plateau_costs(
        tracker_json=args.tracker_json,
        output_json=args.output_json,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
