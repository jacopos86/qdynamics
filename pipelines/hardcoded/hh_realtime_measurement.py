#!/usr/bin/env python3
"""Measurement/cache helpers for realtime checkpoint control."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from pipelines.exact_bench.noise_oracle_runtime import (
    OracleConfig,
    _LOCAL_READOUT_STRATEGIES,
    normalize_mitigation_config,
    normalize_symmetry_mitigation_config,
)
from pipelines.hardcoded.hh_continuation_scoring import (
    MeasurementCacheAudit,
    measurement_group_keys_for_term,
)
from pipelines.hardcoded.hh_continuation_types import MeasurementCacheStats
from pipelines.hardcoded.hh_fixed_manifold_observables import (
    CheckpointObservablePlan,
    ObservableSpec,
    build_checkpoint_observable_plan_from_layout,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    DerivedGeometryKey,
    GeometryValueKey,
    MeasurementTierConfig,
    OracleValueKey,
    RawGroupKey,
    SharedRawGroupKey,
    TemporalPriorKey,
    TemporalPriorRecord,
)

if TYPE_CHECKING:
    from pipelines.hardcoded.hh_fixed_manifold_measured import (
        FixedManifoldMeasuredConfig,
    )


def _qwc_compatible(lhs: str, rhs: str) -> bool:
    left = str(lhs).upper()
    right = str(rhs).upper()
    if len(left) != len(right):
        return False
    for lch, rch in zip(left, right):
        if lch == "I" or rch == "I" or lch == rch:
            continue
        return False
    return True


def _merge_qwc_basis(lhs: str, rhs: str) -> str:
    left = str(lhs).upper()
    right = str(rhs).upper()
    if len(left) != len(right):
        raise ValueError("Cannot merge QWC bases of different lengths.")
    merged: list[str] = []
    for lch, rch in zip(left, right):
        if lch == "I":
            merged.append(rch)
        elif rch == "I" or lch == rch:
            merged.append(lch)
        else:
            raise ValueError(f"Non-QWC merge attempted for {lhs!r} and {rhs!r}.")
    return "".join(merged)


def _group_observable_terms_by_qwc_basis(
    terms: Sequence[tuple[str, complex]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for label, coeff in terms:
        label_s = str(label).upper()
        placed = False
        for group in groups:
            if _qwc_compatible(str(group["basis_label"]), label_s):
                group["basis_label"] = _merge_qwc_basis(str(group["basis_label"]), label_s)
                group["terms"].append((label_s, complex(coeff)))
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "basis_label": str(label_s),
                    "terms": [(str(label_s), complex(coeff))],
                }
            )
    return groups


def _subset_parity_from_counts(counts: Mapping[str, int], classical_indices: Sequence[int]) -> float:
    indices = [int(idx) for idx in classical_indices]
    if not indices:
        return 1.0
    shots = int(sum(int(v) for v in counts.values()))
    if shots <= 0:
        raise RuntimeError("Sampler returned zero total shots.")
    acc = 0.0
    for bitstr_raw, ct in counts.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        width = max(indices) + 1
        if len(bitstr) < width:
            bitstr = bitstr.zfill(width)
        ones = sum(1 for idx in indices if bitstr[-1 - int(idx)] == "1")
        parity = -1.0 if (ones % 2) else 1.0
        acc += float(parity) * float(ct)
    return float(acc / float(shots))


def _subset_parity_from_quasi(quasi: Mapping[str, float], classical_indices: Sequence[int]) -> float:
    indices = [int(idx) for idx in classical_indices]
    if not indices:
        return 1.0
    total = 0.0
    for bitstr_raw, prob in quasi.items():
        bitstr = str(bitstr_raw).replace(" ", "")
        width = max(indices) + 1
        if len(bitstr) < width:
            bitstr = bitstr.zfill(width)
        ones = sum(1 for idx in indices if bitstr[-1 - int(idx)] == "1")
        parity = -1.0 if (ones % 2) else 1.0
        total += float(parity) * float(prob)
    return float(total)


def _weighted_mean_stderr(values: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    vals = np.asarray([float(v) for v in values], dtype=float)
    wts = np.asarray([max(1.0, float(w)) for w in weights], dtype=float)
    if vals.size <= 0:
        return 0.0, float("nan")
    mean = float(np.average(vals, weights=wts))
    if vals.size <= 1:
        return float(mean), 0.0
    var = float(np.average((vals - mean) ** 2, weights=wts))
    stdev = float(np.sqrt(max(var, 0.0)))
    n_eff = float((wts.sum() ** 2) / max(np.sum(wts**2), 1.0))
    stderr = float(stdev / np.sqrt(max(n_eff, 1.0)))
    return float(mean), float(stderr)


def _canonical_raw_group_state_key(key: RawGroupKey) -> str:
    explicit = None if key.state_key is None else str(key.state_key)
    if explicit is not None and explicit.strip() != "":
        return explicit
    return f"observable_family::{str(key.observable_family)}"


def _canonical_shared_raw_group_key(key: RawGroupKey) -> SharedRawGroupKey:
    explicit_state = key.state_key is not None and str(key.state_key).strip() != ""
    return SharedRawGroupKey(
        checkpoint_id=str(key.checkpoint_id),
        candidate_label=(
            None
            if explicit_state or key.candidate_label is None
            else str(key.candidate_label)
        ),
        position_id=(
            None
            if explicit_state or key.position_id is None
            else int(key.position_id)
        ),
        state_key=_canonical_raw_group_state_key(key),
        group_key=str(key.group_key),
    )


def _observable_spec_mean(
    *,
    raw_group_pool: "BackendScheduledRawGroupPool",
    oracle: Any,
    circuit: Any,
    spec: ObservableSpec,
    observable_family: str,
    candidate_label: str | None,
    position_id: int | None,
    state_key: str,
    min_total_shots: int,
    min_samples: int,
) -> dict[str, Any]:
    if bool(spec.is_zero) or spec.sparse_op is None:
        return {
            "mean": 0.0,
            "stderr": 0.0,
            "std": 0.0,
            "stdev": 0.0,
            "n_samples": 0,
            "aggregate": "mean",
            "backend_info": None,
            "term_payloads": [],
        }
    return raw_group_pool.estimate_observable(
        oracle=oracle,
        circuit=circuit,
        observable=spec.sparse_op,
        observable_family=str(observable_family),
        candidate_label=(None if candidate_label is None else str(candidate_label)),
        position_id=(None if position_id is None else int(position_id)),
        min_total_shots=int(min_total_shots),
        min_samples=int(min_samples),
        state_key=str(state_key),
    )


class ExactCheckpointValueCache:
    """Exact same-checkpoint cache keyed by controller geometry identity."""

    def __init__(self, *, checkpoint_id: str, grouping_mode: str) -> None:
        self._checkpoint_id = str(checkpoint_id)
        self._grouping_mode = str(grouping_mode)
        self._entries: dict[GeometryValueKey, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._extensions = 0

    def get_or_compute(
        self,
        key: GeometryValueKey,
        *,
        tier_name: str,
        compute: Callable[[], Any],
    ) -> tuple[Any, bool]:
        if str(key.checkpoint_id) != self._checkpoint_id:
            raise ValueError(
                f"ExactCheckpointValueCache key checkpoint mismatch: {key.checkpoint_id} != {self._checkpoint_id}."
            )
        if str(key.grouping_mode) != self._grouping_mode:
            raise ValueError(
                f"ExactCheckpointValueCache grouping_mode mismatch: {key.grouping_mode} != {self._grouping_mode}."
            )
        record = self._entries.get(key)
        if record is not None:
            self._hits += 1
            if str(tier_name) not in record["tiers"]:
                record["tiers"].add(str(tier_name))
                self._extensions += 1
            return record["value"], True
        self._misses += 1
        value = compute()
        self._entries[key] = {"value": value, "tiers": {str(tier_name)}}
        self._stores += 1
        return value, False

    def summary(self) -> dict[str, Any]:
        total = int(self._hits + self._misses)
        return {
            "checkpoint_id": str(self._checkpoint_id),
            "grouping_mode": str(self._grouping_mode),
            "entries": int(len(self._entries)),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "stores": int(self._stores),
            "extensions": int(self._extensions),
            "hit_rate": (float(self._hits) / float(total) if total > 0 else 0.0),
        }


class OracleCheckpointValueCache:
    """Tier-aware same-checkpoint cache for oracle-side controller evaluations."""

    def __init__(self, *, checkpoint_id: str) -> None:
        self._checkpoint_id = str(checkpoint_id)
        self._entries: dict[OracleValueKey, Any] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0

    def get_or_compute(
        self,
        key: OracleValueKey,
        *,
        compute: Callable[[], Any],
    ) -> tuple[Any, bool]:
        if str(key.checkpoint_id) != self._checkpoint_id:
            raise ValueError(
                f"OracleCheckpointValueCache key checkpoint mismatch: {key.checkpoint_id} != {self._checkpoint_id}."
            )
        if key in self._entries:
            self._hits += 1
            return self._entries[key], True
        self._misses += 1
        value = compute()
        self._entries[key] = value
        self._stores += 1
        return value, False

    def summary(self) -> dict[str, Any]:
        total = int(self._hits + self._misses)
        return {
            "checkpoint_id": str(self._checkpoint_id),
            "entries": int(len(self._entries)),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "stores": int(self._stores),
            "hit_rate": (float(self._hits) / float(total) if total > 0 else 0.0),
        }


class DerivedGeometryMemo:
    """Checkpoint-local derived geometry memo built above exact/raw sources."""

    def __init__(self, *, checkpoint_id: str) -> None:
        self._checkpoint_id = str(checkpoint_id)
        self._entries: dict[DerivedGeometryKey, Any] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0

    def get_or_compute(
        self,
        key: DerivedGeometryKey,
        *,
        compute: Callable[[], Any],
    ) -> tuple[Any, bool]:
        if str(key.checkpoint_id) != self._checkpoint_id:
            raise ValueError(
                f"DerivedGeometryMemo key checkpoint mismatch: {key.checkpoint_id} != {self._checkpoint_id}."
            )
        if key in self._entries:
            self._hits += 1
            return self._entries[key], True
        self._misses += 1
        value = compute()
        self._entries[key] = value
        self._stores += 1
        return value, False

    def summary(self) -> dict[str, Any]:
        total = int(self._hits + self._misses)
        return {
            "checkpoint_id": str(self._checkpoint_id),
            "entries": int(len(self._entries)),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "stores": int(self._stores),
            "hit_rate": (float(self._hits) / float(total) if total > 0 else 0.0),
        }


@dataclass
class BackendScheduledRawSample:
    repeat_index: int
    shots: int
    expectation: float
    counts: dict[str, int]
    basis_label: str = ""
    measured_logical_qubits: tuple[int, ...] = ()
    quasi_probs: dict[str, float] | None = None
    term_details: dict[str, Any] = field(default_factory=dict)
    readout_mitigation: dict[str, Any] = field(default_factory=dict)
    local_gate_twirling: dict[str, Any] = field(default_factory=dict)
    local_dynamical_decoupling: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendScheduledRawGroupRecord:
    key: SharedRawGroupKey
    samples: list[BackendScheduledRawSample] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        return int(len(self.samples))

    @property
    def total_shots(self) -> int:
        return int(sum(int(sample.shots) for sample in self.samples))

    def estimate(self) -> dict[str, Any]:
        if not self.samples:
            return {
                "mean": 0.0,
                "stderr": float("nan"),
                "std": 0.0,
                "stdev": 0.0,
                "n_samples": 0,
                "total_shots": 0,
                "aggregate": "mean",
            }
        vals = np.asarray([float(sample.expectation) for sample in self.samples], dtype=float)
        weights = np.asarray([max(1.0, float(sample.shots)) for sample in self.samples], dtype=float)
        mean = float(np.average(vals, weights=weights))
        if vals.size <= 1:
            stdev = 0.0
            stderr = 0.0
        else:
            var = float(np.average((vals - mean) ** 2, weights=weights))
            stdev = float(np.sqrt(max(var, 0.0)))
            n_eff = float((weights.sum() ** 2) / max(np.sum(weights**2), 1.0))
            stderr = float(stdev / np.sqrt(max(n_eff, 1.0)))
        return {
            "mean": float(mean),
            "stderr": float(stderr),
            "std": float(stdev),
            "stdev": float(stdev),
            "n_samples": int(vals.size),
            "total_shots": int(self.total_shots),
            "aggregate": "mean",
        }


class BackendScheduledRawGroupPool:
    """Same-checkpoint raw term/group reuse for backend_scheduled controller probes."""

    def __init__(self, *, checkpoint_id: str) -> None:
        self._checkpoint_id = str(checkpoint_id)
        self._records: dict[SharedRawGroupKey, BackendScheduledRawGroupRecord] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._extensions = 0

    def get_or_acquire(
        self,
        key: RawGroupKey,
        *,
        min_total_shots: int,
        min_samples: int,
        acquire_sample: Callable[[int], BackendScheduledRawSample | Mapping[str, Any]],
    ) -> tuple[BackendScheduledRawGroupRecord, bool]:
        if str(key.checkpoint_id) != self._checkpoint_id:
            raise ValueError(
                f"BackendScheduledRawGroupPool key checkpoint mismatch: {key.checkpoint_id} != {self._checkpoint_id}."
            )
        shared_key = _canonical_shared_raw_group_key(key)
        record = self._records.get(shared_key)
        reused = record is not None
        if record is None:
            self._misses += 1
            record = BackendScheduledRawGroupRecord(key=shared_key)
            self._records[shared_key] = record
            self._stores += 1
        else:
            self._hits += 1
        target_shots = max(0, int(min_total_shots))
        target_samples = max(1, int(min_samples))
        while int(record.total_shots) < target_shots or int(record.sample_count) < target_samples:
            sample_raw = acquire_sample(int(record.sample_count))
            if isinstance(sample_raw, BackendScheduledRawSample):
                sample = sample_raw
            else:
                sample = BackendScheduledRawSample(
                    repeat_index=int(sample_raw["repeat_index"]),
                    shots=int(sample_raw["shots"]),
                    expectation=float(sample_raw.get("expectation", 0.0)),
                    counts=dict(sample_raw.get("counts", {})),
                    basis_label=str(sample_raw.get("basis_label", "")),
                    measured_logical_qubits=tuple(
                        int(q) for q in sample_raw.get("measured_logical_qubits", ())
                    ),
                    quasi_probs=(
                        None
                        if sample_raw.get("quasi_probs", None) is None
                        else {
                            str(key): float(val)
                            for key, val in dict(sample_raw.get("quasi_probs", {})).items()
                        }
                    ),
                    term_details=dict(sample_raw.get("term_details", {})),
                    readout_mitigation=dict(sample_raw.get("readout_mitigation", {})),
                    local_gate_twirling=dict(sample_raw.get("local_gate_twirling", {})),
                    local_dynamical_decoupling=dict(sample_raw.get("local_dynamical_decoupling", {})),
                )
            record.samples.append(sample)
            self._extensions += 1
        return record, bool(reused)

    def estimate_observable(
        self,
        *,
        oracle: Any,
        circuit: Any,
        observable: Any,
        observable_family: str,
        candidate_label: str | None,
        position_id: int | None,
        min_total_shots: int,
        min_samples: int,
        state_key: str | None = None,
    ) -> dict[str, Any]:
        coeff_total = 0.0 + 0.0j
        variance_terms = 0.0
        term_payloads: list[dict[str, Any]] = []
        non_identity_terms: list[tuple[str, complex]] = []
        for label, coeff in observable.to_list():
            coeff_c = complex(coeff)
            label_s = str(label).upper()
            if all(ch == "I" for ch in label_s):
                coeff_total += coeff_c
                continue
            non_identity_terms.append((str(label_s), complex(coeff_c)))
        for group in _group_observable_terms_by_qwc_basis(non_identity_terms):
            basis_label = str(group["basis_label"])
            key = RawGroupKey(
                checkpoint_id=str(self._checkpoint_id),
                observable_family=str(observable_family),
                candidate_label=(None if candidate_label is None else str(candidate_label)),
                position_id=(None if position_id is None else int(position_id)),
                group_key=str(basis_label),
                state_key=(None if state_key is None else str(state_key)),
            )
            record, _ = self.get_or_acquire(
                key,
                min_total_shots=int(min_total_shots),
                min_samples=int(min_samples),
                acquire_sample=lambda repeat_index, lbl=basis_label: oracle.collect_group_sample(
                    circuit,
                    lbl,
                    repeat_idx=int(repeat_index),
                ),
            )
            for label_s, coeff_c in group["terms"]:
                measured_logical = (
                    tuple(int(q) for q in record.samples[-1].measured_logical_qubits)
                    if record.samples
                    else tuple()
                )
                term_logical = tuple(
                    q for q in range(int(len(str(label_s)))) if str(label_s)[int(len(str(label_s))) - 1 - q] != "I"
                )
                classical_indices = [
                    int(measured_logical.index(int(q)))
                    for q in term_logical
                ]
                sample_values: list[float] = []
                sample_weights: list[float] = []
                for sample in record.samples:
                    sample_values.append(
                        _subset_parity_from_quasi(sample.quasi_probs, classical_indices)
                        if sample.quasi_probs is not None
                        else _subset_parity_from_counts(sample.counts, classical_indices)
                    )
                    sample_weights.append(float(sample.shots))
                term_mean, term_stderr = _weighted_mean_stderr(sample_values, sample_weights)
                coeff_total += coeff_c * complex(float(term_mean), 0.0)
                variance_terms += float(abs(coeff_c) ** 2) * float(term_stderr**2)
                last_sample = record.samples[-1] if record.samples else None
                term_payloads.append(
                    {
                        "label": str(label_s),
                        "basis_label": str(basis_label),
                        "coeff_real": float(np.real(coeff_c)),
                        "coeff_imag": float(np.imag(coeff_c)),
                        "mean": float(term_mean),
                        "stderr": float(term_stderr),
                        "sample_count": int(record.sample_count),
                        "total_shots": int(record.total_shots),
                        "last_term_details": ({} if last_sample is None else dict(last_sample.term_details)),
                    }
                )
        backend_info = {
            "noise_mode": str(oracle.backend_info.noise_mode),
            "estimator_kind": f"{oracle.backend_info.estimator_kind}+raw_group_pool",
            "backend_name": oracle.backend_info.backend_name,
            "using_fake_backend": bool(oracle.backend_info.using_fake_backend),
            "details": {
                **dict(oracle.backend_info.details),
                "raw_group_pool": dict(self.summary()),
                "term_payloads": list(term_payloads),
            },
        }
        stderr_total = float(np.sqrt(max(variance_terms, 0.0)))
        return {
            "mean": float(np.real(coeff_total)),
            "stderr": float(stderr_total),
            "std": float(stderr_total),
            "stdev": float(stderr_total),
            "n_samples": int(sum(int(item["sample_count"]) for item in term_payloads)),
            "aggregate": "mean",
            "backend_info": backend_info,
            "term_payloads": term_payloads,
        }

    def summary(self) -> dict[str, Any]:
        total = int(self._hits + self._misses)
        return {
            "checkpoint_id": str(self._checkpoint_id),
            "entries": int(len(self._records)),
            "hits": int(self._hits),
            "misses": int(self._misses),
            "stores": int(self._stores),
            "extensions": int(self._extensions),
            "total_samples": int(sum(record.sample_count for record in self._records.values())),
            "total_shots": int(sum(record.total_shots for record in self._records.values())),
            "hit_rate": (float(self._hits) / float(total) if total > 0 else 0.0),
        }


"""
incremental_specs(plan, I_c) = candidate means/forces + cross and fringe pair observables only
"""
def _select_incremental_candidate_specs(
    *,
    plan: CheckpointObservablePlan,
    candidate_runtime_indices: Sequence[int],
) -> dict[str, Any]:
    total_runtime = int(len(plan.runtime_rotations))
    candidate_indices = tuple(int(idx) for idx in candidate_runtime_indices)
    if not candidate_indices:
        raise ValueError("candidate_runtime_indices must be non-empty.")
    if len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError("candidate_runtime_indices must be unique.")
    if any(int(idx) < 0 or int(idx) >= int(total_runtime) for idx in candidate_indices):
        raise ValueError(
            f"candidate_runtime_indices {candidate_indices!r} out of range for {total_runtime} rotations."
        )
    candidate_set = {int(idx) for idx in candidate_indices}
    baseline_aug_indices = tuple(idx for idx in range(int(total_runtime)) if idx not in candidate_set)
    candidate_generator_specs = tuple(plan.generator_means[int(idx)] for idx in candidate_indices)
    candidate_force_specs = tuple(plan.force_anticommutators[int(idx)] for idx in candidate_indices)
    baseline_candidate_pair_specs: dict[tuple[int, int], ObservableSpec] = {}
    for baseline_row, aug_baseline_idx in enumerate(baseline_aug_indices):
        for candidate_col, aug_candidate_idx in enumerate(candidate_indices):
            pair_key = (
                (int(aug_baseline_idx), int(aug_candidate_idx))
                if int(aug_baseline_idx) < int(aug_candidate_idx)
                else (int(aug_candidate_idx), int(aug_baseline_idx))
            )
            baseline_candidate_pair_specs[(int(baseline_row), int(candidate_col))] = (
                plan.pair_anticommutators[pair_key]
            )
    candidate_pair_specs: dict[tuple[int, int], ObservableSpec] = {}
    for left in range(len(candidate_indices)):
        for right in range(left + 1, len(candidate_indices)):
            aug_left = int(candidate_indices[int(left)])
            aug_right = int(candidate_indices[int(right)])
            pair_key = (
                (int(aug_left), int(aug_right))
                if int(aug_left) < int(aug_right)
                else (int(aug_right), int(aug_left))
            )
            candidate_pair_specs[(int(left), int(right))] = plan.pair_anticommutators[pair_key]
    return {
        "candidate_runtime_indices": tuple(candidate_indices),
        "baseline_aug_indices": tuple(baseline_aug_indices),
        "candidate_generator_specs": tuple(candidate_generator_specs),
        "candidate_force_specs": tuple(candidate_force_specs),
        "baseline_candidate_pair_specs": dict(baseline_candidate_pair_specs),
        "candidate_pair_specs": dict(candidate_pair_specs),
    }


"""
B_{bc}=c_b c_c(s_{bc}-a_b a_c), C_{cd}=c_c c_d(s_{cd}-a_c a_d), q_c=c_c(h_c-E a_c)
"""
def _assemble_measured_incremental_candidate_block(
    *,
    baseline_measured: Mapping[str, Any],
    baseline_coeffs: Sequence[float],
    candidate_coeffs: Sequence[float],
    candidate_generator_means: Sequence[float],
    candidate_force_expectations: Sequence[float],
    baseline_candidate_pair_expectations: Mapping[tuple[int, int], float],
    candidate_pair_expectations: Mapping[tuple[int, int], float],
    runtime_insert_position: int,
    candidate_regularization_lambda: float,
    pinv_rcond: float,
) -> dict[str, Any]:
    baseline_a = np.asarray(baseline_measured.get("generator_means", ()), dtype=float).reshape(-1)
    baseline_coeff_arr = np.asarray([float(val) for val in baseline_coeffs], dtype=float).reshape(-1)
    if int(baseline_a.size) != int(baseline_coeff_arr.size):
        raise ValueError(
            f"baseline generator/coeff size mismatch: {baseline_a.size} vs {baseline_coeff_arr.size}."
        )
    candidate_a = np.asarray([float(val) for val in candidate_generator_means], dtype=float).reshape(-1)
    candidate_h = np.asarray([float(val) for val in candidate_force_expectations], dtype=float).reshape(-1)
    candidate_coeff_arr = np.asarray([float(val) for val in candidate_coeffs], dtype=float).reshape(-1)
    if int(candidate_a.size) != int(candidate_coeff_arr.size):
        raise ValueError(
            f"candidate generator/coeff size mismatch: {candidate_a.size} vs {candidate_coeff_arr.size}."
        )
    if int(candidate_h.size) != int(candidate_coeff_arr.size):
        raise ValueError(
            f"candidate force/coeff size mismatch: {candidate_h.size} vs {candidate_coeff_arr.size}."
        )
    baseline_K = np.asarray(baseline_measured.get("K", np.zeros((0, 0), dtype=float)), dtype=float)
    baseline_theta_dot = np.asarray(
        baseline_measured.get("theta_dot_step", np.zeros(0, dtype=float)),
        dtype=float,
    ).reshape(-1)
    baseline_f = np.asarray(baseline_measured.get("f", np.zeros(0, dtype=float)), dtype=float).reshape(-1)
    if baseline_K.shape != (int(baseline_coeff_arr.size), int(baseline_coeff_arr.size)):
        raise ValueError(
            f"baseline K shape mismatch: {baseline_K.shape} vs {(baseline_coeff_arr.size, baseline_coeff_arr.size)}."
        )
    if int(baseline_theta_dot.size) != int(baseline_coeff_arr.size):
        raise ValueError(
            f"baseline theta_dot_step size mismatch: {baseline_theta_dot.size} vs {baseline_coeff_arr.size}."
        )
    K_pinv = (
        np.linalg.pinv(baseline_K, rcond=float(pinv_rcond))
        if baseline_K.size
        else np.zeros((0, 0), dtype=float)
    )
    m = int(baseline_coeff_arr.size)
    k = int(candidate_coeff_arr.size)
    B = np.zeros((m, k), dtype=float)
    for baseline_row in range(m):
        for candidate_col in range(k):
            sij = float(
                baseline_candidate_pair_expectations[(int(baseline_row), int(candidate_col))]
            )
            B[baseline_row, candidate_col] = float(
                baseline_coeff_arr[baseline_row]
                * candidate_coeff_arr[candidate_col]
                * (sij - float(baseline_a[baseline_row] * candidate_a[candidate_col]))
            )
    C = np.zeros((k, k), dtype=float)
    for idx in range(k):
        C[idx, idx] = float(
            (candidate_coeff_arr[idx] ** 2) * max(0.0, 1.0 - float(candidate_a[idx] * candidate_a[idx]))
        )
    for left in range(k):
        for right in range(left + 1, k):
            sij = float(candidate_pair_expectations[(int(left), int(right))])
            gij = float(
                candidate_coeff_arr[left]
                * candidate_coeff_arr[right]
                * (sij - float(candidate_a[left] * candidate_a[right]))
            )
            C[left, right] = gij
            C[right, left] = gij
    baseline_energy = float(baseline_measured.get("energy", 0.0))
    q = np.asarray(
        candidate_coeff_arr * (candidate_h - float(baseline_energy) * candidate_a),
        dtype=float,
    ).reshape(-1)
    S = np.asarray(
        C
        + float(candidate_regularization_lambda) * np.eye(int(k), dtype=float)
        - B.T @ K_pinv @ B,
        dtype=float,
    )
    S_pinv = np.linalg.pinv(S, rcond=float(pinv_rcond)) if S.size else np.zeros((0, 0), dtype=float)
    w = np.asarray(q - B.T @ baseline_theta_dot, dtype=float).reshape(-1)
    eta_dot = np.asarray(S_pinv @ w, dtype=float).reshape(-1) if S.size else np.zeros(0, dtype=float)
    gain_exact = float(max(0.0, float(w @ eta_dot))) if w.size else 0.0
    baseline_variance = float(max(float(baseline_measured.get("variance", 0.0)), 1.0e-14))
    gain_ratio = float(gain_exact / baseline_variance)
    theta_dot_aug_existing = np.asarray(
        baseline_theta_dot - K_pinv @ B @ eta_dot,
        dtype=float,
    ).reshape(-1)
    runtime_pos = int(runtime_insert_position)
    if runtime_pos < 0 or runtime_pos > int(theta_dot_aug_existing.size):
        raise ValueError(
            f"runtime_insert_position {runtime_pos} invalid for baseline size {theta_dot_aug_existing.size}."
        )
    theta_dot_step = np.concatenate(
        [
            theta_dot_aug_existing[: int(runtime_pos)],
            eta_dot,
            theta_dot_aug_existing[int(runtime_pos) :],
        ]
    )
    baseline_objective = float(baseline_f @ baseline_theta_dot) if baseline_f.size else 0.0
    return {
        "B": np.asarray(B, dtype=float),
        "C": np.asarray(C, dtype=float),
        "q": np.asarray(q, dtype=float),
        "S": np.asarray(S, dtype=float),
        "w": np.asarray(w, dtype=float),
        "eta_dot": np.asarray(eta_dot, dtype=float),
        "gain_exact": float(gain_exact),
        "gain_ratio": float(gain_ratio),
        "theta_dot_aug_existing": np.asarray(theta_dot_aug_existing, dtype=float),
        "theta_dot_step": np.asarray(theta_dot_step, dtype=float),
        "step_objective_value": float(baseline_objective + gain_exact),
    }


def estimate_grouped_raw_mclachlan_geometry(
    *,
    oracle: Any,
    raw_group_pool: BackendScheduledRawGroupPool,
    layout: Any,
    theta_runtime: np.ndarray | Sequence[float],
    psi_ref: np.ndarray | Sequence[complex],
    h_poly: Any,
    geom_cfg: FixedManifoldMeasuredConfig,
    observable_family_prefix: str,
    candidate_label: str | None,
    position_id: int | None,
    state_key: str,
    min_total_shots: int,
    min_samples: int,
) -> dict[str, Any]:
    from pipelines.hardcoded.hh_fixed_manifold_measured import (
        assemble_measured_geometry,
    )

    plan = build_checkpoint_observable_plan_from_layout(
        layout,
        theta_runtime,
        psi_ref=np.asarray(psi_ref, dtype=complex).reshape(-1),
        h_poly=h_poly,
        drop_abs_tol=float(geom_cfg.observable_drop_abs_tol),
        hermiticity_tol=float(geom_cfg.observable_hermiticity_tol),
        max_observable_terms=int(geom_cfg.observable_max_terms),
    )
    observable_estimates: dict[str, dict[str, Any]] = {}
    backend_infos: list[dict[str, Any]] = []

    def _estimate(spec: ObservableSpec) -> float:
        est = _observable_spec_mean(
            raw_group_pool=raw_group_pool,
            oracle=oracle,
            circuit=plan.circuit,
            spec=spec,
            observable_family=f"{str(observable_family_prefix)}:{str(spec.name)}",
            candidate_label=(None if candidate_label is None else str(candidate_label)),
            position_id=(None if position_id is None else int(position_id)),
            state_key=str(state_key),
            min_total_shots=int(min_total_shots),
            min_samples=int(min_samples),
        )
        observable_estimates[str(spec.name)] = {
            "mean": float(est.get("mean", 0.0)),
            "stderr": float(est.get("stderr", 0.0)),
            "n_samples": int(est.get("n_samples", 0)),
            "term_payloads": list(est.get("term_payloads", [])),
        }
        backend_info = est.get("backend_info")
        if isinstance(backend_info, Mapping):
            backend_infos.append(dict(backend_info))
        return float(est.get("mean", 0.0))

    energy = float(_estimate(plan.energy))
    h2 = float(_estimate(plan.variance_h2))
    generator_means = [float(_estimate(spec)) for spec in plan.generator_means]
    pair_expectations = {
        tuple(pair): float(_estimate(spec))
        for pair, spec in plan.pair_anticommutators.items()
    }
    force_expectations = [float(_estimate(spec)) for spec in plan.force_anticommutators]
    geometry = assemble_measured_geometry(
        plan=plan,
        energy=float(energy),
        h2=float(h2),
        generator_means=generator_means,
        pair_expectations=pair_expectations,
        force_expectations=force_expectations,
        geom_cfg=geom_cfg,
    )
    theta_dot_step = np.asarray(geometry["theta_dot_step"], dtype=float).reshape(-1)
    f_vec = np.asarray(geometry["f"], dtype=float).reshape(-1)
    step_objective_value = float(f_vec @ theta_dot_step) if f_vec.size > 0 else 0.0
    backend_info = {
        "noise_mode": str(getattr(oracle.backend_info, "noise_mode", "unknown")),
        "estimator_kind": f"{str(getattr(oracle.backend_info, 'estimator_kind', 'unknown'))}+grouped_geometry",
        "backend_name": getattr(oracle.backend_info, "backend_name", None),
        "using_fake_backend": bool(getattr(oracle.backend_info, "using_fake_backend", False)),
        "details": {
            **dict(getattr(oracle.backend_info, "details", {})),
            "plan_stats": dict(plan.stats),
            "raw_group_pool": dict(raw_group_pool.summary()),
            "state_key": str(state_key),
            "observable_count": int(len(observable_estimates)),
        },
    }
    if backend_infos:
        last_backend = dict(backend_infos[-1])
        backend_info = {
            **backend_info,
            "details": {
                **dict(last_backend.get("details", {})),
                **dict(backend_info.get("details", {})),
            },
        }
    return {
        "geometry": geometry,
        "plan": plan,
        "plan_stats": dict(plan.stats),
        "observable_estimates": observable_estimates,
        "backend_info": backend_info,
        "raw_group_pool_summary": dict(raw_group_pool.summary()),
        "step_objective_value": float(step_objective_value),
        "state_key": str(state_key),
    }


def estimate_grouped_raw_mclachlan_incremental_block(
    *,
    oracle: Any,
    raw_group_pool: Any,
    baseline_measured: Mapping[str, Any],
    layout: Any,
    theta_runtime: np.ndarray | Sequence[float],
    psi_ref: np.ndarray | Sequence[complex],
    h_poly: Any,
    candidate_runtime_indices: Sequence[int],
    runtime_insert_position: int,
    geom_cfg: FixedManifoldMeasuredConfig,
    candidate_regularization_lambda: float,
    pinv_rcond: float,
    observable_family_prefix: str,
    candidate_label: str | None,
    position_id: int | None,
    state_key: str,
    min_total_shots: int,
    min_samples: int,
) -> dict[str, Any]:
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    candidate_indices = tuple(int(idx) for idx in candidate_runtime_indices)
    expected_indices = tuple(
        range(int(runtime_insert_position), int(runtime_insert_position) + len(candidate_indices))
    )
    if tuple(candidate_indices) != tuple(sorted(candidate_indices)):
        raise ValueError("candidate_runtime_indices must be sorted in ascending runtime order.")
    if tuple(candidate_indices) != tuple(expected_indices):
        raise ValueError(
            "candidate_runtime_indices must match the contiguous inserted runtime block."
        )
    zero_tol = max(1.0e-12, float(geom_cfg.observable_drop_abs_tol))
    if any(abs(float(theta_arr[int(idx)])) > float(zero_tol) for idx in candidate_indices):
        raise ValueError(
            "incremental candidate measurement requires zero inserted candidate angles."
        )
    plan = build_checkpoint_observable_plan_from_layout(
        layout,
        theta_arr,
        psi_ref=np.asarray(psi_ref, dtype=complex).reshape(-1),
        h_poly=h_poly,
        drop_abs_tol=float(geom_cfg.observable_drop_abs_tol),
        hermiticity_tol=float(geom_cfg.observable_hermiticity_tol),
        max_observable_terms=int(geom_cfg.observable_max_terms),
    )
    selected = _select_incremental_candidate_specs(
        plan=plan,
        candidate_runtime_indices=candidate_runtime_indices,
    )
    observable_estimates: dict[str, dict[str, Any]] = {}
    backend_infos: list[dict[str, Any]] = []

    def _estimate(spec: ObservableSpec) -> float:
        est = _observable_spec_mean(
            raw_group_pool=raw_group_pool,
            oracle=oracle,
            circuit=plan.circuit,
            spec=spec,
            observable_family=f"{str(observable_family_prefix)}:{str(spec.name)}",
            candidate_label=(None if candidate_label is None else str(candidate_label)),
            position_id=(None if position_id is None else int(position_id)),
            state_key=str(state_key),
            min_total_shots=int(min_total_shots),
            min_samples=int(min_samples),
        )
        observable_estimates[str(spec.name)] = {
            "mean": float(est.get("mean", 0.0)),
            "stderr": float(est.get("stderr", 0.0)),
            "n_samples": int(est.get("n_samples", 0)),
            "term_payloads": list(est.get("term_payloads", [])),
        }
        backend_info = est.get("backend_info")
        if isinstance(backend_info, Mapping):
            backend_infos.append(dict(backend_info))
        return float(est.get("mean", 0.0))

    candidate_generator_means = [
        float(_estimate(spec)) for spec in selected["candidate_generator_specs"]
    ]
    candidate_force_expectations = [
        float(_estimate(spec)) for spec in selected["candidate_force_specs"]
    ]
    baseline_candidate_pair_expectations = {
        tuple(key): float(_estimate(spec))
        for key, spec in selected["baseline_candidate_pair_specs"].items()
    }
    candidate_pair_expectations = {
        tuple(key): float(_estimate(spec))
        for key, spec in selected["candidate_pair_specs"].items()
    }
    baseline_coeffs = [
        float(plan.runtime_rotations[int(idx)].coeff_real)
        for idx in selected["baseline_aug_indices"]
    ]
    candidate_coeffs = [
        float(plan.runtime_rotations[int(idx)].coeff_real)
        for idx in selected["candidate_runtime_indices"]
    ]
    incremental_block = _assemble_measured_incremental_candidate_block(
        baseline_measured=baseline_measured,
        baseline_coeffs=baseline_coeffs,
        candidate_coeffs=candidate_coeffs,
        candidate_generator_means=candidate_generator_means,
        candidate_force_expectations=candidate_force_expectations,
        baseline_candidate_pair_expectations=baseline_candidate_pair_expectations,
        candidate_pair_expectations=candidate_pair_expectations,
        runtime_insert_position=int(runtime_insert_position),
        candidate_regularization_lambda=float(candidate_regularization_lambda),
        pinv_rcond=float(pinv_rcond),
    )
    plan_stats = {
        **dict(plan.stats),
        "selected_baseline_count": int(len(selected["baseline_aug_indices"])),
        "selected_candidate_count": int(len(selected["candidate_runtime_indices"])),
        "selected_cross_pair_count": int(len(selected["baseline_candidate_pair_specs"])),
        "selected_candidate_pair_count": int(len(selected["candidate_pair_specs"])),
        "selected_observable_count": int(len(observable_estimates)),
    }
    backend_info = {
        "noise_mode": str(getattr(oracle.backend_info, "noise_mode", "unknown")),
        "estimator_kind": f"{str(getattr(oracle.backend_info, 'estimator_kind', 'unknown'))}+grouped_incremental_block",
        "backend_name": getattr(oracle.backend_info, "backend_name", None),
        "using_fake_backend": bool(getattr(oracle.backend_info, "using_fake_backend", False)),
        "details": {
            **dict(getattr(oracle.backend_info, "details", {})),
            "plan_stats": dict(plan_stats),
            "raw_group_pool": dict(raw_group_pool.summary()),
            "state_key": str(state_key),
            "observable_count": int(len(observable_estimates)),
        },
    }
    if backend_infos:
        last_backend = dict(backend_infos[-1])
        backend_info = {
            **backend_info,
            "details": {
                **dict(last_backend.get("details", {})),
                **dict(backend_info.get("details", {})),
            },
        }
    return {
        "incremental_block": incremental_block,
        "plan": plan,
        "plan_stats": plan_stats,
        "observable_estimates": observable_estimates,
        "backend_info": backend_info,
        "raw_group_pool_summary": dict(raw_group_pool.summary()),
        "state_key": str(state_key),
        "selected_observable_names": list(observable_estimates.keys()),
    }


class TemporalMeasurementLedger:
    """Cross-checkpoint stale priors used only for scheduling/refresh hints."""

    def __init__(self) -> None:
        self._priors: dict[TemporalPriorKey, TemporalPriorRecord] = {}
        self._last_checkpoint_summary: dict[str, Any] | None = None

    def candidate_probe_bonus(
        self,
        *,
        candidate_identity: str,
        position_id: int,
        predicted_displacement: float,
    ) -> float:
        key = TemporalPriorKey(candidate_identity=str(candidate_identity), position_id=int(position_id))
        prior = self._priors.get(key)
        if prior is None:
            return 0.0
        if float(predicted_displacement) > 0.05:
            return 0.0
        if str(prior.last_refresh_pressure).strip().lower() != "low":
            return 0.0
        return float(min(0.05, 0.01 * float(prior.times_selected)))

    def refresh_pressure(
        self,
        *,
        predicted_displacement: float,
        rho_miss: float,
        condition_number: float,
    ) -> str:
        if float(predicted_displacement) >= 0.20 or float(rho_miss) >= 0.20 or float(condition_number) >= 1.0e6:
            return "high"
        if float(predicted_displacement) <= 0.05 and float(rho_miss) <= 0.05 and float(condition_number) <= 1.0e4:
            return "low"
        return "medium"

    def record_checkpoint(
        self,
        *,
        checkpoint_index: int,
        selected_candidate_identity: str | None,
        selected_position_id: int | None,
        selected_groups_new: float,
        selected_gain_ratio: float,
        predicted_displacement: float,
        refresh_pressure: str,
    ) -> None:
        self._last_checkpoint_summary = {
            "checkpoint_index": int(checkpoint_index),
            "selected_candidate_identity": (None if selected_candidate_identity is None else str(selected_candidate_identity)),
            "selected_position_id": (None if selected_position_id is None else int(selected_position_id)),
            "predicted_displacement": float(predicted_displacement),
            "refresh_pressure": str(refresh_pressure),
        }
        if selected_candidate_identity is None or selected_position_id is None:
            return
        key = TemporalPriorKey(
            candidate_identity=str(selected_candidate_identity),
            position_id=int(selected_position_id),
        )
        prev = self._priors.get(key)
        self._priors[key] = TemporalPriorRecord(
            candidate_identity=str(selected_candidate_identity),
            position_id=int(selected_position_id),
            last_checkpoint_index=int(checkpoint_index),
            times_selected=(1 if prev is None else int(prev.times_selected) + 1),
            last_groups_new=float(selected_groups_new),
            last_gain_ratio=float(selected_gain_ratio),
            last_predicted_displacement=float(predicted_displacement),
            last_refresh_pressure=str(refresh_pressure),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "prior_entries": int(len(self._priors)),
            "last_checkpoint": (None if self._last_checkpoint_summary is None else dict(self._last_checkpoint_summary)),
        }


def validate_controller_tiers_mean_only(tiers: tuple[MeasurementTierConfig, ...]) -> None:
    for tier in tiers:
        aggregate = tier.oracle_aggregate
        if aggregate is not None and str(aggregate).strip().lower() != "mean":
            raise ValueError(
                f"Realtime checkpoint controller currently supports only mean aggregate tiers; got {aggregate!r} for {tier.tier_name}."
            )


"""
Oracle base config validation: mean aggregate; local/noisy modes only for controller v1.
"""
def validate_controller_oracle_base_config(base_config: OracleConfig) -> None:
    aggregate = str(base_config.oracle_aggregate).strip().lower()
    if aggregate != "mean":
        raise ValueError(
            f"checkpoint controller oracle_v1 requires oracle_aggregate='mean'; got {base_config.oracle_aggregate!r}."
        )
    noise_mode = str(base_config.noise_mode).strip().lower()
    if noise_mode not in {"ideal", "shots", "aer_noise", "runtime", "backend_scheduled"}:
        raise ValueError(
            f"checkpoint controller oracle_v1 unsupported noise_mode {base_config.noise_mode!r}; expected one of ['aer_noise', 'backend_scheduled', 'ideal', 'runtime', 'shots']."
        )
    mitigation_cfg = normalize_mitigation_config(getattr(base_config, "mitigation", "none"))
    symmetry_cfg = normalize_symmetry_mitigation_config(
        getattr(base_config, "symmetry_mitigation", "off")
    )
    if noise_mode == "backend_scheduled":
        if not bool(base_config.use_fake_backend):
            raise ValueError(
                "checkpoint controller oracle_v1 requires use_fake_backend=True for backend_scheduled mode."
            )
        mitigation_mode = str(mitigation_cfg.get("mode", "none"))
        if mitigation_mode not in {"none", "readout"}:
            raise ValueError(
                "checkpoint controller oracle_v1 backend_scheduled mode supports only mitigation modes 'none' or 'readout'."
            )
        if mitigation_mode == "readout":
            if str(symmetry_cfg.get("mode", "off")) not in {"off", "verify_only"}:
                raise ValueError(
                    "checkpoint controller oracle_v1 backend_scheduled readout mitigation is not combinable with active symmetry mitigation."
                )
            strategy = str(mitigation_cfg.get("local_readout_strategy") or "mthree")
            if strategy not in _LOCAL_READOUT_STRATEGIES:
                raise ValueError(
                    f"checkpoint controller oracle_v1 unsupported backend_scheduled readout strategy {strategy!r}; expected one of {sorted(_LOCAL_READOUT_STRATEGIES)}."
                )


def controller_oracle_supports_raw_group_sampling(base_config: OracleConfig) -> bool:
    noise_mode = str(base_config.noise_mode).strip().lower()
    mitigation_cfg = normalize_mitigation_config(getattr(base_config, "mitigation", "none"))
    symmetry_cfg = normalize_symmetry_mitigation_config(
        getattr(base_config, "symmetry_mitigation", "off")
    )
    if noise_mode == "backend_scheduled":
        return True
    if noise_mode != "runtime":
        return False
    if str(mitigation_cfg.get("mode", "none")) != "none":
        return False
    if str(symmetry_cfg.get("mode", "off")) not in {"off", "verify_only"}:
        return False
    return True


"""
Tier materialization: clone base OracleConfig per controller tier; keep policy above ExpectationOracle.
"""
def build_controller_oracle_tier_configs(
    base_config: OracleConfig,
    tiers: tuple[MeasurementTierConfig, ...],
) -> dict[str, OracleConfig]:
    validate_controller_tiers_mean_only(tiers)
    validate_controller_oracle_base_config(base_config)
    base_shots = max(1, int(base_config.shots))
    base_repeats = max(1, int(base_config.oracle_repeats))
    out: dict[str, OracleConfig] = {}
    confirm_shots_default = max(1024, int(round(float(base_shots) * 0.50)))
    for tier in tiers:
        tier_name = str(tier.tier_name)
        if tier_name == "scout":
            default_shots = max(256, int(round(float(base_shots) * 0.25)))
            default_repeats = 1
        elif tier_name == "confirm":
            default_shots = confirm_shots_default
            default_repeats = max(1, base_repeats)
        elif tier_name == "commit":
            default_shots = max(base_shots, confirm_shots_default)
            default_repeats = max(2, base_repeats)
        else:
            default_shots = base_shots
            default_repeats = base_repeats
        out[tier_name] = replace(
            base_config,
            shots=(int(tier.oracle_shots) if tier.oracle_shots is not None else int(default_shots)),
            oracle_repeats=(
                int(tier.oracle_repeats)
                if tier.oracle_repeats is not None
                else int(default_repeats)
            ),
            oracle_aggregate=(
                str(tier.oracle_aggregate)
                if tier.oracle_aggregate is not None
                else "mean"
            ),
        )
    return out


def planning_group_keys_for_term(term: Any) -> list[str]:
    return list(measurement_group_keys_for_term(term))


def planning_stats_for_term(term: Any, audit: MeasurementCacheAudit) -> MeasurementCacheStats:
    return audit.estimate(planning_group_keys_for_term(term))


__all__ = [
    "BackendScheduledRawGroupPool",
    "controller_oracle_supports_raw_group_sampling",
    "DerivedGeometryMemo",
    "ExactCheckpointValueCache",
    "OracleCheckpointValueCache",
    "TemporalMeasurementLedger",
    "build_controller_oracle_tier_configs",
    "estimate_grouped_raw_mclachlan_incremental_block",
    "estimate_grouped_raw_mclachlan_geometry",
    "planning_group_keys_for_term",
    "planning_stats_for_term",
    "validate_controller_oracle_base_config",
    "validate_controller_tiers_mean_only",
]
