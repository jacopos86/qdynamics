#!/usr/bin/env python3
"""Certified joint delete--add exchange maintenance for static QSE bases.

Implements the Paper III C3 contribution (audit-approved formulation): a
candidate patch ``(B^-, B^+)`` on the selected response-operator support is
committed **atomically** only after recomputation of the projected
nonorthogonal pencil verifies improvement of the declared target-root
objective and compiled hardware cost while satisfying overlap-conditioning
and rank guards. Deletion and insertion are evaluated as one coupled move —
never as sequential pruning followed by append-only growth — so a patch may
be accepted whose halves would each fail alone.

Scope and boundaries:

- The declared target-root objective is the lowest retained Ritz root of the
  (policy-projected) pencil — under the q0 policy this is the lowest
  orthogonal Ritz root, i.e. the first-excitation candidate, and lowering it
  is variational within the manifold.
- Compiled costs are injected per pool element (Paper I oracle upstream,
  e.g. the ``two_qubit_only_v1`` scalarization); this module never compiles.
- Certification is exact re-solve, not a proxy: every evaluated patch runs
  ``compute_qse_spectra`` on the patched support. Shortlists only bound how
  many patches are evaluated; they never bypass certification.
- Acceptance is dominance with tolerances: a patch must not degrade either
  coordinate beyond its tolerance and must strictly improve at least one
  (root objective by ``min_root_improvement``, or compiled cost). Conditioning
  may not worsen beyond ``condition_slack_factor`` (or an absolute cap), and
  the retained rank may not fall below the configured fraction.
- Statevector diagnostic; telemetry records every evaluated patch and never
  feeds controller decisions.

Related prior art acknowledged in the Paper III claims-boundary doc:
restarted/thick-restart subspace iteration, SQD add--remove maintenance, and
QSE pruning are established; the certified coupled-patch transaction on a
measured nonorthogonal operator pencil with a compiled-cost gate is the
claimed residual.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.qse_spectra.core import (
    QSEBasisElement,
    _apply_basis_element,
    _apply_polynomial_operator,
    _config as _core_config,
    compute_qse_spectra,
    normalize_statevector,
)

QSE_EXCHANGE_MAINTENANCE_SCHEMA_VERSION = "qse_exchange_maintenance_v1"


@dataclass(frozen=True)
class QSEExchangeConfig:
    """Acceptance functional and enumeration bounds for exchange maintenance."""

    max_rounds: int = 20
    deletion_shortlist_size: int = 5
    insertion_shortlist_size: int = 8
    allow_pure_deletion: bool = True
    allow_pure_insertion: bool = False
    target_root_count: int = 1
    accuracy_tolerance: float = 1.0e-9
    min_root_improvement: float = 1.0e-10
    cost_budget: float | None = None
    max_overlap_condition: float | None = None
    condition_slack_factor: float = 10.0
    min_retained_rank_fraction: float = 1.0

    def __post_init__(self) -> None:
        if int(self.max_rounds) < 1:
            raise ValueError("max_rounds must be >= 1.")
        if int(self.target_root_count) < 1:
            raise ValueError("target_root_count must be >= 1.")
        if int(self.deletion_shortlist_size) < 0 or int(self.insertion_shortlist_size) < 0:
            raise ValueError("shortlist sizes must be >= 0.")
        for name in ("accuracy_tolerance", "min_root_improvement"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0.")
        if self.cost_budget is not None:
            value = float(self.cost_budget)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("cost_budget must be finite and >= 0 when supplied.")
        if self.max_overlap_condition is not None:
            value = float(self.max_overlap_condition)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("max_overlap_condition must be finite and > 0 when supplied.")
        slack = float(self.condition_slack_factor)
        if not math.isfinite(slack) or slack < 1.0:
            raise ValueError("condition_slack_factor must be finite and >= 1.")
        fraction = float(self.min_retained_rank_fraction)
        if not math.isfinite(fraction) or fraction <= 0.0 or fraction > 1.0:
            raise ValueError("min_retained_rank_fraction must be in (0, 1].")


@dataclass(frozen=True)
class QSEExchangeResult:
    """Final support plus the full per-patch certification telemetry."""

    initial_indices: tuple[int, ...]
    final_indices: tuple[int, ...]
    initial_summary: dict[str, Any]
    final_summary: dict[str, Any]
    rounds: tuple[dict[str, Any], ...]
    config: QSEExchangeConfig = field(default_factory=QSEExchangeConfig)


def _solve_summary(
    pool: Sequence[QSEBasisElement],
    indices: Sequence[int],
    costs: Sequence[float],
    *,
    hamiltonian: Any,
    prepared_state: np.ndarray,
    qse_config: Any,
    basis_vector_policy: Any,
    target_root_count: int = 1,
) -> dict[str, Any]:
    result = compute_qse_spectra(
        hamiltonian,
        prepared_state,
        tuple(pool[int(index)] for index in indices),
        config=qse_config,
        basis_vector_policy=basis_vector_policy,
    )
    energies = np.asarray(result.eigenvalues, dtype=float).reshape(-1)
    if energies.size == 0:
        raise ValueError("Exchange certification solve retained no roots.")
    condition = result.overlap_condition_estimate
    roots = [float(value) for value in energies[: max(int(target_root_count), 1)]]
    return {
        "indices": tuple(int(index) for index in indices),
        "root0_energy": float(energies[0]),
        # Ky Fan trace over the lowest R retained roots: a variational upper
        # bound on the sum of the true lowest R sector excitations.
        "root_energies": roots,
        "objective_energy": float(sum(roots)),
        "retained_rank": int(result.retained_rank),
        "overlap_condition_estimate": float(condition) if condition is not None else None,
        "total_compiled_cost": float(sum(float(costs[int(index)]) for index in indices)),
        "result": result,
    }


def _public_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "result"}


def _deletion_shortlist(
    summary: Mapping[str, Any],
    costs: Sequence[float],
    *,
    size: int,
    target_root_count: int = 1,
) -> list[int]:
    """Lowest target-root coefficient magnitude, then highest cost, deduplicated."""

    indices = list(summary["indices"])
    result = summary["result"]
    matrix = np.abs(np.asarray(result.eigenvectors_basis))
    root_count = min(max(int(target_root_count), 1), matrix.shape[1])
    coefficients = matrix[:, :root_count].max(axis=1).reshape(-1)
    by_weight = sorted(range(len(indices)), key=lambda pos: float(coefficients[pos]))
    by_cost = sorted(
        range(len(indices)), key=lambda pos: -float(costs[int(indices[pos])])
    )
    shortlist: list[int] = []
    for pos in list(by_weight[: int(size)]) + list(by_cost[: int(size)]):
        index = int(indices[pos])
        if index not in shortlist:
            shortlist.append(index)
    return shortlist[: 2 * int(size)]


def _insertion_shortlist(
    pool: Sequence[QSEBasisElement],
    summary: Mapping[str, Any],
    *,
    hamiltonian: Any,
    prepared_state: np.ndarray,
    qse_config: Any,
    basis_vector_policy: Any,
    size: int,
    target_root_count: int = 1,
) -> list[int]:
    """Rank unused pool members by novelty and target-root residual capture.

    Heuristic pre-screen only; every shortlisted patch still passes full
    pencil certification before commit.
    """

    cfg = _core_config(qse_config)
    psi, _, nq = normalize_statevector(np.asarray(prepared_state, dtype=complex).reshape(-1))
    cache: dict[str, Any] = {}

    def _project_reference(vec: np.ndarray) -> np.ndarray:
        out = np.asarray(vec, dtype=complex).reshape(-1)
        if basis_vector_policy is not None and str(
            getattr(basis_vector_policy, "reference_projection", "none")
        ) == "q0":
            out = out - complex(np.vdot(psi, out)) * psi
        return out

    current_indices = set(int(index) for index in summary["indices"])
    result = summary["result"]
    matrix_vectors = [np.asarray(v, dtype=complex).reshape(-1) for v in result.matrices.basis_matrix_vectors]
    basis_units: list[np.ndarray] = []
    for vector in matrix_vectors:
        projected = vector.copy()
        for unit in basis_units:
            projected -= complex(np.vdot(unit, projected)) * unit
        norm = float(np.linalg.norm(projected))
        if norm > 1.0e-12:
            basis_units.append(projected / norm)

    eigenvector_matrix = np.asarray(result.eigenvectors_basis)
    root_count = min(max(int(target_root_count), 1), eigenvector_matrix.shape[1])
    residual_hats: list[np.ndarray] = []
    for root in range(root_count):
        coefficients = eigenvector_matrix[:, root].reshape(-1)
        root_state = np.zeros_like(psi)
        for position, vector in enumerate(matrix_vectors):
            root_state += complex(coefficients[position]) * vector
        root_norm = float(np.linalg.norm(root_state))
        if root_norm <= 1.0e-12:
            continue
        root_state = root_state / root_norm
        h_root = np.asarray(
            _apply_polynomial_operator(
                hamiltonian, root_state, nq=int(nq), name="hamiltonian",
                config=cfg, pauli_action_cache=cache,
            ),
            dtype=complex,
        ).reshape(-1)
        residual = h_root - complex(np.vdot(root_state, h_root)) * root_state
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm > 1.0e-12:
            residual_hats.append(residual / residual_norm)

    scored: list[tuple[float, int]] = []
    for index, element in enumerate(pool):
        if int(index) in current_indices:
            continue
        image = _project_reference(
            _apply_basis_element(element, psi, nq=int(nq), config=cfg, pauli_action_cache=cache)
        )
        norm_sq = float(np.vdot(image, image).real)
        if norm_sq <= 0.0:
            continue
        projected = image.copy()
        for unit in basis_units:
            projected -= complex(np.vdot(unit, projected)) * unit
        p_norm_sq = float(np.vdot(projected, projected).real)
        novelty = max(0.0, p_norm_sq / norm_sq)
        capture = 0.0
        if residual_hats and p_norm_sq > 0.0:
            unit_vec = projected / math.sqrt(p_norm_sq)
            for hat in residual_hats:
                capture = max(capture, float(abs(complex(np.vdot(unit_vec, hat)))))
        scored.append((0.25 * novelty + capture, int(index)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [index for _score, index in scored[: int(size)]]


def _certify_patch(
    *,
    current: Mapping[str, Any],
    candidate: Mapping[str, Any],
    config: QSEExchangeConfig,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    current_rank = int(current["retained_rank"])
    rank_floor = max(1, int(math.ceil(float(config.min_retained_rank_fraction) * current_rank)))
    if int(candidate["retained_rank"]) < rank_floor:
        reasons.append("retained_rank_guard")

    candidate_condition = candidate["overlap_condition_estimate"]
    current_condition = current["overlap_condition_estimate"]
    if candidate_condition is not None:
        if config.max_overlap_condition is not None and float(candidate_condition) > float(
            config.max_overlap_condition
        ):
            reasons.append("absolute_condition_guard")
        if (
            current_condition is not None
            and float(candidate_condition)
            > float(current_condition) * float(config.condition_slack_factor)
        ):
            reasons.append("relative_condition_guard")

    shared_roots = min(len(current["root_energies"]), len(candidate["root_energies"]))
    delta_root = float(
        sum(candidate["root_energies"][:shared_roots]) - sum(current["root_energies"][:shared_roots])
    )
    delta_cost = float(candidate["total_compiled_cost"]) - float(current["total_compiled_cost"])
    improves_root = delta_root < -float(config.min_root_improvement)
    improves_cost = delta_cost < 0.0
    root_ok = delta_root <= float(config.accuracy_tolerance)
    cost_ok = delta_cost <= 0.0
    if config.cost_budget is not None:
        # Budgeted variant: root-improving patches may spend cost while the
        # patched total stays within the declared budget; cost-saving patches
        # keep the plain dominance rule.
        within_budget = float(candidate["total_compiled_cost"]) <= float(config.cost_budget)
        if not ((root_ok and improves_cost) or (improves_root and within_budget)):
            reasons.append("budgeted_dominance_gate")
    elif not ((root_ok and improves_cost) or (improves_root and cost_ok)):
        reasons.append("dominance_gate")
    return (not reasons), reasons


def run_qse_exchange_maintenance(
    pool: Sequence[QSEBasisElement],
    selected_indices: Sequence[int],
    compiled_costs: Sequence[float],
    *,
    hamiltonian: Any,
    prepared_state: np.ndarray,
    qse_config: Any = None,
    basis_vector_policy: Any = None,
    config: QSEExchangeConfig | None = None,
) -> QSEExchangeResult:
    """Iterate certified joint delete--add patches until no admissible move."""

    cfg = config if config is not None else QSEExchangeConfig()
    pool_tuple = tuple(pool)
    costs = [float(value) for value in compiled_costs]
    if len(costs) != len(pool_tuple):
        raise ValueError(
            f"compiled_costs length {len(costs)} does not match pool size {len(pool_tuple)}."
        )
    for value in costs:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("compiled_costs entries must be finite and >= 0.")
    current_indices = [int(index) for index in selected_indices]
    if len(set(current_indices)) != len(current_indices):
        raise ValueError("selected_indices must be unique.")

    solve_kwargs = dict(
        hamiltonian=hamiltonian,
        prepared_state=prepared_state,
        qse_config=qse_config,
        basis_vector_policy=basis_vector_policy,
        target_root_count=int(cfg.target_root_count),
    )
    current = _solve_summary(pool_tuple, current_indices, costs, **solve_kwargs)
    initial_summary = _public_summary(current)

    rounds: list[dict[str, Any]] = []
    for round_index in range(int(cfg.max_rounds)):
        deletions = _deletion_shortlist(
            current,
            costs,
            size=int(cfg.deletion_shortlist_size),
            target_root_count=int(cfg.target_root_count),
        )
        insertions = _insertion_shortlist(
            pool_tuple,
            current,
            hamiltonian=hamiltonian,
            prepared_state=prepared_state,
            qse_config=qse_config,
            basis_vector_policy=basis_vector_policy,
            size=int(cfg.insertion_shortlist_size),
            target_root_count=int(cfg.target_root_count),
        )

        patches: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for delete in deletions:
            for insert in insertions:
                patches.append(((delete,), (insert,)))
        if cfg.allow_pure_deletion and len(current["indices"]) > 1:
            patches.extend(((delete,), ()) for delete in deletions)
        if cfg.allow_pure_insertion:
            patches.extend(((), (insert,)) for insert in insertions)

        evaluated: list[dict[str, Any]] = []
        admissible: list[tuple[float, float, int, dict[str, Any]]] = []
        for patch_number, (b_minus, b_plus) in enumerate(patches):
            patched_indices = [
                index for index in current["indices"] if index not in set(b_minus)
            ] + list(b_plus)
            try:
                candidate = _solve_summary(pool_tuple, patched_indices, costs, **solve_kwargs)
            except ValueError as exc:
                evaluated.append(
                    {
                        "delete_indices": list(b_minus),
                        "insert_indices": list(b_plus),
                        "accepted": False,
                        "rejection_reasons": ["solve_failed"],
                        "solve_error": str(exc),
                    }
                )
                continue
            accepted, reasons = _certify_patch(current=current, candidate=candidate, config=cfg)
            record = {
                "delete_indices": list(b_minus),
                "delete_names": [str(pool_tuple[index].name) for index in b_minus],
                "insert_indices": list(b_plus),
                "insert_names": [str(pool_tuple[index].name) for index in b_plus],
                "root0_energy": candidate["root0_energy"],
                "delta_root0": candidate["root0_energy"] - current["root0_energy"],
                "root_energies": list(candidate["root_energies"]),
                "objective_energy": candidate["objective_energy"],
                "delta_objective": candidate["objective_energy"] - current["objective_energy"],
                "total_compiled_cost": candidate["total_compiled_cost"],
                "delta_cost": candidate["total_compiled_cost"] - current["total_compiled_cost"],
                "overlap_condition_estimate": candidate["overlap_condition_estimate"],
                "retained_rank": candidate["retained_rank"],
                "accepted": bool(accepted),
                "rejection_reasons": reasons,
            }
            evaluated.append(record)
            if accepted:
                admissible.append(
                    (
                        float(record["delta_objective"]),
                        float(record["delta_cost"]),
                        patch_number,
                        {"record": record, "candidate": candidate},
                    )
                )

        round_record: dict[str, Any] = {
            "round": int(round_index),
            "deletion_shortlist": list(deletions),
            "insertion_shortlist": list(insertions),
            "evaluated_patch_count": len(evaluated),
            "evaluated_patches": evaluated,
            "committed_patch": None,
        }
        if not admissible:
            rounds.append(round_record)
            break
        admissible.sort(key=lambda item: (item[0], item[1], item[2]))
        best = admissible[0][3]
        # Atomic commit: the support only ever changes to a fully certified
        # patched pencil; rejected candidates leave no partial state behind.
        current = best["candidate"]
        round_record["committed_patch"] = {
            key: value for key, value in best["record"].items() if key != "accepted"
        }
        rounds.append(round_record)

    return QSEExchangeResult(
        initial_indices=tuple(int(index) for index in selected_indices),
        final_indices=tuple(int(index) for index in current["indices"]),
        initial_summary=initial_summary,
        final_summary=_public_summary(current),
        rounds=tuple(rounds),
        config=cfg,
    )


def exchange_maintenance_payload(result: QSEExchangeResult) -> dict[str, Any]:
    """Manifest-ready telemetry payload for one exchange-maintenance run."""

    return {
        "schema_version": QSE_EXCHANGE_MAINTENANCE_SCHEMA_VERSION,
        "policy": "diagnostic_only_certified_joint_delete_add_exchange",
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "post_run_diagnostic_only": True,
        },
        "config": asdict(result.config),
        "initial": dict(result.initial_summary),
        "final": dict(result.final_summary),
        "committed_patch_count": sum(
            1 for round_record in result.rounds if round_record["committed_patch"] is not None
        ),
        "rounds": [dict(round_record) for round_record in result.rounds],
    }
