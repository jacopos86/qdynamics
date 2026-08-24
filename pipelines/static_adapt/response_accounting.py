"""Estimator-charge accounting for the default no-prune Paper-I route.

This module owns estimator identities, ledger charges, and cache replay.  It
does not compute or return gradient, Hessian, or Gram values.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from threading import local
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.estimator_call_ledger import (
    CALL_KEY_SCHEMA_V2,
    EstimatorCallKey,
    EstimatorCallLedger,
    PhysicalTangentOperandIdentity,
    projective_state_fingerprint as estimator_projective_state_fingerprint,
)
from pipelines.static_adapt.resume_scaffold import digest_jsonable
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _physical_generator_block_payload(
    layout: AnsatzParameterLayout,
    logical_index: int,
) -> dict[str, Any]:
    """Return the label-free physical generator carried by one block."""

    index = int(logical_index)
    if index < 0 or index >= int(layout.logical_parameter_count):
        raise IndexError(
            "logical tangent index is outside the accepted parameter layout: "
            f"index={index}, count={layout.logical_parameter_count}."
        )
    block = layout.blocks[index]
    return {
        "execution_mode": str(block.execution_mode),
        "runtime_terms_exyz": [
            {
                "pauli_exyz": str(spec.pauli_exyz),
                "coeff_real": float(spec.coeff_real),
                "nq": int(spec.nq),
            }
            for spec in block.terms
        ],
    }


def _physical_tangent_operand_identity(
    selected_ops_now: Sequence[Any],
    logical_index: int,
    *,
    logical_theta_now: Sequence[float] | np.ndarray,
    parameterization_mode: str,
    zero_amplitude_indices: Sequence[int] = (),
) -> PhysicalTangentOperandIdentity:
    """Build a route- and label-independent tangent estimator identity.

    The ordered derivative-circuit and tie-map fingerprints describe the
    accepted execution chart.  Candidate/accepted labels, branch IDs,
    optimizer scopes, and whitening-frame gauges are deliberately excluded.
    """

    ops = list(selected_ops_now)
    layout = build_parameter_layout(
        ops,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    logical_theta = np.asarray(logical_theta_now, dtype=float).reshape(-1)
    if int(logical_theta.size) != int(layout.logical_parameter_count):
        raise ValueError(
            "logical tangent identity requires one angle per accepted block: "
            f"theta={int(logical_theta.size)}, "
            f"blocks={int(layout.logical_parameter_count)}."
        )
    if not np.all(np.isfinite(logical_theta)):
        raise ValueError("logical tangent identity angles must be finite.")
    index = int(logical_index)
    zero_set = {
        int(value)
        for value in zero_amplitude_indices
        if 0 <= int(value) < int(layout.logical_parameter_count)
    }
    # A newly inserted zero-amplitude block is the identity circuit.  Remove
    # it from the surrounding derivative chart for inherited tangents, while
    # retaining it when it is itself the differentiated coordinate.  This is
    # the canonical same-ray extension rule and prevents index shifts from
    # turning reused old-old geometry into fictitious new measurements.
    canonical_indices = [
        offset
        for offset in range(int(layout.logical_parameter_count))
        if offset not in zero_set or offset == index
    ]
    canonical_position = int(canonical_indices.index(index))
    generator_blocks = [
        _physical_generator_block_payload(layout, offset)
        for offset in canonical_indices
    ]
    generator_payload = _physical_generator_block_payload(layout, index)
    mode = str(parameterization_mode).strip().lower()
    derivative_circuit_fingerprint = digest_jsonable(
        {
            "schema": "ordered_physical_derivative_circuit_v1",
            "parameterization_mode": mode,
            "ordered_generator_blocks": generator_blocks,
            "ordered_logical_angles": [
                float(logical_theta[offset])
                for offset in canonical_indices
            ],
        }
    )
    tie_map_fingerprint = digest_jsonable(
        {
            "schema": "physical_parameterization_tie_map_v1",
            "parameterization_mode": mode,
            "layout_mode": str(layout.mode),
            "logical_parameter_count": int(len(canonical_indices)),
            "runtime_parameter_count": int(
                sum(
                    int(layout.blocks[offset].runtime_count)
                    for offset in canonical_indices
                )
            ),
            "blocks": [
                {
                    "logical_index": int(canonical_offset),
                    "runtime_start": int(
                        sum(
                            int(layout.blocks[prior].runtime_count)
                            for prior in canonical_indices[:canonical_offset]
                        )
                    ),
                    "runtime_count": int(
                        layout.blocks[source_offset].runtime_count
                    ),
                }
                for canonical_offset, source_offset in enumerate(
                    canonical_indices
                )
            ],
        }
    )
    return PhysicalTangentOperandIdentity(
        derivative_circuit_fingerprint=str(derivative_circuit_fingerprint),
        generator_fingerprint=digest_jsonable(
            {
                "schema": "physical_generator_block_v1",
                **generator_payload,
            }
        ),
        insertion_position=canonical_position,
        parameterization_tie_map_fingerprint=str(tie_map_fingerprint),
    )


def _candidate_physical_tangent_operand_identity(
    selected_ops_now: Sequence[Any],
    logical_theta_now: Sequence[float] | np.ndarray,
    candidate_term: Any,
    *,
    insertion_position: int,
    parameterization_mode: str,
) -> PhysicalTangentOperandIdentity:
    """Identify a zero-amplitude candidate in its post-admission chart."""

    ops = list(selected_ops_now)
    logical_theta = np.asarray(logical_theta_now, dtype=float).reshape(-1)
    if int(logical_theta.size) != len(ops):
        raise ValueError(
            "candidate tangent identity requires one angle per accepted block."
        )
    position = max(0, min(int(insertion_position), len(ops)))
    ops.insert(position, candidate_term)
    augmented_theta = np.insert(logical_theta, position, 0.0)
    return _physical_tangent_operand_identity(
        ops,
        position,
        logical_theta_now=augmented_theta,
        parameterization_mode=str(parameterization_mode),
        zero_amplitude_indices=(position,),
    )


_CANDIDATE_CACHE_ESTIMATOR_REPLAY_SCHEMA = (
    "static_adapt_candidate_record_estimator_replay_v1"
)
_CANDIDATE_CACHE_ESTIMATOR_REPLAY_FIELD = (
    "_candidate_record_estimator_replay"
)


@dataclass(slots=True)
class _DefaultNoPruneEstimatorService:
    """State-keyed estimator accounting used by the exact numerical session."""

    ledger: EstimatorCallLedger
    hamiltonian_fingerprint: str
    backend_fingerprint: str
    precision_contract: str
    parameterization_mode: str
    pool: tuple[AnsatzTerm, ...]
    pool_family_ids: tuple[str, ...]
    phase1_residual_indices: frozenset[int]
    call_context: Any = field(default_factory=local)

    def _estimator_consumer_branch_id(self, explicit_branch_id: str | int | None=None) -> str | None:
        if explicit_branch_id is not None:
            return str(explicit_branch_id)
        branch_value = getattr(self.call_context, 'branch_id', None)
        return None if branch_value is None else str(branch_value)

    def _begin_candidate_cache_estimator_capture(self, *, cache_key: str) -> None:
        """Capture estimator-call identities skipped by a future cache hit.

            Candidate-record caching avoids the quantum-estimation work that built
            a feature record.  It must not erase that work from a new run's query
            ledger: a disk cache is an execution accelerator, not evidence that the
            historical measurements were free.  The capture is thread-local so
            parallel candidate workers retain their own deterministic replay list.
            """
        self.call_context.candidate_cache_estimator_capture = {'schema': _CANDIDATE_CACHE_ESTIMATOR_REPLAY_SCHEMA, 'cache_key': str(cache_key), 'entries': []}

    def _finish_candidate_cache_estimator_capture(self, *, cache_key: str) -> dict[str, Any]:
        capture = getattr(self.call_context, 'candidate_cache_estimator_capture', None)
        if not isinstance(capture, Mapping):
            raise RuntimeError('Candidate-record cache store lacks its estimator replay capture.')
        if str(capture.get('schema', '')) != _CANDIDATE_CACHE_ESTIMATOR_REPLAY_SCHEMA:
            raise RuntimeError('Candidate-record estimator replay capture has an invalid schema.')
        if str(capture.get('cache_key', '')) != str(cache_key):
            raise RuntimeError('Candidate-record estimator replay capture changed cache keys.')
        entries = capture.get('entries')
        if not isinstance(entries, list):
            raise RuntimeError('Candidate-record estimator replay capture lacks its entry list.')
        delattr(self.call_context, 'candidate_cache_estimator_capture')
        return {'schema': _CANDIDATE_CACHE_ESTIMATOR_REPLAY_SCHEMA, 'cache_key': str(cache_key), 'entry_count': int(len(entries)), 'entries': copy.deepcopy(entries)}

    def _replay_candidate_cache_estimator_calls(self, cached_record: dict[str, Any], *, cache_key: str) -> dict[str, Any]:
        replay_payload = cached_record.pop(_CANDIDATE_CACHE_ESTIMATOR_REPLAY_FIELD, None)
        if not isinstance(replay_payload, Mapping):
            raise RuntimeError('Candidate-record cache hit lacks the estimator-call replay receipt required by its cache code version.')
        if str(replay_payload.get('schema', '')) != _CANDIDATE_CACHE_ESTIMATOR_REPLAY_SCHEMA:
            raise RuntimeError('Candidate-record cache estimator replay receipt has an invalid schema.')
        if str(replay_payload.get('cache_key', '')) != str(cache_key):
            raise RuntimeError('Candidate-record cache estimator replay receipt changed cache keys.')
        entries = replay_payload.get('entries')
        if not isinstance(entries, list) or int(replay_payload.get('entry_count', -1)) != len(entries):
            raise RuntimeError('Candidate-record cache estimator replay receipt has an invalid entry count.')
        replayed_ids: list[str] = []
        if self.ledger is not None:
            for raw_entry in entries:
                if not isinstance(raw_entry, Mapping):
                    raise RuntimeError('Candidate-record cache estimator replay entry is malformed.')
                identity_payload = raw_entry.get('identity')
                if not isinstance(identity_payload, Mapping):
                    raise RuntimeError('Candidate-record cache estimator replay entry lacks its call identity.')
                identity = EstimatorCallKey.from_dict(identity_payload)
                expected_primitive_id = str(raw_entry.get('primitive_id', ''))
                if identity.primitive_id != expected_primitive_id:
                    raise RuntimeError('Candidate-record cache estimator replay primitive ID does not match its serialized call identity.')
                receipt = self.ledger.record_call(identity, component=str(raw_entry.get('component', '')), consumer_scope=str(raw_entry.get('consumer_scope', '')), branch_id=None if raw_entry.get('branch_id') is None else str(raw_entry.get('branch_id')))
                if str(receipt.primitive_id) != expected_primitive_id:
                    raise RuntimeError('Candidate-record cache estimator replay registered an unexpected primitive ID.')
                replayed_ids.append(str(receipt.primitive_id))
        return {'schema': _CANDIDATE_CACHE_ESTIMATOR_REPLAY_SCHEMA, 'status': 'replayed' if self.ledger is not None else 'ledger_disabled', 'entry_count': int(len(entries)), 'replayed_entry_count': int(len(replayed_ids)), 'primitive_ids': list(replayed_ids)}

    def _record_estimator_primitive(self, *, state: np.ndarray, component: str, consumer_scope: str, primitive_kind: str, observable_or_formula_identity: str, operand_identity: PhysicalTangentOperandIdentity | str | None=None, symmetric_pair: tuple[Any, Any] | None=None, branch_id: str | int | None=None) -> Any:
        capture = getattr(self.call_context, 'candidate_cache_estimator_capture', None)
        if self.ledger is None and (not isinstance(capture, Mapping)):
            return None
        uses_physical_tangent_identity = bool(operand_identity is not None or symmetric_pair is not None)
        key_kwargs: dict[str, Any] = {'projective_state_fingerprint': estimator_projective_state_fingerprint(np.asarray(state, dtype=complex).reshape(-1)), 'hamiltonian_fingerprint': str(self.hamiltonian_fingerprint), 'backend_fingerprint': str(self.backend_fingerprint), 'precision_contract': str(self.precision_contract), 'primitive_kind': str(primitive_kind), 'observable_or_formula_identity': str(observable_or_formula_identity), 'operand_identity': operand_identity, 'symmetric_pair': symmetric_pair}
        if uses_physical_tangent_identity:
            key_kwargs['schema'] = CALL_KEY_SCHEMA_V2
        identity = EstimatorCallKey(**key_kwargs)
        resolved_branch_id = self._estimator_consumer_branch_id(branch_id)
        if isinstance(capture, Mapping):
            capture_entries = capture.get('entries')
            if not isinstance(capture_entries, list):
                raise RuntimeError('Candidate-record estimator capture lost its entry list.')
            capture_entries.append({'identity': identity.as_dict(), 'primitive_id': str(identity.primitive_id), 'component': str(component), 'consumer_scope': str(consumer_scope), 'branch_id': resolved_branch_id})
        if self.ledger is None:
            return None
        return self.ledger.record_call(identity, component=str(component), consumer_scope=str(consumer_scope), branch_id=resolved_branch_id)

    def _active_physical_tangent(self, selected_ops_now: Sequence[Any], logical_theta_now: Sequence[float] | np.ndarray, logical_index: int, *, zero_amplitude_indices: Sequence[int]=()) -> PhysicalTangentOperandIdentity:
        return _physical_tangent_operand_identity(selected_ops_now, int(logical_index), logical_theta_now=logical_theta_now, parameterization_mode=str(self.parameterization_mode), zero_amplitude_indices=zero_amplitude_indices)

    def _candidate_physical_tangent(self, selected_ops_now: Sequence[Any], logical_theta_now: Sequence[float] | np.ndarray, candidate_term: Any, *, insertion_position: int | None=None) -> PhysicalTangentOperandIdentity:
        position = len(selected_ops_now) if insertion_position is None else int(insertion_position)
        return _candidate_physical_tangent_operand_identity(selected_ops_now, logical_theta_now, candidate_term, insertion_position=position, parameterization_mode=str(self.parameterization_mode))

    def _record_active_gradient_primitives(self, *, state: np.ndarray, selected_ops_now: Sequence[Any], logical_theta_now: Sequence[float] | np.ndarray, active_indices: Sequence[int], consumer_scope: str, branch_id: str | int | None=None) -> dict[str, Any]:
        """Charge the active gradient vector consumed by a joint response."""
        primitive_ids: list[str] = []
        newly_charged = 0
        for logical_index in [int(value) for value in active_indices]:
            receipt = self._record_estimator_primitive(state=np.asarray(state, dtype=complex), component='N_grad', consumer_scope=str(consumer_scope), primitive_kind='coordinate_gradient', observable_or_formula_identity='coordinate_energy_gradient_v2', operand_identity=self._active_physical_tangent(selected_ops_now, logical_theta_now, logical_index), branch_id=branch_id)
            if receipt is not None:
                primitive_ids.append(str(receipt.primitive_id))
                if bool(receipt.charged):
                    newly_charged += 1
        return {'schema': 'phase3_active_gradient_query_accounting_v1', 'active_coordinate_count': int(len(active_indices)), 'new_unique_gradients_charged': int(newly_charged), 'deduplicated_or_ledger_disabled_count': int(len(active_indices) - newly_charged), 'primitive_ids': list(primitive_ids), 'component': 'N_grad', 'consumer_scope': str(consumer_scope)}

    def _record_gradient_surface_primitives(self, *, state: np.ndarray, available_indices_now: Sequence[int] | set[int], selected_ops_now: Sequence[Any], logical_theta_now: Sequence[float] | np.ndarray, consumer_scope: str, branch_id: str | int | None=None) -> None:
        append_position = int(len(selected_ops_now))
        for pool_index in sorted((int(value) for value in available_indices_now)):
            coordinate = self._candidate_physical_tangent(selected_ops_now, logical_theta_now, self.pool[int(pool_index)], insertion_position=append_position)
            self._record_estimator_primitive(state=np.asarray(state, dtype=complex), component='N_grad', consumer_scope=str(consumer_scope), primitive_kind='coordinate_gradient', observable_or_formula_identity='coordinate_energy_gradient_v2', operand_identity=coordinate, branch_id=branch_id)

    def _record_candidate_self_metric_primitive(self, *, state: np.ndarray, selected_ops_now: Sequence[Any], logical_theta_now: Sequence[float] | np.ndarray, candidate_term: Any, consumer_scope: str, insertion_position: int | None=None, branch_id: str | int | None=None) -> str | None:
        coordinate = self._candidate_physical_tangent(selected_ops_now, logical_theta_now, candidate_term, insertion_position=insertion_position)
        receipt = self._record_estimator_primitive(state=np.asarray(state, dtype=complex), component='N_metric', consumer_scope=str(consumer_scope), primitive_kind='metric_element', observable_or_formula_identity='fubini_study_metric_v2', symmetric_pair=(coordinate, coordinate), branch_id=branch_id)
        return None if receipt is None else str(receipt.primitive_id)

    def _record_scaffold_geometry_primitives(self, *, state: np.ndarray, selected_ops_now: Sequence[Any], logical_theta_now: Sequence[float] | np.ndarray, refit_window_indices: Sequence[int], consumer_scope: str, branch_id: str | int | None=None, record_metric: bool=True, record_hessian: bool=True, primitive_ids_by_kind: dict[str, list[str]] | None=None) -> tuple[str, ...]:
        primitive_ids: list[str] = []
        coordinates = [self._active_physical_tangent(selected_ops_now, logical_theta_now, int(index)) for index in refit_window_indices]
        for left_offset, left in enumerate(coordinates):
            for right in coordinates[left_offset:]:
                primitive_specs: list[tuple[str, str]] = []
                if bool(record_metric):
                    primitive_specs.append(('metric_element', 'fubini_study_metric_v2'))
                if bool(record_hessian):
                    primitive_specs.append(('hessian_element', 'energy_hessian_v2'))
                for primitive_kind, formula in primitive_specs:
                    receipt = self._record_estimator_primitive(state=np.asarray(state, dtype=complex), component='N_metric', consumer_scope=str(consumer_scope), primitive_kind=primitive_kind, observable_or_formula_identity=formula, symmetric_pair=(left, right), branch_id=branch_id)
                    if receipt is not None:
                        primitive_id = str(receipt.primitive_id)
                        primitive_ids.append(primitive_id)
                        if primitive_ids_by_kind is not None:
                            primitive_ids_by_kind.setdefault(str(primitive_kind), []).append(primitive_id)
        return tuple(sorted(set(primitive_ids)))

    def _record_candidate_geometry_primitives(self, *, state: np.ndarray, selected_ops_now: Sequence[Any], logical_theta_now: Sequence[float] | np.ndarray, refit_window_indices: Sequence[int], candidate_term: Any, consumer_scope: str, branch_id: str | int | None=None, insertion_position: int | None=None, record_metric: bool=True, record_hessian: bool=True, record_self_metric: bool | None=None, record_self_hessian: bool | None=None, primitive_ids_by_kind: dict[str, list[str]] | None=None) -> tuple[str, ...]:
        primitive_ids: list[str] = []
        candidate_position = len(selected_ops_now) if insertion_position is None else max(0, min(int(insertion_position), len(selected_ops_now)))
        augmented_ops = list(selected_ops_now)
        augmented_ops.insert(candidate_position, candidate_term)
        augmented_theta = np.insert(np.asarray(logical_theta_now, dtype=float).reshape(-1), candidate_position, 0.0)
        candidate_coordinate = self._active_physical_tangent(augmented_ops, augmented_theta, candidate_position, zero_amplitude_indices=(candidate_position,))
        active_coordinates = [self._active_physical_tangent(augmented_ops, augmented_theta, int(index) + (1 if candidate_position <= int(index) else 0), zero_amplitude_indices=(candidate_position,)) for index in refit_window_indices]
        for active_coordinate in active_coordinates:
            primitive_specs = []
            if bool(record_metric):
                primitive_specs.append(('metric_element', 'fubini_study_metric_v2'))
            if bool(record_hessian):
                primitive_specs.append(('hessian_element', 'energy_hessian_v2'))
            for primitive_kind, formula in primitive_specs:
                receipt = self._record_estimator_primitive(state=np.asarray(state, dtype=complex), component='N_metric', consumer_scope=str(consumer_scope), primitive_kind=primitive_kind, observable_or_formula_identity=formula, symmetric_pair=(active_coordinate, candidate_coordinate), branch_id=branch_id)
                if receipt is not None:
                    primitive_id = str(receipt.primitive_id)
                    primitive_ids.append(primitive_id)
                    if primitive_ids_by_kind is not None:
                        primitive_ids_by_kind.setdefault(str(primitive_kind), []).append(primitive_id)
        self_metric = bool(record_metric) if record_self_metric is None else bool(record_self_metric)
        self_hessian = bool(record_hessian) if record_self_hessian is None else bool(record_self_hessian)
        primitive_specs = []
        if self_metric:
            primitive_specs.append(('metric_element', 'fubini_study_metric_v2'))
        if self_hessian:
            primitive_specs.append(('hessian_element', 'energy_hessian_v2'))
        for primitive_kind, formula in primitive_specs:
            receipt = self._record_estimator_primitive(state=np.asarray(state, dtype=complex), component='N_metric', consumer_scope=str(consumer_scope), primitive_kind=primitive_kind, observable_or_formula_identity=formula, symmetric_pair=(candidate_coordinate, candidate_coordinate), branch_id=branch_id)
            if receipt is not None:
                primitive_id = str(receipt.primitive_id)
                primitive_ids.append(primitive_id)
                if primitive_ids_by_kind is not None:
                    primitive_ids_by_kind.setdefault(str(primitive_kind), []).append(primitive_id)
        return tuple(sorted(set(primitive_ids)))

    def _record_candidate_pair_geometry_primitives(
        self,
        *,
        state: np.ndarray,
        selected_ops_now: Sequence[Any],
        logical_theta_now: Sequence[float] | np.ndarray,
        left_record: Mapping[str, Any],
        right_record: Mapping[str, Any],
        consumer_scope: str,
        pair_cache_key: str,
        winning_pair: bool,
        batch_kind: str,
    ) -> dict[str, Any]:
        """Record acquisition metadata; never recompute the measured pair."""

        if batch_kind == "greedy":
            strategy_label = "Greedy"
            accounting_schema = (
                "greedy_batch_pair_estimator_accounting_v1"
            )
        elif batch_kind == "combinatorial":
            strategy_label = "Combinatorial"
            accounting_schema = (
                "combinatorial_batch_pair_estimator_accounting_v1"
            )
        else:
            raise ValueError(
                f"Unsupported pair-accounting batch kind: {batch_kind!r}"
            )
        records = (dict(left_record), dict(right_record))
        positions = tuple(
            int(record.get("position_id", len(selected_ops_now)))
            for record in records
        )
        candidates_by_position: dict[int, list[int]] = {}
        for member_index, position in enumerate(positions):
            if position < 0 or position > len(selected_ops_now):
                raise RuntimeError(
                    f"{strategy_label} pair accounting received an invalid "
                    "insertion position."
                )
            candidates_by_position.setdefault(position, []).append(
                member_index
            )
        combined_ops: list[Any] = []
        combined_theta: list[float] = []
        candidate_indices: dict[int, int] = {}
        logical_theta = np.asarray(
            logical_theta_now,
            dtype=float,
        ).reshape(-1)
        if logical_theta.size != len(selected_ops_now):
            raise RuntimeError(
                f"{strategy_label} pair accounting requires one logical "
                "coordinate per accepted operator."
            )
        for position in range(len(selected_ops_now) + 1):
            for member_index in candidates_by_position.get(position, ()):
                candidate_term = records[member_index].get("candidate_term")
                if candidate_term is None:
                    raise RuntimeError(
                        f"{strategy_label} pair accounting lost candidate_term."
                    )
                candidate_indices[member_index] = len(combined_ops)
                combined_ops.append(candidate_term)
                combined_theta.append(0.0)
            if position < len(selected_ops_now):
                combined_ops.append(selected_ops_now[position])
                combined_theta.append(float(logical_theta[position]))
        zero_indices = tuple(
            candidate_indices[index] for index in range(len(records))
        )
        coordinates = tuple(
            self._active_physical_tangent(
                combined_ops,
                combined_theta,
                candidate_indices[index],
                zero_amplitude_indices=zero_indices,
            )
            for index in range(len(records))
        )
        primitive_rows: list[dict[str, Any]] = []
        for primitive_kind, formula in (
            ("metric_element", "fubini_study_metric_v2"),
            ("hessian_element", "energy_hessian_v2"),
        ):
            receipt = self._record_estimator_primitive(
                state=np.asarray(state, dtype=complex),
                component="N_metric",
                consumer_scope=str(consumer_scope),
                primitive_kind=primitive_kind,
                observable_or_formula_identity=formula,
                symmetric_pair=coordinates,
                branch_id=None,
            )
            primitive_rows.append(
                {
                    "primitive_kind": primitive_kind,
                    "primitive_id": (
                        None
                        if receipt is None
                        else str(receipt.primitive_id)
                    ),
                    "charged": (
                        False
                        if receipt is None
                        else bool(receipt.charged)
                    ),
                }
            )
        return {
            "schema": accounting_schema,
            "pair_cache_key": str(pair_cache_key),
            "left_candidate_pool_index": int(
                left_record.get("candidate_pool_index", -1)
            ),
            "right_candidate_pool_index": int(
                right_record.get("candidate_pool_index", -1)
            ),
            "left_position_id": int(left_record.get("position_id", -1)),
            "right_position_id": int(right_record.get("position_id", -1)),
            "winning_pair": bool(winning_pair),
            "consumer_scope": str(consumer_scope),
            "primitive_rows": primitive_rows,
        }
