"""Compiled marginal Qiskit cost, Paper-I Eq. (compiled_marginal).

For a candidate ``A`` inserted at position ``p*`` into the active ansatz
``U``, the three raw coordinates are the compiled marginals

    C_x(r) = x(U (+)_{p*} A) - x(U)

for x in {two-qubit count, two-qubit depth, total depth}, both circuits
transpiled under the sealed results contract.  Every variational angle is
held at a common nonzero value so the candidate gate cannot be simplified
away and the coordinates are amplitude-independent.

Within each scored phase population, every coordinate is centered on its
median and scaled by its median absolute deviation, then compressed with
``(2/pi) atan``.  The three normalized coordinates are combined with the
sealed weights (0.30, 0.30, 0.25), and the master phase score is
``Delta E_t * (1 - C_bar/2)``.  This is exactly ``Delta E_t / K_t`` for
``K_t = 2 / (2 - C_bar)``.  Unchanged records reuse the compilation cache
across phases, so fresh transpilation is incurred only at Phase I.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

MARGINAL_QISKIT_COST_POLICY_V1 = (
    "phase123_compiled_marginal_qiskit_population_relative_v2"
)
MARGINAL_QISKIT_COORDINATE_WEIGHTS = {
    "delta_count_2q": 0.30,
    "delta_depth_2q": 0.30,
    "delta_depth": 0.25,
}
MARGINAL_QISKIT_SCALE_FLOOR = 1.0e-12
# Sec. Results compile contract.  These are not tunable here: a run that
# needs different values declares them on its route, and a mismatch is a
# contract violation rather than a local override.
COMPILE_BACKEND_NAME = "FakeMarrakesh"
COMPILE_OPTIMIZATION_LEVEL = 1
COMPILE_TRANSPILE_SEED = 7
# "Cost compilation holds every variational angle at a common nonzero
# value" -- amplitude independence depends on this being nonzero.
STRUCTURE_THETA_VALUE = 1.0


@dataclass(frozen=True, slots=True)
class CompiledMarginal:
    """One candidate's three compiled marginal coordinates."""

    delta_count_2q: int
    delta_depth_2q: int
    delta_depth: int

    def routed_cost_inputs(self) -> dict[str, float]:
        """Return the penalty inputs in the routed-oracle key space.

        The 1q and size coordinates exist in the construction and carry
        zero weight in this cost model, so they are routed as exact zeros
        rather than omitted -- ``costexp_signed_penalty_total`` indexes
        all five keys unconditionally.
        """

        return {
            "raw_delta_compiled_count_2q": float(self.delta_count_2q),
            "raw_delta_compiled_depth_2q": float(self.delta_depth_2q),
            "raw_delta_compiled_depth": float(self.delta_depth),
            "raw_delta_compiled_count_1q": 0.0,
            "raw_delta_compiled_size": 0.0,
        }


def marginal_cost_K(
    marginal: CompiledMarginal,
    penalty_weights: Mapping[str, float],
) -> float:
    """Return K_t for one candidate under the sealed weight set."""

    from pipelines.static_adapt.ra_adapt.costexp_realization import (
        costexp_signed_penalty_total,
    )

    value = costexp_signed_penalty_total(
        marginal.routed_cost_inputs(),
        penalty_weights,
    )
    if not math.isfinite(value):
        raise ValueError("Marginal Qiskit cost is not finite.")
    return float(value)


def discounted_phase_score(
    active_score: float,
    marginal: CompiledMarginal,
    penalty_weights: Mapping[str, float],
) -> float:
    """Return Delta E_t / (1 + K_t), the master phase score.

    A negative marginal is a genuine compile saving and is preserved with
    its sign, so a candidate that shortens the compiled circuit is
    rewarded rather than clamped to zero.  K_t <= -1 would invert the
    score, so it is refused instead.
    """

    score = float(active_score)
    denominator = 1.0 + marginal_cost_K(marginal, penalty_weights)
    if denominator <= 0.0:
        raise ValueError(
            "Marginal Qiskit cost drove the phase-score denominator "
            "non-positive."
        )
    return score / denominator


def _population_coordinate_statistics(
    marginals: Sequence[CompiledMarginal],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the manuscript median and MAD for the three coordinates."""

    import numpy as np

    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in MARGINAL_QISKIT_COORDINATE_WEIGHTS:
        values = np.asarray(
            [float(getattr(marginal, name)) for marginal in marginals],
            dtype=float,
        )
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        medians[name] = median
        scales[name] = max(MARGINAL_QISKIT_SCALE_FLOOR, mad)
    return medians, scales


def population_relative_score_factors(
    marginals: Sequence[CompiledMarginal],
) -> tuple[list[float], dict[str, Any]]:
    """Return the bounded manuscript score factor for one phase population."""

    population = list(marginals)
    if not population:
        raise ValueError("Marginal Qiskit population must not be empty.")
    medians, scales = _population_coordinate_statistics(population)
    weight_sum = float(sum(MARGINAL_QISKIT_COORDINATE_WEIGHTS.values()))
    rows: list[dict[str, Any]] = []
    factors: list[float] = []
    for marginal in population:
        normalized = {
            name: (2.0 / math.pi)
            * math.atan(
                (float(getattr(marginal, name)) - medians[name])
                / scales[name]
            )
            for name in MARGINAL_QISKIT_COORDINATE_WEIGHTS
        }
        aggregate = float(
            sum(
                MARGINAL_QISKIT_COORDINATE_WEIGHTS[name]
                * normalized[name]
                for name in MARGINAL_QISKIT_COORDINATE_WEIGHTS
            )
            / weight_sum
        )
        factor = float(1.0 - 0.5 * aggregate)
        if not math.isfinite(factor) or factor < 0.5 or factor > 1.5:
            raise RuntimeError("Marginal Qiskit score factor left [0.5, 1.5].")
        factors.append(factor)
        rows.append(
            {
                "raw": {
                    name: float(getattr(marginal, name))
                    for name in MARGINAL_QISKIT_COORDINATE_WEIGHTS
                },
                "normalized": normalized,
                "aggregate": aggregate,
                "score_factor": factor,
            }
        )
    receipt_base = {
        "schema": "paper_i_compiled_marginal_population_relative_v1",
        "policy": MARGINAL_QISKIT_COST_POLICY_V1,
        "coordinates": list(MARGINAL_QISKIT_COORDINATE_WEIGHTS),
        "weights": dict(MARGINAL_QISKIT_COORDINATE_WEIGHTS),
        "medians": medians,
        "mad_scales": scales,
        "scale_floor": MARGINAL_QISKIT_SCALE_FLOOR,
        "score_factor_formula": "1-0.5*weighted_mean(normalized_coordinates)",
        "score_factor_bounds": [0.5, 1.5],
        "rows": rows,
    }
    encoded = json.dumps(
        receipt_base,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    receipt = {
        **receipt_base,
        "population_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return factors, receipt


def marginal_from_feature(feature: Any) -> CompiledMarginal:
    """Recover the three exact compiled marginals carried by a feature."""

    backend = getattr(feature, "compiled_position_cost_backend", None)
    if not isinstance(backend, Mapping):
        raise ValueError("Compiled marginal feature lacks backend telemetry.")
    try:
        values = (
            float(backend["raw_delta_compiled_count_2q"]),
            float(backend["raw_delta_compiled_depth_2q"]),
            float(backend["raw_delta_compiled_depth"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Compiled marginal feature is incomplete.") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Compiled marginal feature contains a nonfinite value.")
    return CompiledMarginal(
        delta_count_2q=int(values[0]),
        delta_depth_2q=int(values[1]),
        delta_depth=int(values[2]),
    )


@dataclass
class CompiledMarginalCache:
    """Per-round transpile cache keyed by base structure and insertion.

    Paper-I incurs fresh transpilation only at Phase I; Phases II and III
    re-score the same identities and hit this cache.  ``base_key`` changes
    when the accepted ansatz changes, which is what retires an entry.
    """

    _entries: dict[tuple[str, str, int], CompiledMarginal] = field(
        default_factory=dict
    )
    transpile_calls: int = 0
    cache_hits: int = 0

    def clear(self) -> None:
        self._entries.clear()

    def get_or_compute(
        self,
        *,
        base_key: str,
        candidate_label: str,
        insertion_position: int,
        compute: Any,
    ) -> CompiledMarginal:
        key = (str(base_key), str(candidate_label), int(insertion_position))
        hit = self._entries.get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        value = compute()
        if not isinstance(value, CompiledMarginal):
            raise TypeError("Compiled marginal provider returned a bad type.")
        self._entries[key] = value
        self.transpile_calls += 1
        return value


def compiled_marginal_for_candidate(
    *,
    base_terms: Sequence[Any],
    candidate_term: Any,
    insertion_position: int,
    num_qubits: int,
    reference_state: Any,
    backend_obj: Any,
) -> CompiledMarginal:
    """Transpile base and trial circuits and return the three marginals."""

    from pipelines.hardcoded.adapt_circuit_execution import (
        build_structural_ansatz_circuit,
    )
    from pipelines.qiskit_backend_tools import (
        compile_circuit_for_backend,
        compiled_gate_stats,
        safe_circuit_depth,
    )

    def _coords(terms: Sequence[Any]) -> tuple[int, int, int]:
        _logical, circuit = build_structural_ansatz_circuit(
            list(terms),
            nq=int(num_qubits),
            ref_state=reference_state,
            structure_theta_value=STRUCTURE_THETA_VALUE,
        )
        compiled = compile_circuit_for_backend(
            circuit,
            backend_obj,
            seed_transpiler=COMPILE_TRANSPILE_SEED,
            optimization_level=COMPILE_OPTIMIZATION_LEVEL,
            initial_layout=None,
        )["compiled"]
        stats = compiled_gate_stats(compiled)
        return (
            int(stats["compiled_count_2q"]),
            int(stats["compiled_depth_2q"]),
            int(safe_circuit_depth(compiled)),
        )

    position = int(insertion_position)
    base = list(base_terms)
    if position < 0 or position > len(base):
        raise ValueError("Insertion position lies outside the active ansatz.")
    trial = base[:position] + [candidate_term] + base[position:]
    base_coords = _coords(base)
    trial_coords = _coords(trial)
    return CompiledMarginal(
        delta_count_2q=trial_coords[0] - base_coords[0],
        delta_depth_2q=trial_coords[1] - base_coords[1],
        delta_depth=trial_coords[2] - base_coords[2],
    )


class TranspiledMarginalCostOracle:
    """Drop-in ``estimate_insertion`` that transpiles instead of estimating.

    The graph-span oracle reports ``no_transpile`` and scores from c_hat
    span estimates, leaving every ``raw_delta_compiled_*`` coordinate
    unset.  This wrapper delegates everything else to it and replaces the
    insertion estimate with the compiled marginals of
    Eq. (compiled_marginal), populating those coordinates and the
    lambda-weighted ``penalty_total`` that every phase reads.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._cache = CompiledMarginalCache()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def cache(self) -> CompiledMarginalCache:
        return self._cache

    def estimate_insertion(
        self,
        snapshot: Any,
        *,
        candidate_term: Any,
        position_id: int,
        proxy_baseline: Any | None = None,
    ) -> Any:
        import dataclasses

        base = self._inner.estimate_insertion(
            snapshot,
            candidate_term=candidate_term,
            position_id=position_id,
            proxy_baseline=proxy_baseline,
        )
        base_ops = tuple(snapshot.base_ops)
        position = int(position_id)
        # Positions index insertion sites, so an endpoint append is a
        # legal position equal to the current logical depth.
        position = max(0, min(position, len(base_ops)))
        label = str(getattr(candidate_term, "label", candidate_term))
        marginal = self._cache.get_or_compute(
            base_key=f"{len(base_ops)}:{'|'.join(str(getattr(t, 'label', t)) for t in base_ops)}",
            candidate_label=label,
            insertion_position=position,
            compute=lambda: compiled_marginal_for_candidate(
                base_terms=base_ops,
                candidate_term=candidate_term,
                insertion_position=position,
                num_qubits=int(self._inner.num_qubits),
                reference_state=self._inner.ref_state,
                backend_obj=self._inner.targets[0].backend_obj,
            ),
        )
        weights = {
            "penalty_weight_2q": float(self._inner.config.weight_2q),
            "penalty_weight_depth": float(self._inner.config.weight_depth),
            "penalty_weight_size": float(self._inner.config.weight_size),
            "penalty_weight_depth_2q": float(
                self._inner.config.weight_depth_2q
            ),
            "penalty_weight_1q": float(self._inner.config.weight_1q),
        }
        penalty = marginal_cost_K(marginal, weights)
        return dataclasses.replace(
            base,
            penalty_total=float(penalty),
            depth_surrogate=float(marginal.delta_depth),
            hardware_cost_source=MARGINAL_QISKIT_COST_POLICY_V1,
            source_mode=MARGINAL_QISKIT_COST_POLICY_V1,
            raw_delta_compiled_count_2q=float(marginal.delta_count_2q),
            delta_compiled_count_2q=float(marginal.delta_count_2q),
            raw_delta_compiled_depth_2q=float(marginal.delta_depth_2q),
            delta_compiled_depth_2q=float(marginal.delta_depth_2q),
            raw_delta_compiled_depth=float(marginal.delta_depth),
            delta_compiled_depth=float(marginal.delta_depth),
            raw_delta_compiled_size=0.0,
            delta_compiled_size=0.0,
        )
