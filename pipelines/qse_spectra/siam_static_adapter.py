#!/usr/bin/env python3
"""Paper III static-spectroscopy adapter for the finite-box SIAM benchmark.

The SIAM scientific layer (``src/quantum/siam_{model,observables,pool,
reference}.py``) is ported verbatim from the verified Paper II implementation
at commit ``da7b26f3cd13d262407e6f373c449dedb2eb65c2`` (branch
``codex/paper-ii-search-profiles-20260824``). This adapter exposes that layer
to the Paper III static-QSE campaign without importing any time-dynamics
runtime: no time grids, no quench observables, no dynamics qualification.

Three contracts this module owns:

1. **Physics identity.** A :class:`SIAMPhysicsKey` is the immutable typed
   identity of one SIAM case (bath size, ``U``, ``V``, ``t_b``, JW layout,
   conserved sector) with a canonical JSON digest. The spectroscopy
   Hamiltonian is always the *interacting* ``V = 1/sqrt(2)`` Hamiltonian.
2. **Ground-state semantics.** The production spectroscopy seed must be a
   declared ground-state artifact of the interacting Hamiltonian. The
   Paper II decoupled bath/quench state (``build_initial_state``: the ``V=0``
   Fermi sea with one impurity electron) is structurally rejected as a
   production seed. An exact-sector ground state is available only as a
   clearly labeled diagnostic fixture for bounded smokes.
3. **Cost-proxy fidelity receipt.** ``graph_span_compiled_cost_receipt``
   compares the deterministic graph-span ``K`` against Qiskit-compiled
   two-qubit counts on a bounded, deterministic pool sample. It reports
   fidelity; it never queries exact spectra and never changes the online
   cost.

Reporting/diagnostic use only; nothing here feeds controller decisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.siam_model import (
    DEFAULT_HOPPING,
    QUENCHED_HYBRIDIZATION,
    SIAMLayout,
    build_initial_state,
    build_siam_hamiltonian,
    embed_sector_vector,
    sector_indices,
    sector_vector,
)
from src.quantum.siam_pool import build_siam_pool

SIAM_STATIC_ADAPTER_SCHEMA_VERSION = "paper_iii_siam_static_adapter_v1"
SIAM_SOURCE_COMMIT = "da7b26f3cd13d262407e6f373c449dedb2eb65c2"
SIAM_MODEL_FAMILY = "siam_finite_box"

GROUND_SOURCE_IMPORTED = "imported_interacting_ground_state_artifact"
GROUND_SOURCE_DIAGNOSTIC = "exact_sector_diagnostic_fixture"

_SECTOR_LEAK_TOLERANCE = 1.0e-10
_QUENCH_SEED_OVERLAP_REJECT = 1.0 - 1.0e-9
_QUENCH_SEED_PROVENANCE_MARKERS = (
    "quench",
    "decoupled",
    "v0_seed",
)

_CASE_TABLE: dict[str, dict[str, Any]] = {
    "siam_nb2_fixture": {"n_bath": 2, "u": 1.5, "role": "fixture"},
    "siam_nb4_pilot": {"n_bath": 4, "u": 1.5, "role": "pilot"},
}


class SIAMStaticAdapterError(ValueError):
    """Raised on SIAM identity, provenance, or ground-state violations."""


# ---------------------------------------------------------------------------
# Physics identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIAMPhysicsKey:
    """Immutable typed identity of one SIAM static-spectroscopy case."""

    case_id: str
    n_bath: int
    u: float
    v: float
    hopping: float
    n_up: int
    n_down: int
    n_qubits: int
    jw_order: str = "spin_blocked_no_taper"
    model_family: str = SIAM_MODEL_FAMILY

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "model_family": str(self.model_family),
            "case_id": str(self.case_id),
            "n_bath": int(self.n_bath),
            "u": float(self.u),
            "v": float(self.v),
            "hopping": float(self.hopping),
            "jw_order": str(self.jw_order),
            "sector": {"n_up": int(self.n_up), "n_down": int(self.n_down)},
            "n_qubits": int(self.n_qubits),
            "cutoff_kind": "n_bath",
            "schema": SIAM_STATIC_ADAPTER_SCHEMA_VERSION,
        }

    def digest(self) -> str:
        blob = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def siam_physics_key(case_id: str) -> SIAMPhysicsKey:
    """Resolve one of the declared SIAM cases into its typed identity."""

    spec = _CASE_TABLE.get(str(case_id))
    if spec is None:
        raise SIAMStaticAdapterError(
            f"unknown SIAM case {case_id!r}; declared cases: {sorted(_CASE_TABLE)}"
        )
    layout = SIAMLayout(int(spec["n_bath"]))
    n_up, n_down = layout.electron_counts()
    return SIAMPhysicsKey(
        case_id=str(case_id),
        n_bath=int(spec["n_bath"]),
        u=float(spec["u"]),
        v=float(QUENCHED_HYBRIDIZATION),
        hopping=float(DEFAULT_HOPPING),
        n_up=int(n_up),
        n_down=int(n_down),
        n_qubits=int(layout.n_qubits),
    )


def declared_siam_cases() -> tuple[str, ...]:
    return tuple(sorted(_CASE_TABLE))


# ---------------------------------------------------------------------------
# Pool and Hamiltonian
# ---------------------------------------------------------------------------


def build_siam_hamiltonian_for_key(key: SIAMPhysicsKey):
    """The interacting spectroscopy Hamiltonian at the key's ``U, V, t_b``."""

    return build_siam_hamiltonian(
        int(key.n_bath), u=float(key.u), v=float(key.v), hopping=float(key.hopping)
    )


def build_siam_pool_elements(key: SIAMPhysicsKey) -> tuple[Any, ...]:
    """The deterministic logical pool as Paper III QSE basis elements."""

    from pipelines.qse_spectra.core import polynomial_basis_element

    blocks = build_siam_pool(int(key.n_bath))
    elements = tuple(
        polynomial_basis_element(
            block.operator,
            name=f"siam::{block.family}::{block.label}",
            metadata={
                "family": str(block.family),
                "label": str(block.label),
                "source_commit": SIAM_SOURCE_COMMIT,
            },
        )
        for block in blocks
    )
    names = [str(element.name) for element in elements]
    if len(set(names)) != len(names):  # pragma: no cover - pool labels are unique
        raise SIAMStaticAdapterError("SIAM pool element names are not unique.")
    return elements


def ordered_pool_digest(elements: Sequence[Any]) -> str:
    names = [str(element.name) for element in elements]
    blob = json.dumps(names, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:24]


def siam_sector_spectrum(
    key: SIAMPhysicsKey, *, count: int
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Exact sector ground vector (embedded) and lowest ``count`` energies.

    Post-run reference data only: it annotates completed traces and scores
    fixed-accuracy crossings. It never selects a record, triggers a fallback,
    or stops an online algorithm.
    """

    from src.quantum.siam_reference import sector_hamiltonian

    sector_h = sector_hamiltonian(int(key.n_bath), u=float(key.u), v=float(key.v))
    dim = sector_h.shape[0]
    if int(count) < 1 or int(count) > dim:
        raise SIAMStaticAdapterError(
            f"reference count must lie in 1..{dim}; got {count}."
        )
    energies, vectors = np.linalg.eigh(sector_h)
    ground = embed_sector_vector(vectors[:, 0], int(key.n_bath))
    return np.asarray(ground, dtype=complex), tuple(
        float(x) for x in energies[: int(count)]
    )


# ---------------------------------------------------------------------------
# Ground-state semantics
# ---------------------------------------------------------------------------


def _sector_leakage(key: SIAMPhysicsKey, psi: np.ndarray) -> float:
    inside = sector_vector(psi, int(key.n_bath))
    total = float(np.vdot(psi, psi).real)
    kept = float(np.vdot(inside, inside).real)
    return max(0.0, total - kept)


def validate_spectroscopy_ground_state(
    key: SIAMPhysicsKey,
    psi: np.ndarray,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a production spectroscopy seed against the physics key.

    Rejects wrong register size, non-normalized states, sector leakage, and
    the Paper II decoupled bath/quench state (both by declared provenance
    markers and by direct overlap with the constructed ``V=0`` seed).
    """

    state = np.asarray(psi, dtype=complex).reshape(-1)
    expected = 1 << int(key.n_qubits)
    if state.size != expected:
        raise SIAMStaticAdapterError(
            f"state length {state.size} does not match register size {expected} "
            f"for case {key.case_id!r}."
        )
    norm = float(np.linalg.norm(state))
    if abs(norm - 1.0) > 1.0e-9:
        raise SIAMStaticAdapterError(
            f"spectroscopy seed must be normalized; got norm {norm!r}."
        )
    leakage = _sector_leakage(key, state)
    if leakage > _SECTOR_LEAK_TOLERANCE:
        raise SIAMStaticAdapterError(
            f"spectroscopy seed leaks {leakage:.3e} outside the "
            f"(N_up={key.n_up}, N_down={key.n_down}) sector."
        )

    declared = json.dumps(dict(provenance or {}), default=str).lower()
    for marker in _QUENCH_SEED_PROVENANCE_MARKERS:
        if marker in declared:
            raise SIAMStaticAdapterError(
                "provenance declares a decoupled/quench seed "
                f"(marker {marker!r}); the Paper II quench state is not the "
                "interacting spectroscopy ground state."
            )

    quench_seed = build_initial_state(int(key.n_bath), hopping=float(key.hopping))
    overlap = abs(complex(np.vdot(quench_seed, state)))
    if overlap >= _QUENCH_SEED_OVERLAP_REJECT:
        raise SIAMStaticAdapterError(
            "state matches the Paper II decoupled bath/quench seed "
            f"(|overlap| = {overlap:.12f}); production spectroscopy requires "
            "the declared interacting ground-state artifact."
        )

    return {
        "norm": norm,
        "sector_leakage": leakage,
        "quench_seed_overlap": overlap,
        "sector": {"n_up": int(key.n_up), "n_down": int(key.n_down)},
    }


def state_payload_sha256(psi: np.ndarray) -> str:
    state = np.ascontiguousarray(np.asarray(psi, dtype=complex).reshape(-1))
    return hashlib.sha256(state.tobytes()).hexdigest()


def load_imported_ground_state(
    key: SIAMPhysicsKey,
    manifest_path: Path,
    *,
    state_key: str = "auto",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate a declared interacting ground-state artifact."""

    from pipelines.qse_spectra.io import load_state_json

    path = Path(manifest_path)
    psi, io_provenance = load_state_json(
        path, expected_nq=int(key.n_qubits), state_key=state_key
    )
    checks = validate_spectroscopy_ground_state(key, psi, provenance=io_provenance)
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = {
        "ground_source": GROUND_SOURCE_IMPORTED,
        "diagnostic_only": False,
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha,
        "state_payload_sha256": state_payload_sha256(psi),
        "physics_digest": key.digest(),
        "validation": checks,
        "io_provenance": dict(io_provenance),
    }
    return psi, provenance


def exact_sector_ground_fixture(key: SIAMPhysicsKey) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact-sector interacting ground state, labeled diagnostic-only.

    Bounded smokes may seed QSE from this fixture; it is never paper-facing
    evidence and its provenance says so explicitly.
    """

    ground, energies = siam_sector_spectrum(key, count=1)
    provenance = {
        "ground_source": GROUND_SOURCE_DIAGNOSTIC,
        "diagnostic_only": True,
        "fixture_label": "exact_sector_ground_state_diagnostic_fixture",
        "paper_facing": False,
        "state_payload_sha256": state_payload_sha256(ground),
        "physics_digest": key.digest(),
        "ground_energy": float(energies[0]),
    }
    return ground, provenance


# ---------------------------------------------------------------------------
# Assembled static problem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SIAMStaticProblem:
    """One SIAM case's physics, shaped like the shared Paper III problem."""

    key: SIAMPhysicsKey
    hamiltonian: Any
    basis: tuple[Any, ...]
    costs: tuple[float, ...]
    resources: tuple[dict[str, float], ...]
    ground: np.ndarray
    ground_provenance: dict[str, Any]
    spectrum: tuple[float, ...]
    pool_digest: str

    @property
    def num_qubits(self) -> int:
        return int(self.key.n_qubits)

    @property
    def references(self) -> tuple[float, ...]:
        return tuple(self.spectrum[1:])

    @property
    def ground_energy(self) -> float:
        return float(self.spectrum[0])

    def resource_triple(self, indices: Sequence[int]) -> dict[str, float]:
        n2q = float(sum(self.resources[int(i)]["c_hat_2q"] for i in indices))
        d2q = float(sum(self.resources[int(i)]["c_hat_d"] for i in indices))
        one = float(sum(self.resources[int(i)]["c_hat_1q"] for i in indices))
        return {"n2q": n2q, "d2q": d2q, "dc": d2q + one}

    def arm_receipt(self) -> dict[str, Any]:
        return {
            "model_family": SIAM_MODEL_FAMILY,
            "case_id": str(self.key.case_id),
            "physics_digest": self.key.digest(),
            "pool_digest": self.pool_digest,
            "pool_size": len(self.basis),
            "num_qubits": int(self.key.n_qubits),
            "ground_source": str(self.ground_provenance.get("ground_source")),
            "diagnostic_only": bool(self.ground_provenance.get("diagnostic_only")),
            "granularity": "macro_records_uniform",
            "source_commit": SIAM_SOURCE_COMMIT,
        }


def load_siam_problem(
    case_id: str,
    *,
    ground_state_manifest: Path | None = None,
    allow_exact_diagnostic_fixture: bool = False,
    target_roots: int = 6,
) -> SIAMStaticProblem:
    """Assemble one SIAM static-spectroscopy problem.

    Production requires ``ground_state_manifest`` (a declared interacting
    ground-state artifact). ``allow_exact_diagnostic_fixture=True`` instead
    seeds from the exact-sector fixture, which stays labeled diagnostic-only.
    """

    from pipelines.qse_spectra.compiled_costs import (
        ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        annotate_basis_with_compiled_costs,
        resolve_cost_weights_preset,
    )
    from pipelines.qse_spectra.paper_iii_problem import assert_uniform_granularity

    key = siam_physics_key(case_id)
    if ground_state_manifest is not None:
        ground, ground_provenance = load_imported_ground_state(
            key, Path(ground_state_manifest)
        )
    elif allow_exact_diagnostic_fixture:
        ground, ground_provenance = exact_sector_ground_fixture(key)
    else:
        raise SIAMStaticAdapterError(
            "production SIAM spectroscopy requires a declared interacting "
            "ground-state artifact; pass ground_state_manifest, or opt into "
            "the diagnostic-only fixture explicitly."
        )

    hamiltonian = build_siam_hamiltonian_for_key(key)
    basis = build_siam_pool_elements(key)
    assert_uniform_granularity(basis)

    rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=int(key.n_qubits),
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=resolve_cost_weights_preset("two_qubit_only_v1"),
    )
    costs = tuple(float(r.scalarized_canonical_cost) for r in rows)
    resources = tuple(
        {
            "c_hat_2q": float(r.estimate.c_hat_2q),
            "c_hat_d": float(r.estimate.c_hat_d),
            "c_hat_1q": float(r.estimate.c_hat_1q),
        }
        for r in rows
    )

    _ground_embedded, spectrum = siam_sector_spectrum(
        key, count=int(target_roots) + 1
    )

    return SIAMStaticProblem(
        key=key,
        hamiltonian=hamiltonian,
        basis=basis,
        costs=costs,
        resources=resources,
        ground=np.asarray(ground, dtype=complex),
        ground_provenance=dict(ground_provenance),
        spectrum=spectrum,
        pool_digest=ordered_pool_digest(basis),
    )


# ---------------------------------------------------------------------------
# Graph-span vs compiled-cost receipt
# ---------------------------------------------------------------------------


def _family_stratified_sample(
    elements: Sequence[Any], *, per_family: int
) -> list[int]:
    """Deterministic per-family sample indices, in pool order."""

    by_family: dict[str, list[int]] = {}
    for index, element in enumerate(elements):
        family = str((element.metadata or {}).get("family", "unknown"))
        by_family.setdefault(family, []).append(index)
    chosen: list[int] = []
    for family in sorted(by_family):
        members = by_family[family]
        take = min(int(per_family), len(members))
        if take == len(members):
            chosen.extend(members)
            continue
        # evenly spaced across the family's pool order, endpoints included
        positions = [
            round(i * (len(members) - 1) / (take - 1)) if take > 1 else 0
            for i in range(take)
        ]
        chosen.extend(members[p] for p in sorted(set(positions)))
    return sorted(set(chosen))


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    result = spearmanr(list(xs), list(ys))
    return float(result.correlation)


def _top_decile_overlap(xs: Sequence[float], ys: Sequence[float]) -> float:
    count = len(xs)
    take = max(1, count // 10)
    top_x = set(sorted(range(count), key=lambda i: (-float(xs[i]), i))[:take])
    top_y = set(sorted(range(count), key=lambda i: (-float(ys[i]), i))[:take])
    return len(top_x & top_y) / float(take)


def graph_span_compiled_cost_receipt(
    case_id: str,
    *,
    per_family_sample: int = 4,
    compiled_two_qubit_counts: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Bounded offline ranking-stability receipt for the SIAM cost proxy.

    Compares graph-span two-qubit counts against Qiskit-compiled two-qubit
    counts on the full ``N_b=2`` pool or a deterministic family-stratified
    ``N_b=4`` sample. Records the correlation, top-decile overlap, backend
    and transpiler seed, and every compared record.

    ``compiled_two_qubit_counts`` lets a test inject compiled values for the
    sampled elements (in sample order) instead of running the transpiler.
    The receipt reports proxy fidelity only: it never queries exact spectra
    and never changes the online cost.
    """

    from pipelines.qse_spectra.compiled_costs import (
        ORACLE_KIND_BACKEND_TRANSPILE,
        ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        annotate_basis_with_compiled_costs,
        resolve_cost_weights_preset,
    )
    from pipelines.static_adapt.hh_backend_compile_oracle import BackendCompileConfig

    key = siam_physics_key(case_id)
    elements = build_siam_pool_elements(key)
    if int(key.n_bath) <= 2:
        sample = list(range(len(elements)))
        sample_policy = "full_pool"
    else:
        sample = _family_stratified_sample(elements, per_family=int(per_family_sample))
        sample_policy = f"family_stratified_per_family_{int(per_family_sample)}"
    sampled = [elements[i] for i in sample]

    weights = resolve_cost_weights_preset("two_qubit_only_v1")
    span_rows = annotate_basis_with_compiled_costs(
        sampled,
        num_qubits=int(key.n_qubits),
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
        cost_weights=weights,
    )
    span_2q = [float(r.estimate.c_hat_2q) for r in span_rows]

    backend_config = BackendCompileConfig()
    if compiled_two_qubit_counts is not None:
        if len(compiled_two_qubit_counts) != len(sampled):
            raise SIAMStaticAdapterError(
                f"injected compiled counts have length "
                f"{len(compiled_two_qubit_counts)}; sample has {len(sampled)}."
            )
        compiled_2q = [float(x) for x in compiled_two_qubit_counts]
        compiled_source = "injected_for_test"
    else:
        compiled_rows = annotate_basis_with_compiled_costs(
            sampled,
            num_qubits=int(key.n_qubits),
            oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE,
            oracle_config=backend_config,
            cost_weights=weights,
        )
        compiled_2q = [float(r.estimate.c_hat_2q) for r in compiled_rows]
        compiled_source = "backend_transpile_single_v1"

    records = [
        {
            "pool_index": int(pool_index),
            "name": str(element.name),
            "family": str((element.metadata or {}).get("family", "unknown")),
            "graph_span_2q": float(span),
            "compiled_2q": float(compiled),
        }
        for pool_index, element, span, compiled in zip(
            sample, sampled, span_2q, compiled_2q
        )
    ]

    return {
        "schema_version": "siam_graph_span_compiled_cost_receipt_v1",
        "case_id": str(key.case_id),
        "physics_digest": key.digest(),
        "sample_policy": sample_policy,
        "sample_size": len(sample),
        "pool_size": len(elements),
        "spearman_rank_correlation": _spearman(span_2q, compiled_2q),
        "top_decile_overlap": _top_decile_overlap(span_2q, compiled_2q),
        "compiled_source": compiled_source,
        "backend": {
            "requested_backend_name": str(backend_config.requested_backend_name),
            "seed_transpiler": int(backend_config.seed_transpiler),
            "optimization_level": int(backend_config.optimization_level),
        },
        "records": records,
        "controller_boundary": {
            "feeds_controller_decisions": False,
            "controller_usable": False,
            "changes_online_cost": False,
            "queries_exact_spectra": False,
        },
    }


__all__ = [
    "GROUND_SOURCE_DIAGNOSTIC",
    "GROUND_SOURCE_IMPORTED",
    "SIAMPhysicsKey",
    "SIAMStaticAdapterError",
    "SIAMStaticProblem",
    "SIAM_MODEL_FAMILY",
    "SIAM_SOURCE_COMMIT",
    "SIAM_STATIC_ADAPTER_SCHEMA_VERSION",
    "build_siam_hamiltonian_for_key",
    "build_siam_pool_elements",
    "declared_siam_cases",
    "exact_sector_ground_fixture",
    "graph_span_compiled_cost_receipt",
    "load_imported_ground_state",
    "load_siam_problem",
    "ordered_pool_digest",
    "siam_physics_key",
    "siam_sector_spectrum",
    "state_payload_sha256",
    "validate_spectroscopy_ground_state",
]
