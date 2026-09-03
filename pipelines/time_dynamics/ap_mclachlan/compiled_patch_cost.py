"""Complete-patch compiled hardware cost for the generalized-exchange selector.

Paper II's compiled cost model (``compiled_marrakesh_population_v1``) prices a
candidate patch ``B = (D, A)`` by transpiling the *complete* patched word, not
by summing per-atom proxies.  This module reproduces the Paper-I compile
precedent exactly (``pipelines.static_adapt.ra_adapt.marginal_qiskit_cost``):
the same circuit builder, the same structural angle on every runtime
coordinate, the same backend/optimization-level/seed contract, and the same
three resource extractors.

Math.  Let ``W = (w_1, ..., w_n)`` be the active word, one term per runtime
coordinate in runtime order.  For a patch that deletes the coordinate set
``D`` and inserts the positioned terms ``A = {(p_k, a_k)}`` (``p_k`` an
*original-layout* cut), the patched word is

    W^B = splice( W \\ D, { (remap_D(p_k), a_k) } ),

where ``remap_D(p) = p - |{d in D : d < p}|`` is the survivor-count remap
shared with certification, and several insertions at one remapped cut keep
their given order (the same rule ``state_with_inserted_runtime_coordinates``
applies when the finalist is materialized, so the compiled word is the
committed word).

Every word is compiled at the structural angle ``theta_i = theta_s`` for all
``i`` (``2 c_ij theta_s`` per Pauli rotation), and the resource triple

    x(W) = (N_2q(W), D_2q(W), D_c(W))

is read from the transpiled circuit: two-qubit gate count (arity based,
excluding barrier/delay/id/measure/reset), two-qubit depth, and total depth.
The signed complete-patch marginal is

    Delta x(B) = x(W^B) - x(W),

negative for a genuine compile relief (a deletion that shortens the circuit)
and preserved with its sign.  Population normalization of the marginals lives
in :mod:`pipelines.time_dynamics.ap_mclachlan.population_scoring`; this module
only produces the raw triples and marginals.

Placement.  For a candidate term ``a`` and the ordered active blocks, cuts
related by certified commuting swaps form equivalence classes (the relation is
:func:`pipelines.time_dynamics.ap_mclachlan.commutation.generator_blocks_commute`,
identical to Paper I's insertion-position reduction).  Paper I retains the
*earliest* member of each class as the insertion cut.  Paper II deviates here,
as its plan requires: every member of the class is compiled and the member
with the smallest weighted raw marginal

    c(p) = (w_n2q Delta N_2q + w_d2q Delta D_2q + w_dc Delta D_c) / (w_n2q + w_d2q + w_dc)

is chosen, ties broken by the raw tuple ``(Delta N_2q, Delta D_2q, Delta D_c)``
ascending and then by the cut ascending.  Commuting members implement the same
unitary, so the choice changes only the compiled resource, never the physics.

Failure semantics.  Any Qiskit/transpile exception, any non-finite or missing
resource, and any word that fails to build raise
:class:`CompiledPatchCostError`.  There is no proxy fallback and no ``K = 1``
fallback: a bundle that declares compiled cost either prices every patch by
compilation or does not price it at all.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import pipelines.hardcoded.adapt_circuit_execution as circuit_execution
import pipelines.qiskit_backend_tools as backend_tools
from pipelines.time_dynamics.ap_mclachlan.commutation import (
    DEFAULT_COEFFICIENT_TOLERANCE,
    block_commutation_crossings,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    remap_cuts_after_deletion,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_config import (
    COMPILE_COORDINATE_WEIGHTS,
    CompileContract,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanState,
    _per_coordinate_terms,
)
from src.quantum.ansatz_parameterization import (
    RotationTermSpec,
    iter_runtime_rotation_terms,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


COMPILED_PATCH_COST_SCHEMA_V1 = "paper_ii_compiled_patch_cost_v1"
WORD_IDENTITY_SCHEMA_V1 = "paper_ii_compiled_word_identity_v1"
COMPILE_BACKEND_PROVENANCE_SCHEMA_V1 = "paper_ii_compile_backend_provenance_v1"
COMPILE_RESOURCE_PROVENANCE_SCHEMA_V1 = "paper_ii_compile_resource_provenance_v1"
PLACEMENT_SCHEMA_V1 = "paper_ii_compiled_placement_v1"
PLACEMENT_RULE_ID = "paper_ii_min_weighted_raw_marginal_placement_v1"
PAPER_I_PLACEMENT_RULE_ID = "paper_i_earliest_class_member"

# The circuit builder's expansion settings
# (``build_structural_ansatz_circuit``: ``build_parameter_layout(...,
# ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)``).
# ``word_identity`` must expand a term exactly as the builder does, so the
# same three values are named here; ``test_word_identity_expansion_matches_
# builder_layout`` pins the parity.
BUILDER_IGNORE_IDENTITY = True
BUILDER_COEFFICIENT_TOLERANCE = 1.0e-12
BUILDER_SORT_TERMS = True

# Coordinate order of the weighted marginal: fixed by the sealed weight map.
_WEIGHT_KEYS: tuple[str, ...] = tuple(COMPILE_COORDINATE_WEIGHTS)


class CompiledPatchCostError(ValueError):
    """Compiled cost is unavailable for a patch; the caller must not fall back."""


# ---------------------------------------------------------------------------
# Resource triple
# ---------------------------------------------------------------------------


def _coerce_resource(name: str, value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise CompiledPatchCostError(f"compiled resource {name!r} is missing.")
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise CompiledPatchCostError(
            f"compiled resource {name!r} is not numeric: {value!r}."
        ) from exc
    if not math.isfinite(as_float):
        raise CompiledPatchCostError(
            f"compiled resource {name!r} is not finite: {value!r}."
        )
    if not as_float.is_integer():
        raise CompiledPatchCostError(
            f"compiled resource {name!r} is not an integer: {value!r}."
        )
    return int(as_float)


@dataclass(frozen=True, slots=True)
class CompiledResourceTriple:
    """``(N_2q, D_2q, D_c)`` of one compiled word, or a signed difference."""

    n2q: int
    d2q: int
    dc: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "n2q", _coerce_resource("n2q", self.n2q))
        object.__setattr__(self, "d2q", _coerce_resource("d2q", self.d2q))
        object.__setattr__(self, "dc", _coerce_resource("dc", self.dc))

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.n2q), int(self.d2q), int(self.dc))

    def as_dict(self) -> dict[str, int]:
        return {"n2q": int(self.n2q), "d2q": int(self.d2q), "dc": int(self.dc)}

    def as_delta_dict(self) -> dict[str, int]:
        """The triple keyed like the sealed coordinate weights."""

        return {
            "delta_n2q": int(self.n2q),
            "delta_d2q": int(self.d2q),
            "delta_dc": int(self.dc),
        }

    def minus(self, other: "CompiledResourceTriple") -> "CompiledResourceTriple":
        """Signed coordinate-wise difference ``self - other``."""

        return CompiledResourceTriple(
            n2q=int(self.n2q) - int(other.n2q),
            d2q=int(self.d2q) - int(other.d2q),
            dc=int(self.dc) - int(other.dc),
        )

    def __sub__(self, other: "CompiledResourceTriple") -> "CompiledResourceTriple":
        return self.minus(other)


ZERO_MARGINAL = CompiledResourceTriple(n2q=0, d2q=0, dc=0)


def _backend_name(backend_obj: Any) -> str | None:
    name_attr = getattr(backend_obj, "name", None)
    if callable(name_attr):
        try:
            return str(name_attr())
        except Exception:
            return None
    return None if name_attr is None else str(name_attr)


def compile_resource_triple(
    circuit: Any,
    backend_obj: Any,
    contract: CompileContract,
) -> tuple[CompiledResourceTriple, dict[str, Any]]:
    """Transpile ``circuit`` under ``contract`` and read ``(N_2q, D_2q, D_c)``.

    The three extractors are the Paper-I ones:
    ``compiled_gate_stats`` for ``compiled_count_2q`` and
    ``compiled_depth_2q`` and ``safe_circuit_depth`` for the total depth.
    Every failure surfaces as :class:`CompiledPatchCostError`.
    """

    try:
        compiled_payload = backend_tools.compile_circuit_for_backend(
            circuit,
            backend_obj,
            seed_transpiler=int(contract.transpile_seed),
            optimization_level=int(contract.optimization_level),
            initial_layout=None,
        )
        compiled = compiled_payload["compiled"]
        stats = backend_tools.compiled_gate_stats(compiled)
        total_depth = backend_tools.safe_circuit_depth(compiled)
    except CompiledPatchCostError:
        raise
    except Exception as exc:
        raise CompiledPatchCostError(
            f"compile failed under {contract.backend_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(stats, Mapping):
        raise CompiledPatchCostError("compiled gate statistics are not a mapping.")
    triple = CompiledResourceTriple(
        n2q=_coerce_resource("compiled_count_2q", stats.get("compiled_count_2q")),
        d2q=_coerce_resource("compiled_depth_2q", stats.get("compiled_depth_2q")),
        dc=_coerce_resource("compiled_depth", total_depth),
    )
    provenance: dict[str, Any] = {
        "schema": COMPILE_RESOURCE_PROVENANCE_SCHEMA_V1,
        "backend_name": _backend_name(backend_obj),
        "optimization_level": int(contract.optimization_level),
        "transpile_seed": int(contract.transpile_seed),
        "structural_angle": float(contract.structural_angle),
        "logical_num_qubits": int(getattr(circuit, "num_qubits", 0) or 0),
        "compiled_num_qubits": (
            None
            if not isinstance(compiled_payload, Mapping)
            else compiled_payload.get("compiled_num_qubits")
        ),
        "logical_to_physical": (
            None
            if not isinstance(compiled_payload, Mapping)
            else compiled_payload.get("logical_to_physical")
        ),
        "compiled_op_counts": dict(stats.get("compiled_op_counts", {}) or {}),
        "resource_semantics": {
            "n2q": "compiled_count_2q: post-transpile two-qubit ops by arity, "
            "excluding barrier/delay/id/measure/reset",
            "d2q": "compiled_depth_2q: depth over two-qubit ops",
            "dc": "safe_circuit_depth: total transpiled depth",
        },
    }
    return triple, provenance


# ---------------------------------------------------------------------------
# Words and their identity
# ---------------------------------------------------------------------------


def reference_state_sha256(ref_state: Any | None) -> str:
    """``sha256`` of the complex128 amplitudes (``"none"`` without a state)."""

    if ref_state is None:
        return "none"
    arr = np.asarray(ref_state, dtype=np.complex128).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def word_rotation_specs(
    terms: Sequence[Any],
) -> tuple[tuple[RotationTermSpec, ...], ...]:
    """Expand each term to the Pauli rotations the circuit builder emits."""

    out: list[tuple[RotationTermSpec, ...]] = []
    for index, term in enumerate(terms):
        polynomial = getattr(term, "polynomial", None)
        if polynomial is None:
            raise CompiledPatchCostError(
                f"word term {index} ({getattr(term, 'label', term)!r}) has no polynomial."
            )
        try:
            specs = iter_runtime_rotation_terms(
                polynomial,
                ignore_identity=BUILDER_IGNORE_IDENTITY,
                coefficient_tolerance=BUILDER_COEFFICIENT_TOLERANCE,
                sort_terms=BUILDER_SORT_TERMS,
            )
        except Exception as exc:
            raise CompiledPatchCostError(
                f"word term {index} ({getattr(term, 'label', term)!r}) failed to expand: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        out.append(tuple(specs))
    return tuple(out)


def word_identity(
    terms: Sequence[Any],
    nq: int,
    ref_state_sha: str,
    structural_angle: float,
) -> str:
    """``sha256`` of the ordered word plus reference state and angle.

    Mirrors the fields of the Paper-I ``_structure_key``: qubit count,
    reference-state hash, structural angle, and the ordered blocks of
    ``(pauli_exyz, coeff_re, nq)`` rotation specs.  Labels never enter: two
    words with the same rotations compile to the same circuit.
    """

    nq_i = int(nq)
    blocks = []
    for index, specs in enumerate(word_rotation_specs(terms)):
        for spec in specs:
            if int(spec.nq) != nq_i:
                raise CompiledPatchCostError(
                    f"word term {index} acts on {spec.nq} qubits, expected {nq_i}."
                )
        blocks.append(
            {
                "runtime_count": int(len(specs)),
                "runtime_terms_exyz": [
                    {
                        "pauli_exyz": str(spec.pauli_exyz),
                        "coeff_re": float(spec.coeff_real),
                        "nq": int(spec.nq),
                    }
                    for spec in specs
                ],
            }
        )
    payload = {
        "schema": WORD_IDENTITY_SCHEMA_V1,
        "num_qubits": nq_i,
        "ref_state_hash": str(ref_state_sha),
        "structure_theta_value": float(structural_angle),
        "initial_layout": None,
        "segment_kind": "full",
        "layout": {
            "term_order": "sorted" if BUILDER_SORT_TERMS else "native",
            "ignore_identity": bool(BUILDER_IGNORE_IDENTITY),
            "coefficient_tolerance": float(BUILDER_COEFFICIENT_TOLERANCE),
            "blocks": blocks,
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_num_qubits(state: APMcLachlanState) -> int:
    """Qubit count of the state's word (executor first, Hilbert dimension else)."""

    psi = np.asarray(state.psi_ref, dtype=complex).reshape(-1)
    dim = int(psi.size)
    nq = getattr(getattr(state, "executor", None), "nq", None)
    if nq is None:
        if dim <= 0:
            raise CompiledPatchCostError("reference state is empty.")
        nq = int(round(math.log2(dim)))
    nq = int(nq)
    if nq < 0 or (1 << nq) != dim:
        raise CompiledPatchCostError(
            f"reference-state dimension {dim} is not 2**{nq}."
        )
    return nq


def base_word(state: APMcLachlanState) -> tuple[Any, ...]:
    """The active word ``W``: one term per runtime coordinate, runtime order."""

    return tuple(_per_coordinate_terms(state))


def _relabel(term: Any, label: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=getattr(term, "polynomial"),
        execution_mode=str(
            getattr(term, "execution_mode", "termwise_product") or "termwise_product"
        ),
    )


def structural_word(
    state: APMcLachlanState,
    removed_runtime_indices: Sequence[int],
    insertions: Sequence[tuple[int, Any, str]],
) -> tuple[Any, ...]:
    """The complete patched word ``W^B``.

    Start from :func:`base_word`, delete the removed coordinates, remap every
    insertion cut over the deletion set with
    :func:`remap_cuts_after_deletion`, and splice the inserted terms at the
    remapped cuts (cut ``p`` sits before survivor ``p``; equal remapped cuts
    keep the given order).  ``insertions`` are ``(original_layout_cut, term,
    label)`` entries exactly as ``exchange_selector.candidate_insertions``
    produces them; inserted terms carry their given label.
    """

    base = base_word(state)
    n = int(len(base))
    removed = tuple(sorted(set(int(i) for i in removed_runtime_indices)))
    bad = [i for i in removed if i < 0 or i >= n]
    if bad:
        raise CompiledPatchCostError(
            f"removed runtime indices {bad} lie outside [0, {n})."
        )
    entries: list[tuple[int, Any, str]] = []
    for position, entry in enumerate(insertions):
        try:
            cut, term, label = entry
        except (TypeError, ValueError) as exc:
            raise CompiledPatchCostError(
                f"insertion {position} is not a (cut, term, label) entry: {entry!r}."
            ) from exc
        cut_i = int(cut)
        if cut_i < 0 or cut_i > n:
            raise CompiledPatchCostError(
                f"insertion cut {cut_i} for {label!r} lies outside [0, {n}]."
            )
        entries.append((cut_i, term, str(label)))
    return edit_word(base, removed, entries)


def edit_word(
    base: Sequence[Any],
    removed: Sequence[int],
    entries: Sequence[tuple[int, Any, str]],
) -> tuple[Any, ...]:
    """Pure word edit: delete ``removed`` from ``base`` and splice ``entries``.

    ``entries`` carry ORIGINAL-layout cuts; they are remapped over the
    deletion set exactly as :func:`structural_word` does.  Pure so a worker
    process can rebuild a word from the base, the removed set, and the
    inserted terms alone.
    """

    base = tuple(base)
    removed = tuple(sorted(set(int(i) for i in removed)))
    removed_set = set(removed)
    survivors = [term for index, term in enumerate(base) if index not in removed_set]
    remapped = remap_cuts_after_deletion([int(cut) for cut, _t, _l in entries], removed)
    by_cut: dict[int, list[Any]] = {}
    for new_cut, (_old, term, label) in zip(remapped, entries):
        by_cut.setdefault(int(new_cut), []).append(_relabel(term, str(label)))
    word: list[Any] = []
    for position in range(len(survivors) + 1):
        word.extend(by_cut.get(position, ()))
        if position < len(survivors):
            word.append(survivors[position])
    return tuple(word)


def word_key(
    terms: Sequence[Any],
    *,
    nq: int,
    ref_state: Any | None,
    contract: CompileContract,
) -> str:
    """The cache key :class:`CompiledPatchCostCache` uses for ``terms``."""

    return word_identity(
        terms,
        int(nq),
        reference_state_sha256(ref_state),
        float(contract.structural_angle),
    )


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


def compile_word_resource_triple(
    terms: Sequence[Any],
    *,
    nq: int,
    ref_state: Any | None,
    contract: CompileContract,
    backend_obj: Any,
) -> tuple[CompiledResourceTriple, dict[str, Any]]:
    """Build the structural circuit of ``terms`` and compile it (no cache)."""

    try:
        _layout, circuit = circuit_execution.build_structural_ansatz_circuit(
            list(terms),
            nq=int(nq),
            ref_state=ref_state,
            structure_theta_value=float(contract.structural_angle),
        )
    except CompiledPatchCostError:
        raise
    except Exception as exc:
        raise CompiledPatchCostError(
            f"structural word of {len(terms)} terms failed to build: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    triple, provenance = compile_resource_triple(circuit, backend_obj, contract)
    provenance["word_length"] = int(len(terms))
    return triple, provenance


WordCompiler = Callable[..., tuple[CompiledResourceTriple, dict[str, Any]]]


class CompiledPatchCostCache:
    """Memoizes ``x(W)`` by :func:`word_identity`.

    The base word of a checkpoint and every trial word are looked up by the
    identity of their rotations, so a word reached by two different patches
    (or by two placements that swap commuting blocks into the same order) is
    compiled once.  ``compile_calls`` counts real compilations,
    ``cache_hits`` counts served lookups.  Dictionary and counter access is
    guarded by a lock so the cache can sit behind a worker pool; a duplicate
    compile raced by two workers is benign because compilation is
    deterministic.
    """

    def __init__(self, *, word_compiler: WordCompiler = compile_word_resource_triple) -> None:
        self._entries: dict[str, CompiledResourceTriple] = {}
        self._lock = threading.Lock()
        self._word_compiler = word_compiler
        self.compile_calls = 0
        self.cache_hits = 0

    def __len__(self) -> int:
        with self._lock:
            return int(len(self._entries))

    def prime(self, word_sha256: str, triple: CompiledResourceTriple) -> bool:
        """Insert a triple compiled elsewhere; returns True when new.

        Used by the process-pool warm: a worker compiled the word under the
        same sealed contract, so the entry is exactly what
        :meth:`compile_word_record` would have produced.  Counted as a
        compile call because a real transpile happened (in the worker).
        """

        if not isinstance(triple, CompiledResourceTriple):
            raise CompiledPatchCostError("prime() requires a CompiledResourceTriple.")
        with self._lock:
            if str(word_sha256) in self._entries:
                return False
            self._entries[str(word_sha256)] = triple
            self.compile_calls += 1
            return True

    def lookup(self, word_sha256: str) -> CompiledResourceTriple | None:
        """Peek without touching the counters."""

        with self._lock:
            return self._entries.get(str(word_sha256))

    def compile_word_record(
        self,
        terms: Sequence[Any],
        *,
        nq: int,
        ref_state: Any | None,
        contract: CompileContract,
        backend_obj: Any,
    ) -> tuple[str, CompiledResourceTriple, bool]:
        """Return ``(word_sha256, triple, from_cache)`` for ``terms``."""

        key = word_identity(
            terms,
            int(nq),
            reference_state_sha256(ref_state),
            float(contract.structural_angle),
        )
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self.cache_hits += 1
                return key, hit, True
        triple, _provenance = self._word_compiler(
            terms,
            nq=int(nq),
            ref_state=ref_state,
            contract=contract,
            backend_obj=backend_obj,
        )
        if not isinstance(triple, CompiledResourceTriple):
            raise CompiledPatchCostError(
                "word compiler returned "
                f"{type(triple).__name__}, expected CompiledResourceTriple."
            )
        with self._lock:
            self.compile_calls += 1
            stored = self._entries.setdefault(key, triple)
            return key, stored, False

    def compile_word(
        self,
        terms: Sequence[Any],
        *,
        nq: int,
        ref_state: Any | None,
        contract: CompileContract,
        backend_obj: Any,
    ) -> CompiledResourceTriple:
        return self.compile_word_record(
            terms,
            nq=nq,
            ref_state=ref_state,
            contract=contract,
            backend_obj=backend_obj,
        )[1]


# ---------------------------------------------------------------------------
# Commutation classes and placement
# ---------------------------------------------------------------------------


def insertion_class_members(
    term: Any,
    blocks: Sequence[Any],
    *,
    coefficient_tolerance: float = DEFAULT_COEFFICIENT_TOLERANCE,
) -> dict[int, tuple[int, ...]]:
    """Map each retained representative cut to its full commutation class.

    Keys are exactly ``singleton_insertion_cuts(term, blocks)`` in the same
    order; the class of representative ``p`` is ``p, p+1, ..., q-1`` where
    ``q`` is the next representative (the first cut whose preceding block the
    term does not certifiably commute with), so the classes partition
    ``range(len(blocks) + 1)``.  The crossing certificate is
    :func:`block_commutation_crossings`, the single relation both lanes share.
    """

    crossings = block_commutation_crossings(
        term, blocks, coefficient_tolerance=coefficient_tolerance
    )
    classes: dict[int, tuple[int, ...]] = {}
    representative = 0
    members = [0]
    for index, commutes in enumerate(crossings):
        cut = int(index + 1)
        if commutes:
            members.append(cut)
            continue
        classes[representative] = tuple(members)
        representative = cut
        members = [cut]
    classes[representative] = tuple(members)
    return classes


def weighted_raw_marginal(
    marginal: CompiledResourceTriple,
    contract: CompileContract,
) -> float:
    """``sum_x w_x Delta x / sum_x w_x`` over the sealed coordinate weights."""

    weights = dict(contract.coordinate_weights)
    delta = marginal.as_delta_dict()
    missing = [key for key in _WEIGHT_KEYS if key not in weights]
    if missing:
        raise CompiledPatchCostError(f"compile contract lacks weights {missing}.")
    total = 0.0
    for key in _WEIGHT_KEYS:
        total += float(weights[key])
    if not math.isfinite(total) or total <= 0.0:
        raise CompiledPatchCostError("compile contract weights must sum to > 0.")
    value = 0.0
    for key in _WEIGHT_KEYS:
        value += float(weights[key]) * float(delta[key])
    value = value / total
    if not math.isfinite(value):
        raise CompiledPatchCostError("weighted raw marginal is not finite.")
    return float(value)


@dataclass(frozen=True)
class PlacementCandidate:
    """One evaluated class member."""

    cut: int
    triple: CompiledResourceTriple
    marginal: CompiledResourceTriple
    weighted_marginal: float
    from_cache: bool

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "cut": int(self.cut),
            "triple": self.triple.as_dict(),
            "marginal": self.marginal.as_delta_dict(),
            "weighted_marginal": float(self.weighted_marginal),
            "from_cache": bool(self.from_cache),
        }


@dataclass(frozen=True)
class Placement:
    """The chosen cut of a candidate inside one commutation class."""

    chosen_cut: int
    marginal: CompiledResourceTriple
    candidates_evaluated: tuple[PlacementCandidate, ...]
    base_triple: CompiledResourceTriple
    compile_calls_used: int
    placement_rule_id: str = PLACEMENT_RULE_ID

    @property
    def candidate_count(self) -> int:
        return int(len(self.candidates_evaluated))

    @property
    def chosen(self) -> PlacementCandidate:
        for candidate in self.candidates_evaluated:
            if int(candidate.cut) == int(self.chosen_cut):
                return candidate
        raise CompiledPatchCostError("placement lost its chosen candidate.")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": PLACEMENT_SCHEMA_V1,
            "placement_rule_id": str(self.placement_rule_id),
            "paper_i_rule_id": PAPER_I_PLACEMENT_RULE_ID,
            "chosen_cut": int(self.chosen_cut),
            "marginal": self.marginal.as_delta_dict(),
            "base_triple": self.base_triple.as_dict(),
            "compile_calls_used": int(self.compile_calls_used),
            "candidates_evaluated": [
                candidate.to_json_dict() for candidate in self.candidates_evaluated
            ],
        }


def choose_placement(
    term: Any,
    class_positions: Sequence[int],
    *,
    base_terms: Sequence[Any],
    nq: int,
    ref_state: Any | None,
    contract: CompileContract,
    backend_obj: Any,
    cache: CompiledPatchCostCache,
) -> Placement:
    """Pick the cut of one commutation class by minimum weighted raw marginal.

    Paper I retains the earliest member of each commutation class as the
    insertion cut.  This is the Paper-II deviation the plan requires: the base
    word is compiled once (through ``cache``), every position ``p`` in
    ``class_positions`` is compiled as ``base[:p] + [term] + base[p:]``, the
    signed marginal ``x(trial) - x(base)`` is formed per coordinate, and the
    position minimizing :func:`weighted_raw_marginal` wins.  Ties are broken
    by the raw tuple ``(dn2q, dd2q, ddc)`` ascending, then by the cut
    ascending.  Rule id: ``paper_ii_min_weighted_raw_marginal_placement_v1``.
    """

    base = tuple(base_terms)
    positions = tuple(sorted(set(int(p) for p in class_positions)))
    if not positions:
        raise CompiledPatchCostError("a placement class must contain a cut.")
    bad = [p for p in positions if p < 0 or p > len(base)]
    if bad:
        raise CompiledPatchCostError(
            f"placement cuts {bad} lie outside [0, {len(base)}]."
        )
    _base_key, base_triple, base_hit = cache.compile_word_record(
        base, nq=nq, ref_state=ref_state, contract=contract, backend_obj=backend_obj
    )
    calls = 0 if base_hit else 1
    candidates: list[PlacementCandidate] = []
    for cut in positions:
        trial = base[:cut] + (term,) + base[cut:]
        _key, triple, hit = cache.compile_word_record(
            trial, nq=nq, ref_state=ref_state, contract=contract, backend_obj=backend_obj
        )
        calls += 0 if hit else 1
        marginal = triple.minus(base_triple)
        candidates.append(
            PlacementCandidate(
                cut=int(cut),
                triple=triple,
                marginal=marginal,
                weighted_marginal=weighted_raw_marginal(marginal, contract),
                from_cache=bool(hit),
            )
        )
    best = min(
        candidates,
        key=lambda c: (float(c.weighted_marginal), c.marginal.as_tuple(), int(c.cut)),
    )
    return Placement(
        chosen_cut=int(best.cut),
        marginal=best.marginal,
        candidates_evaluated=tuple(candidates),
        base_triple=base_triple,
        compile_calls_used=int(calls),
    )


# ---------------------------------------------------------------------------
# Complete patch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchCostRecord:
    """Compiled resource of one complete patched word and its signed marginal."""

    word_sha256: str
    triple: CompiledResourceTriple
    marginal: CompiledResourceTriple
    compile_calls_used: int
    from_cache: bool
    word_length: int
    removed_runtime_indices: tuple[int, ...]
    insertion_cuts_original: tuple[int, ...]
    insertion_cuts_remapped: tuple[int, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPILED_PATCH_COST_SCHEMA_V1,
            "word_sha256": str(self.word_sha256),
            "triple": self.triple.as_dict(),
            "marginal": self.marginal.as_delta_dict(),
            "compile_calls_used": int(self.compile_calls_used),
            "from_cache": bool(self.from_cache),
            "word_length": int(self.word_length),
            "removed_runtime_indices": [int(i) for i in self.removed_runtime_indices],
            "insertion_cuts_original": [int(c) for c in self.insertion_cuts_original],
            "insertion_cuts_remapped": [int(c) for c in self.insertion_cuts_remapped],
        }


def compile_base_word(
    state: APMcLachlanState,
    *,
    contract: CompileContract,
    backend_obj: Any,
    cache: CompiledPatchCostCache,
) -> PatchCostRecord:
    """Compile the checkpoint's base word ``W`` (marginal identically zero)."""

    nq = state_num_qubits(state)
    ref_state = np.asarray(state.psi_ref, dtype=complex).reshape(-1)
    word = base_word(state)
    key, triple, hit = cache.compile_word_record(
        word, nq=nq, ref_state=ref_state, contract=contract, backend_obj=backend_obj
    )
    return PatchCostRecord(
        word_sha256=key,
        triple=triple,
        marginal=ZERO_MARGINAL,
        compile_calls_used=0 if hit else 1,
        from_cache=bool(hit),
        word_length=int(len(word)),
        removed_runtime_indices=(),
        insertion_cuts_original=(),
        insertion_cuts_remapped=(),
    )


def compile_complete_patch(
    state: APMcLachlanState,
    *,
    removed_runtime_indices: Sequence[int],
    insertions: Sequence[tuple[int, Any, str]],
    contract: CompileContract,
    backend_obj: Any,
    cache: CompiledPatchCostCache,
    base_triple: CompiledResourceTriple | None,
) -> PatchCostRecord:
    """Compile ``W^B`` and return ``x(W^B)`` with ``Delta x = x(W^B) - x(W)``.

    ``base_triple`` is the checkpoint's ``x(W)`` from :func:`compile_base_word`
    (compiled once per checkpoint and reused); passing ``None`` fetches it
    through ``cache`` instead.  ``compile_calls_used`` counts the real
    compilations this call incurred (patched word plus, when fetched here, a
    base miss); ``from_cache`` refers to the patched word.
    """

    nq = state_num_qubits(state)
    ref_state = np.asarray(state.psi_ref, dtype=complex).reshape(-1)
    removed = tuple(sorted(set(int(i) for i in removed_runtime_indices)))
    word = structural_word(state, removed, insertions)
    original_cuts = tuple(int(cut) for cut, _t, _l in insertions)
    remapped_cuts = remap_cuts_after_deletion(original_cuts, removed)
    calls = 0
    if base_triple is None:
        _base_key, base_triple, base_hit = cache.compile_word_record(
            base_word(state),
            nq=nq,
            ref_state=ref_state,
            contract=contract,
            backend_obj=backend_obj,
        )
        calls += 0 if base_hit else 1
    key, triple, hit = cache.compile_word_record(
        word, nq=nq, ref_state=ref_state, contract=contract, backend_obj=backend_obj
    )
    calls += 0 if hit else 1
    return PatchCostRecord(
        word_sha256=key,
        triple=triple,
        marginal=triple.minus(base_triple),
        compile_calls_used=int(calls),
        from_cache=bool(hit),
        word_length=int(len(word)),
        removed_runtime_indices=removed,
        insertion_cuts_original=original_cuts,
        insertion_cuts_remapped=tuple(int(c) for c in remapped_cuts),
    )


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def _distribution_version(name: str, *, required: bool) -> str | None:
    try:
        return str(importlib_metadata.version(name))
    except importlib_metadata.PackageNotFoundError as exc:
        if required:
            raise CompiledPatchCostError(
                f"required distribution {name!r} is not installed."
            ) from exc
        return None


def resolve_compile_backend(
    contract: CompileContract,
) -> tuple[Any, dict[str, Any]]:
    """Resolve the contract's backend fail-closed and capture provenance.

    Exactly one local fake target must resolve for the requested name (no
    preferred fallback, no runtime lookup), as the Paper-I
    ``BackendCompileOracle`` requires.
    """

    requested = str(contract.backend_name)
    try:
        targets, audit = backend_tools.resolve_backend_targets(
            requested_names=(requested,),
            preferred_fake_backends=(),
            allow_preferred_fallback=False,
            fallback_mode="single",
            allow_runtime_lookup=False,
        )
    except Exception as exc:
        raise CompiledPatchCostError(
            f"backend resolution for {requested!r} raised {type(exc).__name__}: {exc}"
        ) from exc
    if len(targets) != 1:
        raise CompiledPatchCostError(
            f"Could not resolve requested compile backend {requested!r} "
            f"(resolved {len(targets)} targets)."
        )
    target = targets[0]
    provenance: dict[str, Any] = {
        "schema": COMPILE_BACKEND_PROVENANCE_SCHEMA_V1,
        "requested_backend_name": requested,
        "backend_name": str(target.resolved_name),
        "resolution_kind": str(target.resolution_kind),
        "using_fake_backend": bool(target.using_fake_backend),
        "target_snapshot": dict(target.target_snapshot),
        "optimization_level": int(contract.optimization_level),
        "transpile_seed": int(contract.transpile_seed),
        "structural_angle": float(contract.structural_angle),
        "coordinate_weights": {
            str(k): float(v) for k, v in dict(contract.coordinate_weights).items()
        },
        "scale_floor": float(contract.scale_floor),
        "placement_rule_id": PLACEMENT_RULE_ID,
        "qiskit_version": _distribution_version("qiskit", required=True),
        "qiskit_ibm_runtime_version": _distribution_version(
            "qiskit-ibm-runtime", required=False
        ),
        "resolution_audit": [dict(row) for row in audit],
    }
    return target.backend_obj, provenance



# ---------------------------------------------------------------------------
# Process-pool warm: deterministic fan-out of word compiles
# ---------------------------------------------------------------------------
#
# Transpilation holds the GIL (measured: 4 threads give 0.83x), so useful
# concurrency for the compile stage needs processes.  Workers receive the
# checkpoint context once (base word, candidate atom terms, contract) through
# the pool initializer, jobs describe a word as an EDIT of that context, and
# the parent consumes results in submission order and primes the cache.  The
# serial scoring replay then hits the cache for every word, so completion
# order cannot influence any value; with seed_transpiler sealed, a worker's
# triple is the triple the parent would have compiled.

_POOL_CONTEXT: dict[str, Any] = {}

BLAS_THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _pool_initializer(
    contract: CompileContract,
    base_terms: tuple[Any, ...],
    atom_terms: dict[str, Any],
    nq: int,
    ref_state: Any,
) -> None:
    backend, _provenance = resolve_compile_backend(contract)
    _POOL_CONTEXT.clear()
    _POOL_CONTEXT.update(
        backend=backend,
        contract=contract,
        base=tuple(base_terms),
        atoms=dict(atom_terms),
        nq=int(nq),
        ref_state=ref_state,
    )


def _pool_word_for_job(job: tuple) -> tuple[Any, ...]:
    kind = job[0]
    base = _POOL_CONTEXT["base"]
    atoms = _POOL_CONTEXT["atoms"]
    if kind == "trial":
        _kind, atom_id, cut = job
        return base[: int(cut)] + (atoms[str(atom_id)],) + base[int(cut) :]
    if kind == "patch":
        _kind, removed, entries = job
        return edit_word(
            base,
            tuple(int(i) for i in removed),
            [(int(cut), atoms[str(atom_id)], str(label)) for cut, atom_id, label in entries],
        )
    raise CompiledPatchCostError(f"unknown warm job kind {kind!r}.")


def _pool_compile_job(job: tuple) -> tuple[str, tuple[int, int, int]]:
    word = _pool_word_for_job(job)
    contract = _POOL_CONTEXT["contract"]
    key = word_key(
        word, nq=_POOL_CONTEXT["nq"], ref_state=_POOL_CONTEXT["ref_state"], contract=contract
    )
    triple, _provenance = compile_word_resource_triple(
        word,
        nq=_POOL_CONTEXT["nq"],
        ref_state=_POOL_CONTEXT["ref_state"],
        contract=contract,
        backend_obj=_POOL_CONTEXT["backend"],
    )
    return key, triple.as_tuple()


class CompileWarmPool:
    """One checkpoint's process pool for compile warming.

    Created lazily on the first warm with more than one missing word and
    closed by the owner at the end of the checkpoint.  Pinning the five BLAS
    thread variables to 1 in the parent environment before spawning keeps
    the workers single-threaded (the parent's own numpy is already loaded,
    so the parent's numerics are unaffected).
    """

    def __init__(
        self,
        *,
        contract: CompileContract,
        base_terms: Sequence[Any],
        atom_terms: Mapping[str, Any],
        nq: int,
        ref_state: Any,
        workers: int,
    ) -> None:
        import concurrent.futures
        import os

        self.workers = max(1, int(workers))
        # Spawned workers inherit the parent's environment at spawn time, so
        # the pins must be present in the parent before the pool starts.
        # The parent's numpy is already loaded, so its numerics are
        # unaffected; the pinned values are recorded so later selections'
        # telemetry reflects the environment that was actually in force.
        self.pinned_env: dict[str, str] = {}
        for name in BLAS_THREAD_ENV_VARS:
            if os.environ.get(name) is None:
                os.environ[name] = "1"
                self.pinned_env[name] = "1"
        self._executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_pool_initializer,
            initargs=(contract, tuple(base_terms), dict(atom_terms), int(nq), ref_state),
        )
        self.jobs_run = 0

    def compile_jobs(
        self, jobs: Sequence[tuple], expected_keys: Sequence[str]
    ) -> list[tuple[str, CompiledResourceTriple]]:
        """Run jobs and return ``(key, triple)`` in submission order."""

        out: list[tuple[str, CompiledResourceTriple]] = []
        for expected, (key, triple) in zip(
            expected_keys, self._executor.map(_pool_compile_job, list(jobs))
        ):
            if str(key) != str(expected):
                raise CompiledPatchCostError(
                    "worker word identity disagrees with the coordinator's; "
                    "the pool context is stale."
                )
            out.append((str(key), CompiledResourceTriple(*triple)))
        self.jobs_run += len(out)
        return out

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def warm_word_cache(
    cache: CompiledPatchCostCache,
    pool: CompileWarmPool,
    jobs: Sequence[tuple[str, tuple]],
) -> int:
    """Compile every not-yet-cached ``(key, job)`` in the pool; prime the cache.

    Duplicate keys are compiled once; already-cached keys are skipped.
    Returns the number of words primed.
    """

    seen: set[str] = set()
    missing: list[tuple[str, tuple]] = []
    for key, job in jobs:
        key = str(key)
        if key in seen or cache.lookup(key) is not None:
            continue
        seen.add(key)
        missing.append((key, job))
    if not missing:
        return 0
    results = pool.compile_jobs([job for _k, job in missing], [k for k, _j in missing])
    primed = 0
    for key, triple in results:
        if cache.prime(key, triple):
            primed += 1
    return primed

__all__ = [
    "BLAS_THREAD_ENV_VARS",
    "CompileWarmPool",
    "edit_word",
    "warm_word_cache",
    "word_key",
    "BUILDER_COEFFICIENT_TOLERANCE",
    "BUILDER_IGNORE_IDENTITY",
    "BUILDER_SORT_TERMS",
    "COMPILED_PATCH_COST_SCHEMA_V1",
    "COMPILE_BACKEND_PROVENANCE_SCHEMA_V1",
    "COMPILE_RESOURCE_PROVENANCE_SCHEMA_V1",
    "PAPER_I_PLACEMENT_RULE_ID",
    "PLACEMENT_RULE_ID",
    "PLACEMENT_SCHEMA_V1",
    "WORD_IDENTITY_SCHEMA_V1",
    "ZERO_MARGINAL",
    "CompiledPatchCostCache",
    "CompiledPatchCostError",
    "CompiledResourceTriple",
    "PatchCostRecord",
    "Placement",
    "PlacementCandidate",
    "base_word",
    "choose_placement",
    "compile_base_word",
    "compile_complete_patch",
    "compile_resource_triple",
    "compile_word_resource_triple",
    "insertion_class_members",
    "reference_state_sha256",
    "resolve_compile_backend",
    "state_num_qubits",
    "structural_word",
    "weighted_raw_marginal",
    "word_identity",
    "word_rotation_specs",
]
