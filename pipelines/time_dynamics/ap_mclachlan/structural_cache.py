"""Checkpoint-local frozen-ray geometry cache for the exchange selector.

Every structural candidate — pure deletion, positioned insertion, or true
exchange — is scored on one frozen checkpoint ray ``|psi_k>``.  Deletion
removes active tangent columns from the candidate solve; positioned insertion
adds the zero-angle tangent of a child occurrence at an original-layout cut:

    tau_{a,p} = horizontal_{psi_k}[ U_{>p} (-i c_a P_a) U_{<=p} |phi_ref> ].

Because inserted coordinates are zero angle, ``tau_{a,p}`` is independent of
the deletion set: deletion conditions *which rows* of the assembled Gram enter
a candidate solve, never the columns themselves.  This module exploits that:

* all positioned tangents are built in one ascending pass over cuts, batching
  the suffix rotations over the accumulated column matrix (the same
  suffix-propagation pattern the compiled executor uses for its all-tangents
  path), and only for the retained cuts of each candidate;
* the active--inserted cross Gram, inserted--inserted Gram, and inserted force
  entries are computed once per checkpoint; a candidate geometry
  ``(G^{D,I}, f^{D,I})`` is then pure index selection plus block assembly in
  the order induced by the typed plan word.

Nothing here materializes or refits an ANZATS; that remains the certification
stage's job.  The cache is valid for exactly one checkpoint identity and must
be rebuilt after any accepted commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.geometry_eval import GeometryEvaluation
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    solve_theta_dot,
)
from pipelines.time_dynamics.ap_mclachlan.state import APMcLachlanState
from src.quantum.ansatz_parameterization import iter_runtime_rotation_terms
from src.quantum.pauli_actions import (
    apply_compiled_pauli,
    apply_exp_term,
    compile_pauli_action_exyz,
)


STRUCTURAL_CACHE_SCHEMA_V1 = "ap_mclachlan_structural_cache_v1"


def _runtime_rotation_sequence(
    state: APMcLachlanState,
) -> tuple[tuple[Any, float], ...]:
    """Per runtime coordinate, its compiled Pauli action and real coefficient."""

    out: list[tuple[Any, float]] = []
    for term, block in zip(state.terms, state.layout.blocks):
        for spec in block.terms:
            action = state.executor.pauli_action_cache.get(str(spec.pauli_exyz))
            if action is None:
                action = compile_pauli_action_exyz(str(spec.pauli_exyz), int(spec.nq))
                state.executor.pauli_action_cache[str(spec.pauli_exyz)] = action
            out.append((action, float(spec.coeff_real)))
    return tuple(out)


def _single_child_action(state: APMcLachlanState, atom: Any) -> tuple[Any, float]:
    """Compiled action and coefficient for a single-child candidate atom."""

    specs = iter_runtime_rotation_terms(
        getattr(atom.term, "polynomial"),
        ignore_identity=bool(state.executor.ignore_identity),
        coefficient_tolerance=float(state.executor.coefficient_tolerance),
        sort_terms=bool(state.executor.sort_terms),
    )
    if len(specs) != 1:
        raise ValueError(
            "structural cache requires one runtime Pauli child per candidate "
            f"atom; got {len(specs)} for {atom.atom_id!r}."
        )
    spec = specs[0]
    action = state.executor.pauli_action_cache.get(str(spec.pauli_exyz))
    if action is None:
        action = compile_pauli_action_exyz(str(spec.pauli_exyz), int(spec.nq))
        state.executor.pauli_action_cache[str(spec.pauli_exyz)] = action
    return action, float(spec.coeff_real)


@dataclass(frozen=True)
class StructuralInsertionCache:
    """Frozen-ray candidate geometry for one checkpoint.

    Columns are indexed by ``(atom_id, cut)``; ``column_index`` maps that key
    to its column in ``tangent_matrix``, ``cross`` (active x inserted),
    ``gram_inserted`` (inserted x inserted, symmetric), and ``force_inserted``.
    ``coordinate_keys`` is the frozen ordered active coordinate identity the
    row indices of ``cross`` refer to.
    """

    checkpoint_key: tuple
    coordinate_keys: tuple[str, ...]
    column_index: Mapping[tuple[str, int], int]
    tangent_matrix: np.ndarray
    cross: np.ndarray
    gram_inserted: np.ndarray
    force_inserted: np.ndarray
    solve_memo: dict = field(default_factory=dict)

    @property
    def column_count(self) -> int:
        return int(self.tangent_matrix.shape[1])


def build_structural_insertion_cache(
    *,
    state: APMcLachlanState,
    evaluation: GeometryEvaluation,
    cuts_by_atom: Mapping[str, Sequence[int]],
    atoms_by_id: Mapping[str, Any],
    checkpoint_key: tuple = (),
) -> StructuralInsertionCache:
    """Build every positioned tangent and its Gram/force blocks in one pass.

    ``evaluation`` must carry the frozen tangent matrix
    (``include_tangent_matrix=True``); ``cuts_by_atom`` maps each candidate
    atom id to its retained original-layout cuts (from the commutation
    quotient).  The ascending pass keeps one prefix state and applies each
    block's rotation to the accumulated raw-column matrix, so each column
    added at cut ``p`` receives exactly the suffix ``U_{>p}``.
    """

    if evaluation.tangent_matrix is None:
        raise ValueError(
            "structural cache requires evaluation.tangent_matrix "
            "(include_tangent_matrix=True)."
        )
    psi = np.asarray(evaluation.psi, dtype=complex).reshape(-1)
    b_bar = np.asarray(
        -1.0j
        * (
            np.asarray(evaluation.h_psi, dtype=complex).reshape(-1)
            - float(evaluation.energy_expectation) * psi
        ),
        dtype=complex,
    )
    T_active = np.asarray(evaluation.tangent_matrix, dtype=complex)
    n = int(state.runtime_parameter_count)
    if T_active.shape != (int(psi.size), n):
        raise ValueError(
            f"tangent matrix shape {T_active.shape} does not match "
            f"({psi.size}, {n})."
        )

    rotations = _runtime_rotation_sequence(state)
    if len(rotations) != n:
        raise ValueError(
            f"runtime rotation sequence length {len(rotations)} != {n}."
        )
    theta = np.asarray(state.theta_runtime, dtype=float).reshape(-1)

    # Group requested (atom, cut) pairs by cut; validate cuts.
    ordered_keys: list[tuple[str, int]] = []
    by_cut: dict[int, list[str]] = {}
    for atom_id in sorted(cuts_by_atom):
        atom = atoms_by_id[str(atom_id)]
        for cut in sorted({int(c) for c in cuts_by_atom[atom_id]}):
            if cut < 0 or cut > n:
                raise ValueError(
                    f"cut {cut} out of range [0, {n}] for atom {atom_id!r}."
                )
            ordered_keys.append((str(atom_id), int(cut)))
            by_cut.setdefault(int(cut), []).append(str(atom_id))
        # Compile eagerly so a bad atom fails before the pass runs.
        _single_child_action(state, atom)

    dim = int(psi.size)
    columns = np.zeros((dim, len(ordered_keys)), dtype=complex)
    column_index = {key: i for i, key in enumerate(ordered_keys)}

    phi = np.asarray(state.psi_ref, dtype=complex).reshape(-1).copy()
    emitted: list[int] = []
    for cut in range(n + 1):
        for atom_id in by_cut.get(cut, ()):
            action, coeff = _single_child_action(state, atoms_by_id[atom_id])
            index = column_index[(atom_id, cut)]
            columns[:, index] = -1.0j * coeff * apply_compiled_pauli(phi, action)
            emitted.append(index)
        if cut < n:
            action, coeff = rotations[cut]
            dt = float(theta[cut])
            if dt != 0.0:
                phi = apply_exp_term(
                    phi,
                    action,
                    coeff=complex(coeff),
                    dt=dt,
                    tol=state.executor.coefficient_tolerance,
                )
                if emitted:
                    sub = columns[:, emitted]
                    pauli_sub = np.column_stack(
                        [apply_compiled_pauli(sub[:, j], action) for j in range(sub.shape[1])]
                    )
                    angle = dt * float(coeff)
                    columns[:, emitted] = (
                        np.cos(angle) * sub - 1.0j * np.sin(angle) * pauli_sub
                    )

    # Horizontalize at the frozen ray.
    overlaps = psi.conj() @ columns
    columns = columns - np.outer(psi, overlaps)

    cross = np.asarray(np.real(T_active.conj().T @ columns), dtype=float)
    gram_inserted = np.asarray(np.real(columns.conj().T @ columns), dtype=float)
    gram_inserted = 0.5 * (gram_inserted + gram_inserted.T)
    force_inserted = np.asarray(
        np.real(columns.conj().T @ b_bar), dtype=float
    ).reshape(-1)

    if not (
        np.all(np.isfinite(columns))
        and np.all(np.isfinite(cross))
        and np.all(np.isfinite(gram_inserted))
        and np.all(np.isfinite(force_inserted))
    ):
        raise ValueError("structural insertion cache contains non-finite values.")

    return StructuralInsertionCache(
        checkpoint_key=tuple(checkpoint_key),
        coordinate_keys=tuple(str(k) for k in state.runtime_coordinate_labels),
        column_index=dict(column_index),
        tangent_matrix=columns,
        cross=cross,
        gram_inserted=gram_inserted,
        force_inserted=force_inserted,
    )


def assemble_candidate_geometry(
    *,
    cache: StructuralInsertionCache,
    base_K: np.ndarray,
    base_f: np.ndarray,
    keep_indices: Sequence[int],
    inserted_selection: Sequence[tuple[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``(G^{D,I}, f^{D,I})`` for one candidate from cached blocks.

    ``keep_indices`` are the surviving active coordinates (``J_k \\ D``) in
    frozen order; ``inserted_selection`` lists the plan's ``(atom_id, cut)``
    occurrences in the order induced by the typed plan word, which is the
    column order of the returned solve.  Survivor columns precede inserted
    columns, matching the candidate tangent matrix
    ``[T_{J\\D} | T_I]`` of the specification.
    """

    keep = list(int(i) for i in keep_indices)
    cols = [cache.column_index[(str(a), int(p))] for a, p in inserted_selection]
    m = len(cols)
    k = len(keep)
    K_base = np.asarray(base_K, dtype=float)
    f_base = np.asarray(base_f, dtype=float).reshape(-1)

    G = np.zeros((k + m, k + m), dtype=float)
    f = np.zeros(k + m, dtype=float)
    if k:
        G[:k, :k] = K_base[np.ix_(keep, keep)]
        f[:k] = f_base[keep]
    if m:
        G[k:, k:] = cache.gram_inserted[np.ix_(cols, cols)]
        f[k:] = cache.force_inserted[cols]
        if k:
            cross = cache.cross[np.ix_(keep, cols)]
            G[:k, k:] = cross
            G[k:, :k] = cross.T
    G = 0.5 * (G + G.T)
    return G, f


def structural_candidate_solve(
    *,
    cache: StructuralInsertionCache,
    base_K: np.ndarray,
    base_f: np.ndarray,
    norm_b_sq: float,
    keep_indices: Sequence[int],
    inserted_selection: Sequence[tuple[str, int]],
    inverse_policy: McLachlanInversePolicy,
    epsilon_norm: float,
    memo_key: tuple | None = None,
) -> tuple[float, float]:
    """Return ``(Q, q)`` for one structural candidate under the actual policy.

    ``Q`` is the realized captured drift of the candidate solve and
    ``q = Q / (||b||^2 + eps)`` the dimensionless score input.  Results are
    memoized on ``memo_key`` (canonical plan identity plus deletion set) when
    provided.
    """

    if memo_key is not None:
        cached = cache.solve_memo.get(memo_key)
        if cached is not None:
            return cached
    G, f = assemble_candidate_geometry(
        cache=cache,
        base_K=base_K,
        base_f=base_f,
        keep_indices=keep_indices,
        inserted_selection=inserted_selection,
    )
    if int(f.size) == 0:
        result = (0.0, 0.0)
    else:
        solve = solve_theta_dot(G, f, policy=inverse_policy)
        Q = float(solve.captured_drift)
        denom = float(norm_b_sq) + max(0.0, float(epsilon_norm))
        result = (Q, float(Q / denom))
    if memo_key is not None:
        cache.solve_memo[memo_key] = result
    return result


__all__ = [
    "STRUCTURAL_CACHE_SCHEMA_V1",
    "StructuralInsertionCache",
    "assemble_candidate_geometry",
    "build_structural_insertion_cache",
    "structural_candidate_solve",
]
