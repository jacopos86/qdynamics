"""QSE boundary adapter for static-ADAPT implementation calls.

QSE production code should import static-ADAPT implementation modules only here.
Data-only HH full-meta contract symbols live in ``pipelines.contracts``; this
adapter owns the remaining true implementation calls needed to reconstruct
Hamiltonians and static full-meta pools from artifacts.
"""

from __future__ import annotations

from typing import Any, Callable

from pipelines.static_adapt.builders.hh_pool_presets import (
    _build_hh_full_meta_pool,
    build_hh_pool_by_key,
)
from pipelines.static_adapt.builders.problem_setup import build_problem_hamiltonian


def build_artifact_problem_hamiltonian(**kwargs: Any) -> Any:
    """Rebuild a problem Hamiltonian from artifact settings."""

    return build_problem_hamiltonian(**kwargs)


def build_hh_full_meta_pool_for_qse(**kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
    """Build the direct HH full-meta pool used by non-canonical QSE sidecars."""

    return _build_hh_full_meta_pool(**kwargs)


def build_canonical_hh_full_meta_pool_for_qse(
    *,
    ai_log: Callable[..., None] | None = None,
    **kwargs: Any,
) -> tuple[list[Any], str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Build the canonical HH full-meta pool with static-ADAPT production filters."""

    return build_hh_pool_by_key(
        ai_log=ai_log,
        include_legal_subspace_filter_meta=True,
        **kwargs,
    )


__all__ = [
    "build_artifact_problem_hamiltonian",
    "build_canonical_hh_full_meta_pool_for_qse",
    "build_hh_full_meta_pool_for_qse",
]
