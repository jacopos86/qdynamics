"""Pure mathematical policy for Paper-II generalized exchange.

This module knows nothing about ansatz classes, Pauli generators, circuits,
measurement backends, or AP-McLachlan runtime state.  Those objects belong to
an adapter.  Mathematically, a proposal is only a pair ``(D, I)`` of deletions
and positioned insertions, together with realized objective values supplied by
an external evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


EXCHANGE_FACE_FULL = "full"
EXCHANGE_FACE_INSERT_ONLY = "insert_only"
EXCHANGE_FACE_DELETE_ONLY = "delete_only"
EXCHANGE_FACE_STAY_ONLY = "stay_only"

EXCHANGE_RANKING_COST_AWARE = "cost_aware"
EXCHANGE_RANKING_SIGNED_DRIFT = "signed_realized_drift"

REALIZED_ACCEPT = "accept"
REALIZED_RETRY_INSERT_FACE = "retry_insert_face"
REALIZED_REFUSE = "refuse"


@dataclass(frozen=True)
class GeneralizedPatch:
    """A support edit ``(D, I)`` independent of its physical realization."""

    deletions: tuple[Hashable, ...] = ()
    insertions: tuple[Hashable, ...] = ()

    @property
    def kind(self) -> str:
        if self.deletions and self.insertions:
            return "exchange"
        if self.deletions:
            return "delete"
        if self.insertions:
            return "insert"
        return "stay"


@dataclass(frozen=True)
class GeneralizedExchangeDomain:
    """The admissible face and ordering at one checkpoint."""

    checkpoint_l2: float
    l2_cut: float
    accuracy_debt: bool
    insertion_face_open: bool
    deletion_face_open: bool
    true_exchange_face_open: bool
    face: str
    ranking: str
    debt_policy: str
    support_floor: int
    insertion_cardinality_cap: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "checkpoint_l2": float(self.checkpoint_l2),
            "l2_cut": float(self.l2_cut),
            "accuracy_debt": bool(self.accuracy_debt),
            "insertion_face_open": bool(self.insertion_face_open),
            "deletion_face_open": bool(self.deletion_face_open),
            "true_exchange_face_open": bool(self.true_exchange_face_open),
            "face": str(self.face),
            "ranking": str(self.ranking),
            "debt_policy": str(self.debt_policy),
            "support_floor": int(self.support_floor),
            "insertion_cardinality_cap": int(self.insertion_cardinality_cap),
        }


@dataclass(frozen=True)
class GeneralizedExchange:
    """Mathematical generalized-exchange rule.

    ``debt_policy='drift_ranked'`` is the Paper-II operating rule.
    ``debt_policy='insertion_only'`` is a boundary-face ablation.
    ``debt_policy='any_improving'`` is retained only to reproduce historical
    diagnostics.
    """

    l2_cut: float
    debt_policy: str = "drift_ranked"
    support_floor: int = 1
    insertion_cardinality_cap: int = 1
    l2_debt_enabled: bool = True

    def __post_init__(self) -> None:
        if float(self.l2_cut) < 0.0:
            raise ValueError("l2_cut must be non-negative.")
        if self.debt_policy not in {
            "drift_ranked",
            "insertion_only",
            "any_improving",
        }:
            raise ValueError(f"Unknown debt_policy {self.debt_policy!r}.")
        if int(self.support_floor) < 0:
            raise ValueError("support_floor must be non-negative.")
        if int(self.insertion_cardinality_cap) < 0:
            raise ValueError("insertion_cardinality_cap must be non-negative.")

    def domain(
        self,
        *,
        checkpoint_l2: float,
        insertion_gate_open: bool,
        deletion_candidate_count: int,
    ) -> GeneralizedExchangeDomain:
        """Return the admissible face without inspecting an ansatz object."""

        accuracy_debt = bool(
            self.l2_debt_enabled and float(checkpoint_l2) > float(self.l2_cut)
        )
        insertion_open = bool(
            insertion_gate_open and int(self.insertion_cardinality_cap) > 0
        )
        deletion_open = int(deletion_candidate_count) > 0
        if accuracy_debt and self.debt_policy == "insertion_only":
            deletion_open = False

        if insertion_open and deletion_open:
            face = EXCHANGE_FACE_FULL
        elif insertion_open:
            face = EXCHANGE_FACE_INSERT_ONLY
        elif deletion_open:
            face = EXCHANGE_FACE_DELETE_ONLY
        else:
            face = EXCHANGE_FACE_STAY_ONLY

        ranking = (
            EXCHANGE_RANKING_SIGNED_DRIFT
            if accuracy_debt and self.debt_policy == "drift_ranked"
            else EXCHANGE_RANKING_COST_AWARE
        )
        return GeneralizedExchangeDomain(
            checkpoint_l2=float(checkpoint_l2),
            l2_cut=float(self.l2_cut),
            accuracy_debt=accuracy_debt,
            insertion_face_open=insertion_open,
            deletion_face_open=deletion_open,
            true_exchange_face_open=bool(insertion_open and deletion_open),
            face=face,
            ranking=ranking,
            debt_policy=str(self.debt_policy),
            support_floor=int(self.support_floor),
            insertion_cardinality_cap=int(self.insertion_cardinality_cap),
        )

    @staticmethod
    def assess_realized_candidate(
        *,
        domain: GeneralizedExchangeDomain,
        patch: GeneralizedPatch,
        candidate_l2: float,
    ) -> str:
        """Apply the signed realized-L2 commit rule under accuracy debt."""

        if not domain.accuracy_debt or float(candidate_l2) < domain.checkpoint_l2:
            return REALIZED_ACCEPT
        if (
            patch.deletions
            and domain.insertion_face_open
            and domain.deletion_face_open
        ):
            return REALIZED_RETRY_INSERT_FACE
        return REALIZED_REFUSE


__all__ = [
    "EXCHANGE_FACE_DELETE_ONLY",
    "EXCHANGE_FACE_FULL",
    "EXCHANGE_FACE_INSERT_ONLY",
    "EXCHANGE_FACE_STAY_ONLY",
    "EXCHANGE_RANKING_COST_AWARE",
    "EXCHANGE_RANKING_SIGNED_DRIFT",
    "GeneralizedExchange",
    "GeneralizedExchangeDomain",
    "GeneralizedPatch",
    "REALIZED_ACCEPT",
    "REALIZED_REFUSE",
    "REALIZED_RETRY_INSERT_FACE",
]
