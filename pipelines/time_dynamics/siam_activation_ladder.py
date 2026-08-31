"""Versioned activation-cut contracts for the SIAM ``n_bath=4`` pilot.

The first SIAM pilot packages predate the parallel-benchmark decision memo.
Those packages used ``1e-3`` as rung ``r00``.  The frozen campaign now owns a
coarser ladder beginning at ``1e-2``.  Historical action identifiers and
embedded package digests are immutable evidence, so migration means mapping
their *cut value* onto the canonical ladder; it never means rewriting ``r00``
to ``r02`` or accepting the historical index as the canonical index.

This module is deliberately independent of the planner and package builder.
It is the single compatibility seam used by both.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Sequence


SIAM_NB4_CANONICAL_LADDER_V2 = "siam_nb4_activation_cut_ladder_v2"
SIAM_NB4_LEGACY_LADDER_V1 = "siam_nb4_activation_cut_ladder_legacy_v1"

# The canonical sequence is the exact sequence frozen in the 2026-08-28
# parallel-benchmark handoff.  The 3.333... values use twelve significant
# decimal digits, matching the repository's serialized cut convention.
SIAM_NB4_ACTIVATION_CUT_LADDER: tuple[float, ...] = (
    1.0e-2,
    3.33333333333e-3,
    1.0e-3,
    3.33333333333e-4,
    1.0e-4,
    3.33333333333e-5,
    1.0e-5,
    3.33333333333e-6,
    1.0e-6,
    3.33333333333e-7,
    1.0e-7,
)

# Immutable package evidence produced before the handoff.  Keep this exact;
# changing it would make the embedded action/manifest digests uninterpretable.
SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER: tuple[float, ...] = (
    1.0e-3,
    3.0e-4,
    1.0e-4,
    3.0e-5,
    1.0e-5,
    3.0e-6,
    1.0e-6,
    3.0e-7,
    1.0e-7,
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def ladder_payload(values: Sequence[float]) -> tuple[str, ...]:
    """Return the historical package serialization for one ladder."""

    return tuple(repr(float(value)) for value in values)


def ladder_sha256(values: Sequence[float]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(ladder_payload(values)))).hexdigest()


SIAM_NB4_CANONICAL_LADDER_SHA256 = ladder_sha256(
    SIAM_NB4_ACTIVATION_CUT_LADDER
)
SIAM_NB4_LEGACY_LADDER_SHA256 = ladder_sha256(
    SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER
)


@dataclass(frozen=True)
class SIAMLadderResolution:
    """One cut resolved without conflating package and canonical positions."""

    activation_cut: float
    canonical_index: int | None
    package_index: int
    package_contract: str
    package_ladder_sha256: str

    @property
    def migrated(self) -> bool:
        return self.package_contract == SIAM_NB4_LEGACY_LADDER_V1

    @property
    def has_canonical_equivalent(self) -> bool:
        return self.canonical_index is not None


def canonical_registered_cut(value: float) -> tuple[float, int]:
    """Resolve one exact cut on the canonical v2 ladder."""

    cut = float(value)
    if not math.isfinite(cut) or cut <= 0.0:
        raise ValueError("SIAM activation cut must be finite and positive")
    for index, rung in enumerate(SIAM_NB4_ACTIVATION_CUT_LADDER):
        if cut == rung:
            return rung, index
    raise ValueError(
        "SIAM activation cut is not one exact canonical rung: "
        f"{cut!r}; ladder={SIAM_NB4_ACTIVATION_CUT_LADDER!r}"
    )


def canonical_index(value: float) -> int:
    return canonical_registered_cut(value)[1]


def resolve_embedded_ladder(
    *,
    activation_cut: float,
    package_index: Any,
    action_ladder_sha256: Any,
    manifest_ladder_sha256: Any,
    manifest_ladder: Any,
) -> SIAMLadderResolution:
    """Validate a canonical or immutable legacy embedded package ladder.

    New packages must use the canonical contract.  The legacy contract remains
    readable solely so already-produced raw receipts keep their original bytes,
    source epoch, action ID, and package identity.
    """

    if isinstance(package_index, bool) or not isinstance(package_index, int):
        raise ValueError("SIAM package ladder index is invalid")
    if not isinstance(manifest_ladder, list) or not all(
        isinstance(value, str) for value in manifest_ladder
    ):
        raise ValueError("SIAM package ladder payload is invalid")
    if action_ladder_sha256 != manifest_ladder_sha256:
        raise ValueError("SIAM action/manifest ladder digests differ")

    candidates = (
        (
            SIAM_NB4_CANONICAL_LADDER_V2,
            SIAM_NB4_ACTIVATION_CUT_LADDER,
            SIAM_NB4_CANONICAL_LADDER_SHA256,
        ),
        (
            SIAM_NB4_LEGACY_LADDER_V1,
            SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER,
            SIAM_NB4_LEGACY_LADDER_SHA256,
        ),
    )
    for contract, values, digest in candidates:
        if manifest_ladder != list(ladder_payload(values)):
            continue
        if action_ladder_sha256 != digest:
            raise ValueError("SIAM embedded ladder payload/digest mismatch")
        if package_index < 0 or package_index >= len(values):
            raise ValueError("SIAM embedded ladder index is out of range")
        if float(activation_cut) != values[package_index]:
            raise ValueError("SIAM embedded ladder index/cut mismatch")
        try:
            _cut, current_index = canonical_registered_cut(activation_cut)
        except ValueError:
            # A legacy package at, for example, 3e-4 remains valid historical
            # evidence even though the canonical v2 scientific ladder uses
            # 3.333...e-4.  Such evidence is readable/reproducible, but it has
            # no canonical planner position and therefore cannot cover a v2
            # action.  The special r00/cut1e-3 migration does have an exact
            # canonical equivalent at index two.
            current_index = None
        return SIAMLadderResolution(
            activation_cut=float(activation_cut),
            canonical_index=current_index,
            package_index=package_index,
            package_contract=contract,
            package_ladder_sha256=digest,
        )
    raise ValueError("SIAM package embeds an unknown activation-cut ladder")


def ladder_contract_json() -> dict[str, Any]:
    """Machine-readable migration contract for planner/report receipts."""

    return {
        "schema": SIAM_NB4_CANONICAL_LADDER_V2,
        "canonical_values": list(SIAM_NB4_ACTIVATION_CUT_LADDER),
        "canonical_sha256": SIAM_NB4_CANONICAL_LADDER_SHA256,
        "legacy_compatibility": {
            "schema": SIAM_NB4_LEGACY_LADDER_V1,
            "values": list(SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER),
            "sha256": SIAM_NB4_LEGACY_LADDER_SHA256,
            "noncanonical_cut_semantics": (
                "readable_historical_evidence_without_canonical_action_coverage"
            ),
            "immutable_action_aliases": {
                "r00_cut1e-3": {
                    "activation_cut": 1.0e-3,
                    "legacy_index": 0,
                    "canonical_index": canonical_index(1.0e-3),
                    "relabel": False,
                }
            },
        },
    }


__all__ = [
    "SIAM_NB4_ACTIVATION_CUT_LADDER",
    "SIAM_NB4_CANONICAL_LADDER_SHA256",
    "SIAM_NB4_CANONICAL_LADDER_V2",
    "SIAM_NB4_LEGACY_ACTIVATION_CUT_LADDER",
    "SIAM_NB4_LEGACY_LADDER_SHA256",
    "SIAM_NB4_LEGACY_LADDER_V1",
    "SIAMLadderResolution",
    "canonical_index",
    "canonical_registered_cut",
    "ladder_contract_json",
    "ladder_payload",
    "ladder_sha256",
    "resolve_embedded_ladder",
]
