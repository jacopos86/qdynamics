"""Population-relative compiled-cost factor and shifted Paper-I score.

Paper II AP-McLachlan generalized-exchange selector.  Each checkpoint
enumerates candidate patches ``B = (D, A)`` (``D`` a deletion set of active
coordinate indices, ``A`` positioned insertions), evaluates one physical
merit ordinate ``m(B)`` per patch, and needs a population-relative
compiled-cost factor ``K(B)`` plus a shifted Paper-I score.  Both are
sealed once per certification level and never renormalized after a
finalist is rejected.  This module is pure: numpy-only, deterministic,
no I/O.

Cost coordinates.  For a patch ``B`` applied to the base ansatz ``U`` the
three signed compiled marginals are

    Delta x(B) = x(U (+) B) - x(U),   x in {n_2q, d_2q, d_c}

for two-qubit count, two-qubit depth, and total circuit depth.

Population normalization (Paper-I convention, see
``pipelines.static_adapt.ra_adapt.marginal_qiskit_cost``).  Over the
ordered population ``P`` of competing patches, per coordinate ``x``:

    med_x     = median_{B in P} Delta x(B)
    MAD_x     = median_{B in P} |Delta x(B) - med_x|
    s_x       = max(MAD_x, s_min),                 s_min = 1e-12
    Cbar_x(B) = (2/pi) atan((Delta x(B) - med_x) / s_x)     in (-1, 1)

    Cbar(B)   = (0.30 Cbar_n2q + 0.30 Cbar_d2q + 0.25 Cbar_dc) / 0.85
    K(B)      = 2 / (2 - Cbar(B))                           in (2/3, 2)

A compile burden (``Cbar > 0``) gives ``K > 1``; a compile relief
(``Cbar < 0``) gives ``K < 1``.  ``m / K = m (1 - Cbar/2)`` is the Paper-I
master-score identity.

Neutrality.  A singleton population has ``Delta x = med_x`` for every
coordinate, so ``Cbar_x = (2/pi) atan(0) = 0`` and ``K = 1`` exactly.  A
coordinate whose values are all equal likewise has ``MAD_x = 0`` and
``Delta x - med_x = 0`` for every record, so it contributes ``atan(0) = 0``
regardless of the floored scale; the code sets that bounded coordinate to
``0.0`` explicitly rather than dividing ``0 / s_min``.  A coordinate with
``MAD_x = 0`` but unequal values (more than half the population tied at the
median) keeps the Paper-I floor ``s_x = s_min``, which saturates the
off-median records at ``Cbar_x -> +/-1``; in double precision that limit
is reached exactly, so ``K`` can equal an endpoint numerically even though
the interval is open in exact arithmetic.  An empty population raises
``ValueError``.

Stay.  The stay patch ``B = (emptyset, emptyset)`` is the reference, not an
observation: it is never included in the population statistics and always
carries ``K = 1`` (``k_for_stay``), merit ``m = 0``, and score ``S = 0``.

Shifted Paper-I score.  With ``tau >= 0`` and ``m >= -tau``,

    S(m, K, tau) = (m + tau) / K - tau

so ``K = 1`` gives ``S = m``; ``tau = 0`` gives ``S = m / K``; and
``m = -tau`` gives ``S = -tau`` for every ``K``.  Regimes:

* ``LARGE_MISS``: ``tau = 0``, eligibility ``m > 0`` (strict);
* ``MAINTENANCE``: ``tau = tau_del >= 0``, eligibility ``m >= -tau_del``;
  a candidate must have ``S > 0`` to beat the stay reference, so a
  zero-merit deletion with compile relief ``K < 1`` scores
  ``S = tau_del (1/K - 1) > 0`` and beats stay when ``tau_del > 0``.

Rank order is descending ``S``, then descending raw merit, then ascending
``K``, then ascending canonical patch key: a deterministic total order.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1,
    AppendCostRawEstimate,
    AppendCostSettings,
    AppendCostTelemetry,
    append_cost_telemetry_for_family,
)

POPULATION_COST_NORMALIZATION_POLICY_V1 = (
    "paper_ii_population_cost_normalization_v1"
)
POPULATION_COST_NORMALIZATION_SCHEMA_V1 = (
    "paper_ii_population_cost_normalization_receipt_v1"
)
POPULATION_COST_COORDINATES: tuple[str, ...] = (
    "delta_n2q",
    "delta_d2q",
    "delta_dc",
)
POPULATION_COST_COORDINATE_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"delta_n2q": 0.30, "delta_d2q": 0.30, "delta_dc": 0.25}
)
POPULATION_COST_SCALE_FLOOR = 1.0e-12
STAY_COST_FACTOR = 1.0
STAY_MERIT = 0.0
STAY_SCORE = 0.0
# ``StructuralCandidate.order_key`` for the stay patch: no deletions, no
# insertions.  Reused so stay compares with every other canonical key.
STAY_PATCH_IDENTITY: tuple[tuple[int, ...], tuple[tuple[str, ...], ...]] = ((), ())

EXCLUDED_MERIT_NOT_POSITIVE = "merit_not_positive"
EXCLUDED_MERIT_BELOW_MINUS_TAU = "merit_below_minus_tau"
EXCLUDED_SCORE_NOT_ABOVE_STAY = "score_not_above_stay"

_LEGACY_COMPONENTS = ("2q", "d", "1q", "theta", "shot")
_LEGACY_COMPONENT_SOURCE = "paper_ii_population_repair_input"


# ---------------------------------------------------------------------------
# Patch identity
# ---------------------------------------------------------------------------


def patch_identity(
    removed_runtime_indices: Sequence[int],
    plan_key: Sequence[Sequence[str]],
) -> tuple[tuple[int, ...], tuple[tuple[str, ...], ...]]:
    """Return the canonical patch identity in ``StructuralCandidate.order_key`` shape.

    ``(removed_runtime_indices tuple, plan_key tuple)`` where every plan-key
    entry is a tuple of strings.  The same shape serves as ``patch_id`` for
    sealing and as ``canonical_key`` for ranking, so ties compare across
    candidate shapes exactly as they do in the structural enumeration.
    """

    removed = tuple(int(index) for index in removed_runtime_indices)
    plan = tuple(tuple(str(part) for part in entry) for entry in plan_key)
    return (removed, plan)


def _json_ready(value: Any) -> Any:
    """Convert nested tuples/mappings/scalars to a canonical JSON shape.

    Fails closed on unsupported types so an identity that cannot be
    serialized cannot silently alter the receipt.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    raise TypeError(
        f"patch identity component of type {type(value).__name__!r} is not "
        "JSON-canonical."
    )


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _finite_float(value: Any, name: str) -> float:
    value_f = float(value)
    if not math.isfinite(value_f):
        raise ValueError(f"{name} must be finite.")
    return value_f


# ---------------------------------------------------------------------------
# Compiled resource marginals and the sealed population
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompiledResourceMarginal:
    """Signed compiled marginals ``x(patched) - x(base)`` for one patch.

    ``delta_n2q``: two-qubit gate count; ``delta_d2q``: two-qubit depth;
    ``delta_dc``: total circuit depth.  Negative values are genuine compile
    relief (a deletion that shortens the circuit) and are preserved.
    """

    delta_n2q: int
    delta_d2q: int
    delta_dc: int

    def __post_init__(self) -> None:
        for name in POPULATION_COST_COORDINATES:
            raw = getattr(self, name)
            if isinstance(raw, float) and not raw.is_integer():
                raise ValueError(f"{name} must be an integer marginal.")
            object.__setattr__(self, name, int(raw))

    def as_tuple(self) -> tuple[int, int, int]:
        return (int(self.delta_n2q), int(self.delta_d2q), int(self.delta_dc))

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name)) for name in POPULATION_COST_COORDINATES
        }


@dataclass(frozen=True, slots=True)
class CoordinateStatistics:
    """Population median, MAD, and floored scale for one cost coordinate."""

    name: str
    median: float
    mad: float
    scale: float
    all_equal: bool


@dataclass(frozen=True, slots=True)
class SealedRecord:
    """One patch's raw marginals, bounded coordinates, ``Cbar`` and ``K``."""

    patch_id: Any
    marginal: CompiledResourceMarginal
    bounded_n2q: float
    bounded_d2q: float
    bounded_dc: float
    cbar: float
    K: float

    def bounded(self) -> dict[str, float]:
        return {
            "delta_n2q": float(self.bounded_n2q),
            "delta_d2q": float(self.bounded_d2q),
            "delta_dc": float(self.bounded_dc),
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "patch_id": _json_ready(self.patch_id),
            "raw": {k: int(v) for k, v in self.marginal.as_dict().items()},
            "normalized": self.bounded(),
            "cbar": float(self.cbar),
            "K": float(self.K),
        }


@dataclass(frozen=True)
class SealedPopulation:
    """Immutable population-relative cost normalization for one level.

    Built by :meth:`PopulationCostNormalization.seal` and never modified:
    rejecting a finalist does not reseal, so every surviving patch keeps
    the ``K`` it was assigned when the population was first scored.
    """

    policy: str
    schema: str
    weights: tuple[tuple[str, float], ...]
    scale_floor: float
    statistics: tuple[CoordinateStatistics, ...]
    records: tuple[SealedRecord, ...]
    population_sha256: str
    _index: Mapping[Any, int] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        index: dict[Any, int] = {}
        for position, record in enumerate(self.records):
            if record.patch_id in index:
                raise ValueError(
                    f"duplicate patch_id in sealed population: {record.patch_id!r}"
                )
            index[record.patch_id] = int(position)
        object.__setattr__(self, "_index", MappingProxyType(index))

    # -- shape ------------------------------------------------------------

    @property
    def size(self) -> int:
        return int(len(self.records))

    @property
    def patch_ids(self) -> tuple[Any, ...]:
        return tuple(record.patch_id for record in self.records)

    @property
    def marginals(self) -> tuple[CompiledResourceMarginal, ...]:
        return tuple(record.marginal for record in self.records)

    @property
    def medians(self) -> dict[str, float]:
        return {stat.name: float(stat.median) for stat in self.statistics}

    @property
    def mads(self) -> dict[str, float]:
        return {stat.name: float(stat.mad) for stat in self.statistics}

    @property
    def scales(self) -> dict[str, float]:
        return {stat.name: float(stat.scale) for stat in self.statistics}

    @property
    def cbar(self) -> tuple[float, ...]:
        return tuple(float(record.cbar) for record in self.records)

    @property
    def K(self) -> tuple[float, ...]:
        return tuple(float(record.K) for record in self.records)

    # -- lookup -----------------------------------------------------------

    def record_of(self, patch_id: Any) -> SealedRecord:
        try:
            return self.records[self._index[patch_id]]
        except KeyError as exc:
            raise KeyError(
                f"patch_id {patch_id!r} is not in the sealed population."
            ) from exc

    def k_of(self, patch_id: Any) -> float:
        """Return the sealed ``K`` for ``patch_id``; ``KeyError`` if unknown."""

        return float(self.record_of(patch_id).K)

    def cbar_of(self, patch_id: Any) -> float:
        return float(self.record_of(patch_id).cbar)

    @staticmethod
    def k_for_stay() -> float:
        """Stay is the reference, not an observation: ``K = 1`` always."""

        return float(STAY_COST_FACTOR)

    # -- receipt ----------------------------------------------------------

    def receipt_payload(self) -> dict[str, Any]:
        """The hashed payload (without the hash itself)."""

        return _receipt_payload(
            schema=self.schema,
            policy=self.policy,
            weights=dict(self.weights),
            scale_floor=self.scale_floor,
            statistics=self.statistics,
            records=self.records,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            **self.receipt_payload(),
            "population_sha256": str(self.population_sha256),
        }


def _receipt_payload(
    *,
    schema: str,
    policy: str,
    weights: Mapping[str, float],
    scale_floor: float,
    statistics: Sequence[CoordinateStatistics],
    records: Sequence[SealedRecord],
) -> dict[str, Any]:
    return {
        "schema": str(schema),
        "policy": str(policy),
        "coordinates": list(POPULATION_COST_COORDINATES),
        "weights": {str(k): float(v) for k, v in weights.items()},
        "weight_sum": float(sum(float(v) for v in weights.values())),
        "scale_floor": float(scale_floor),
        "medians": {stat.name: float(stat.median) for stat in statistics},
        "mads": {stat.name: float(stat.mad) for stat in statistics},
        "mad_scales": {stat.name: float(stat.scale) for stat in statistics},
        "all_equal": {stat.name: bool(stat.all_equal) for stat in statistics},
        "bounded_formula": "(2/pi)*atan((delta-median)/scale)",
        "cost_factor_formula": "2/(2-weighted_mean(normalized_coordinates))",
        "cost_factor_bounds_exclusive": [2.0 / 3.0, 2.0],
        "stay_cost_factor": float(STAY_COST_FACTOR),
        "size": int(len(records)),
        "records": [record.to_json_dict() for record in records],
    }


@dataclass(frozen=True)
class PopulationCostNormalization:
    """Seal a population's compiled marginals into per-patch ``K`` factors.

    Numerical conventions match Paper-I ``population_relative_score_factors``
    exactly: ``numpy`` median, MAD about that median, scale floor
    ``1e-12``, ``(2/pi) atan`` compression, weights ``(0.30, 0.30, 0.25)``
    divided by their sum ``0.85``, and a canonical sorted-key JSON receipt
    hashed with SHA-256.
    """

    weights: tuple[tuple[str, float], ...] = tuple(
        (name, float(POPULATION_COST_COORDINATE_WEIGHTS[name]))
        for name in POPULATION_COST_COORDINATES
    )
    scale_floor: float = POPULATION_COST_SCALE_FLOOR
    policy: str = POPULATION_COST_NORMALIZATION_POLICY_V1
    schema: str = POPULATION_COST_NORMALIZATION_SCHEMA_V1

    def __post_init__(self) -> None:
        weights = tuple(
            (str(name), _finite_float(value, f"weight[{name}]"))
            for name, value in dict(self.weights).items()
        )
        names = tuple(name for name, _ in weights)
        if names != POPULATION_COST_COORDINATES:
            raise ValueError(
                "weights must be keyed exactly by "
                f"{POPULATION_COST_COORDINATES} in order; got {names}."
            )
        if any(value <= 0.0 for _, value in weights):
            raise ValueError("weights must be positive.")
        object.__setattr__(self, "weights", weights)
        floor = _finite_float(self.scale_floor, "scale_floor")
        if floor <= 0.0:
            raise ValueError("scale_floor must be positive.")
        object.__setattr__(self, "scale_floor", floor)

    @property
    def weight_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.weights}

    @property
    def weight_sum(self) -> float:
        return float(sum(float(value) for _, value in self.weights))

    @staticmethod
    def k_for_stay() -> float:
        """Stay is the reference, not an observation: ``K = 1`` always."""

        return float(STAY_COST_FACTOR)

    def coordinate_statistics(
        self, marginals: Sequence[CompiledResourceMarginal]
    ) -> tuple[CoordinateStatistics, ...]:
        """Median, MAD, floored scale, and all-equal flag per coordinate."""

        population = tuple(marginals)
        if not population:
            raise ValueError("population must not be empty.")
        out: list[CoordinateStatistics] = []
        for name in POPULATION_COST_COORDINATES:
            values = np.asarray(
                [float(getattr(marginal, name)) for marginal in population],
                dtype=float,
            )
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            all_equal = bool(np.all(values == values[0]))
            out.append(
                CoordinateStatistics(
                    name=str(name),
                    median=median,
                    mad=mad,
                    scale=float(max(float(self.scale_floor), mad)),
                    all_equal=all_equal,
                )
            )
        return tuple(out)

    def seal(
        self,
        records: Sequence[tuple[Any, CompiledResourceMarginal]],
    ) -> SealedPopulation:
        """Seal an ORDERED population of ``(patch_id, marginal)`` records.

        Order is part of the identity: the receipt hash is over the ordered
        record list, so the same patches in a different order seal to a
        different ``population_sha256`` even though every ``K`` is equal.
        """

        ordered = tuple(records)
        if not ordered:
            raise ValueError("population must not be empty.")
        pairs: list[tuple[Any, CompiledResourceMarginal]] = []
        for entry in ordered:
            patch_id, marginal = entry
            if not isinstance(marginal, CompiledResourceMarginal):
                raise TypeError(
                    "population records must carry CompiledResourceMarginal; "
                    f"got {type(marginal).__name__!r}."
                )
            pairs.append((patch_id, marginal))
        statistics = self.coordinate_statistics([m for _, m in pairs])
        stats_by_name = {stat.name: stat for stat in statistics}
        weights = self.weight_map
        weight_sum = self.weight_sum
        sealed: list[SealedRecord] = []
        for patch_id, marginal in pairs:
            bounded: dict[str, float] = {}
            for name in POPULATION_COST_COORDINATES:
                stat = stats_by_name[name]
                if stat.all_equal:
                    # atan(0): every record sits on the median.  Explicit so
                    # the neutral coordinate never touches the floored scale.
                    bounded[name] = 0.0
                    continue
                bounded[name] = (2.0 / math.pi) * math.atan(
                    (float(getattr(marginal, name)) - stat.median) / stat.scale
                )
            cbar = float(
                sum(weights[name] * bounded[name] for name in POPULATION_COST_COORDINATES)
                / weight_sum
            )
            K = float(2.0 / (2.0 - cbar))
            # Mathematically K in (2/3, 2).  In double precision (2/pi)*atan
            # saturates to exactly +/-1 once |delta - median| / scale exceeds
            # ~1e16 (a floored scale with an integer outlier does this), so
            # the guard is inclusive at the endpoints, as in Paper-I.
            if not math.isfinite(K) or K < 2.0 / 3.0 or K > 2.0:
                raise RuntimeError(
                    "population cost factor left the interval [2/3, 2]."
                )
            sealed.append(
                SealedRecord(
                    patch_id=patch_id,
                    marginal=marginal,
                    bounded_n2q=float(bounded["delta_n2q"]),
                    bounded_d2q=float(bounded["delta_d2q"]),
                    bounded_dc=float(bounded["delta_dc"]),
                    cbar=cbar,
                    K=K,
                )
            )
        payload = _receipt_payload(
            schema=self.schema,
            policy=self.policy,
            weights=weights,
            scale_floor=self.scale_floor,
            statistics=statistics,
            records=sealed,
        )
        digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
        return SealedPopulation(
            policy=str(self.policy),
            schema=str(self.schema),
            weights=tuple(self.weights),
            scale_floor=float(self.scale_floor),
            statistics=tuple(statistics),
            records=tuple(sealed),
            population_sha256=str(digest),
        )

    __call__ = seal


def seal_population(
    records: Sequence[tuple[Any, CompiledResourceMarginal]],
) -> SealedPopulation:
    """Seal ``records`` under the default v1 policy."""

    return PopulationCostNormalization().seal(records)


# ---------------------------------------------------------------------------
# Shifted Paper-I score and ranking
# ---------------------------------------------------------------------------


class RankRegime(str, enum.Enum):
    """Which shift and eligibility rule the checkpoint is under."""

    LARGE_MISS = "large_miss"
    MAINTENANCE = "maintenance"


def shifted_score(merit: float, K: float, tau: float = 0.0) -> float:
    """Return ``S = (m + tau) / K - tau`` for ``m >= -tau``.

    ``K = 1`` gives ``S = m``; ``tau = 0`` gives ``S = m / K``; ``m = -tau``
    gives ``S = -tau`` for every ``K``.  ``m < -tau`` is outside the
    admissible domain and raises ``ValueError``.
    """

    m = _finite_float(merit, "merit")
    k = _finite_float(K, "K")
    t = _finite_float(tau, "tau")
    if k <= 0.0:
        raise ValueError("K must be positive.")
    if t < 0.0:
        raise ValueError("tau must be non-negative.")
    if m < -t:
        raise ValueError(
            f"merit {m!r} is below -tau = {-t!r}; shifted score undefined."
        )
    return float((m + t) / k - t)


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """One patch's ranking inputs: merit ``m``, sealed ``K``, tie-breaks."""

    patch_id: Any
    merit: float
    K: float
    raw_merit_tiebreak: float
    canonical_key: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "merit", _finite_float(self.merit, "merit"))
        k = _finite_float(self.K, "K")
        if k <= 0.0:
            raise ValueError("K must be positive.")
        object.__setattr__(self, "K", k)
        object.__setattr__(
            self,
            "raw_merit_tiebreak",
            _finite_float(self.raw_merit_tiebreak, "raw_merit_tiebreak"),
        )


@dataclass(frozen=True, slots=True)
class RankedPatch:
    """One eligible patch in rank order."""

    patch_id: Any
    merit: float
    K: float
    score: float
    eligible: bool
    rank_key: tuple


@dataclass(frozen=True, slots=True)
class ExcludedPatch:
    """One ineligible patch with the reason it was excluded."""

    patch_id: Any
    merit: float
    K: float
    score: float | None
    reason: str


@dataclass(frozen=True)
class RankingResult:
    """Ordered eligible patches plus the excluded remainder."""

    regime: RankRegime
    tau: float
    ranked: tuple[RankedPatch, ...]
    excluded: tuple[ExcludedPatch, ...]

    @property
    def best(self) -> RankedPatch | None:
        return self.ranked[0] if self.ranked else None

    @property
    def stay(self) -> RankedPatch:
        return stay_reference()


def stay_reference() -> RankedPatch:
    """The stay reference: ``m = 0``, ``K = 1``, ``S = 0``."""

    return RankedPatch(
        patch_id=STAY_PATCH_IDENTITY,
        merit=float(STAY_MERIT),
        K=float(STAY_COST_FACTOR),
        score=float(STAY_SCORE),
        eligible=True,
        rank_key=(
            -float(STAY_SCORE),
            -float(STAY_MERIT),
            float(STAY_COST_FACTOR),
            STAY_PATCH_IDENTITY,
        ),
    )


def _coerce_score_record(entry: Any) -> ScoreRecord:
    if isinstance(entry, ScoreRecord):
        return entry
    parts = tuple(entry)
    if len(parts) != 5:
        raise ValueError(
            "score records must be ScoreRecord or "
            "(patch_id, merit, K, raw_merit_tiebreak, canonical_key)."
        )
    return ScoreRecord(*parts)


def score_and_rank(
    records: Sequence[Any],
    regime: RankRegime,
    tau_del: float = 0.0,
) -> RankingResult:
    """Score and rank an ORDERED population of patches under ``regime``.

    Eligibility: ``LARGE_MISS`` uses ``tau = 0`` (``tau_del`` is not
    consulted) and requires ``m > 0``; ``MAINTENANCE`` uses
    ``tau = tau_del`` and requires ``m >= -tau_del`` and ``S > 0`` (the stay
    reference has ``S = 0``).  Rank key is
    ``(-S, -raw_merit_tiebreak, K, canonical_key)`` ascending, which is a
    total order because patch ids and canonical keys must be unique.
    """

    regime = RankRegime(regime)
    coerced = tuple(_coerce_score_record(entry) for entry in records)
    tau_input = _finite_float(tau_del, "tau_del")
    if tau_input < 0.0:
        raise ValueError("tau_del must be non-negative.")
    tau = 0.0 if regime is RankRegime.LARGE_MISS else float(tau_input)

    seen_ids: set[Any] = set()
    seen_keys: set[Any] = set()
    for record in coerced:
        if record.patch_id in seen_ids:
            raise ValueError(f"duplicate patch_id in ranking: {record.patch_id!r}")
        if record.canonical_key in seen_keys:
            raise ValueError(
                f"duplicate canonical_key in ranking: {record.canonical_key!r}"
            )
        seen_ids.add(record.patch_id)
        seen_keys.add(record.canonical_key)

    ranked: list[RankedPatch] = []
    excluded: list[ExcludedPatch] = []
    for record in coerced:
        m = float(record.merit)
        k = float(record.K)
        if regime is RankRegime.LARGE_MISS and not m > 0.0:
            excluded.append(
                ExcludedPatch(
                    patch_id=record.patch_id,
                    merit=m,
                    K=k,
                    score=None,
                    reason=EXCLUDED_MERIT_NOT_POSITIVE,
                )
            )
            continue
        if regime is RankRegime.MAINTENANCE and m < -tau:
            excluded.append(
                ExcludedPatch(
                    patch_id=record.patch_id,
                    merit=m,
                    K=k,
                    score=None,
                    reason=EXCLUDED_MERIT_BELOW_MINUS_TAU,
                )
            )
            continue
        score = shifted_score(m, k, tau)
        if not score > float(STAY_SCORE):
            excluded.append(
                ExcludedPatch(
                    patch_id=record.patch_id,
                    merit=m,
                    K=k,
                    score=float(score),
                    reason=EXCLUDED_SCORE_NOT_ABOVE_STAY,
                )
            )
            continue
        rank_key = (
            -float(score),
            -float(record.raw_merit_tiebreak),
            float(k),
            record.canonical_key,
        )
        ranked.append(
            RankedPatch(
                patch_id=record.patch_id,
                merit=m,
                K=k,
                score=float(score),
                eligible=True,
                rank_key=rank_key,
            )
        )
    ranked.sort(key=lambda patch: patch.rank_key)
    return RankingResult(
        regime=regime,
        tau=float(tau),
        ranked=tuple(ranked),
        excluded=tuple(excluded),
    )


# ---------------------------------------------------------------------------
# Legacy compatibility: family_robust_v1 over the actual population
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegacyRobustDenominator:
    """``append_cost`` family_robust_v1 denominators for one patch."""

    patch_id: Any
    raw_components: Mapping[str, float]
    bar_components: Mapping[str, float]
    hardware_cost_excess_sum: float
    hardware_cost_denominator: float
    utility_denominator: float
    telemetry: AppendCostTelemetry


def legacy_family_robust_denominators(
    records: Sequence[tuple[Any, Mapping[str, float]]],
    *,
    settings: AppendCostSettings | None = None,
) -> tuple[LegacyRobustDenominator, ...]:
    """Historical proxy repair: family_robust_v1 over the ACTUAL population.

    The historical path normalized each candidate against a singleton
    family, which makes every denominator exactly ``1.0``.  This helper
    feeds the whole competing population to
    ``append_cost_telemetry_for_family`` unchanged, so the arithmetic is
    the module's own: per component ``bar = asinh(max(0, v - median) /
    max(scale_floor, family_scale))``, ``excess = sum lambda_c bar_c``,
    ``denominator = max(1, 1 + excess)``, ``utility_denominator =
    max(scale_floor, denominator ** alpha)``.  The clip is one-sided, so
    records at or below the median keep denominator ``1.0`` exactly.
    """

    ordered = tuple(records)
    if not ordered:
        raise ValueError("population must not be empty.")
    cfg = settings if settings is not None else AppendCostSettings()
    if cfg.cost_normalization_mode != AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1:
        raise ValueError(
            "legacy repair requires cost_normalization_mode "
            f"{AP_APPEND_COST_NORMALIZATION_FAMILY_ROBUST_V1!r}; got "
            f"{cfg.cost_normalization_mode!r}."
        )
    estimates = tuple(
        legacy_raw_estimate(patch_id, raw_components)
        for patch_id, raw_components in ordered
    )
    telemetry = append_cost_telemetry_for_family(
        estimates,
        insertion_gains=tuple(None for _ in estimates),
        settings=cfg,
    )
    out: list[LegacyRobustDenominator] = []
    for (patch_id, _), item in zip(ordered, telemetry):
        out.append(
            LegacyRobustDenominator(
                patch_id=patch_id,
                raw_components=MappingProxyType(
                    {str(k): float(v) for k, v in item.raw_components.items()}
                ),
                bar_components=MappingProxyType(
                    {str(k): float(v) for k, v in item.bar_components.items()}
                ),
                hardware_cost_excess_sum=float(item.hardware_cost_excess_sum),
                hardware_cost_denominator=float(item.hardware_cost_denominator),
                utility_denominator=float(item.utility_denominator),
                telemetry=item,
            )
        )
    return tuple(out)


def legacy_raw_estimate(
    patch_id: Any, raw_components: Mapping[str, float]
) -> AppendCostRawEstimate:
    """Wrap a five-component dict as the ``AppendCostRawEstimate`` input shape."""

    components = {
        key: float(raw_components.get(key, 0.0)) for key in _LEGACY_COMPONENTS
    }
    unknown = set(str(k) for k in raw_components) - set(_LEGACY_COMPONENTS)
    if unknown:
        raise ValueError(
            f"unknown legacy cost components {sorted(unknown)}; expected "
            f"{_LEGACY_COMPONENTS}."
        )
    return AppendCostRawEstimate(
        raw_components=components,
        component_sources={key: _LEGACY_COMPONENT_SOURCE for key in _LEGACY_COMPONENTS},
        inserted_runtime_count=int(max(0, round(components["theta"]))),
        atom_ids=(str(patch_id),),
    )


__all__ = [
    "EXCLUDED_MERIT_BELOW_MINUS_TAU",
    "EXCLUDED_MERIT_NOT_POSITIVE",
    "EXCLUDED_SCORE_NOT_ABOVE_STAY",
    "POPULATION_COST_COORDINATES",
    "POPULATION_COST_COORDINATE_WEIGHTS",
    "POPULATION_COST_NORMALIZATION_POLICY_V1",
    "POPULATION_COST_NORMALIZATION_SCHEMA_V1",
    "POPULATION_COST_SCALE_FLOOR",
    "STAY_COST_FACTOR",
    "STAY_MERIT",
    "STAY_PATCH_IDENTITY",
    "STAY_SCORE",
    "CompiledResourceMarginal",
    "CoordinateStatistics",
    "ExcludedPatch",
    "LegacyRobustDenominator",
    "PopulationCostNormalization",
    "RankRegime",
    "RankedPatch",
    "RankingResult",
    "ScoreRecord",
    "SealedPopulation",
    "SealedRecord",
    "legacy_family_robust_denominators",
    "legacy_raw_estimate",
    "patch_identity",
    "score_and_rank",
    "seal_population",
    "shifted_score",
    "stay_reference",
]
