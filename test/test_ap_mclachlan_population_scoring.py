"""Population-relative cost factor K and shifted Paper-I score (Paper II)."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from pipelines.time_dynamics.ap_mclachlan.append_cost import (
    AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1,
    AppendCostRawEstimate,
    AppendCostSettings,
    append_cost_telemetry_for_family,
)
from pipelines.time_dynamics.ap_mclachlan.population_scoring import (
    EXCLUDED_MERIT_BELOW_MINUS_TAU,
    EXCLUDED_MERIT_NOT_POSITIVE,
    EXCLUDED_SCORE_NOT_ABOVE_STAY,
    POPULATION_COST_COORDINATES,
    POPULATION_COST_NORMALIZATION_POLICY_V1,
    POPULATION_COST_SCALE_FLOOR,
    STAY_PATCH_IDENTITY,
    CompiledResourceMarginal,
    PopulationCostNormalization,
    RankRegime,
    ScoreRecord,
    SealedPopulation,
    legacy_family_robust_denominators,
    patch_identity,
    score_and_rank,
    seal_population,
    shifted_score,
    stay_reference,
)

K_LOWER = 2.0 / 3.0
K_UPPER = 2.0


def _m(value: int) -> CompiledResourceMarginal:
    return CompiledResourceMarginal(value, value, value)


def _three() -> list[tuple[str, CompiledResourceMarginal]]:
    return [("cheap", _m(1)), ("mid", _m(10)), ("dear", _m(100))]


# ---------------------------------------------------------------------------
# CompiledResourceMarginal
# ---------------------------------------------------------------------------


def test_compiled_resource_marginal_coerces_and_exposes_tuple() -> None:
    marginal = CompiledResourceMarginal(delta_n2q=3.0, delta_d2q=-2, delta_dc=7)
    assert marginal.as_tuple() == (3, -2, 7)
    assert all(isinstance(v, int) for v in marginal.as_tuple())
    assert marginal.as_dict() == {"delta_n2q": 3, "delta_d2q": -2, "delta_dc": 7}
    with pytest.raises(ValueError):
        CompiledResourceMarginal(1.5, 0, 0)


# ---------------------------------------------------------------------------
# Population normalization: neutrality, discrimination, bounds, receipt
# ---------------------------------------------------------------------------


def test_singleton_population_is_exactly_neutral() -> None:
    sealed = seal_population([("only", _m(37))])
    assert sealed.size == 1
    assert sealed.k_of("only") == 1.0
    assert sealed.cbar_of("only") == 0.0
    record = sealed.record_of("only")
    assert record.bounded() == {name: 0.0 for name in POPULATION_COST_COORDINATES}
    assert sealed.mads == {name: 0.0 for name in POPULATION_COST_COORDINATES}
    assert sealed.scales == {
        name: POPULATION_COST_SCALE_FLOOR for name in POPULATION_COST_COORDINATES
    }


def test_three_record_population_discriminates_cheap_median_dear() -> None:
    sealed = seal_population(_three())
    k_cheap, k_mid, k_dear = sealed.K
    assert k_cheap < 1.0 < k_dear
    assert k_mid == 1.0
    # Hand computation: median 10, MAD = median(9, 0, 90) = 9.
    # cheap: (2/pi) atan(-9/9) = -0.5 on every coordinate -> Cbar = -0.5,
    # K = 2/(2.5) = 0.8.  dear: (2/pi) atan(90/9) on every coordinate.
    assert k_cheap == pytest.approx(0.8, abs=1e-15)
    cbar_dear = (2.0 / math.pi) * math.atan(10.0)
    assert sealed.cbar_of("dear") == pytest.approx(cbar_dear, abs=1e-15)
    assert k_dear == pytest.approx(2.0 / (2.0 - cbar_dear), abs=1e-15)
    assert sealed.medians == {name: 10.0 for name in POPULATION_COST_COORDINATES}
    assert sealed.mads == {name: 9.0 for name in POPULATION_COST_COORDINATES}


def test_weighted_combination_uses_sealed_weights_over_their_sum() -> None:
    # Only delta_dc varies: Cbar = 0.25 * bounded_dc / 0.85.
    sealed = seal_population(
        [
            ("a", CompiledResourceMarginal(4, 4, 1)),
            ("b", CompiledResourceMarginal(4, 4, 10)),
            ("c", CompiledResourceMarginal(4, 4, 100)),
        ]
    )
    rec = sealed.record_of("a")
    assert rec.bounded_n2q == 0.0 and rec.bounded_d2q == 0.0
    assert rec.bounded_dc == pytest.approx(-0.5, abs=1e-15)
    assert rec.cbar == pytest.approx(0.25 * -0.5 / 0.85, abs=1e-15)
    assert rec.K == pytest.approx(2.0 / (2.0 - rec.cbar), abs=1e-15)


def test_zero_mad_all_equal_population_is_neutral() -> None:
    sealed = seal_population([("a", _m(5)), ("b", _m(5)), ("c", _m(5)), ("d", _m(5))])
    assert sealed.K == (1.0, 1.0, 1.0, 1.0)
    assert sealed.cbar == (0.0, 0.0, 0.0, 0.0)
    assert all(stat.all_equal for stat in sealed.statistics)


def test_zero_mad_with_outlier_keeps_paper_i_floor_and_saturates() -> None:
    # More than half tied at the median: MAD = 0, scale = floor, outlier -> ~+1.
    sealed = seal_population([("a", _m(5)), ("b", _m(5)), ("c", _m(5)), ("x", _m(6))])
    assert sealed.K[:3] == (1.0, 1.0, 1.0)
    assert sealed.cbar_of("x") == pytest.approx(1.0, abs=1e-9)
    assert sealed.k_of("x") == pytest.approx(2.0, abs=1e-9)
    assert sealed.k_of("x") < K_UPPER


def test_sign_symmetry_relief_below_one_burden_above_one() -> None:
    sealed = seal_population([("relief", _m(-8)), ("base", _m(0)), ("burden", _m(8))])
    assert sealed.k_of("relief") < 1.0
    assert sealed.k_of("base") == 1.0
    assert sealed.k_of("burden") > 1.0
    assert sealed.cbar_of("relief") == pytest.approx(-sealed.cbar_of("burden"), abs=1e-15)
    # Paper-I identity: 1/K = 1 - Cbar/2.
    for patch_id in ("relief", "base", "burden"):
        assert 1.0 / sealed.k_of(patch_id) == pytest.approx(
            1.0 - 0.5 * sealed.cbar_of(patch_id), abs=1e-15
        )


def test_k_bounds_are_exclusive_for_extreme_marginals() -> None:
    # Nonzero MAD (= 1): the extremes sit at (2/pi) atan(1e9), strictly
    # inside (-1, 1), so K is strictly inside (2/3, 2).
    sealed = seal_population(
        [
            ("floor", _m(-10**9)),
            ("a", _m(-1)),
            ("b", _m(0)),
            ("c", _m(1)),
            ("ceiling", _m(10**9)),
        ]
    )
    for k in sealed.K:
        assert K_LOWER < k < K_UPPER
    assert sealed.k_of("floor") == pytest.approx(K_LOWER, abs=1e-8)
    assert sealed.k_of("ceiling") == pytest.approx(K_UPPER, abs=1e-8)
    assert sealed.k_of("b") == 1.0


def test_k_endpoint_is_reachable_only_by_floating_saturation() -> None:
    # MAD = 0 with two outliers: scale = 1e-12, atan(1e21) saturates to
    # pi/2 exactly in double precision, so K lands on the closed endpoint
    # instead of raising.
    sealed = seal_population(
        [
            ("floor", _m(-10**9)),
            ("a", _m(0)),
            ("b", _m(0)),
            ("c", _m(0)),
            ("ceiling", _m(10**9)),
        ]
    )
    assert sealed.k_of("ceiling") == K_UPPER
    assert sealed.k_of("floor") == pytest.approx(K_LOWER, abs=1e-15)
    for k in sealed.K:
        assert K_LOWER <= k <= K_UPPER


def test_empty_population_raises_value_error() -> None:
    with pytest.raises(ValueError):
        seal_population([])
    with pytest.raises(ValueError):
        PopulationCostNormalization().seal(())


def test_stay_cost_factor_is_one_and_stay_is_not_an_observation() -> None:
    assert PopulationCostNormalization.k_for_stay() == 1.0
    sealed = seal_population(_three())
    assert sealed.k_for_stay() == 1.0
    assert STAY_PATCH_IDENTITY not in sealed.patch_ids
    with pytest.raises(KeyError):
        sealed.k_of(STAY_PATCH_IDENTITY)


def test_k_of_unknown_id_raises_and_duplicates_are_rejected() -> None:
    sealed = seal_population(_three())
    with pytest.raises(KeyError):
        sealed.k_of("missing")
    with pytest.raises(ValueError):
        seal_population([("dup", _m(1)), ("dup", _m(2))])


def test_sealed_population_is_immutable() -> None:
    sealed = seal_population(_three())
    with pytest.raises(AttributeError):
        sealed.records = ()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        sealed.records[0].K = 5.0  # type: ignore[misc]
    assert isinstance(sealed, SealedPopulation)
    assert isinstance(sealed.records, tuple)


def test_sha256_is_deterministic_and_order_sensitive_but_value_stable() -> None:
    a = seal_population(_three())
    b = seal_population(_three())
    assert a.population_sha256 == b.population_sha256
    assert len(a.population_sha256) == 64
    reordered = seal_population(list(reversed(_three())))
    assert reordered.population_sha256 != a.population_sha256
    # Same values, same K per id, different ordered receipt.
    for patch_id in ("cheap", "mid", "dear"):
        assert reordered.k_of(patch_id) == a.k_of(patch_id)
    # Fresh but equal marginal objects hash identically.
    rebuilt = seal_population(
        [("cheap", _m(1)), ("mid", _m(10)), ("dear", _m(100))]
    )
    assert rebuilt.population_sha256 == a.population_sha256
    # Changing one value changes the hash.
    changed = seal_population([("cheap", _m(1)), ("mid", _m(10)), ("dear", _m(101))])
    assert changed.population_sha256 != a.population_sha256


def test_receipt_hash_reproduces_from_canonical_json() -> None:
    sealed = seal_population(_three())
    receipt = sealed.to_json_dict()
    assert receipt["policy"] == POPULATION_COST_NORMALIZATION_POLICY_V1
    assert receipt["weights"] == {"delta_n2q": 0.30, "delta_d2q": 0.30, "delta_dc": 0.25}
    assert receipt["scale_floor"] == POPULATION_COST_SCALE_FLOOR
    assert receipt["size"] == 3
    assert [row["patch_id"] for row in receipt["records"]] == ["cheap", "mid", "dear"]
    assert set(receipt["records"][0]) == {"patch_id", "raw", "normalized", "cbar", "K"}
    payload = {k: v for k, v in receipt.items() if k != "population_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == sealed.population_sha256


def test_patch_identity_matches_order_key_shape_and_serializes() -> None:
    identity = patch_identity((2, 0), (("sel", "atom:a", "000003"),))
    assert identity == ((2, 0), (("sel", "atom:a", "000003"),))
    other = patch_identity((), (("tok", "x", "000001", "k"),))
    sealed = seal_population([(identity, _m(1)), (other, _m(3))])
    assert sealed.k_of(identity) < 1.0 < sealed.k_of(other)
    receipt = sealed.to_json_dict()
    assert receipt["records"][0]["patch_id"] == [[2, 0], [["sel", "atom:a", "000003"]]]
    assert STAY_PATCH_IDENTITY == patch_identity((), ())


# ---------------------------------------------------------------------------
# Shifted Paper-I score
# ---------------------------------------------------------------------------


def test_shifted_score_identities() -> None:
    # K = 1 -> S = m.
    for m in (-0.25, 0.0, 0.3, 2.0):
        assert shifted_score(m, 1.0, tau=0.5) == pytest.approx(m, abs=1e-15)
    # tau = 0 and m > 0 -> S = m / K.
    for k in (0.7, 1.0, 1.9):
        assert shifted_score(0.4, k, tau=0.0) == pytest.approx(0.4 / k, abs=1e-15)
    # m = -tau -> S = -tau for any K.
    for k in (0.7, 1.0, 1.9):
        assert shifted_score(-0.3, k, tau=0.3) == pytest.approx(-0.3, abs=1e-15)
    # Stay reference.
    assert shifted_score(0.0, 1.0, tau=0.0) == 0.0
    assert shifted_score(0.0, 1.0, tau=0.7) == 0.0


def test_shifted_score_rejects_merit_below_minus_tau_and_bad_inputs() -> None:
    with pytest.raises(ValueError):
        shifted_score(-0.31, 1.0, tau=0.3)
    with pytest.raises(ValueError):
        shifted_score(-1e-9, 1.0, tau=0.0)
    with pytest.raises(ValueError):
        shifted_score(0.1, 0.0, tau=0.0)
    with pytest.raises(ValueError):
        shifted_score(0.1, 1.0, tau=-0.1)
    with pytest.raises(ValueError):
        shifted_score(math.nan, 1.0)


def test_zero_merit_deletion_with_relief_beats_stay_in_maintenance() -> None:
    tau = 0.2
    score = shifted_score(0.0, 0.8, tau=tau)
    assert score == pytest.approx(tau * (1.0 / 0.8 - 1.0), abs=1e-15)
    assert score > stay_reference().score == 0.0
    # Same deletion with a burden K > 1 does not beat stay.
    assert shifted_score(0.0, 1.25, tau=tau) < 0.0
    # And with tau = 0 (LARGE_MISS) a zero-merit patch never beats stay.
    assert shifted_score(0.0, 0.8, tau=0.0) == 0.0


def test_stay_reference_is_zero_with_unit_cost() -> None:
    stay = stay_reference()
    assert stay.patch_id == STAY_PATCH_IDENTITY
    assert stay.merit == 0.0 and stay.K == 1.0 and stay.score == 0.0
    assert stay.eligible is True


# ---------------------------------------------------------------------------
# score_and_rank
# ---------------------------------------------------------------------------


def _rec(pid: str, m: float, k: float, raw: float, key: tuple) -> ScoreRecord:
    return ScoreRecord(patch_id=pid, merit=m, K=k, raw_merit_tiebreak=raw, canonical_key=key)


def test_rank_is_deterministic_and_applies_tiebreaks_in_order() -> None:
    records = [
        # score 1.0 / raw 0.5 / K 1.0 / key ((1,),())  -- winner of the tie chain
        _rec("p_key_low", 1.0, 1.0, 0.5, ((1,), ())),
        # score 1.0 / raw 0.5 / K 1.0 / key ((2,),())  -- loses on canonical key
        _rec("p_key_high", 1.0, 1.0, 0.5, ((2,), ())),
        # score 1.0 / raw 0.5 / K 1.5 (m = 1.5)        -- loses on K
        _rec("p_k_high", 1.5, 1.5, 0.5, ((0,), ())),
        # score 1.0 / raw 0.9                           -- wins on raw merit
        _rec("p_raw_high", 1.0, 1.0, 0.9, ((9,), ())),
        # score 2.0                                     -- wins outright
        _rec("p_top", 2.0, 1.0, 0.0, ((5,), ())),
        # score 0.5                                     -- last
        _rec("p_low", 0.5, 1.0, 9.0, ((3,), ())),
    ]
    expected = ["p_top", "p_raw_high", "p_key_low", "p_key_high", "p_k_high", "p_low"]
    result = score_and_rank(records, RankRegime.LARGE_MISS)
    assert [p.patch_id for p in result.ranked] == expected
    assert result.excluded == ()
    assert result.tau == 0.0
    assert result.best is not None and result.best.patch_id == "p_top"
    # Input order must not matter.
    for permutation in (list(reversed(records)), records[2:] + records[:2]):
        again = score_and_rank(permutation, RankRegime.LARGE_MISS)
        assert [p.patch_id for p in again.ranked] == expected
        assert [p.rank_key for p in again.ranked] == [p.rank_key for p in result.ranked]
    # Plain 5-tuples are accepted as records.
    tuples = [(r.patch_id, r.merit, r.K, r.raw_merit_tiebreak, r.canonical_key) for r in records]
    assert [p.patch_id for p in score_and_rank(tuples, RankRegime.LARGE_MISS).ranked] == expected
    # Ranked scores are the shifted scores and every ranked patch is eligible.
    for patch in result.ranked:
        assert patch.eligible is True
        assert patch.score == pytest.approx(patch.merit / patch.K, abs=1e-15)
        assert patch.rank_key[0] == -patch.score


def test_large_miss_excludes_nonpositive_merit_and_ignores_tau_del() -> None:
    records = [
        _rec("pos", 0.2, 0.9, 0.2, ((1,), ())),
        _rec("zero", 0.0, 0.8, 0.0, ((2,), ())),
        _rec("neg", -0.05, 0.8, -0.05, ((3,), ())),
    ]
    result = score_and_rank(records, RankRegime.LARGE_MISS, tau_del=0.5)
    assert result.tau == 0.0
    assert [p.patch_id for p in result.ranked] == ["pos"]
    assert result.ranked[0].score == pytest.approx(0.2 / 0.9, abs=1e-15)
    assert {(e.patch_id, e.reason) for e in result.excluded} == {
        ("zero", EXCLUDED_MERIT_NOT_POSITIVE),
        ("neg", EXCLUDED_MERIT_NOT_POSITIVE),
    }
    assert all(e.score is None for e in result.excluded)


def test_maintenance_excludes_below_minus_tau_and_not_above_stay() -> None:
    tau = 0.2
    records = [
        _rec("gain", 0.1, 1.0, 0.1, ((1,), ())),           # S = 0.1 > 0
        _rec("relief_del", 0.0, 0.8, 0.0, ((2,), ())),     # S = 0.2*(1.25-1) = 0.05 > 0
        _rec("neutral_del", 0.0, 1.0, 0.0, ((3,), ())),    # S = 0 -> not above stay
        _rec("burden_del", 0.0, 1.2, 0.0, ((4,), ())),     # S < 0 -> not above stay
        _rec("at_edge", -0.2, 0.7, -0.2, ((5,), ())),      # m = -tau -> S = -tau
        _rec("below", -0.2000001, 0.7, -0.2, ((6,), ())),  # m < -tau -> excluded
    ]
    result = score_and_rank(records, RankRegime.MAINTENANCE, tau_del=tau)
    assert result.tau == tau
    assert [p.patch_id for p in result.ranked] == ["gain", "relief_del"]
    assert result.ranked[1].score == pytest.approx(tau * (1.0 / 0.8 - 1.0), abs=1e-15)
    reasons = {e.patch_id: e.reason for e in result.excluded}
    assert reasons == {
        "neutral_del": EXCLUDED_SCORE_NOT_ABOVE_STAY,
        "burden_del": EXCLUDED_SCORE_NOT_ABOVE_STAY,
        "at_edge": EXCLUDED_SCORE_NOT_ABOVE_STAY,
        "below": EXCLUDED_MERIT_BELOW_MINUS_TAU,
    }
    scores = {e.patch_id: e.score for e in result.excluded}
    assert scores["neutral_del"] == 0.0
    assert scores["burden_del"] < 0.0
    assert scores["at_edge"] == pytest.approx(-tau, abs=1e-15)
    assert scores["below"] is None


def test_score_and_rank_rejects_duplicates_and_bad_tau() -> None:
    records = [_rec("a", 0.1, 1.0, 0.1, ((1,), ())), _rec("a", 0.2, 1.0, 0.2, ((2,), ()))]
    with pytest.raises(ValueError):
        score_and_rank(records, RankRegime.LARGE_MISS)
    records = [_rec("a", 0.1, 1.0, 0.1, ((1,), ())), _rec("b", 0.2, 1.0, 0.2, ((1,), ()))]
    with pytest.raises(ValueError):
        score_and_rank(records, RankRegime.LARGE_MISS)
    with pytest.raises(ValueError):
        score_and_rank([_rec("a", 0.1, 1.0, 0.1, ((1,), ()))], RankRegime.MAINTENANCE, tau_del=-0.1)
    with pytest.raises(ValueError):
        ScoreRecord("a", 0.1, 0.0, 0.1, ((1,), ()))
    assert score_and_rank([], RankRegime.MAINTENANCE, tau_del=0.1).ranked == ()


def test_sealed_k_feeds_ranking_end_to_end() -> None:
    sealed = seal_population(_three())
    records = [
        (pid, 0.1, sealed.k_of(pid), 0.1, patch_identity((i,), ()))
        for i, pid in enumerate(("cheap", "mid", "dear"))
    ]
    result = score_and_rank(records, RankRegime.LARGE_MISS)
    # Equal merit: cheaper compile wins through smaller K.
    assert [p.patch_id for p in result.ranked] == ["cheap", "mid", "dear"]


# ---------------------------------------------------------------------------
# Legacy repair: family_robust_v1 over the actual population
# ---------------------------------------------------------------------------


def _components(two_q: float, depth: float) -> dict[str, float]:
    return {"2q": two_q, "d": depth, "1q": 3.0, "theta": 1.0, "shot": 0.0}


def test_legacy_singleton_reproduces_unit_denominator() -> None:
    out = legacy_family_robust_denominators([("only", _components(2.0, 2.0))])
    assert len(out) == 1
    assert out[0].patch_id == "only"
    assert out[0].hardware_cost_excess_sum == 0.0
    assert out[0].hardware_cost_denominator == 1.0
    assert out[0].utility_denominator == 1.0
    assert all(v == 0.0 for v in out[0].bar_components.values())


def test_legacy_population_penalizes_only_the_expensive_record() -> None:
    records = [
        ("cheap", _components(1.0, 1.0)),
        ("mid", _components(10.0, 10.0)),
        ("dear", _components(100.0, 100.0)),
    ]
    settings = AppendCostSettings()
    out = legacy_family_robust_denominators(records, settings=settings)
    by_id = {row.patch_id: row for row in out}
    assert by_id["cheap"].hardware_cost_denominator == 1.0
    assert by_id["mid"].hardware_cost_denominator == 1.0
    assert by_id["dear"].hardware_cost_denominator > 1.0
    # Median 10, excess scale median([90]) = 90: bar = asinh(90/90) = asinh(1).
    expected_bar = math.asinh(1.0)
    assert by_id["dear"].bar_components["2q"] == pytest.approx(expected_bar, abs=1e-15)
    assert by_id["dear"].bar_components["d"] == pytest.approx(expected_bar, abs=1e-15)
    expected_denominator = 1.0 + (settings.lambda_2q + settings.lambda_d) * expected_bar
    assert by_id["dear"].hardware_cost_denominator == pytest.approx(
        expected_denominator, abs=1e-15
    )
    # Exactly what append_cost_telemetry_for_family gives for the same family.
    estimates = tuple(
        AppendCostRawEstimate(
            raw_components=dict(components),
            component_sources={k: "x" for k in components},
            inserted_runtime_count=1,
            atom_ids=(pid,),
        )
        for pid, components in records
    )
    reference = append_cost_telemetry_for_family(
        estimates, insertion_gains=(None, None, None), settings=settings
    )
    for row, ref in zip(out, reference):
        assert row.hardware_cost_denominator == ref.hardware_cost_denominator
        assert row.utility_denominator == ref.utility_denominator
        assert dict(row.bar_components) == dict(ref.bar_components)
        assert row.telemetry.hardware_cost_excess_sum == ref.hardware_cost_excess_sum


def test_legacy_repair_rejects_empty_raw_mode_and_unknown_components() -> None:
    with pytest.raises(ValueError):
        legacy_family_robust_denominators([])
    with pytest.raises(ValueError):
        legacy_family_robust_denominators(
            [("a", _components(1.0, 1.0))],
            settings=AppendCostSettings(
                cost_normalization_mode=AP_APPEND_COST_NORMALIZATION_RAW_LEGACY_V1
            ),
        )
    with pytest.raises(ValueError):
        legacy_family_robust_denominators([("a", {"2q": 1.0, "bogus": 2.0})])
