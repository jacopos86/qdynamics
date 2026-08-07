from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines.exact_bench import generic_static_adapt_variants as variants
from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS,
    TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
    TABLE_I_STATIC_SUITE_PROFILE_ENV,
    paper_i_hh_completion_case_id,
    table_i_canonical_specs,
    table_i_suite_profile,
)
from pipelines.static_adapt.builders.shared_pauli_pool_contract import (
    SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
    SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
    SharedPauliPoolParent,
    build_shared_pauli_child_pool,
)
from pipelines.static_adapt.builders import shared_pauli_pool_contract as pool_contract
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _candidate(label: str, terms: list[tuple[str, float]]) -> variants._PoolCandidate:
    polynomial = PauliPolynomial(
        "JW",
        [PauliTerm(8, ps=pauli, pc=coefficient) for pauli, coefficient in terms],
    )
    return variants._PoolCandidate(
        label=label,
        polynomial=polynomial,
        support=tuple(
            sorted(
                {
                    qubit
                    for pauli, _coefficient in terms
                    for qubit in range(8)
                    if pauli[7 - qubit] != "e"
                }
            )
        ),
        pauli_labels_exyz=tuple(pauli for pauli, _coefficient in terms),
        construction="full_meta::full_meta",
    )


def _hh_nph2_context() -> SimpleNamespace:
    return SimpleNamespace(
        family_key="hh",
        request=SimpleNamespace(
            problem_key="hh",
            num_sites=2,
            ordering="blocked",
            n_ph_max=2,
            boson_encoding="binary",
        ),
        layout=SimpleNamespace(total_qubits=8, fermion_qubits=4),
        sector=SimpleNamespace(num_particles=(1, 1)),
    )


def _base_args(spec) -> dict[str, str]:  # noqa: ANN001
    args = tuple(str(value) for value in spec.base_pipeline_args)
    assert len(args) % 2 == 0
    return {args[index]: args[index + 1] for index in range(0, len(args), 2)}


def test_completion_same_cutoff_profile_has_exact_six_case_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (
        ("weak-weak", "0.25", "0.353553390593", 3, -0.918380919994822, "a10820b35d82ea3bd29599b5"),
        ("intermediate-weak", "1.25", "0.353553390593", 3, -0.4950053491813613, "8c5f49d0f545a12f898be7ba"),
        ("strong-weak", "8.0", "0.353553390593", 3, 0.5264586847939736, "2218571998ef766037aa4d0f"),
        ("weak-strong", "0.25", "0.790569415042", 7, -1.1387206380749124, "42872c0f1988ea8bdbd99b79"),
        ("intermediate-strong", "1.25", "0.790569415042", 7, -0.6239396137518493, "99397703afad40a7bd87403c"),
        ("strong-strong", "8.0", "0.790569415042", 7, 0.5205762765682517, "b941d7eae8f318acfc831c86"),
    )
    assert table_i_suite_profile("hh_completion_same_cutoff") == (
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE
    )
    specs = table_i_canonical_specs(
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE
    )
    assert tuple(str(spec.benchmark_id) for spec in specs) == tuple(
        TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS.values()
    )
    assert len(specs) == 6
    for spec, (
        regime,
        u_value,
        g_value,
        n_ph,
        expected_exact_energy,
        expected_reference_hash,
    ) in zip(specs, expected, strict=True):
        args = _base_args(spec)
        assert paper_i_hh_completion_case_id(regime) == str(spec.benchmark_id)
        assert paper_i_hh_completion_case_id(regime.replace("-", "_")) == str(
            spec.benchmark_id
        )
        assert args == {
            "--problem": "hh",
            "--L": "2",
            "--t": "1.0",
            "--u": u_value,
            "--dv": "0.0",
            "--omega0": "1.0",
            "--g-ep": g_value,
            "--n-ph-max": str(n_ph),
            "--boson-encoding": "binary",
            "--ordering": "blocked",
            "--boundary": "open",
            "--v-nn": "0.0",
            "--t-prime": "0.0",
        }
        assert int(spec.exact_reference_n_ph_max) == n_ph
        assert "same_cutoff_reference" in spec.tags
        assert f"regime_alias:{regime}" in spec.tags
        exact_energy, reference_hash, _reference_key = exact_energy_for_spec(
            spec, n_ph_max=n_ph
        )
        assert exact_energy == pytest.approx(
            expected_exact_energy, rel=0.0, abs=1.0e-12
        )
        assert reference_hash == expected_reference_hash

    monkeypatch.setenv(
        TABLE_I_STATIC_SUITE_PROFILE_ENV,
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
    )
    for algorithm_id in (
        "static_geo_adapt_vqe",
        "static_full_meta_append_adapt_vqe",
    ):
        assert variants.default_static_adapt_variant_case_ids(
            "hh", algorithm_id
        ) == tuple(TABLE_I_PAPER_I_HH_COMPLETION_REGIME_CASE_IDS.values())


def test_completion_regime_alias_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown Paper-I HH completion regime alias"):
        paper_i_hh_completion_case_id("weakish-strongish")


def test_completion_cutoff_and_unfiltered_cache_provenance_are_serialized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        TABLE_I_STATIC_SUITE_PROFILE_ENV,
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
    )
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "OFF")
    monkeypatch.setenv(
        "STATIC_ADAPT_HH_POOL_CACHE_SCOPE", "paper-i-holstein-sector"
    )
    specs = table_i_canonical_specs(
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE
    )
    spec = specs[0]
    context = variants._resolve_context_from_spec(spec)
    cutoff = variants._resolve_cutoff_provenance(
        spec=spec,
        context=context,
        exact_reference_n_ph_max=None,
    )
    assert cutoff == {
        "resolved_exact_reference_n_ph_max": 3,
        "n_ph_work": 3,
        "n_ph_reference": 3,
        "exact_reference_n_ph_max": 3,
        "same_cutoff_reference": True,
        "cutoff_pair": {
            "n_ph_work": 3,
            "n_ph_ref": 3,
            "reference_role": "same_cutoff_exact_reference",
        },
    }

    payload = variants.run_generic_static_adapt_variant_single(
        family="hh",
        case_id=str(spec.benchmark_id),
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "completion_provenance",
        max_adapt_iterations=0,
        optimizer_maxiter=1,
        hh_adaptive_pool_profile="full_meta_unfiltered",
        exact_reference_n_ph_max=3,
    )
    assert payload["status"] == "completed"
    row = payload["result"]
    for receipt in (payload, row):
        assert receipt["n_ph_work"] == 3
        assert receipt["n_ph_reference"] == 3
        assert receipt["exact_reference_n_ph_max"] == 3
        assert receipt["same_cutoff_reference"] is True
        assert receipt["cutoff_pair"] == cutoff["cutoff_pair"]
    assert payload["comparator_source"]["cutoff_pair"] == cutoff["cutoff_pair"]
    assert row["hh_pool_cache_mode_requested"] == "OFF"
    assert row["hh_pool_cache_mode_effective"] == "off"
    assert row["hh_pool_cache_scope_requested_raw"] == (
        "paper-i-holstein-sector"
    )
    assert row["hh_pool_cache_scope_requested"] == (
        "paper_i_holstein_sector"
    )
    # The cache implementation supports the structural sector scope only for
    # its historical n_ph=2/4 contract, so this n_ph=3 row correctly resolves
    # to exact-key scope without changing cache behavior.
    assert row["hh_pool_cache_scope_effective"] == "exact"
    assert row["hh_pool_cache_policy"]["schema"] == (
        "generic_static_hh_pool_cache_policy_receipt_v1"
    )
    assert row["hh_pool_cache_events"] == []
    assert row["hh_pool_cache_event_names"] == []


def test_generic_comparator_seed_zero_is_valid_end_to_end() -> None:
    _config, settings = variants._effective_optimizer_settings_for_config(
        variants._get_config("static_geo_adapt_vqe"),
        optimizer_maxiter=1,
        seed=0,
    )
    assert settings["spsa_seed_base"] == 0


def test_completion_profile_real_projected_singleton_pool_hashes() -> None:
    expected = {
        "hh_L2_nph3_completion_weak_weak": (123, 1631, 1622, 9, "5bfc80b4c205dcf3f3b22a876ef9ada4ca903d2e62fd7f0cb4854c35ab29df6b", "679f0fd6309f2a7e6732f557a8c4907a25856be59512fc91795d93555a975315", "3bb76ec68fb2bbde7a46428c9bce3bd343dd337c4087a0f827aabdf5e54d837e"),
        "hh_L2_nph3_completion_intermediate_weak": (123, 1631, 1622, 9, "a7c040f58615125044309ad80b07e7e08b04593eb4ceba567bd2d306aec0026d", "679f0fd6309f2a7e6732f557a8c4907a25856be59512fc91795d93555a975315", "bec3c8a523f251f34e00c8d671e7f658d8e7c4f1acbb0518515fb8467885a976"),
        "hh_L2_nph3_completion_strong_weak": (123, 1631, 1622, 9, "9a11fb2a11b8b7c92780602cfeccc5d7b70890cee6b9fb8bad9c652314103f0e", "679f0fd6309f2a7e6732f557a8c4907a25856be59512fc91795d93555a975315", "e1dbe2831c5ceeabc4d064623614cc660617232441d90eb2a13c7e3d8cec8638"),
        "hh_L2_nph7_completion_weak_strong": (171, 8757, 8748, 9, "7e466a02ad38e425f89cb3423819f93dc6bbd912cf35cdd0817b2f6a6563fdcf", "8d2b1a15366f1c87dbb84fa440b7ac435ceaa131eb68debcbe94df9b1fb8a865", "67d70e6aec1c4e2933194c3a311ecf13b70a58425c8392bc78fd034de2c93e0c"),
        "hh_L2_nph7_completion_intermediate_strong": (171, 8757, 8748, 9, "60728fc55cadd3a652b048f8073fcb913a9d5cb3c46393f271ca9a1bc74c58fb", "8d2b1a15366f1c87dbb84fa440b7ac435ceaa131eb68debcbe94df9b1fb8a865", "d7dd2368af0bc6a31f73b780f36676068d32c429f79eb070ad704c347d775b43"),
        "hh_L2_nph7_completion_strong_strong": (171, 8757, 8748, 9, "c1d7d953864bca3ab13749fcb1d70f980b4ba2246278e58d62108f9622ed878d", "8d2b1a15366f1c87dbb84fa440b7ac435ceaa131eb68debcbe94df9b1fb8a865", "ba4d96953d34d4e079ba7bebe3aadd111e567fdecf83f2b4749c9cff525ee7c7"),
    }
    specs = table_i_canonical_specs(
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE
    )
    for index, spec in enumerate(specs):
        context = variants._resolve_context_from_spec(spec)
        parent_pool = variants.build_full_meta_candidate_pool(context, max_terms=None)
        algorithm_id = (
            "static_geo_adapt_vqe"
            if index % 2 == 0
            else "static_full_meta_append_adapt_vqe"
        )
        expanded, meta = variants._expand_pool_with_shared_pauli_children(
            pool=parent_pool,
            context=context,
            config=variants._get_config(algorithm_id),
            mode=SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
            symmetry_policy="hard_guard",
            max_subset_size=1,
            max_terms=9000,
        )
        (
            parent_count,
            raw_term_count,
            candidate_count,
            identity_null_count,
            source_parent_hash,
            label_hash,
            pool_hash,
        ) = expected[str(spec.benchmark_id)]
        assert len(parent_pool) == parent_count
        assert len(expanded) == candidate_count
        assert meta["projected_singleton_source_term_count"] == raw_term_count
        assert meta["projected_singleton_null_identity_count"] == identity_null_count
        assert meta["projected_singleton_null_count"] == identity_null_count
        assert len(meta["projected_singleton_null_exclusions"]) == identity_null_count
        assert meta["projected_singleton_projection_zero_exclusions"] == []
        assert meta["source_parent_ordered_pool_hash"] == source_parent_hash
        assert meta["ordered_label_hash"] == label_hash
        assert meta["ordered_pool_hash"] == pool_hash


def test_exact_projected_identity_child_is_excluded_with_lineage() -> None:
    parent = _candidate(
        "full_meta::identity_and_z",
        [("eeeeeeee", 0.5), ("eeeeeeez", 0.5)],
    )
    expanded, meta = variants._expand_pool_with_shared_pauli_children(
        pool=(parent,),
        context=_hh_nph2_context(),
        config=variants._get_config("static_geo_adapt_vqe"),
        mode=SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
        symmetry_policy="hard_guard",
        max_subset_size=1,
        max_terms=16,
    )
    assert len(expanded) == 1
    assert expanded[0].pauli_labels_exyz == ("eeeeeeez",)
    assert meta["projected_singleton_null_identity_count"] == 1
    assert meta["projected_singleton_null_count"] == 1
    assert meta["projected_singleton_null_exclusions"][0]["reason"] == (
        "exact_projection_is_identity_global_phase_direction"
    )


def test_exact_projection_zero_exclusion_preserves_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parents = (
        _candidate("full_meta::projection_zero", [("zeeeeeee", 0.5)]),
        _candidate("full_meta::retained", [("eeeeeeez", 0.5)]),
    )
    original = pool_contract._project_singleton_children
    call_count = 0

    def _project_with_one_zero(**kwargs):  # noqa: ANN003, ANN202
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            child = kwargs["children"][0]
            return [], {
                "projection_input_count": 1,
                "projection_zero_rejection_count": 1,
                "deduplicated_candidate_count": 0,
                "zero_rejections": [
                    {
                        "status": "rejected",
                        "reason": "projection_is_zero",
                        "parent_label": "full_meta::projection_zero",
                        "raw_candidate_label": str(child["child_label"]),
                        "raw_child_indices": [int(child["child_index"])],
                        "raw_child_labels": [str(child["child_label"])],
                        "projection": {"reason": "projection_is_zero"},
                    }
                ],
            }
        return original(**kwargs)

    monkeypatch.setattr(
        pool_contract,
        "_project_singleton_children",
        _project_with_one_zero,
    )
    result = build_shared_pauli_child_pool(
        parents=[
            SharedPauliPoolParent(
                label=parent.label,
                polynomial=parent.polynomial,
                family_id="hh",
                stage_family="full_meta",
                construction="full_meta",
            )
            for parent in parents
        ],
        mode=SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
        symmetry_policy="hard_guard",
        max_subset_size=1,
        problem_key="hh",
        num_sites=2,
        ordering="blocked",
        qpb=2,
        n_ph_max=2,
        boson_encoding="binary",
        total_register_width=8,
        fixed_num_particles=(1, 1),
        max_terms=16,
    )
    assert len(result.candidates) == 1
    assert result.meta["projected_singleton_projection_zero_count"] == 1
    assert result.meta["projected_singleton_null_count"] == 1
    exclusion = result.meta["projected_singleton_projection_zero_exclusions"][0]
    assert exclusion["null_kind"] == "exact_projection_zero"
    assert exclusion["parent_label"] == "full_meta::projection_zero"
    assert exclusion["raw_child_indices"] == [0]


@pytest.mark.parametrize(
    "algorithm_id",
    ["static_geo_adapt_vqe", "static_full_meta_append_adapt_vqe"],
)
def test_projected_singleton_pool_is_children_only_for_geo_and_append(
    algorithm_id: str,
) -> None:
    parents = (
        _candidate(
            "full_meta::boson_pair",
            [
                ("zeeeeeee", 0.25),
                # X on an n_ph=2 binary register must be exactly projected.
                # That one raw child becomes one grouped logical direction.
                ("xeeeeeee", 0.5),
            ],
        ),
        # An already-singleton full_meta parent must be reclassified as a child,
        # not leaked into the pool with representation='parent'.
        _candidate("full_meta::fermion_z", [("eeeeeeez", 0.75)]),
    )
    expanded, meta = variants._expand_pool_with_shared_pauli_children(
        pool=parents,
        context=_hh_nph2_context(),
        config=variants._get_config(algorithm_id),
        mode=SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
        symmetry_policy="hard_guard",
        max_subset_size=1,
        max_terms=16,
    )

    assert len(expanded) == 3
    assert all(
        candidate.runtime_split_representation == "projected_singleton_child"
        for candidate in expanded
    )
    assert all(candidate.parent_label is not None for candidate in expanded)
    assert all(len(candidate.runtime_split_child_indices) == 1 for candidate in expanded)
    assert all(candidate.runtime_split_symmetry_gate["checked"] is True for candidate in expanded)
    assert all(candidate.runtime_split_symmetry_gate["passed"] is True for candidate in expanded)
    assert all(candidate.execution_mode == "grouped_exact" for candidate in expanded)
    assert [candidate.pauli_labels_exyz for candidate in expanded] == [
        ("xzeeeeee", "xeeeeeee"),
        ("zeeeeeee",),
        ("eeeeeeez",),
    ]
    assert meta["mode"] == SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
    assert meta["base_pool_term_count"] == 2
    assert meta["expanded_pool_term_count"] == 3
    assert meta["projected_singleton_source_term_count"] == 3
    assert meta["projected_singleton_candidate_count"] == 3
    assert meta["projected_singleton_projection_input_count"] == 3
    assert meta["projected_singleton_projection_zero_count"] == 0
    assert meta["projected_singleton_projection_zero_exclusions"] == []
    assert meta["projected_singleton_projection_deduplicated_count"] == 0
    assert meta["projected_singleton_grouped_term_count"] == 4
    assert meta["candidate_representation_counts"] == {
        "parent": 0,
        "child_set": 0,
        "projected_singleton_child": 3,
    }
    assert len(meta["source_parent_ordered_pool_hash"]) == 64
    assert len(meta["source_parent_ordered_label_hash"]) == 64
    assert len(meta["ordered_pool_hash"]) == 64
    assert len(meta["ordered_label_hash"]) == 64
    assert meta["contract_identity"]["candidate_representation_policy"] == (
        "projected_singleton_child_only_v1"
    )
    assert meta["contract_identity"]["padding_policy"] == (
        "exact_projected_grouped_v1"
    )
    assert meta["source_parent_ordered_pool_hash"] == (
        "4f045919cafc73f1c51e207215f2dda95963df31d55e2c8411d93ffe7973bd8f"
    )
    assert meta["ordered_pool_hash"] == (
        "c2695df13f9314c796f75e591911fe840ae3e1910309181a32d340fdb4538608"
    )


def test_projected_singleton_pool_hash_is_identical_across_geo_and_append() -> None:
    parents = (
        _candidate("full_meta::z_pair", [("zeeeeeee", 0.25), ("ezeeeeee", 0.5)]),
        _candidate("full_meta::fermion_z", [("eeeeeeez", 0.75)]),
    )
    outputs = []
    for algorithm_id in ("static_geo_adapt_vqe", "static_full_meta_append_adapt_vqe"):
        expanded, meta = variants._expand_pool_with_shared_pauli_children(
            pool=parents,
            context=_hh_nph2_context(),
            config=variants._get_config(algorithm_id),
            mode=SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
            symmetry_policy="hard_guard",
            max_subset_size=1,
            max_terms=16,
        )
        outputs.append(
            (
                [candidate.label for candidate in expanded],
                meta["source_parent_ordered_pool_hash"],
                meta["ordered_pool_hash"],
                meta["ordered_candidate_count"],
            )
        )
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("symmetry_policy", "max_subset_size", "message"),
    [
        ("off", 1, "requires symmetry_policy=hard_guard"),
        ("hard_guard", 2, "requires exact subset size 1"),
    ],
)
def test_projected_singleton_pool_fails_closed_on_policy_mixing(
    symmetry_policy: str,
    max_subset_size: int,
    message: str,
) -> None:
    parent = _candidate("full_meta::z_pair", [("zeeeeeee", 0.25), ("ezeeeeee", 0.5)])
    with pytest.raises(ValueError, match=message):
        variants._expand_pool_with_shared_pauli_children(
            pool=(parent,),
            context=_hh_nph2_context(),
            config=variants._get_config("static_geo_adapt_vqe"),
            mode=SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
            symmetry_policy=symmetry_policy,
            max_subset_size=max_subset_size,
            max_terms=16,
        )


def test_historical_parent_plus_children_manifest_hash_is_unchanged() -> None:
    parent = SharedPauliPoolParent(
        label="full_meta::pair_hop",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="zeee", pc=0.25),
                PauliTerm(4, ps="eexy", pc=0.5),
                PauliTerm(4, ps="eeyx", pc=-0.5),
            ],
        ),
        family_id="hh",
        stage_family="full_meta",
        construction="full_meta",
    )
    result = build_shared_pauli_child_pool(
        parents=[parent],
        mode=SHARED_PAULI_POOL_MODE_CHILD_SETS_V1,
        symmetry_policy="hard_guard",
        max_subset_size=3,
        problem_key="hh",
        num_sites=2,
        ordering="blocked",
        qpb=1,
        max_terms=8,
    )
    assert [candidate.representation for candidate in result.candidates] == [
        "parent",
        "child_set",
        "child_set",
    ]
    assert result.manifest["ordered_pool_hash"] == (
        "5e3cd591181c7d7d8ac887c001dc042ddfb3883d0e209c74b871b912c142f9dc"
    )


@pytest.mark.parametrize(
    "algorithm_id",
    ["static_geo_adapt_vqe", "static_full_meta_append_adapt_vqe"],
)
@pytest.mark.parametrize("seed", [0, 7])
def test_generic_benchmark_threads_projected_singleton_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    algorithm_id: str,
    seed: int,
) -> None:
    from pipelines.exact_bench import generic_static_benchmark as benchmark

    captured: dict[str, object] = {}

    def _fake_runner(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        return {
            "schema": "generic_static_adapt_variants_v4",
            "status": "completed",
            "rows": [{"status": "ok"}],
        }

    monkeypatch.setattr(variants, "run_generic_static_adapt_variant_single", _fake_runner)
    monkeypatch.setattr(
        benchmark,
        "evaluate_algorithm_for_family",
        lambda algorithm_id, family: SimpleNamespace(
            status="runnable", resolved_pool_key="full_meta"
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "_dispatch_kind",
        lambda **kwargs: "generic_static_adapt_variants",
    )
    monkeypatch.setenv(
        "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE",
        SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1,
    )
    monkeypatch.setenv(
        "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY", "hard_guard"
    )
    monkeypatch.setenv("GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE", "1")
    monkeypatch.setenv("GENERIC_STATIC_TABLE_ADAPT_SEED", str(seed))

    payload = benchmark.run_single(
        family="hh",
        case_id="hh_L2_nph2_three_model_sym_weak_weak",
        algorithm_id=algorithm_id,
        output_dir=tmp_path / algorithm_id,
    )

    assert payload["status"] == "completed"
    assert captured["shared_pauli_pool_mode"] == (
        SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
    )
    assert captured["shared_pauli_pool_symmetry_policy"] == "hard_guard"
    assert captured["shared_pauli_pool_max_subset_size"] == 1
    assert captured["seed"] == seed
    assert payload["shared_pauli_pool_env_overlay"] == {
        "shared_pauli_pool_mode": (
            SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
        ),
        "shared_pauli_pool_symmetry_policy": "hard_guard",
        "shared_pauli_pool_max_subset_size": 1,
    }


def test_comparator_cell_threads_projected_singleton_mode_and_seed7(
    tmp_path: Path,
) -> None:
    from chtc.phase3_optuna import (
        run_paper_i_hh_spsa_budget_ladder_cell as cell_runner,
    )

    row = {
        "method_key": "geo",
        "suite_profile": TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
        "adapt_optimizer_kind": "powell",
        "max_depth": "2",
        "budget": "200",
        "same_cutoff_exact_gs_energy": "-0.918380919994822",
        "exact_reference_energy": "-0.918380919994822",
        "exact_reference_n_ph_max": "3",
        "shared_pauli_pool_mode": (
            SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
        ),
        "shared_pauli_pool_symmetry_policy": "hard_guard",
        "shared_pauli_pool_max_subset_size": "1",
        "pool_contract": "full_meta_unfiltered",
        "adapt_pool_class_filter_json": "off",
        "seed": "7",
    }

    env = cell_runner.append_geo_env(row, tmp_path / "out")

    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE"] == (
        SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
    )
    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY"] == (
        "hard_guard"
    )
    assert env["GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE"] == "1"
    assert env["GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE"] == (
        "full_meta_unfiltered"
    )
    assert env["GENERIC_STATIC_TABLE_ADAPT_SEED"] == "7"
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE" not in env

    conflicting = dict(row)
    conflicting["adapt_seed"] = "8"
    with pytest.raises(ValueError, match="seed and adapt_seed disagree"):
        cell_runner.append_geo_env(conflicting, tmp_path / "conflicting")


def test_projected_singleton_route_requires_unfiltered_full_meta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        TABLE_I_STATIC_SUITE_PROFILE_ENV,
        TABLE_I_PAPER_I_HH_COMPLETION_SAME_CUTOFF_PROFILE,
    )
    payload = variants.run_generic_static_adapt_variant_single(
        family="hh",
        case_id="hh_L2_nph3_completion_weak_weak",
        algorithm_id="static_geo_adapt_vqe",
        output_dir=tmp_path / "must_fail",
        max_adapt_iterations=1,
        optimizer_maxiter=1,
        shared_pauli_pool_mode=(
            SHARED_PAULI_POOL_MODE_PROJECTED_SINGLETON_CHILDREN_ONLY_V1
        ),
        shared_pauli_pool_symmetry_policy="hard_guard",
        shared_pauli_pool_max_subset_size=1,
    )
    assert payload["status"] == "failed"
    assert "requires hh_adaptive_pool_profile=full_meta_unfiltered" in str(
        payload["reason"]
    )
