from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _duplicate_child_metadata(*, owner_sha256: str) -> dict[str, object]:
    label = "guarded_singleton::eeyeeezeee"
    parent_id = "parent:d0df81502d83973c"
    physical_generator_id = "child:82a7c379dca861bd"
    return {
        "generator_id": physical_generator_id,
        "compile_cache_identity": {
            "generator_id": f"{physical_generator_id}::pool[30]",
            "pool_index": 30,
        },
        "parent_generator_id": parent_id,
        "ra_retained_parent_owner": {
            "schema": "ra_retained_parent_owner_v1",
            "child_label": label,
            "parent_label": "paop_full:paop_disp(site=1)",
            "parent_generator_identity": parent_id,
            "sha256": owner_sha256,
        },
        "compile_metadata": {
            "runtime_terms_exyz": [
                {
                    "pauli_exyz": "x",
                    "coefficient_real": 1.0,
                    "coefficient_imaginary": 0.0,
                }
            ],
            "symmetry_gate": {"checked": True, "passed": True},
        },
    }


def test_active_prefix_serializes_duplicate_label_owner_by_occurrence(
    monkeypatch,
) -> None:
    """A later duplicate must not overwrite an earlier owner's signature."""

    label = "guarded_singleton::eeyeeezeee"
    operator = AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(1, ps="x", pc=1.0)],
        ),
        execution_mode="termwise_product",
    )
    earlier = _duplicate_child_metadata(owner_sha256="1" * 64)
    later = _duplicate_child_metadata(owner_sha256="2" * 64)
    layout = SimpleNamespace(
        logical_parameter_count=0,
        runtime_parameter_count=0,
        blocks=(),
    )
    transition_services = SimpleNamespace(
        build_selected_layout=lambda _operators: layout,
        canonicalize_runtime_theta_for_selected_layout=(
            lambda theta, _layout: np.asarray(theta, dtype=float)
        ),
        controller_noise_runtime=None,
    )
    state_service = SimpleNamespace(
        prepare=lambda **_kwargs: np.asarray([1.0 + 0.0j]),
        fixed_count_auditor=SimpleNamespace(
            assert_valid=lambda *_args, **_kwargs: {"passed": True}
        ),
    )
    context = SimpleNamespace(
        transition_services=transition_services,
        state_service=state_service,
        problem_key="fixture",
        parameterization_mode="logical_shared",
        parameterization_mode_source="fixture",
        optimizer_name="POWELL",
        optimizer_powell_coordinate_chart_policy="fixture",
        route_profile="fixture",
        route_contract_sha256="a" * 64,
        phase1_score_mode="fixture",
        phase2_curvature_policy="fixture",
        phase_shortlist_runtime=SimpleNamespace(
            phase2_score_cfg=SimpleNamespace(
                phase2_cheap_curvature_proxy_policy="fixture"
            )
        ),
        generator_sector_contract={"schema": "fixture"},
    )
    cursor = SimpleNamespace(
        controller_round=46,
        selected_ops=[operator, operator],
        selected_operator_metadata=[earlier, later],
        theta=np.zeros(0, dtype=float),
        selected_executor=None,
        pool_generator_registry={label: later},
        estimator_call_ledger=object(),
        estimator_prefix_checkpoint_cursor={},
        estimator_active_prefix_ledger_receipts_all=[],
    )
    candidate_sector_auditor = SimpleNamespace(
        active_contract=lambda *_args, **_kwargs: {"passed": True}
    )

    monkeypatch.setattr(
        adapt_pipeline,
        "_logical_theta_alias",
        lambda theta, _layout: np.asarray(theta, dtype=float),
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_default_no_prune_strict_replay",
        lambda **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_default_no_prune_fixed_spin_sector_probability",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "projective_state_fingerprint",
        lambda _state: "state-fingerprint",
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "estimator_projective_state_fingerprint",
        lambda _state: "estimator-state-fingerprint",
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_default_no_prune_active_prefix_estimator_ledger_receipt",
        lambda **_kwargs: {"checkpoint_sequence": 1},
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_optimizer_coordinate_chart_payload",
        lambda **_kwargs: {"schema": "fixture"},
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "serialize_layout",
        lambda _layout: {"schema": "fixture"},
    )

    checkpoint = adapt_pipeline._default_no_prune_active_prefix_checkpoint(
        context=context,
        cursor=cursor,
        candidate_sector_auditor=candidate_sector_auditor,
        event=None,
        history_row={
            "phase1_energy_model": "fixture",
            "selected_ops": [label],
            "selected_positions": [1],
            "selected_effective_positions": [1],
            "post_admission_prune": {"deleted_indices": []},
        },
    )

    owners = [
        row["ra_retained_parent_owner"]["sha256"]
        for row in checkpoint["ordered_active_operators"]
    ]
    assert owners == ["1" * 64, "2" * 64]
    physical_ids = [
        row["generator_id"]
        for row in checkpoint["ordered_active_operators"]
    ]
    assert physical_ids == [
        "child:82a7c379dca861bd",
        "child:82a7c379dca861bd",
    ]
    assert all("::pool[" not in generator_id for generator_id in physical_ids)
