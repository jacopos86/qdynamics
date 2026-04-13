from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt import adapt_pipeline_legacy_20260322 as legacy_loader
from pipelines.static_adapt import adapt_pipeline as current_adapt
from pipelines.scaffold.hh_continuation_scoring import raw_f_metric_from_state
from src.quantum.compiled_polynomial import apply_compiled_polynomial, compile_polynomial_action
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.pauli_words import PauliTerm


def _term(label: str) -> object:
    return type(
        "_DummyLegacyAnsatzTerm",
        (),
        {
            "label": str(label),
            "polynomial": PauliPolynomial("JW", [PauliTerm(len(str(label)), ps=str(label), pc=1.0)]),
        },
    )()


def _helper_module_names() -> list[str]:
    return [name for name, _ in legacy_loader._LEGACY_HELPER_MODULES] + [legacy_loader.LEGACY_SCORER_PRIVATE_MODULE]


@pytest.fixture()
def restore_legacy_modules() -> None:
    module_names = _helper_module_names()
    saved = {name: sys.modules.get(name) for name in module_names}
    legacy_loader._load_legacy_module.cache_clear()
    yield
    legacy_loader._load_legacy_module.cache_clear()
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_bridge(mode: str):
    legacy_loader._preload_legacy_helpers(mode)
    return importlib.import_module("pipelines.hardcoded.hh_continuation_scoring")


def test_legacy_entrypoint_hidden_geometry_flag_parses(restore_legacy_modules: None) -> None:
    args = legacy_loader.parse_args(["--phase3-legacy-geometry-mode", "exact_reduced"])
    assert str(args.phase3_legacy_geometry_mode) == "exact_reduced"


def test_legacy_entrypoint_hidden_geometry_flag_defaults_to_proxy(restore_legacy_modules: None) -> None:
    args = legacy_loader.parse_args([])
    assert str(args.phase3_legacy_geometry_mode) == "proxy_reduced"


def test_extract_legacy_geometry_mode_none_strips_hidden_flag_from_sys_argv(
    restore_legacy_modules: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adapt_pipeline_legacy_20260322.py",
            "--L",
            "2",
            "--phase3-legacy-geometry-mode",
            "exact_reduced",
            "--problem",
            "hh",
        ],
    )
    mode, stripped = legacy_loader._extract_legacy_geometry_mode(None)
    assert mode == "exact_reduced"
    assert stripped == ["--L", "2", "--problem", "hh"]


def test_legacy_bridge_proxy_reduced_records_selector_debug(restore_legacy_modules: None) -> None:
    scoring = _load_bridge("proxy_reduced")
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = scoring.Phase1CompileCostOracle()
    meas = scoring.MeasurementCacheAudit()
    candidate_term = _term("x")
    metric_exact = raw_f_metric_from_state(
        psi_state=psi_ref,
        candidate_label="x",
        candidate_term=candidate_term,
        compiled_cache={},
        pauli_action_cache={},
    )
    base = scoring.build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.3,
        metric_proxy=float(metric_exact),
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=1, append_position=1, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=scoring.SimpleScoreConfig(),
    )
    h_compiled = compile_polynomial_action(_term("z").polynomial, pauli_action_cache={})
    feat = scoring.build_full_candidate_features(
        base_feature=base,
        psi_state=np.asarray(psi_ref, dtype=complex),
        candidate_term=candidate_term,
        window_terms=[_term("x")],
        window_labels=["x"],
        cfg=scoring.FullScoreConfig(shortlist_size=2),
        novelty_oracle=scoring.Phase2NoveltyOracle(),
        curvature_oracle=scoring.Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
        legacy_selected_ops=[_term("x")],
        legacy_theta=np.asarray([0.0], dtype=float),
        legacy_psi_ref=np.asarray(psi_ref, dtype=complex),
        legacy_h_compiled=h_compiled,
    )
    debug = dict((feat.compiled_position_cost_backend or {}).get("phase3_geometry_debug", {}))
    assert debug["phase3_legacy_geometry_mode"] == "proxy_reduced"
    assert float(debug["legacy_proxy_full_v2_score"]) == pytest.approx(float(feat.full_v2_score or 0.0))
    assert float(debug["selector_score"]) == pytest.approx(float(feat.full_v2_score or 0.0))


def test_legacy_bridge_exact_reduced_records_both_score_families(restore_legacy_modules: None) -> None:
    scoring = _load_bridge("exact_reduced")
    psi_ref = np.zeros(2, dtype=complex)
    psi_ref[0] = 1.0 + 0.0j
    oracle = scoring.Phase1CompileCostOracle()
    meas = scoring.MeasurementCacheAudit()
    selected_ops = [_term("x")]
    candidate_term = _term("x")
    metric_exact = raw_f_metric_from_state(
        psi_state=psi_ref,
        candidate_label="x",
        candidate_term=candidate_term,
        compiled_cache={},
        pauli_action_cache={},
    )
    base = scoring.build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=1,
        append_position=1,
        positions_considered=[1],
        gradient_signed=0.3,
        metric_proxy=float(metric_exact),
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=1, append_position=1, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=scoring.SimpleScoreConfig(),
    )
    h_compiled = compile_polynomial_action(_term("z").polynomial, pauli_action_cache={})
    hpsi = apply_compiled_polynomial(np.asarray(psi_ref, dtype=complex), h_compiled)
    assert np.isfinite(np.linalg.norm(hpsi))
    feat = scoring.build_full_candidate_features(
        base_feature=base,
        psi_state=np.asarray(psi_ref, dtype=complex),
        candidate_term=candidate_term,
        window_terms=list(selected_ops),
        window_labels=["x"],
        cfg=scoring.FullScoreConfig(shortlist_size=2),
        novelty_oracle=scoring.Phase2NoveltyOracle(),
        curvature_oracle=scoring.Phase2CurvatureOracle(),
        compiled_cache={},
        pauli_action_cache={},
        optimizer_memory=None,
        legacy_selected_ops=list(selected_ops),
        legacy_theta=np.asarray([0.0], dtype=float),
        legacy_psi_ref=np.asarray(psi_ref, dtype=complex),
        legacy_h_compiled=h_compiled,
    )
    debug = dict((feat.compiled_position_cost_backend or {}).get("phase3_geometry_debug", {}))
    assert debug["phase3_legacy_geometry_mode"] == "exact_reduced"
    assert "legacy_proxy_full_v2_score" in debug
    assert "exact_reduced_full_v2_score" in debug
    assert float(debug["selector_score"]) == pytest.approx(float(debug["exact_reduced_full_v2_score"]))
    assert float(feat.full_v2_score or 0.0) == pytest.approx(float(debug["exact_reduced_full_v2_score"]))


def test_current_shadow_legacy_bridge_loader_is_private(restore_legacy_modules: None) -> None:
    public_scoring = importlib.import_module("pipelines.hardcoded.hh_continuation_scoring")
    bridge = current_adapt._load_shadow_legacy_geometry_bridge("proxy_reduced")
    assert callable(getattr(bridge, "build_full_candidate_features", None))
    assert str(bridge.__name__).startswith("pipelines.static_adapt._shadow_hh_continuation_scoring_bridge_")
    assert public_scoring is importlib.import_module("pipelines.hardcoded.hh_continuation_scoring")


def test_attach_shadow_legacy_geometry_debug_preserves_current_scores() -> None:
    scoring = importlib.import_module("pipelines.hardcoded.hh_continuation_scoring")
    oracle = scoring.Phase1CompileCostOracle()
    meas = scoring.MeasurementCacheAudit()
    base = scoring.build_candidate_features(
        stage_name="core",
        candidate_label="x",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        gradient_signed=0.25,
        metric_proxy=0.25,
        sigma_hat=0.0,
        refit_window_indices=[0],
        compile_cost=oracle.estimate(candidate_term_count=1, position_id=0, append_position=0, refit_active_count=1),
        measurement_stats=meas.estimate(["x"]),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=scoring.SimpleScoreConfig(),
    )
    current_feat = scoring.CandidateFeatures(
        **{
            **base.__dict__,
            "full_v2_score": 0.125,
            "phase2_raw_score": 0.25,
            "selector_score": 0.125,
            "novelty": 0.9,
            "novelty_mode": "current_mode",
            "curvature_mode": "current_curv",
        }
    )
    shadow_feat = scoring.CandidateFeatures(
        **{
            **base.__dict__,
            "full_v2_score": 0.5,
            "phase2_raw_score": 0.75,
            "novelty": 0.1,
            "novelty_mode": "legacy_mode",
            "curvature_mode": "legacy_curv",
            "compiled_position_cost_backend": {
                "phase3_geometry_debug": {
                    "phase3_legacy_geometry_mode": "proxy_reduced",
                    "selector_score": 0.5,
                }
            },
        }
    )
    payload = current_adapt._shadow_legacy_geometry_debug_payload(
        current_feat=current_feat,
        shadow_feat=shadow_feat,
        shadow_mode="proxy_reduced",
    )
    updated = current_adapt._attach_shadow_legacy_geometry_debug(current_feat, payload)
    assert float(updated.full_v2_score or 0.0) == pytest.approx(0.125)
    assert float(updated.phase2_raw_score or 0.0) == pytest.approx(0.25)
    debug = dict((updated.compiled_position_cost_backend or {}).get("shadow_legacy_geometry_debug", {}))
    assert debug["mode"] == "proxy_reduced"
    assert float(debug["shadow_full_v2_score"]) == pytest.approx(0.5)
    assert float(debug["current_full_v2_score"]) == pytest.approx(0.125)
