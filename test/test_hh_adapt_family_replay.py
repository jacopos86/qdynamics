from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    RunConfig,
    build_replay_scaffold_context,
    _build_pool_for_family,
    _resolve_family,
    _resolve_family_from_metadata,
    parse_args,
)
from src.quantum.ansatz_parameterization import build_parameter_layout, serialize_layout
from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _mk_cfg(tmp_path: Path, *, generator_family: str = "match_adapt", fallback_family: str = "full_meta") -> RunConfig:
    return RunConfig(
        adapt_input_json=tmp_path / "in.json",
        output_json=tmp_path / "out.json",
        output_csv=tmp_path / "out.csv",
        output_md=tmp_path / "out.md",
        output_log=tmp_path / "out.log",
        tag="test",
        generator_family=generator_family,
        fallback_family=fallback_family,
        legacy_paop_key="paop_lf_std",
        replay_seed_policy="auto",
        replay_continuation_mode="legacy",
        L=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        sector_n_up=1,
        sector_n_dn=1,
        reps=2,
        restarts=2,
        maxiter=20,
        method="SPSA",
        seed=7,
        energy_backend="one_apply_compiled",
        progress_every_s=60.0,
        wallclock_cap_s=600,
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
        spsa_a=0.2,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=0,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        replay_freeze_fraction=0.2,
        replay_unfreeze_fraction=0.3,
        replay_full_fraction=0.5,
        replay_qn_spsa_refresh_every=5,
        replay_qn_spsa_refresh_mode="diag_rms_grad",
        phase3_symmetry_mitigation_mode="off",
    )


def test_parse_defaults_match_adapt_and_spsa() -> None:
    args = parse_args(["--adapt-input-json", "dummy.json"])
    assert str(args.generator_family) == "match_adapt"
    assert str(args.fallback_family) == "full_meta"
    assert str(args.method) == "SPSA"
    assert str(args.replay_continuation_mode) == "legacy"


def test_parse_rejects_auto_replay_continuation_mode() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--adapt-input-json", "dummy.json", "--replay-continuation-mode", "auto"])


def test_parse_accepts_phase2_replay_continuation_mode() -> None:
    args = parse_args(["--adapt-input-json", "dummy.json", "--replay-continuation-mode", "phase2_v1"])
    assert str(args.replay_continuation_mode) == "phase2_v1"


def test_parse_accepts_phase3_replay_continuation_mode() -> None:
    args = parse_args(["--adapt-input-json", "dummy.json", "--replay-continuation-mode", "phase3_v1"])
    assert str(args.replay_continuation_mode) == "phase3_v1"


def _hh_h_poly(*, L: int = 2, n_ph_max: int = 1):
    return build_hubbard_holstein_hamiltonian(
        dims=L,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=n_ph_max,
        boson_encoding="binary",
        v_t=None,
        v0=0.0,
        t_eval=None,
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )


def test_resolve_family_prefers_adapt_vqe_pool_type() -> None:
    fam, src = _resolve_family_from_metadata({"adapt_vqe": {"pool_type": "full_meta"}})
    assert fam == "full_meta"
    assert src == "adapt_vqe.pool_type"


def test_resolve_family_prefers_pareto_lean_pool_type() -> None:
    fam, src = _resolve_family_from_metadata({"adapt_vqe": {"pool_type": "pareto_lean"}})
    assert fam == "pareto_lean"
    assert src == "adapt_vqe.pool_type"


def test_resolve_family_prefers_pareto_lean_l2_pool_type() -> None:
    fam, src = _resolve_family_from_metadata({"adapt_vqe": {"pool_type": "pareto_lean_l2"}})
    assert fam == "pareto_lean_l2"
    assert src == "adapt_vqe.pool_type"


def test_resolve_family_uses_settings_adapt_pool() -> None:
    fam, src = _resolve_family_from_metadata({"settings": {"adapt_pool": "uccsd_paop_lf_full"}})
    assert fam == "uccsd_paop_lf_full"
    assert src == "settings.adapt_pool"


def test_resolve_family_maps_legacy_pool_variant() -> None:
    fam, src = _resolve_family_from_metadata({"meta": {"pool_variant": "B"}})
    assert fam == "pool_b"
    assert src == "meta.pool_variant"


def test_resolve_family_match_adapt_falls_back_when_missing(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    info = _resolve_family(cfg, {})
    assert info["requested"] == "match_adapt"
    assert info["resolved"] == "full_meta"
    assert bool(info["fallback_used"]) is True
    assert info["resolution_source"] == "fallback_family"


def test_resolve_family_honors_explicit_cli_pareto_lean_l2(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="pareto_lean_l2", fallback_family="full_meta")
    info = _resolve_family(cfg, {})
    assert info["requested"] == "pareto_lean_l2"
    assert info["resolved"] == "pareto_lean_l2"
    assert bool(info["fallback_used"]) is False
    assert info["resolution_source"] == "cli.generator_family"


def test_build_pool_for_family_supports_pareto_lean_l2(tmp_path: Path) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="pareto_lean_l2")
    pool, meta = _build_pool_for_family(cfg, family="pareto_lean_l2", h_poly=_hh_h_poly())
    assert len(pool) > 0
    assert meta["family"] == "pareto_lean_l2"
    assert int(meta["dedup_total"]) == len(pool)


def test_build_pool_for_family_pareto_lean_l2_rejects_non_l2(tmp_path: Path) -> None:
    cfg = replace(_mk_cfg(tmp_path, generator_family="pareto_lean_l2"), L=3, sector_n_up=2, sector_n_dn=1)
    with pytest.raises(ValueError, match="only valid for L=2"):
        _build_pool_for_family(cfg, family="pareto_lean_l2", h_poly=_hh_h_poly(L=3))


def test_build_pool_for_family_pareto_lean_l2_rejects_nphmax_not_1(tmp_path: Path) -> None:
    cfg = replace(_mk_cfg(tmp_path, generator_family="pareto_lean_l2"), n_ph_max=2)
    with pytest.raises(ValueError, match="only valid for n_ph_max=1"):
        _build_pool_for_family(cfg, family="pareto_lean_l2", h_poly=_hh_h_poly(n_ph_max=2))


def test_build_replay_scaffold_context_honors_payload_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    pool = [
        AnsatzTerm(
            label="op_1",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="xx", pc=1.0), PauliTerm(2, ps="zz", pc=0.5)],
            ),
        ),
        AnsatzTerm(
            label="op_2",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="yy", pc=1.0)]),
        ),
    ]
    wf_layout = build_parameter_layout(
        pool,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    payload_layout = serialize_layout(wf_layout)
    payload = {
        "adapt_vqe": {
            "pool_type": "pareto_lean_l2",
            "operators": ["op_1", "op_2"],
            "optimal_point": [0.1, 0.15, -0.2],
            "logical_optimal_point": [0.125, -0.2],
            "parameterization": payload_layout,
        },
        "initial_state": {
            "handoff_state_kind": "prepared_state",
            "amplitudes_qn_to_q0": {"00": {"re": 1.0, "im": 0.0}},
            "nq_total": 2,
        },
    }
    psi_ref = np.array([1.0 + 0.0j, 0.0, 0.0, 0.0], dtype=complex)

    monkeypatch.setattr(
        "pipelines.hardcoded.hh_vqe_from_adapt_family._build_pool_for_family",
        lambda *_args, **_kwargs: (list(pool), {"family": "pareto_lean_l2", "dedup_total": len(pool)}),
    )

    ctx = build_replay_scaffold_context(
        cfg,
        h_poly=object(),
        psi_ref=psi_ref,
        payload_in=payload,
    )

    assert ctx.family_info["resolved"] == "pareto_lean_l2"
    assert tuple(term.label for term in ctx.replay_terms) == ("op_1", "op_2")
    assert int(ctx.base_layout.runtime_parameter_count) == int(wf_layout.runtime_parameter_count)
    assert np.allclose(ctx.adapt_theta_runtime, np.array([0.1, 0.15, -0.2], dtype=float))
    assert np.allclose(ctx.adapt_theta_logical, np.array([0.125, -0.2], dtype=float))
    assert bool(ctx.pool_meta["candidate_pool_complete"]) is True


def test_build_replay_scaffold_context_uses_metadata_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _mk_cfg(tmp_path, generator_family="match_adapt", fallback_family="full_meta")
    pool = [
        AnsatzTerm(
            label="op_1",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xx", pc=1.0)]),
        )
    ]
    payload = {
        "adapt_vqe": {
            "pool_type": "pareto_lean_l2",
            "operators": ["op_1"],
            "optimal_point": [0.2],
        },
        "initial_state": {
            "handoff_state_kind": "prepared_state",
            "amplitudes_qn_to_q0": {"00": {"re": 1.0, "im": 0.0}},
            "nq_total": 2,
        },
    }
    psi_ref = np.array([1.0 + 0.0j, 0.0, 0.0, 0.0], dtype=complex)

    monkeypatch.setattr(
        "pipelines.hardcoded.hh_vqe_from_adapt_family._build_pool_for_family",
        lambda *_args, **_kwargs: (list(pool), {"family": "pareto_lean_l2", "dedup_total": len(pool)}),
    )

    ctx = build_replay_scaffold_context(
        cfg,
        h_poly=object(),
        psi_ref=psi_ref,
        payload_in=payload,
    )

    assert ctx.family_info["requested"] == "match_adapt"
    assert ctx.family_info["resolved"] == "pareto_lean_l2"
    assert ctx.family_info["resolution_source"] == "adapt_vqe.pool_type"
