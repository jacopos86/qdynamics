from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_vqe_from_adapt_family import (
    RunConfig,
    _resolve_family,
    _resolve_family_from_metadata,
    parse_args,
)


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


def test_resolve_family_prefers_adapt_vqe_pool_type() -> None:
    fam, src = _resolve_family_from_metadata({"adapt_vqe": {"pool_type": "full_meta"}})
    assert fam == "full_meta"
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
