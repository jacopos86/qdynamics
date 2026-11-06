"""The stage-1 installer only installs sealed, qualified, physics-matched seeds."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import prospective_hh_seed_worker as worker
from pipelines.time_dynamics.runners import build_hh_fixed_layer_seed as runner

_SPEC = importlib.util.spec_from_file_location(
    "install_seeds", Path("chtc/paper_ii_regime_seeds_v1/install_seeds.py")
)
install_seeds = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(install_seeds)

TSV = "regime_id\tu\tg_ep\tnph\tdv\toptimizer_seed\nanchor_u4_g0p5_nph1\t4.00\t0.5\t1\t0.0\t97\nweak_weak\t0.25\t0.353553390593\t3\t0.0\t11\n"


def _computation(order, *, signed_error: float) -> worker.SeedComputation:
    return worker.SeedComputation(
        variational_energy=-1.5 + signed_error,
        exact_energy=-1.5,
        exact_state_infidelity=1.0e-9,
        runtime_payload={
            "settings": {"L": 2},
            "adapt_vqe": {
                "parameterization_execution_mode": "per_pauli_term",
                "num_parameters": 216,
            },
        },
        optimizer_receipt={"restarts": 4},
        architecture_binding={"policy": dict(order.architecture)},
    )


def _build(tmp_path: Path, regime_id: str, *, u: float, g: float, nph: int, seed: int, signed_error: float, dv: float = 0.0) -> Path:
    identity = runner.build_seed_identity(
        regime_id=regime_id, u=u, g_ep=g, n_ph_max=nph, optimizer_seed=seed,
        artifact_path="seed.json", dv=dv,
    )
    raw = tmp_path / "fetched" / "raw_outputs" / "paper_ii_regime_seeds_v1" / regime_id
    runner.build_regime_seed(
        regime_id=regime_id, seed=identity, output_json=raw / "seed.json",
        repository_root=tmp_path,
        compute=lambda order: _computation(order, signed_error=signed_error),
    )
    return raw


def test_installs_qualified_rows_and_reports_the_rest(tmp_path: Path) -> None:
    tsv = tmp_path / "regimes.tsv"
    tsv.write_text(TSV, encoding="utf-8")
    _build(tmp_path, "anchor_u4_g0p5_nph1", u=4.0, g=0.5, nph=1, seed=97, signed_error=2.0e-8, dv=0.0)
    dest = tmp_path / "seeds"
    receipt = install_seeds.install(
        fetched_root=tmp_path / "fetched",
        regime_ids=["anchor_u4_g0p5_nph1", "weak_weak"],
        regimes_tsv=tsv,
        dest=dest,
    )
    assert set(receipt["installed"]) == {"anchor_u4_g0p5_nph1"}
    assert "no receipt" in receipt["problems"]["weak_weak"]
    installed = dest / "anchor_u4_g0p5_nph1.json"
    assert installed.is_file()
    sums = (dest / "SHA256SUMS").read_text()
    assert install_seeds._file_sha(installed) in sums
    assert json.loads((dest / "install_receipt.json").read_text())["installed"][
        "anchor_u4_g0p5_nph1"
    ]["energy_error"] == pytest.approx(2.0e-8)
    # Re-running is idempotent for an identical seed.
    again = install_seeds.install(
        fetched_root=tmp_path / "fetched", regime_ids=["anchor_u4_g0p5_nph1"],
        regimes_tsv=tsv, dest=dest,
    )
    assert set(again["installed"]) == {"anchor_u4_g0p5_nph1"}


def test_physics_mismatch_and_tampering_are_refused(tmp_path: Path) -> None:
    tsv = tmp_path / "regimes.tsv"
    tsv.write_text(TSV, encoding="utf-8")
    # Built at the wrong cutoff for its row name.
    _build(tmp_path, "anchor_u4_g0p5_nph1", u=4.0, g=0.5, nph=3, seed=97, signed_error=2.0e-8, dv=0.0)
    receipt = install_seeds.install(
        fetched_root=tmp_path / "fetched", regime_ids=["anchor_u4_g0p5_nph1"],
        regimes_tsv=tsv, dest=tmp_path / "seeds",
    )
    assert "physics" in receipt["problems"]["anchor_u4_g0p5_nph1"]
    # Tamper with the artifact bytes of a correct build.
    raw = _build(tmp_path / "second", "weak_weak", u=0.25, g=0.353553390593, nph=3, seed=11, signed_error=1.0e-8)
    artifact = raw / "seed.json"
    payload = json.loads(artifact.read_text())
    payload["adapt_vqe"]["num_parameters"] = 1
    artifact.write_text(json.dumps(payload))
    receipt = install_seeds.install(
        fetched_root=tmp_path / "second" / "fetched", regime_ids=["weak_weak"],
        regimes_tsv=tsv, dest=tmp_path / "seeds2",
    )
    assert "digest" in receipt["problems"]["weak_weak"]
    assert not (tmp_path / "seeds2" / "weak_weak.json").exists()


def test_unqualified_receipt_is_never_installed(tmp_path: Path) -> None:
    tsv = tmp_path / "regimes.tsv"
    tsv.write_text(TSV, encoding="utf-8")
    identity = runner.build_seed_identity(
        regime_id="weak_weak", u=0.25, g_ep=0.353553390593, n_ph_max=3,
        optimizer_seed=11, artifact_path="seed.json",
    )
    raw = tmp_path / "fetched" / "raw_outputs" / "paper_ii_regime_seeds_v1" / "weak_weak"

    def _failing(order):
        raise worker.ProspectiveExecutionError("seed fails the hard Delta E <= 1e-6 gate: 2e-05")

    runner.build_regime_seed(
        regime_id="weak_weak", seed=identity, output_json=raw / "seed.json",
        repository_root=tmp_path, compute=_failing,
    )
    receipt = install_seeds.install(
        fetched_root=tmp_path / "fetched", regime_ids=["weak_weak"],
        regimes_tsv=tsv, dest=tmp_path / "seeds",
    )
    assert "unqualified" in receipt["problems"]["weak_weak"]
    assert not (tmp_path / "seeds").exists()
