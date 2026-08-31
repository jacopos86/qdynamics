"""Production-wrapper contract for reference-blind SIAM raw trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.paper_ii_runs import build_siam_nb2_proof_run
from pipelines.time_dynamics.run_lock import RUN_LOCK_SCHEMA_V4, assert_comparable
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import main
from pipelines.time_dynamics.siam_screening import SIAMScreeningConfig
from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest


def _runner_args(
    *, seed: Path, output: Path, policy: str = "exchange"
) -> list[str]:
    run = build_siam_nb2_proof_run(
        seed_path=str(seed),
        arm=policy,
        output_json=str(output),
        code_revision="siam-test-revision",
        extra_flags=("--t-final", "0.08", "--num-times", "3"),
    )
    return list(run.argv())


def _seed(path: Path) -> None:
    write_siam_seed_manifest(
        path,
        SIAMScreeningConfig(
            n_bath=2,
            u=1.5,
            t_final=99.0,
            num_times=2,
            label="horizon_not_seed_identity",
        ),
    )


@pytest.mark.parametrize("policy", ("exchange", "avqds"))
def test_siam_raw_run_uses_verified_seed_and_no_exact_reference(
    tmp_path: Path, policy: str
) -> None:
    seed = tmp_path / "siam_seed.json"
    output = tmp_path / f"{policy}.json"
    _seed(seed)

    assert main(_runner_args(seed=seed, output=output, policy=policy)) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["run_lock"]["physics"]["family_key"] == "siam"
    assert payload["run_lock"]["physics"]["seed_artifact_json"] == str(
        seed.resolve()
    )
    assert payload["run_lock"]["schema"] == RUN_LOCK_SCHEMA_V4
    assert payload["run_lock"]["physics"]["seed_sha256"] == hashlib.sha256(
        seed.read_bytes()
    ).hexdigest()
    assert payload["run_lock"]["physics"]["exact_reference_json"] is None
    assert payload["run_lock"]["physics"]["exact_reference_sha256"] is None
    assert payload["run_lock"]["physics"]["time_grid"]["point_count"] == 3
    assert len(
        payload["run_lock"]["physics"]["time_grid"][
            "ordered_values_sha256"
        ]
    ) == 64
    assert payload["run_lock"]["policy"]["structural_policy"] == policy
    assert payload["initial_state"]["parameterization_mode"] == "logical_shared"
    assert payload["summary"]["candidate_pool_complete"] is True
    assert payload["run_lock"]["shared_protocol"]["candidate_pool"][
        "append_occurrence_policy"
    ] == "layer_reuse"
    assert payload["decision_data_flow"]["uses_exact_reference_for_decision"] is False
    assert all("psi_real" in row and "psi_imag" in row for row in payload["plot_rows"])
    metadata = payload["trajectory"]["metadata"]
    assert metadata["input_source_kind"] == "siam_seed_manifest"
    assert metadata["verified_seed_sha256"] == payload["run_lock"]["physics"][
        "seed_sha256"
    ]
    for irrelevant in (
        "tag",
        "generator_family",
        "fallback_family",
        "replay_candidate_pool_mode",
    ):
        assert irrelevant not in metadata


def test_siam_apm_and_avqds_raw_locks_are_comparable(tmp_path: Path) -> None:
    seed = tmp_path / "siam_seed.json"
    _seed(seed)
    payloads = []
    for policy in ("exchange", "avqds"):
        output = tmp_path / f"{policy}.json"
        assert main(_runner_args(seed=seed, output=output, policy=policy)) == 0
        payloads.append(json.loads(output.read_text(encoding="utf-8")))

    locks = [payload["run_lock"] for payload in payloads]
    assert locks[0]["policy"]["structural_policy"] != locks[1]["policy"][
        "structural_policy"
    ]
    assert assert_comparable(locks)


@pytest.mark.parametrize(
    "extra, message",
    (
        (["--reference-energy-json", "poison.json"], "reference-blind"),
        (["--enable-drive"], "initial-condition change"),
    ),
)
def test_siam_runner_rejects_online_reference_and_external_drive(
    tmp_path: Path, extra: list[str], message: str, capsys
) -> None:
    seed = tmp_path / "siam_seed.json"
    _seed(seed)
    with pytest.raises(SystemExit, match="2"):
        main(_runner_args(seed=seed, output=tmp_path / "out.json") + extra)
    assert message in capsys.readouterr().err


def test_siam_runner_requires_statevectors_and_logical_coordinates(
    tmp_path: Path, capsys
) -> None:
    seed = tmp_path / "siam_seed.json"
    _seed(seed)
    args = _runner_args(seed=seed, output=tmp_path / "out.json")
    args.remove("--record-statevector")
    with pytest.raises(SystemExit, match="2"):
        main(args)
    assert "require --record-statevector" in capsys.readouterr().err

    args = _runner_args(seed=seed, output=tmp_path / "out2.json")
    index = args.index("logical_shared")
    args[index] = "per_pauli_term"
    with pytest.raises(SystemExit, match="2"):
        main(args)
    assert "requires --parameterization-mode logical_shared" in capsys.readouterr().err


def test_siam_runner_rejects_seed_changed_after_verified_load(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    from pipelines.time_dynamics import siam_seed as seed_module

    seed = tmp_path / "siam_seed.json"
    output = tmp_path / "out.json"
    _seed(seed)
    real_loader = seed_module.load_siam_runtime_input

    def load_then_tamper(path):
        runtime_input = real_loader(path)
        Path(path).write_text('{"tampered": true}\n', encoding="utf-8")
        return runtime_input

    monkeypatch.setattr(seed_module, "load_siam_runtime_input", load_then_tamper)
    with pytest.raises(SystemExit, match="2"):
        main(_runner_args(seed=seed, output=output))

    assert "changed after verified load" in capsys.readouterr().err
    assert not output.exists()
