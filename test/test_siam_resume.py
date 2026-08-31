"""SIAM logical-macro resume and continuation parity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import SolveRepairConfig
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.paper_ii_runs import build_siam_nb2_proof_run
from pipelines.time_dynamics.resume import (
    RESUME_SCHEMA_V2,
    runtime_input_from_resume_state,
)
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import (
    main,
    run_append_ap_mclachlan_from_runtime_input,
)
from pipelines.time_dynamics.siam_screening import SIAMScreeningConfig
from pipelines.time_dynamics.siam_seed import (
    load_siam_runtime_input,
    write_siam_seed_manifest,
)


def _config(*, t_final: float, num_times: int) -> SIAMScreeningConfig:
    return SIAMScreeningConfig(
        n_bath=2,
        u=1.5,
        t_final=float(t_final),
        num_times=int(num_times),
        label="siam_resume_contract",
    )


def _patch_config() -> SupportPatchControllerConfig:
    return SupportPatchControllerConfig(
        insertion_gate_mode="mclachlan_l2",
        insertion_l2_cut=1.0e-3,
        max_structural_pool_size=64,
        append_schur_max_condition_number=None,
        max_joint_patch_evaluations=50000,
        max_certification_attempts_per_level=12,
    )


def _run(runtime_input, times, *, controller_resume_state=None):
    return run_append_ap_mclachlan_from_runtime_input(
        runtime_input,
        record_statevector=True,
        times=tuple(float(value) for value in times),
        integrator_method="rk4",
        parameterization_mode="logical_shared",
        support_patch_config=_patch_config(),
        solve_repair_config=SolveRepairConfig.minimal_profile(),
        controller_resume_state=controller_resume_state,
    )


def _recorded_state(row) -> np.ndarray:
    return np.asarray(row["psi_real"], dtype=float) + 1.0j * np.asarray(
        row["psi_imag"], dtype=float
    )


def _ordered_active_rotations(term, layout) -> list[dict[str, float | int | str]]:
    pauli_terms = list(term.polynomial.return_polynomial())
    if str(layout.term_order) == "sorted":
        pauli_terms.sort(key=lambda pauli: pauli.pw2strng())
    rotations: list[dict[str, float | int | str]] = []
    tolerance = float(layout.coefficient_tolerance)
    for pauli in pauli_terms:
        word = str(pauli.pw2strng())
        coefficient = complex(pauli.p_coeff)
        if abs(coefficient) < tolerance:
            continue
        if bool(layout.ignore_identity) and word == "e" * int(pauli.nqubit()):
            continue
        assert abs(coefficient.imag) <= tolerance
        rotations.append(
            {
                "pauli_exyz": word,
                "coeff_real": float(coefficient.real),
                "nq": int(pauli.nqubit()),
            }
        )
    return rotations


def _seed_runtime_input(tmp_path: Path, config: SIAMScreeningConfig):
    seed_path = tmp_path / "siam_seed.json"
    write_siam_seed_manifest(seed_path, config)
    return load_siam_runtime_input(seed_path)


def _cli_args(
    *,
    seed: Path,
    output: Path,
    t_final: float,
    num_times: int,
    t_initial: float | None = None,
    resume_from: Path | None = None,
) -> list[str]:
    extra_flags = [
        "--t-final",
        repr(float(t_final)),
        "--num-times",
        str(int(num_times)),
    ]
    if t_initial is not None:
        extra_flags.extend(("--t-initial", repr(float(t_initial))))
    if resume_from is not None:
        extra_flags.extend(("--resume-from-run-json", str(resume_from)))
    run = build_siam_nb2_proof_run(
        seed_path=str(seed),
        arm="exchange",
        output_json=str(output),
        code_revision="siam-resume-test-revision",
        extra_flags=tuple(extra_flags),
    )
    return list(run.argv())


def test_siam_raw_run_emits_usable_logical_macro_resume(
    tmp_path: Path,
) -> None:
    config = _config(t_final=0.2, num_times=6)
    base = _seed_runtime_input(tmp_path, config)

    raw = json.loads(json.dumps(_run(base, config.times), sort_keys=True))
    resume = raw["resume_state"]

    assert resume["schema"] == RESUME_SCHEMA_V2
    assert resume["problem_family"] == "siam"
    assert resume["parameterization_mode"] == "logical_shared"
    assert "unavailable" not in resume
    assert resume["controller_state"]["checkpoint_index"] == 5
    assert len(resume["coordinates"]) == len(resume["theta_runtime"])
    assert resume["theta_runtime"] == pytest.approx(resume["theta_logical"])
    assert len(resume["coordinates"]) > 0
    assert any(
        len(coordinate["rotations"]) > 1
        for coordinate in resume["coordinates"]
    )
    assert [coordinate["label"] for coordinate in resume["coordinates"]] == raw[
        "final_state"
    ]["selected_term_labels"]
    pool_by_label = {
        term.label: term for term in base.candidate_pool_terms
    }
    for coordinate in resume["coordinates"]:
        pool_label = coordinate["label"].split("::insr", maxsplit=1)[0]
        pool_term = pool_by_label[pool_label]
        assert coordinate["execution_mode"] == pool_term.execution_mode
        assert coordinate["rotations"] == _ordered_active_rotations(
            pool_term,
            base.base_layout,
        )

    resumed_input = runtime_input_from_resume_state(
        resume,
    )
    resumed_state = state_from_scaffold_runtime_input(
        resumed_input,
        parameterization_mode="logical_shared",
    )
    expected_boundary = _recorded_state(raw["plot_rows"][-1])

    assert resumed_state.runtime_parameter_count == len(resume["coordinates"])
    assert resumed_state.runtime_pauli_parameter_count > (
        resumed_state.runtime_parameter_count
    )
    expected_expanded_theta = np.concatenate(
        [
            np.full(len(coordinate["rotations"]), theta, dtype=float)
            for coordinate, theta in zip(
                resume["coordinates"], resume["theta_logical"]
            )
        ]
    )
    np.testing.assert_allclose(
        np.asarray(resumed_input.theta_runtime, dtype=float),
        expected_expanded_theta,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(resumed_input.theta_logical, dtype=float),
        np.asarray(resume["theta_logical"], dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    assert resumed_state.runtime_coordinate_labels == tuple(
        coordinate["runtime_label"] for coordinate in resume["coordinates"]
    )
    np.testing.assert_allclose(
        resumed_state.theta_runtime,
        np.asarray(resume["theta_runtime"], dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        resumed_state.psi_initial,
        expected_boundary,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_siam_split_continuation_matches_uninterrupted_boundary_and_final_state(
    tmp_path: Path,
) -> None:
    config = _config(t_final=0.4, num_times=11)
    base = _seed_runtime_input(tmp_path, config)
    split_index = 5

    uninterrupted = _run(base, config.times)
    first = _run(base, config.times[: split_index + 1])
    first_resume = json.loads(json.dumps(first["resume_state"], sort_keys=True))
    resumed_input = runtime_input_from_resume_state(
        first_resume,
        base_runtime_input=base,
    )
    second = _run(
        resumed_input,
        config.times[split_index:],
        controller_resume_state=first_resume["controller_state"],
    )

    first_boundary = _recorded_state(first["plot_rows"][-1])
    resumed_boundary = _recorded_state(second["plot_rows"][0])
    np.testing.assert_allclose(
        resumed_boundary,
        first_boundary,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        _recorded_state(second["plot_rows"][-1]),
        _recorded_state(uninterrupted["plot_rows"][-1]),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert second["plot_rows"][-1]["energy_expectation"] == pytest.approx(
        uninterrupted["plot_rows"][-1]["energy_expectation"],
        abs=1.0e-10,
    )
    assert second["final_state"]["selected_term_labels"] == uninterrupted[
        "final_state"
    ]["selected_term_labels"]
    assert second["trajectory"]["points"][-1]["theta_runtime"] == pytest.approx(
        uninterrupted["trajectory"]["points"][-1]["theta_runtime"],
        abs=1.0e-10,
    )


def test_siam_cli_raw_json_resume_matches_uninterrupted_continuation(
    tmp_path: Path,
) -> None:
    config = _config(t_final=0.08, num_times=3)
    seed = tmp_path / "siam_seed.json"
    write_siam_seed_manifest(seed, config)
    uninterrupted_path = tmp_path / "uninterrupted.json"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    assert main(
        _cli_args(
            seed=seed,
            output=uninterrupted_path,
            t_final=0.08,
            num_times=3,
        )
    ) == 0
    assert main(
        _cli_args(
            seed=seed,
            output=first_path,
            t_final=0.04,
            num_times=2,
        )
    ) == 0
    first = json.loads(first_path.read_text(encoding="utf-8"))
    assert first["resume_state"]["schema"] == RESUME_SCHEMA_V2
    assert main(
        _cli_args(
            seed=seed,
            output=second_path,
            t_initial=0.04,
            t_final=0.08,
            num_times=2,
            resume_from=first_path,
        )
    ) == 0

    uninterrupted = json.loads(uninterrupted_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    lineage = second["segment_lineage"]
    assert lineage["source_run_path"] == str(first_path.resolve())
    assert lineage["source_run_sha256"] == hashlib.sha256(
        first_path.read_bytes()
    ).hexdigest()
    assert lineage["source_resume_payload_sha256"] == hashlib.sha256(
        json.dumps(
            first["resume_state"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert lineage["boundary_link_pass"] is True
    assert lineage["current_initial_state_sha256"] == lineage[
        "source_terminal_state_sha256"
    ]
    assert lineage["current_initial_ansatz_sha256"] == lineage[
        "source_terminal_ansatz_sha256"
    ]
    assert lineage["current_controller_input_sha256"] == lineage[
        "source_controller_state_sha256"
    ]
    assert second["trajectory"]["metadata"]["resume_lineage"][
        "lineage_sha256"
    ] == lineage["lineage_sha256"]
    np.testing.assert_allclose(
        _recorded_state(second["plot_rows"][0]),
        _recorded_state(first["plot_rows"][-1]),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        _recorded_state(second["plot_rows"][-1]),
        _recorded_state(uninterrupted["plot_rows"][-1]),
        rtol=0.0,
        atol=1.0e-10,
    )
    assert second["final_state"]["selected_term_labels"] == uninterrupted[
        "final_state"
    ]["selected_term_labels"]
    assert second["trajectory"]["points"][-1]["theta_runtime"] == pytest.approx(
        uninterrupted["trajectory"]["points"][-1]["theta_runtime"],
        abs=1.0e-10,
    )


def test_siam_run_fails_if_resume_serialization_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(t_final=0.2, num_times=6)
    base = _seed_runtime_input(tmp_path, config)

    def fail_resume_serialization(*args, **kwargs):
        del args, kwargs
        raise ValueError("forced resume serialization failure")

    monkeypatch.setattr(
        "pipelines.time_dynamics.runners.ap_append_from_adapt_artifact."
        "build_resume_state",
        fail_resume_serialization,
    )

    with pytest.raises(RuntimeError, match="SIAM.*resumable"):
        _run(base, config.times)
