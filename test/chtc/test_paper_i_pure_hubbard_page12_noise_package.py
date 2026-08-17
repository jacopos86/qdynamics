from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727" / (
    "paper_i_ra_adapt_pure_hubbard_page12_fullnoise6_r50_20260811_v3_chtc"
)


def _module(path: Path, *, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _package_module(path: Path, *, name: str) -> ModuleType:
    sys.path.insert(0, PACKAGE.as_posix())
    previous = sys.modules.pop("package_contract", None)
    try:
        return _module(path, name=name)
    finally:
        sys.path.remove(PACKAGE.as_posix())
        sys.modules.pop("package_contract", None)
        if previous is not None:
            sys.modules["package_contract"] = previous


def test_contract_is_exact_six_cell_pure_hubbard_noise_matrix() -> None:
    contract = _module(
        PACKAGE / "package_contract.py",
        name="paper_i_pure_hubbard_noise_package_contract",
    )

    assert contract.TARGET_HORIZON == 50
    assert contract.U_VALUES == (1.5, 8.0)
    assert contract.NOISE_TUPLE_ORDER == (
        "sigma_E",
        "p1",
        "p2",
        "epsilon1",
        "epsilon2",
    )
    assert contract.NOISE_LEVELS == (
        ("low", (1.0e-6, 1.0e-8, 1.0e-7, 2.0e-4, 6.0e-4)),
        (
            "high",
            (7.071067811865475e-5, 1.0e-6, 1.0e-5, 2.0e-3, 6.0e-3),
        ),
        ("extreme", (1.0e-2, 1.0e-3, 1.0e-2, 6.0e-2, 6.0e-2)),
    )
    assert len(contract.CELL_ROWS) == 6
    assert len(contract.expected_execution_ids()) == 6
    assert contract.VALUE_NOISE_SEED == 702688422
    assert contract.COHERENT_NOISE_SEED == 20260609
    assert contract.INSERTION_POLICY.endswith("tau1em4_v1")
    assert contract.PLATEAU_THRESHOLD == 1.0e-4
    assert contract.OPTIMIZER == "powell"
    assert contract.OPTIMIZER_MAXITER == 200
    assert contract.RESOURCE_ENVELOPE["request_cpus"] == 2
    assert contract.RESOURCE_ENVELOPE["request_memory_mb"] == 8192
    assert contract.RESOURCE_ENVELOPE["request_disk_mb"] == 12288
    assert contract.RESOURCE_ENVELOPE["max_runtime_seconds"] == 259200


def test_sealed_application_sources_use_platform_stable_v2_authority() -> None:
    source_dir = PACKAGE / "source_authority"
    rows = sorted(source_dir.glob("*.application_source_contract.json"))
    assert len(rows) == 6
    for path in rows:
        source = json.loads(path.read_text(encoding="utf-8"))
        exact = source["same_cutoff_exact_reference"]
        assert source["schema"] == (
            "paper_i_pure_hubbard_noise_page12_application_source_contract_v2"
        )
        assert "energy" not in exact
        assert exact["controller_input"] is False
        assert exact["evaluation_policy_id"] == (
            "runtime_same_cutoff_exact_diagnostic_full_precision_v1"
        )
        assert exact["analytic_reference"]["formula_id"] == (
            "l2_open_half_filled_hubbard_ground_energy_v1"
        )


def test_package_requires_full_noise_runtime_source_and_disables_caches() -> None:
    contract = _module(
        PACKAGE / "package_contract.py",
        name="paper_i_pure_hubbard_noise_source_contract",
    )
    required = set(contract.REQUIRED_ROUTE_SOURCE_PATHS)
    assert {
        "pipelines/exact_bench/noise_oracle_runtime.py",
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/ra_adapt/pure_hubbard_noise_page12.py",
    } <= required

    wrapper = (PACKAGE / "execute_authorized_job.sh").read_text(
        encoding="utf-8"
    )
    assert wrapper.count("STATIC_ADAPT_HH_POOL_CACHE=off") >= 2
    assert wrapper.count("STATIC_ADAPT_CANDIDATE_RECORD_CACHE=off") >= 2
    assert "--cleanenv" in wrapper


def test_builder_requires_real_p3_then_source_extracted_p4() -> None:
    source = (PACKAGE / "build_package.py").read_text(encoding="utf-8")
    p3 = source.index('p3 = _run_preflight("p3")')
    archive = source.index("source_manifest = _write_source_archive(locks)")
    p4 = source.index('p4 = _run_preflight("p4", job_path=p4_job)')
    manifest = source.index('PACKAGE_DIR / "package_manifest.json"')
    assert p3 < archive < p4 < manifest
    assert '"real_noisy_gradient_probe_passed"' in source
    assert '"real_noisy_powell_probe_passed"' in source


def test_numerical_evidence_rejects_declarative_only_payload() -> None:
    module = _package_module(
        PACKAGE / "run_numerical_preflight.py",
        name="paper_i_pure_hubbard_noise_numerical_preflight",
    )
    result = SimpleNamespace(
        run=SimpleNamespace(
            accepted_trajectory=[],
            scientific_replay=[],
            stop=SimpleNamespace(completed_controller_rounds=0),
        ),
        scientific_receipts={
            "controller_noise": {
                "candidate_gradient_scoring": "noisy",
                "powell_refit_objective": "noisy",
            }
        },
        to_dict=lambda: {"status": "declared_only"},
    )
    with pytest.raises(Exception, match="did not accept"):
        module._numerical_evidence(
            result,
            witness_execution_id=(
                "pure_hubbard_page12_fullnoise__u1p5__low"
            ),
        )


def test_numerical_evidence_requires_real_evaluation_records() -> None:
    module = _package_module(
        PACKAGE / "run_numerical_preflight.py",
        name="paper_i_pure_hubbard_noise_trace_preflight",
    )
    rng_state = {"draw_count": 0}
    empty_rows: list[object] = []
    empty_receipts: dict[str, object] = {}
    runtime_delta = {
        "schema": (
            "paper_i_pure_hubbard_controller_noise_transition_delta_v1"
        ),
        "evaluation_count_before": 0,
        "evaluation_count_after": 0,
        "evaluation_records_delta": empty_rows,
        "evaluation_records_delta_sha256": module.canonical_sha256(
            empty_rows
        ),
        "cumulative_evaluation_records_sha256": module.canonical_sha256(
            empty_rows
        ),
        "compiled_noise_receipt_count_before": 0,
        "compiled_noise_receipt_count_after": 0,
        "compiled_noise_receipts_delta": empty_receipts,
        "compiled_noise_receipts_delta_sha256": module.canonical_sha256(
            empty_receipts
        ),
        "cumulative_compiled_noise_receipts_sha256": (
            module.canonical_sha256(empty_receipts)
        ),
        "noise_contract_sha256": "n" * 64,
        "rng_state_after": rng_state,
        "rng_state_after_sha256": module.canonical_sha256(rng_state),
    }
    result = SimpleNamespace(
        run=SimpleNamespace(
            accepted_trajectory=[
                SimpleNamespace(controller_round=1, energy=-1.0)
            ],
            scientific_replay=[
                SimpleNamespace(
                    controller_round=1,
                    accepted_refit=SimpleNamespace(
                        policy="supported_fs_powell_v1",
                        full_ansatz=True,
                        supported_rank=1,
                    ),
                    phase=SimpleNamespace(
                        phase3=SimpleNamespace(
                            coordinate_scope="stationary_source_response_v1",
                            supported_rank=1,
                        )
                    ),
                )
            ],
            stop=SimpleNamespace(completed_controller_rounds=1),
        ),
        scientific_receipts={
            "accepted_round_receipts": [
                {
                    "controller_noise": {
                        "controller_energy_after": -0.9,
                        "exact_diagnostic_energy_after": -1.0,
                        "runtime_delta": runtime_delta,
                    }
                }
            ],
            "controller_noise": {
                "schema": "paper_i_pure_hubbard_controller_noise_receipt_v1",
                "candidate_gradient_scoring": "noisy",
                "powell_refit_objective": "noisy",
                "geometry_and_gram": "exact",
                "reported_energy": "exact_diagnostic",
                "same_circuit_incumbent": True,
                "optimizer_evaluation_order": "serial_v1",
                "candidate_record_cache": "off_fail_closed_v1",
                "accepted_round_count": 1,
                "final_controller_energy": -0.9,
                "final_exact_diagnostic_energy": -1.0,
                "evaluation_count": 0,
                "evaluation_records": [],
                "compiled_noise_receipts": {},
                "noise_contract_sha256": "n" * 64,
                "value_noise": {
                    "model": "gaussian_iid_v1",
                    "draw_count": 0,
                    "rng_state": rng_state,
                },
            },
        },
        to_dict=lambda: {
            "optimizer": "POWELL",
            "candidate_gradient_scoring": "noisy",
        },
    )
    with pytest.raises(Exception, match="evaluation records"):
        module._numerical_evidence(
            result,
            witness_execution_id=(
                "pure_hubbard_page12_fullnoise__u1p5__low"
            ),
        )


def test_package_contract_rejects_cache_and_bytecode_artifacts(tmp_path: Path) -> None:
    contract = _module(
        PACKAGE / "package_contract.py",
        name="paper_i_pure_hubbard_noise_cache_contract",
    )
    clean = tmp_path / "clean"
    clean.mkdir()
    contract.reject_cache_artifacts(clean)

    cache = clean / "__pycache__"
    cache.mkdir()
    with pytest.raises(contract.PackageContractError, match="cache artifacts"):
        contract.reject_cache_artifacts(clean)
    cache.rmdir()
    (clean / "worker.pyc").write_bytes(b"not bytecode")
    with pytest.raises(contract.PackageContractError, match="worker.pyc"):
        contract.reject_cache_artifacts(clean)


def test_unsealed_package_tree_has_no_cache_artifacts() -> None:
    assert not list(PACKAGE.rglob("__pycache__"))
    assert not list(PACKAGE.rglob("*.pyc"))
    assert not list(PACKAGE.rglob("*.pyo"))


def test_control_plane_is_an_exact_byte_closed_inventory(
    tmp_path: Path,
) -> None:
    contract = _module(
        PACKAGE / "package_contract.py",
        name="paper_i_pure_hubbard_noise_control_closure",
    )
    rows = []
    for name in contract.CONTROL_FILES:
        path = tmp_path / name
        payload = f"sealed:{name}\n".encode()
        path.write_bytes(payload)
        rows.append(
            {
                "path": name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest = {"control_files": rows}
    observed = contract.validate_control_file_bindings(tmp_path, manifest)
    assert tuple(observed) == contract.CONTROL_FILES

    (tmp_path / "run_cell.py").write_text("drift\n", encoding="utf-8")
    with pytest.raises(contract.PackageContractError, match="byte binding"):
        contract.validate_control_file_bindings(tmp_path, manifest)


def test_retry_attempt_identity_and_terminal_archive_are_closed() -> None:
    module = _package_module(
        PACKAGE / "run_cell.py",
        name="paper_i_pure_hubbard_noise_scheduler_identity",
    )
    execution_id = "pure_hubbard_page12_fullnoise__u1p5__low"
    scheduler = module._scheduler_identity(
        execution_id=execution_id,
        cluster_id=123,
        proc_id=4,
        attempt_ordinal=2,
    )
    assert scheduler == {
        "schema": "paper_i_chtc_scheduler_attempt_identity_v1",
        "cluster_id": 123,
        "proc_id": 4,
        "attempt_ordinal": 2,
        "terminal_output_archive": (
            "transfer/pure_hubbard_page12_fullnoise__u1p5__low__"
            "123__4.tar.gz"
        ),
        "transfer_policy": "on_exit_only_v1",
    }
    with pytest.raises(Exception, match="Scheduler attempt identity"):
        module._scheduler_identity(
            execution_id=execution_id,
            cluster_id=123,
            proc_id=4,
            attempt_ordinal=0,
        )

    submit = (PACKAGE / "submit.sub.in").read_text(encoding="utf-8")
    wrapper = (PACKAGE / "execute_authorized_job.sh").read_text(
        encoding="utf-8"
    )
    assert "when_to_transfer_output = ON_EXIT\n" in submit
    assert "ON_EXIT_OR_EVICT" not in submit
    assert "__PACKAGE_MANIFEST_FILE_SHA256__" in submit
    assert "__JOB_WRAPPER_SHA256__" in submit
    assert "__RUN_CELL_SHA256__" in submit
    assert "__PACKAGE_CONTRACT_SHA256__" in submit
    assert "_CONDOR_JOB_AD" in wrapper
    assert "NumJobStarts" in wrapper
    assert "attempt_identity.json" in wrapper
    assert "--scheduler-attempt-ordinal" in wrapper

    activation = _package_module(
        PACKAGE / "activate_package.py",
        name="paper_i_pure_hubbard_noise_submit_render",
    )
    rendered = activation._render_submit_template(
        package_relative="chtc/package",
        activation_relative="chtc/activation",
        source_archive_sha256="1" * 64,
        package_manifest_file_sha256="2" * 64,
        job_wrapper_sha256="3" * 64,
        run_cell_sha256="4" * 64,
        package_contract_sha256="5" * 64,
    )
    assert "__PACKAGE_" not in rendered
    assert "__ACTIVATION_" not in rendered
    assert "__SOURCE_" not in rendered
    assert "__JOB_" not in rendered
    assert "__RUN_" not in rendered
    assert "when_to_transfer_output = ON_EXIT\n" in rendered


def test_deep_validation_runs_real_packaged_p4_inside_pinned_image() -> None:
    validator = (PACKAGE / "validate_package.py").read_text(encoding="utf-8")
    authorization = (PACKAGE / "run_cell.py").read_text(encoding="utf-8")
    assert '"run_numerical_preflight.py"' in validator
    assert '"--mode",' in validator and '"p4",' in validator
    assert '"p4_numerical_witness": pinned_p4' in validator
    assert '"deep_pinned_numerical_p4_passed"' in validator
    assert 'probe.get("p4_numerical_witness")' in authorization
    assert 'pinned_p4.get("source_locked_archive_validated") is not True' in (
        authorization
    )

    module = _package_module(
        PACKAGE / "validate_package.py",
        name="paper_i_pure_hubbard_noise_pinned_p4_validation",
    )
    archive_sha = "a" * 64
    payload = module.digested(
        {
            "schema": module.P4_RECEIPT_SCHEMA,
            "status": "passed",
            "scientific_execution_performed": True,
            "source_locked_archive_validated": True,
            "real_noisy_gradient_probe_passed": True,
            "real_noisy_powell_probe_passed": True,
            "completed_controller_rounds": 1,
            "source_archive_sha256": archive_sha,
        }
    )
    assert module._validated_pinned_p4(
        payload,
        manifest={"source_archive": {"sha256": archive_sha}},
    ) == payload
    tampered = dict(payload)
    tampered["real_noisy_powell_probe_passed"] = False
    tampered = module.digested(tampered)
    with pytest.raises(Exception, match="packaged numerical P4 drifted"):
        module._validated_pinned_p4(
            tampered,
            manifest={"source_archive": {"sha256": archive_sha}},
        )
