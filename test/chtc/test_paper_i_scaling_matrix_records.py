from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from chtc.phase3_optuna import preflight_submit, run_task
from chtc.phase3_optuna import generate_paper_i_scaling_matrix_records as generator
from chtc.phase3_optuna import prewarm_paper_i_scaling_hh_pool_cache as prewarm
from chtc.phase3_optuna import run_paper_i_scaling_matrix_cell as runner


def _fake_exact_energy(spec, *, n_ph_max: int):
    case_id = str(spec.benchmark_id)
    digest = hashlib.sha256(f"{case_id}:{n_ph_max}".encode("utf-8")).hexdigest()
    energy = -float(int(digest[:8], 16) % 1_000_000) / 100_000.0
    return energy, digest[:24], {"case_id": case_id, "n_ph_max": int(n_ph_max)}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("paper_i_scaling_matrix")
    output_dir = root / "input" / "paper_i_scaling_matrix_test_v1"
    submit_path = root / "submit_paper_i_scaling_matrix_test_v1.sub"
    manifest = generator.generate(
        batch_id="paper_i_scaling_matrix_test_v1",
        output_dir=output_dir,
        submit_path=submit_path,
        spin_boson_horizon=37,
        bose_hubbard_horizon=41,
        exact_energy_resolver=_fake_exact_energy,
    )
    with (output_dir / "paper_i_scaling_matrix_records.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return manifest, output_dir, submit_path, rows


def test_generator_emits_exact_34_by_3_matrix_with_locked_horizons(generated) -> None:
    manifest, _output_dir, _submit_path, rows = generated
    assert manifest["physical_case_count"] == 34
    assert manifest["method_count"] == 3
    assert manifest["record_count"] == 102
    assert len(rows) == 102
    assert Counter(row["family"] for row in rows) == {
        "hh": 36,
        "hubbard": 18,
        "spin_boson": 24,
        "bose_hubbard": 24,
    }
    assert Counter(row["method_key"] for row in rows) == {"snake": 34, "geo": 34, "append": 34}
    assert all(
        row["n_ph_work"] == row["n_ph_ref"] == row["exact_reference_n_ph_max"] == ""
        for row in rows
        if row["family"] == "hubbard"
    )

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    assert len(by_case) == 34
    assert all({row["method_key"] for row in case_rows} == {"snake", "geo", "append"} for case_rows in by_case.values())
    for row in rows:
        L = int(row["L"])
        horizon = int(row["expected_horizon"])
        if row["family"] == "hh":
            assert horizon == 50
        elif row["family"] == "hubbard":
            assert horizon == (20 if L == 2 else 30)
        elif row["family"] == "spin_boson":
            assert horizon == 37
        else:
            assert horizon == 41
        assert row["max_depth"] == row["phase3_adapt_max_depth"] == str(horizon)


def test_generator_uses_parent_only_powell_and_method_specific_repeat_policy(generated) -> None:
    _manifest, _output_dir, _submit_path, rows = generated
    for row in rows:
        assert row["optimizer"] == "POWELL"
        assert row["adapt_optimizer_kind"] == "powell"
        assert row["budget"] == "200"
        assert row["phase3_refit_maxiter"] == "200"
        assert row["phase3_final_maxiter"] == "200"
        assert row["pool_contract"] == "full_meta_unfiltered"
        assert row["child_policy"] == "macro_only"
        assert row["generic_adapt_runtime_split_mode"] == "off"
        assert row["snake_phase3_runtime_split_mode"] == "off"
        assert row["shared_pauli_pool_mode"] == "off"
        assert row["phase2_batching"] == row["phase3_batching"] == "off"
        assert row["one_accepted_parent_per_outer_iteration"] == "true"
        assert row["resource_qubit_cap"] == "16"
        assert row["resource_pool_term_cap"] == "1024"
        assert row["exact_fidelity_max_qubits"] == "10"
        if row["method_key"] == "snake":
            assert row["adapt_allow_repeats"] == "true"
            assert row["generic_adapt_stop_policy"] == ""
            assert row["request_cpus"] == "4"
            assert row["adapt_parallel_gradient_workers"] == "2"
            assert row["adapt_beam_parent_workers"] == "2"
            assert row["phase3_adapt_parallel_gradient_workers"] == "2"
            assert row["phase3_adapt_beam_parent_workers"] == "2"
        else:
            assert row["adapt_allow_repeats"] == "true"
            assert row["generic_adapt_stop_policy"] == "fixed_horizon_no_target_v1"
            assert row["request_cpus"] == "1"
            assert row["adapt_parallel_gradient_workers"] == "not_applicable"
            assert row["adapt_beam_parent_workers"] == "not_applicable"
        if row["family"] == "hh":
            assert row["hh_pool_cache_mode"] == "disk"
            assert row["hh_pool_cache_scope"] == "exact"
            assert row["hh_generator_registry_cache_mode"] == "disk"
            assert row["hh_generator_registry_cache_required"] == "true"


def test_exact_energy_manifest_is_shared_by_all_three_methods(generated) -> None:
    manifest, output_dir, _submit_path, rows = generated
    exact_path = output_dir / "exact_energy_manifest.json"
    payload = json.loads(exact_path.read_text(encoding="utf-8"))
    assert payload["status"] == "pass"
    assert payload["record_count"] == 34
    assert manifest["exact_energy_manifest_sha256"] == hashlib.sha256(exact_path.read_bytes()).hexdigest()
    for row in rows:
        exact = payload["records"][row["case_id"]]
        assert float(row["same_cutoff_exact_gs_energy"]) == pytest.approx(exact["exact_energy"], abs=0.0)
        assert float(row["exact_reference_energy"]) == pytest.approx(exact["exact_energy"], abs=0.0)
        assert row["exact_energy_key"] == exact["key_hash"]
        assert row["exact_energy_method"] == "same_cutoff_exact_diagonalization"
        if row["family"] == "hubbard":
            assert exact["n_ph_work"] is None
            assert exact["n_ph_applicability"] == "not_applicable_nonbosonic"
            assert exact["resolver_n_ph_max"] is None
            assert exact["compatibility_call_n_ph_max"] == 1


def test_snake_policy_matches_current_forward_controls_except_parent_only(generated) -> None:
    _manifest, output_dir, _submit_path, _rows = generated
    policy = json.loads((output_dir / "paper_i_scaling_matrix_snake_policy.json").read_text(encoding="utf-8"))
    static = policy["static"]
    inner = policy["inner_optimizer"]
    assert policy["pool"]["pool_key"] == "full_meta"
    assert static["static_meta_feature_profile"] == "paper_i_production_v1"
    assert static["static_route_id"] == "route_a"
    assert static["static_lane_route"] == "physical_operator_type"
    assert static["adapt_reopt_policy"] == "full"
    assert static["adapt_insertion_mode"] == "full_commutation_reduced"
    assert static["adapt_full_refit_every"] == 1
    assert static["adapt_final_full_refit"] is True
    assert static["adapt_allow_repeats"] is True
    assert static["adapt_parallel_gradient_workers"] == 2
    assert static["adapt_beam_parent_workers"] == 2
    assert static["adapt_eps_grad"] == 0.0
    assert static["adapt_eps_energy"] == 0.0
    assert static["phase2_enable_batching"] is False
    assert static["phase3_enable_batching"] is False
    assert static["phase3_runtime_split_mode"] == "off"
    assert static["shared_pauli_pool_mode"] == "off"
    assert inner["inner_optimizer"] == inner["final_optimizer_type"] == "POWELL"
    assert inner["refit_maxiter"] == inner["final_maxiter"] == 200


def test_runner_environment_keeps_cache_out_of_transferred_output(generated, tmp_path: Path) -> None:
    _manifest, _output_dir, _submit_path, rows = generated
    snake = next(row for row in rows if row["method_key"] == "snake" and row["family"] == "hubbard")
    geo = next(row for row in rows if row["method_key"] == "geo" and row["family"] == "spin_boson")
    hh = next(row for row in rows if row["method_key"] == "snake" and row["family"] == "hh")
    output_root = tmp_path / "cell"
    runner.validate_record(snake)
    _env, snake_overlay = runner.build_environment(snake, output_root)
    assert snake_overlay["PHASE3_POLICY_INNER_OPTIMIZER"] == "POWELL"
    assert snake_overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS"] == "true"
    assert snake_overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_PARALLEL_GRADIENT_WORKERS"] == "2"
    assert snake_overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_BEAM_PARENT_WORKERS"] == "2"
    assert "GENERIC_STATIC_TABLE_ADAPT_OPTIMIZER_KIND" not in snake_overlay
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE" not in snake_overlay
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY" not in snake_overlay
    assert "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE" not in snake_overlay
    cache_path = Path(snake_overlay["STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR"])
    assert not cache_path.is_relative_to(output_root)
    assert snake_overlay["GENERIC_STATIC_TABLE_SAME_CUTOFF_EXACT_GS_ENERGY"] == snake["same_cutoff_exact_gs_energy"]

    runner.validate_record(geo)
    _env, geo_overlay = runner.build_environment(geo, output_root)
    assert geo_overlay["GENERIC_STATIC_TABLE_ADAPT_OPTIMIZER_KIND"] == "powell"
    assert geo_overlay["GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS"] == "true"
    assert geo_overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY"] == "fixed_horizon_no_target_v1"
    assert geo_overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE"] == "off"
    assert geo_overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY"] == "off"
    assert geo_overlay["GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE"] == "1"

    runner.validate_record(hh)
    _env, hh_overlay = runner.build_environment(hh, output_root)
    assert hh_overlay["STATIC_ADAPT_HH_POOL_CACHE"] == "disk"
    assert hh_overlay["STATIC_ADAPT_HH_POOL_CACHE_SCOPE"] == "exact"
    assert hh_overlay["STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE"] == "disk"
    assert Path(hh_overlay["STATIC_ADAPT_HH_GENERATOR_REGISTRY_CACHE_DIR"]).is_dir()


def test_scaling_runner_exports_powell_cap_policy_only_for_append_repair_row(
    generated,
    tmp_path: Path,
) -> None:
    _manifest, _output_dir, _submit_path, rows = generated
    append = dict(next(row for row in rows if row["method_key"] == "append" and row["family"] == "hubbard"))
    append["powell_maxiter_cap_policy"] = "accept_finite_nonincreasing_v1"

    runner.validate_record(append)
    _env, overlay = runner.build_environment(append, tmp_path / "append_cap")
    assert overlay["GENERIC_STATIC_TABLE_POWELL_MAXITER_CAP_POLICY"] == (
        "accept_finite_nonincreasing_v1"
    )

    geo = dict(next(row for row in rows if row["method_key"] == "geo" and row["family"] == "hubbard"))
    geo["powell_maxiter_cap_policy"] = "accept_finite_nonincreasing_v1"
    with pytest.raises(ValueError, match="restricted to append-only"):
        runner.validate_record(geo)


def test_native_snake_environment_does_not_request_generic_split_overlay(
    generated,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pipelines.exact_bench import generic_static_benchmark as benchmark

    _manifest, _output_dir, _submit_path, rows = generated
    snake = next(row for row in rows if row["method_key"] == "snake")
    _env, overlay = runner.build_environment(snake, tmp_path / "snake")
    for field in (
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY",
        "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE",
    ):
        monkeypatch.delenv(field, raising=False)
    for name, value in overlay.items():
        monkeypatch.setenv(name, value)
    overrides = benchmark._generic_adapt_runtime_split_overrides_from_env()
    assert overrides == {}
    assert benchmark._generic_adapt_runtime_split_overrides_requested(overrides) is False


def test_submit_uses_minimal_batch_scoped_bundle_and_no_old_input_tree(generated) -> None:
    manifest, output_dir, submit_path, _rows = generated
    contract = run_task.parse_submit_contract(submit_path)
    transfers = contract["transfer_input_files"]
    assert set(transfers) == {
        "chtc/phase3_optuna/image.sif",
        "chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh",
        str(output_dir.resolve()),
    }
    blockers = preflight_submit._paper_i_scaling_transfer_blockers(
        transfers,
        records_rel=str((output_dir / "paper_i_scaling_matrix_records.tsv").resolve()),
    )
    assert blockers == []
    assert contract["transfer_output_files"] == [
        "raw_outputs/paper_i_scaling_matrix_test_v1/$(record_id)"
    ]
    assert contract["request_cpus"] == "$(cpus)"

    bundle = Path(manifest["code_bundle"]["path"])
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == manifest["code_bundle"]["sha256"]
    with tarfile.open(bundle, "r:gz") as archive:
        names = set(archive.getnames())
    assert "chtc/phase3_optuna/run_paper_i_scaling_matrix_cell.py" in names
    assert any(name.startswith("pipelines/exact_bench") for name in names)
    assert "pipelines/exact_bench/static_reference_metrics.py" in names
    assert any(name.startswith("src/quantum") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)

    lock_path = Path(manifest["implementation_lock"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert manifest["implementation_lock_sha256"] == hashlib.sha256(lock_path.read_bytes()).hexdigest()
    entries = {entry["path"]: entry for entry in lock["entries"]}
    assert entries["pipelines/exact_bench/static_reference_metrics.py"]["critical_bundle_member"] is True
    assert all(
        entry["bundle_member_sha256"] == entry["sha256"]
        for entry in entries.values()
        if entry["critical_bundle_member"]
    )


def test_preflight_contract_accepts_generated_rows_and_rejects_policy_drift(generated) -> None:
    _manifest, _output_dir, _submit_path, rows = generated
    assert all(preflight_submit._is_paper_i_scaling_matrix(row) for row in rows)
    assert all(preflight_submit._paper_i_scaling_matrix_contract_blockers(row) == [] for row in rows)
    broken = dict(rows[0])
    broken["shared_pauli_pool_mode"] = "shared_pauli_child_sets_v1"
    assert "paper_i_scaling_shared_pauli_pool_mode_mismatch:shared_pauli_child_sets_v1:expected:off" in (
        preflight_submit._paper_i_scaling_matrix_contract_blockers(broken)
    )


def test_preflight_accepts_only_complete_34_case_snake_overlay_repair(generated) -> None:
    _manifest, _output_dir, _submit_path, rows = generated
    repair_scope = "snake_only_all_34_physical_cases_overlay_plumbing_v1"
    repair_rows = []
    for row in rows:
        if row["method_key"] != "snake":
            continue
        repaired = dict(row)
        repaired["batch_id"] = "paper_i_scaling_matrix_snake_overlay_repair_test_v1"
        repaired["record_id"] = repaired["record_id"].replace(
            "paper_i_scaling_matrix_test_v1",
            "paper_i_scaling_matrix_snake_overlay_repair_test_v1",
            1,
        )
        repaired["repair_scope"] = repair_scope
        repaired["repair_source_batch_id"] = "paper_i_scaling_matrix_test_v1"
        repaired["repair_source_record_id"] = row["record_id"]
        repair_rows.append(repaired)
    assert len(repair_rows) == 34
    assert preflight_submit._paper_i_scaling_matrix_bundle_blockers(repair_rows) == []
    assert all(preflight_submit._paper_i_scaling_matrix_contract_blockers(row) == [] for row in repair_rows)

    incomplete = repair_rows[:-1]
    assert any(
        blocker.startswith("paper_i_scaling_matrix_row_count_mismatch:33:expected:34")
        for blocker in preflight_submit._paper_i_scaling_matrix_bundle_blockers(incomplete)
    )

    contaminated = [*repair_rows, dict(next(row for row in rows if row["method_key"] == "geo"))]
    assert preflight_submit._paper_i_scaling_matrix_bundle_blockers(contaminated)

    typo = dict(rows[0])
    typo["suite_profile"] = "paper_i_scaling_matrix_typo"
    assert preflight_submit._is_paper_i_scaling_matrix(typo)
    assert any(
        reason.startswith("paper_i_scaling_suite_profile_mismatch")
        for reason in preflight_submit._paper_i_scaling_matrix_contract_blockers(typo)
    )


def test_preflight_validates_full_matrix_and_effective_submit_contract(generated) -> None:
    _manifest, output_dir, submit_path, rows = generated
    assert preflight_submit._paper_i_scaling_matrix_bundle_blockers(rows) == []
    incomplete = rows[:-1]
    blockers = preflight_submit._paper_i_scaling_matrix_bundle_blockers(incomplete)
    assert any(reason.startswith("paper_i_scaling_matrix_row_count_mismatch") for reason in blockers)
    assert any(reason.startswith("paper_i_scaling_matrix_case_method_set_mismatch") for reason in blockers)

    cpu_drift = [dict(row) for row in rows]
    snake_index = next(index for index, row in enumerate(cpu_drift) if row["method_key"] == "snake")
    cpu_drift[snake_index]["request_cpus"] = "1"
    assert any(
        reason.startswith("paper_i_scaling_matrix_cpu_counts_mismatch")
        for reason in preflight_submit._paper_i_scaling_matrix_bundle_blockers(cpu_drift)
    )

    contract = run_task.parse_submit_contract(submit_path)
    records_rel = str((output_dir / "paper_i_scaling_matrix_records.tsv").resolve())
    assert preflight_submit._paper_i_scaling_submit_contract_blockers(
        contract,
        rows,
        records_rel=records_rel,
    ) == []
    bad_output = dict(contract)
    bad_output["transfer_output_files"] = ["raw_outputs/wrong/$(record_id)"]
    assert any(
        reason.startswith("paper_i_scaling_submit_output_route_mismatch")
        for reason in preflight_submit._paper_i_scaling_submit_contract_blockers(
            bad_output,
            rows,
            records_rel=records_rel,
        )
    )
    bad_cpus = dict(contract)
    bad_cpus["request_cpus"] = "4"
    assert any(
        reason.startswith("paper_i_scaling_submit_request_cpus_mismatch")
        for reason in preflight_submit._paper_i_scaling_submit_contract_blockers(
            bad_cpus,
            rows,
            records_rel=records_rel,
        )
    )


def test_preflight_hashes_and_parses_snake_policy(generated, tmp_path: Path) -> None:
    _manifest, _output_dir, _submit_path, rows = generated
    snake = next(row for row in rows if row["method_key"] == "snake")
    assert preflight_submit._paper_i_scaling_snake_policy_blockers(snake) == []

    source = Path(snake["phase3_policy_json"])
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["static"]["phase3_runtime_split_mode"] = "shortlist_pauli_children_v1"
    tampered = tmp_path / "tampered_policy.json"
    tampered.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    broken = dict(snake)
    broken["phase3_policy_json"] = str(tampered)
    broken["phase3_policy_json_sha256"] = hashlib.sha256(tampered.read_bytes()).hexdigest()
    blockers = preflight_submit._paper_i_scaling_snake_policy_blockers(broken)
    assert any("phase3_runtime_split_mode_mismatch" in reason for reason in blockers)


def test_preflight_binds_implementation_lock_to_bundle_members(generated) -> None:
    _manifest, _output_dir, _submit_path, rows = generated
    assert preflight_submit._paper_i_scaling_implementation_lock_blockers(rows[0]) == []


def test_preflight_detects_queue_resource_tampering(generated, tmp_path: Path) -> None:
    _manifest, output_dir, submit_path, rows = generated
    source_queue = output_dir / "paper_i_scaling_matrix_record_queue.tsv"
    lines = source_queue.read_text(encoding="utf-8").splitlines()
    first = lines[0].split("\t")
    first[2] = "1"
    lines[0] = "\t".join(first)
    tampered_queue = tmp_path / "tampered_queue.tsv"
    tampered_queue.write_text("\n".join(lines) + "\n", encoding="utf-8")
    contract = run_task.parse_submit_contract(submit_path)
    contract["queue_record_id_file"] = str(tampered_queue)
    blockers = preflight_submit._paper_i_scaling_submit_contract_blockers(
        contract,
        rows,
        records_rel=str((output_dir / "paper_i_scaling_matrix_records.tsv").resolve()),
    )
    assert any(reason.startswith("paper_i_scaling_queue_resource_mismatch") for reason in blockers)


def test_preflight_detects_queue_cpu_tampering(generated, tmp_path: Path) -> None:
    _manifest, output_dir, submit_path, rows = generated
    source_queue = output_dir / "paper_i_scaling_matrix_record_queue.tsv"
    lines = source_queue.read_text(encoding="utf-8").splitlines()
    first = lines[0].split("\t")
    assert first[1] == "4"
    first[1] = "1"
    lines[0] = "\t".join(first)
    tampered_queue = tmp_path / "tampered_cpu_queue.tsv"
    tampered_queue.write_text("\n".join(lines) + "\n", encoding="utf-8")
    contract = run_task.parse_submit_contract(submit_path)
    contract["queue_record_id_file"] = str(tampered_queue)
    blockers = preflight_submit._paper_i_scaling_submit_contract_blockers(
        contract,
        rows,
        records_rel=str((output_dir / "paper_i_scaling_matrix_records.tsv").resolve()),
    )
    assert any(reason.startswith("paper_i_scaling_queue_resource_mismatch") for reason in blockers)


def test_seed_cache_copy_records_hash_provenance(tmp_path: Path) -> None:
    seed = tmp_path / "seed"
    target = tmp_path / "target"
    seed.mkdir()
    target.mkdir()
    (seed / "a.pickle").write_bytes(b"a")
    (seed / "b.pickle").write_bytes(b"bb")
    (seed / "ignore.json").write_text("{}", encoding="utf-8")
    payload = prewarm._seed_batch_cache(target, seed)
    assert payload["status"] == "copied"
    assert payload["copied_file_count"] == 2
    assert len(payload["seed_cache_manifest_sha256"]) == 64
    assert sorted(path.name for path in target.iterdir()) == ["a.pickle", "b.pickle"]


def test_dual_cache_preflight_requires_exact_manifested_file_sets(tmp_path: Path) -> None:
    pool_dir = tmp_path / "pool"
    registry_dir = tmp_path / "registry"
    pool_dir.mkdir()
    registry_dir.mkdir()

    def rows_for(directory: Path, prefix: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index in range(12):
            path = directory / f"{prefix}{index:02d}.pickle"
            path.write_bytes(f"{prefix}:{index}".encode("utf-8"))
            rows.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        return rows

    pool_files = rows_for(pool_dir, "p")
    registry_files = rows_for(registry_dir, "r")
    verification = [
        {
            "case_id": f"hh_case_{index}",
            "pool_cache_path": pool_files[index]["path"],
            "generator_registry_cache_path": registry_files[index]["path"],
            "pool_cache_disk_hit_verified": True,
            "generator_registry_cache_disk_hit_verified": True,
        }
        for index in range(12)
    ]
    payload = {
        "schema": "paper_i_scaling_matrix_hh_dual_cache_prewarm_v1",
        "status": "pass",
        "case_count": 12,
        "pool_cache": {
            "mode": "disk",
            "scope": "exact",
            "cache_dir": str(pool_dir),
            "file_count": 12,
            "files": pool_files,
        },
        "generator_registry_cache": {
            "mode": "disk",
            "cache_dir": str(registry_dir),
            "file_count": 12,
            "files": registry_files,
        },
        "disk_hit_verification": verification,
        "total_size_bytes": sum(int(item["size_bytes"]) for item in pool_files + registry_files),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    row = {
        "hh_pool_cache_manifest": str(manifest),
        "hh_pool_cache_dir": str(pool_dir),
        "hh_generator_registry_cache_dir": str(registry_dir),
    }
    preflight_submit._SCALING_HH_CACHE_VALIDATION_MEMO.clear()
    assert preflight_submit._paper_i_scaling_hh_cache_blockers(row) == []

    (pool_dir / "extra.pickle").write_bytes(b"extra")
    preflight_submit._SCALING_HH_CACHE_VALIDATION_MEMO.clear()
    blockers = preflight_submit._paper_i_scaling_hh_cache_blockers(row)
    assert any("pool_cache_directory_set_mismatch" in reason for reason in blockers)


def test_preflight_cli_is_read_only_without_explicit_status_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    records = batch_dir / "paper_i_scaling_matrix_records.tsv"
    records.write_text("record_id\n", encoding="utf-8")
    sentinel = {"record_count": 102, "status": "frozen_sentinel"}
    status_paths = [
        batch_dir / "paper_i_scaling_matrix_manifest.json",
        batch_dir / "submission_audit.json",
    ]
    for path in status_paths:
        path.write_text(json.dumps(sentinel, sort_keys=True) + "\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in status_paths}
    payload = {
        "schema": "phase3_chtc_submit_preflight_bundle_v1",
        "submit_path": str(tmp_path / "submit.sub"),
        "records_path": str(records),
        "record_ids": [f"paper_i_scaling_matrix_test_{index}" for index in range(102)],
        "status": "pass",
        "ok": True,
        "record_count": 102,
        "failed_record_count": 0,
        "blocking_reasons": [],
        "records": [],
    }
    monkeypatch.setattr(preflight_submit, "build_preflight_bundle", lambda **_kwargs: dict(payload))
    output = tmp_path / "external_preflight.json"
    assert preflight_submit.main(["--submit", str(tmp_path / "submit.sub"), "--output-json", str(output)]) == 0
    assert output.is_file()
    assert all(path.read_bytes() == before[path] for path in status_paths)


def test_explicit_scaling_status_update_requires_portable_staged_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    batch_dir = repo_root / "input" / "paper_i_scaling_matrix_test_v1"
    batch_dir.mkdir(parents=True)
    records = batch_dir / "paper_i_scaling_matrix_records.tsv"
    records.write_text("record_id\n", encoding="utf-8")
    status_paths = [
        batch_dir / "paper_i_scaling_matrix_manifest.json",
        batch_dir / "submission_audit.json",
    ]
    for path in status_paths:
        path.write_text(json.dumps({"record_count": 102, "status": "ready_for_preflight"}) + "\n", encoding="utf-8")
    payload = {
        "records_path": str(records),
        "record_ids": [f"paper_i_scaling_matrix_test_{index}" for index in range(102)],
        "ok": True,
        "record_count": 102,
    }

    outside = repo_root / "external_preflight.json"
    outside.write_text("{}\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in status_paths}
    with pytest.raises(ValueError, match="staged batch preflight path"):
        preflight_submit._mark_complete_scaling_preflight_pass(
            payload,
            output_json=outside,
            repo_root=repo_root,
        )
    assert all(path.read_bytes() == before[path] for path in status_paths)

    staged = batch_dir / "preflight.json"
    staged.write_text("{}\n", encoding="utf-8")
    preflight_submit._mark_complete_scaling_preflight_pass(
        payload,
        output_json=staged,
        repo_root=repo_root,
    )
    for path in status_paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        assert artifact["status"] == "preflight_pass"
        assert artifact["preflight"]["path"] == (
            "input/paper_i_scaling_matrix_test_v1/preflight.json"
        )
        assert not Path(artifact["preflight"]["path"]).is_absolute()
