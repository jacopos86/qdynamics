#!/usr/bin/env python3
"""Build the source-locked Append-S weak--strong warm continuation bundle.

The bundle starts from the validated k=50 projected-singleton Append-ADAPT
state, preserves its frozen source and all scientific settings, and changes
only the user-authorized warm-start state, horizon, and first-hit stop rule.
It is intentionally emitted with Condor requirements=False until the narrow
stop-policy extension is reviewed and an exact remote-image preflight passes.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence


BUNDLE_ID = (
    "paper_i_hh_append_projected_singleton_weak_strong_warm_continuation_"
    "k50_to_firsthit_or_k80_20260721_v1_chtc"
)
BUNDLE_CREATED_UTC = "2026-07-21T21:23:38+00:00"
BATCH_NAME = "paper-i-hh-append-proj-singleton-ws-k50-firsthit2e4-k80-v1"
JOB_ID = "append_projected_singleton__weak_strong__warm_k50_firsthit2e4_k80"
BASE_BUNDLE_ID = "paper_i_hh_append_projected_singleton_all_six_r50_20260719_v4_chtc"
BASE_JOB_ID = "append_projected_singleton__weak_strong__r50"
BASE_SOURCE_SHA256 = "8922435b176d635544f6fa2629da05ea7151f457e584c39e47a2ee161de94ecd"
BASE_GENERIC_SHA256 = "33f0e8ffba1d532e86037077e99d8578423fc7a52842e2479a40afea1588ed3d"
BASE_TEST_SHA256 = "790fd3bea888c444883d7677ba0418b2112781fec65e8283fe6a88a9935d1c19"
BASE_TRANSFER_SHA256 = "e6be4d4418d116d05a668c2dbfda6a13523ae7893bb555e003883b64739ee467"
BASE_RESULT_MEMBER_SHA256 = "38531e74e1beb25b11e497b1ed23cae0cca4c820327c19e1da21b2d8a5091a6e"
BASE_RESULT_MEMBER_SIZE = 5_199_975_930
BASE_S_ALG = 1_276_060.0
BASE_DEPTH = 50
BASE_ITERATIONS = 50
BASE_TERMINAL_ERROR = 0.0006059547584833513
BASE_TERMINAL_ENERGY = -1.138114683316429
TARGET_ERROR = 2.0e-4
MAX_TOTAL_ITERATIONS = 80
EXPECTED_IMAGE_SHA256 = "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
STOP_POLICY = "append_warm_start_first_hit_or_max_iterations_v1"
CAP_POLICY = "accept_finite_nonincreasing_v1"
GENERIC_REL = Path("pipelines/exact_bench/generic_static_adapt_variants.py")
TEST_REL = Path("test/test_generic_static_adapt_variants.py")


def _utc_now() -> str:
    return BUNDLE_CREATED_UTC


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _bundle_dir() -> Path:
    return Path(__file__).resolve().parent


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _deterministic_archive(source_root: Path, output: Path) -> None:
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(source_root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(source_root).as_posix()
                    info = tar.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} patch anchor count is {count}, expected 1")
    return text.replace(old, new, 1)


def _patch_generic_source(text: str) -> str:
    text = _replace_once(
        text,
        '_FIXED_HORIZON_NO_TARGET_STOP_POLICY = "fixed_horizon_no_target_v1"\n',
        '_FIXED_HORIZON_NO_TARGET_STOP_POLICY = "fixed_horizon_no_target_v1"\n'
        '_APPEND_WARM_START_FIRST_HIT_OR_MAX_ITERATIONS_STOP_POLICY = (\n'
        '    "append_warm_start_first_hit_or_max_iterations_v1"\n'
        ')\n',
        "stop-policy constant",
    )
    old_guard = '''    benchmark_energy_stop_target = _normalize_energy_stop_target(energy_stop_target)
    benchmark_first_hit_thresholds = _normalize_first_hit_thresholds(first_hit_thresholds)
    generic_adapt_stop_policy_label = str(generic_adapt_stop_policy or "").strip() or None
    powell_cap_policy = _normalize_powell_maxiter_cap_policy(powell_maxiter_cap_policy)
    if powell_cap_policy == _POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING:
        if config.algorithm_id != STATIC_FULL_META_APPEND_ADAPT_VQE:
            raise ValueError(
                "accept_finite_nonincreasing_v1 is restricted to append-only ADAPT repair rows"
            )
        if config.optimizer_kind != "powell":
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires the Powell inner optimizer"
            )
        if generic_adapt_stop_policy_label != _FIXED_HORIZON_NO_TARGET_STOP_POLICY:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires fixed_horizon_no_target_v1"
            )
        if benchmark_energy_stop_target is not None:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires energy_stop_target to be absent; "
                "the repaired row must continue to the fixed outer horizon"
            )
        if float(gradient_threshold) != 0.0:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires gradient_threshold=0 so gradient "
                "rules cannot terminate the repaired fixed-horizon row early"
            )
'''
    new_guard = '''    benchmark_energy_stop_target = _normalize_energy_stop_target(energy_stop_target)
    benchmark_first_hit_thresholds = _normalize_first_hit_thresholds(first_hit_thresholds)
    generic_adapt_stop_policy_label = str(generic_adapt_stop_policy or "").strip() or None
    powell_cap_policy = _normalize_powell_maxiter_cap_policy(powell_maxiter_cap_policy)
    warm_target_policy = (
        generic_adapt_stop_policy_label
        == _APPEND_WARM_START_FIRST_HIT_OR_MAX_ITERATIONS_STOP_POLICY
    )
    if warm_target_policy:
        if powell_cap_policy != _POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING:
            raise ValueError(
                "append warm-start first-hit policy requires accept_finite_nonincreasing_v1"
            )
        if config.algorithm_id != STATIC_FULL_META_APPEND_ADAPT_VQE:
            raise ValueError(
                "append warm-start first-hit policy is restricted to append-only ADAPT"
            )
        if config.optimizer_kind != "powell":
            raise ValueError("append warm-start first-hit policy requires Powell")
        warm_inputs = (
            initial_selected_operator_labels,
            initial_selected_operator_batches,
            initial_theta,
            initial_adapt_history,
        )
        if any(value is None for value in warm_inputs):
            raise ValueError(
                "append warm-start first-hit policy requires labels, batches, theta, and history"
            )
        warm_labels = list(initial_selected_operator_labels or ())
        warm_batches = [list(batch) for batch in (initial_selected_operator_batches or ())]
        warm_theta = list(initial_theta or ())
        warm_history = list(initial_adapt_history or ())
        if not warm_history:
            raise ValueError("append warm-start first-hit policy requires a nonempty prefix")
        if not (
            len(warm_labels)
            == len(warm_batches)
            == len(warm_theta)
            == len(warm_history)
        ):
            raise ValueError(
                "append warm-start first-hit prefix labels/batches/theta/history must align"
            )
        if any(len(batch) != 1 for batch in warm_batches):
            raise ValueError("append warm-start first-hit prefix requires singleton batches")
        if [str(batch[0]) for batch in warm_batches] != [str(label) for label in warm_labels]:
            raise ValueError("append warm-start first-hit batches must flatten to labels")
        if int(max_adapt_iterations) <= len(warm_history):
            raise ValueError("append warm-start first-hit horizon must exceed the prefix")
        if benchmark_energy_stop_target is None:
            raise ValueError("append warm-start first-hit policy requires energy_stop_target")
        if not any(
            math.isclose(
                float(threshold),
                float(benchmark_energy_stop_target),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for threshold in benchmark_first_hit_thresholds
        ):
            raise ValueError(
                "append warm-start first-hit thresholds must include energy_stop_target"
            )
    if powell_cap_policy == _POWELL_MAXITER_CAP_POLICY_ACCEPT_FINITE_NONINCREASING:
        if config.algorithm_id != STATIC_FULL_META_APPEND_ADAPT_VQE:
            raise ValueError(
                "accept_finite_nonincreasing_v1 is restricted to append-only ADAPT repair rows"
            )
        if config.optimizer_kind != "powell":
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires the Powell inner optimizer"
            )
        if generic_adapt_stop_policy_label not in {
            _FIXED_HORIZON_NO_TARGET_STOP_POLICY,
            _APPEND_WARM_START_FIRST_HIT_OR_MAX_ITERATIONS_STOP_POLICY,
        }:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires fixed_horizon_no_target_v1 or "
                "append_warm_start_first_hit_or_max_iterations_v1"
            )
        if (
            generic_adapt_stop_policy_label == _FIXED_HORIZON_NO_TARGET_STOP_POLICY
            and benchmark_energy_stop_target is not None
        ):
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires energy_stop_target to be absent; "
                "the repaired row must continue to the fixed outer horizon"
            )
        if float(gradient_threshold) != 0.0:
            raise ValueError(
                "accept_finite_nonincreasing_v1 requires gradient_threshold=0 so gradient "
                "rules cannot terminate the selected outer-horizon policy early"
            )
'''
    text = _replace_once(text, old_guard, new_guard, "Powell/stop contract")
    old_target = '''    target_stop_enabled = (
        benchmark_energy_stop_target is not None
        and _finite_float_or_none(reference_metrics.get("primary_reference_energy")) is not None
    )

    selected: list[_PoolCandidate] = []
'''
    new_target = '''    target_stop_enabled = (
        benchmark_energy_stop_target is not None
        and _finite_float_or_none(reference_metrics.get("primary_reference_energy")) is not None
    )
    if warm_target_policy and not target_stop_enabled:
        raise ValueError(
            "append warm-start first-hit policy requires a finite primary reference energy"
        )

    selected: list[_PoolCandidate] = []
'''
    text = _replace_once(text, old_target, new_target, "finite reference gate")
    old_output = '''            "adapt_target_stop_policy": (
                "fixed_iteration_horizon"
                if generic_adapt_stop_policy_label == _FIXED_HORIZON_NO_TARGET_STOP_POLICY
                else
                "first_hit_or_max_depth"
                if target_stop_enabled
                else "gradient_threshold_or_pool_exhaustion"
            ),
'''
    new_output = '''            "adapt_target_stop_policy": (
                "fixed_iteration_horizon"
                if generic_adapt_stop_policy_label == _FIXED_HORIZON_NO_TARGET_STOP_POLICY
                else
                "warm_start_first_hit_or_max_iterations"
                if generic_adapt_stop_policy_label
                == _APPEND_WARM_START_FIRST_HIT_OR_MAX_ITERATIONS_STOP_POLICY
                else
                "first_hit_or_max_depth"
                if target_stop_enabled
                else "gradient_threshold_or_pool_exhaustion"
            ),
'''
    return _replace_once(text, old_output, new_output, "output stop-policy label")


def _patch_frozen_tests(text: str) -> str:
    fixed_positive_tail = '''    assert all(
        item["optimizer_cap_acceptance_reason"]
        == "finite_nonincreasing_powell_maxiter_accepted"
        for item in row["adapt_history"]
    )
'''
    warm_positive_tail = fixed_positive_tail + '''

    warm_payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "accepted_cap_warm_first_hit",
        max_adapt_iterations=4,
        optimizer_maxiter=2,
        gradient_threshold=0.0,
        energy_stop_target=10.0,
        first_hit_thresholds=(10.0,),
        same_cutoff_exact_gs_energy=0.0,
        exact_reference_energy=0.0,
        generic_adapt_stop_policy="append_warm_start_first_hit_or_max_iterations_v1",
        powell_maxiter_cap_policy="accept_finite_nonincreasing_v1",
        initial_selected_operator_labels=row["selected_operators"],
        initial_selected_operator_batches=row["selected_operator_batches"],
        initial_theta=row["theta"],
        initial_adapt_history=row["adapt_history"],
    )
    warm_row = warm_payload["result"]
    assert warm_payload["status"] == "completed"
    assert warm_row["adapt_continuation_mode"] == "warm_start_selected_theta_v1"
    assert warm_row["adapt_warm_start_source_iterations"] == 3
    assert warm_row["adapt_warm_start_source_depth"] == 3
    assert warm_row["adapt_num_iterations"] == 4
    assert warm_row["adapt_stop_reason"] == "benchmark_abs_delta_e_target"
    assert warm_row["adapt_target_stop_policy"] == "warm_start_first_hit_or_max_iterations"
    assert warm_row["optimizer_capped_iterations"] == [3]
    assert warm_row["optimizer_capped_accepted_iterations"] == [3]
'''
    text = _replace_once(
        text,
        fixed_positive_tail,
        warm_positive_tail,
        "positive capped-Powell warm continuation test",
    )
    addition = r'''


def test_append_warm_first_hit_policy_contract_accepts_exact_prefix() -> None:
    policy = variants._APPEND_WARM_START_FIRST_HIT_OR_MAX_ITERATIONS_STOP_POLICY
    assert policy == "append_warm_start_first_hit_or_max_iterations_v1"


@pytest.mark.parametrize(
    ("labels", "batches", "theta", "history", "max_iterations", "target", "thresholds", "reason"),
    [
        (None, None, None, None, 80, 2.0e-4, (2.0e-4,), "requires labels"),
        ([], [], [], [], 80, 2.0e-4, (2.0e-4,), "nonempty prefix"),
        (["a"], [["a"]], [0.0], [{"iteration": 0}], 1, 2.0e-4, (2.0e-4,), "horizon"),
        (["a"], [["a"]], [0.0], [{"iteration": 0}], 80, None, (2.0e-4,), "energy_stop_target"),
        (["a"], [["a"]], [0.0], [{"iteration": 0}], 80, 2.0e-4, (1.0e-4,), "thresholds"),
    ],
)
def test_append_warm_first_hit_policy_rejects_incomplete_contract(
    monkeypatch,
    tmp_path: Path,
    labels,
    batches,
    theta,
    history,
    max_iterations,
    target,
    thresholds,
    reason,
) -> None:
    payload = variants.run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / reason.replace(" ", "_"),
        max_adapt_iterations=max_iterations,
        gradient_threshold=0.0,
        energy_stop_target=target,
        first_hit_thresholds=thresholds,
        generic_adapt_stop_policy="append_warm_start_first_hit_or_max_iterations_v1",
        powell_maxiter_cap_policy="accept_finite_nonincreasing_v1",
        initial_selected_operator_labels=labels,
        initial_selected_operator_batches=batches,
        initial_theta=theta,
        initial_adapt_history=history,
    )
    assert payload["status"] == "failed"
    assert reason in payload["reason"]
'''
    if "test_append_warm_first_hit_policy_contract_accepts_exact_prefix" in text:
        raise RuntimeError("warm continuation tests already present in frozen source")
    return text.rstrip() + addition + "\n"


def _freeze_source(repo: Path, bundle: Path) -> dict[str, Any]:
    base_archive = repo / "chtc" / "phase3_optuna" / "input" / BASE_BUNDLE_ID / "source_locked.tar.gz"
    if _sha256(base_archive) != BASE_SOURCE_SHA256:
        raise RuntimeError("frozen Append parent archive hash mismatch")
    with tempfile.TemporaryDirectory(prefix="append-warm-source-") as tmp_name:
        source = Path(tmp_name) / "source"
        source.mkdir(parents=True)
        with tarfile.open(base_archive, "r:gz") as tar:
            tar.extractall(source, filter="data")
        generic = source / GENERIC_REL
        frozen_test = source / TEST_REL
        if _sha256(generic) != BASE_GENERIC_SHA256:
            raise RuntimeError("frozen generic runner member hash mismatch")
        if _sha256(frozen_test) != BASE_TEST_SHA256:
            raise RuntimeError("frozen generic runner test member hash mismatch")
        original_generic = generic.read_text(encoding="utf-8")
        original_test = frozen_test.read_text(encoding="utf-8")
        generic.write_text(_patch_generic_source(original_generic), encoding="utf-8")
        frozen_test.write_text(_patch_frozen_tests(original_test), encoding="utf-8")
        output = bundle / "source_locked.tar.gz"
        _deterministic_archive(source, output)
        inventory = []
        for path in sorted(source.rglob("*")):
            if path.is_file():
                inventory.append(
                    {
                        "path": path.relative_to(source).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    payload = {
        "schema": "paper_i_source_locked_archive_manifest_v1",
        "created_utc": _utc_now(),
        "archive": output.name,
        "archive_sha256": _sha256(output),
        "parent_archive_sha256": BASE_SOURCE_SHA256,
        "construction": "exact_parent_archive_plus_two_surgical_member_patches_v1",
        "changed_members": [
            {
                "path": GENERIC_REL.as_posix(),
                "before_sha256": BASE_GENERIC_SHA256,
                "after_sha256": next(row["sha256"] for row in inventory if row["path"] == GENERIC_REL.as_posix()),
                "change": "explicit warm-continuation first-hit stop contract only",
            },
            {
                "path": TEST_REL.as_posix(),
                "before_sha256": BASE_TEST_SHA256,
                "after_sha256": next(row["sha256"] for row in inventory if row["path"] == TEST_REL.as_posix()),
                "change": "focused fail-closed stop-contract tests only",
            },
        ],
        "inventory": inventory,
    }
    _write_json(bundle / "source_archive_manifest.json", payload)
    return payload


def _read_members(archive: Path, wanted: Sequence[str]) -> dict[str, bytes]:
    wanted_set = set(wanted)
    found: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.name not in wanted_set:
                continue
            handle = tar.extractfile(member)
            if handle is None:
                raise RuntimeError(f"unable to extract {member.name}")
            found[member.name] = handle.read()
            if len(found) == len(wanted_set):
                break
    missing = wanted_set - set(found)
    if missing:
        raise RuntimeError(f"missing source evidence members: {sorted(missing)}")
    return found


def _compact_prefix_history(progress_bytes: bytes, trajectory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    complete = []
    scored = []
    for line in progress_bytes.decode("utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "iteration_complete":
            complete.append(row)
        elif row.get("event") == "iteration_scored":
            scored.append(row)
    if len(complete) != BASE_ITERATIONS:
        raise RuntimeError(f"expected 50 iteration-complete events, found {len(complete)}")
    if len(scored) != BASE_ITERATIONS:
        raise RuntimeError(f"expected 50 iteration-scored events, found {len(scored)}")
    if len(trajectory) != BASE_ITERATIONS:
        raise RuntimeError("tracking trajectory is not exactly 50 rounds")
    history: list[dict[str, Any]] = []
    for index, (score, row, tracked) in enumerate(zip(scored, complete, trajectory, strict=True)):
        error = float(row["abs_delta_e"])
        if (
            int(score["iteration"]) != index
            or int(row["iteration"]) != index
            or int(row["depth_after"]) != index + 1
        ):
            raise RuntimeError(f"prefix ordering failed at iteration {index}")
        if int(tracked["round"]) != index + 1 or float(tracked["error"]) != error:
            raise RuntimeError(f"prefix trajectory mismatch at iteration {index}")
        history.append(
            {
                "schema": "paper_i_append_warm_prefix_history_row_v1",
                "iteration": index,
                "history_position": index,
                "depth_before": index,
                "depth_after": index + 1,
                "appended_operator_count": 1,
                "appended_operator_labels": list(row["appended_batch_labels"]),
                "selected_candidate_labels": list(score["selected_candidate_labels"]),
                "selected_batch_labels": list(row["selected_batch_labels"]),
                "batch_size": int(row["batch_size"]),
                "energy_before": float(score["energy_before"]),
                "energy_after": float(row["energy_after"]),
                "abs_delta_e_after": error,
                "abs_delta_e_same_cutoff_after": error,
                "delta_E_abs_after": error,
                "primary_energy_metric_after": "same_cutoff_abs_delta_e",
                "max_abs_gradient": float(score["max_abs_gradient"]),
                "gradient_l2_norm": float(score["gradient_l2_norm"]),
                "best_selector_score": float(score["best_selector_score"]),
                "candidate_count_scored": int(score["candidate_count_scored"]),
                "optimizer": str(row["optimizer"]),
                "optimizer_success": bool(row["optimizer_success"]),
                "optimizer_raw_success": bool(row["optimizer_raw_success"]),
                "optimizer_capped": bool(row["optimizer_capped"]),
                "optimizer_capped_accepted": bool(row["optimizer_capped_accepted"]),
                "optimizer_cap_acceptance_reason": str(row["optimizer_cap_acceptance_reason"]),
                "optimizer_cap_policy": CAP_POLICY,
                "optimizer_nfev": int(row["optimizer_nfev"]),
                "optimizer_nit": row["optimizer_nit"],
                "prefix_nfev_total_after": int(row["nfev_total"]),
                "prefix_nit_total_after": int(row["nit_total"]),
                "eligible_for_first_hit": True,
                "estimator_call_round_receipt_included": False,
                "prefix_source": "validated_r50_iteration_progress_v1",
            }
        )
    return history


def _build_warm_state(repo: Path, bundle: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence_archive = repo / (
        "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/retrieval_20260720T131959Z/"
        "append_projected_singleton__weak_strong__r50_transfer.tar.gz"
    )
    tracking_path = repo / (
        "raw_outputs/chtc_fetch_paper_i_hh_sr_20260720/retrieval_20260720T131959Z/"
        "append_projected_singleton__weak_strong__r50_tracking_summary.json"
    )
    if _sha256(evidence_archive) != BASE_TRANSFER_SHA256:
        raise RuntimeError("validated r50 transfer archive hash mismatch")
    tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
    if tracking["result_member"]["sha256"] != BASE_RESULT_MEMBER_SHA256:
        raise RuntimeError("validated result-member hash mismatch")
    prefix = f"raw_outputs/{BASE_BUNDLE_ID}/{BASE_JOB_ID}"
    names = {
        "runtime_seed": f"{prefix}/runtime_seed.json",
        "progress": f"{prefix}/adapt_iteration_progress.jsonl",
        "manifest": f"{prefix}/normalized_job_manifest.json",
        "validation": f"{prefix}/validation_receipt.json",
    }
    members = _read_members(evidence_archive, list(names.values()))
    runtime_seed = json.loads(members[names["runtime_seed"]])
    original_manifest = json.loads(members[names["manifest"]])
    original_validation = json.loads(members[names["validation"]])
    adapt = runtime_seed["adapt_vqe"]
    labels = [str(value) for value in adapt["operators"]]
    batches = [[str(value) for value in batch] for batch in adapt["selected_operator_batches"]]
    theta = [float(value) for value in adapt["theta"]]
    if not (len(labels) == len(batches) == len(theta) == BASE_DEPTH):
        raise RuntimeError("r50 warm-state labels/batches/theta do not align")
    if [batch[0] for batch in batches if len(batch) == 1] != labels:
        raise RuntimeError("r50 warm-state batches do not flatten to labels")
    logical = [float(value) for value in adapt["logical_optimal_point"]]
    if logical != theta:
        raise RuntimeError("r50 logical optimal point does not equal theta")
    history = _compact_prefix_history(
        members[names["progress"]], tracking["result"]["trajectory"]
    )
    if [row["appended_operator_labels"][0] for row in history] != labels:
        raise RuntimeError("r50 progress operator order does not match runtime seed")
    state_core = {
        "selected_operator_labels": labels,
        "selected_operator_batches": batches,
        "theta": theta,
        "adapt_history": history,
    }
    prefix_identity = {
        "schema": "paper_i_append_warm_prefix_identity_v1",
        "status": "pass",
        "source_transfer_archive": {
            "path": evidence_archive.relative_to(repo).as_posix(),
            "sha256": BASE_TRANSFER_SHA256,
            "size_bytes": evidence_archive.stat().st_size,
        },
        "source_tracking_summary": {
            "path": tracking_path.relative_to(repo).as_posix(),
            "sha256": _sha256(tracking_path),
        },
        "source_result_member": {
            "name": tracking["result_member"]["name"],
            "sha256": BASE_RESULT_MEMBER_SHA256,
            "size_bytes": BASE_RESULT_MEMBER_SIZE,
        },
        "source_members": {
            key: {
                "name": names[key],
                "sha256": _sha256_bytes(members[names[key]]),
                "size_bytes": len(members[names[key]]),
            }
            for key in names
        },
        "source_bundle_id": BASE_BUNDLE_ID,
        "source_job_id": BASE_JOB_ID,
        "source_iterations": BASE_ITERATIONS,
        "source_depth": BASE_DEPTH,
        "source_terminal_energy": BASE_TERMINAL_ENERGY,
        "source_same_cutoff_abs_error": BASE_TERMINAL_ERROR,
        "source_S_alg": BASE_S_ALG,
        "ordered_operator_labels_sha256": _canonical_sha256(labels),
        "selected_operator_batches_sha256": _canonical_sha256(batches),
        "theta_sha256": _canonical_sha256(theta),
        "compact_history_sha256": _canonical_sha256(history),
        "selected_generator_semantics_sha256": adapt["selected_generator_semantics_sha256"],
        "rng_state_serialized": False,
        "determinism_basis": [
            "exact_statevector_backend",
            "Powell_seed_7",
            "frozen_source_archive",
            "frozen_container_image",
        ],
    }
    prefix_identity["prefix_identity_sha256"] = _canonical_sha256(prefix_identity)
    state = {
        "schema": "paper_i_append_projected_singleton_warm_start_state_v1",
        "created_utc": _utc_now(),
        "prefix_identity": prefix_identity,
        **state_core,
    }
    state["state_core_sha256"] = _canonical_sha256(state_core)
    _write_json(bundle / "source_prefix_identity.json", prefix_identity)
    _write_json(bundle / "warm_start_state.json", state)
    if float(original_validation["S_alg"]) != BASE_S_ALG:
        raise RuntimeError("source validation S_alg mismatch")
    return state, original_manifest, original_validation


def _flatten_diff(before: Any, after: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        rows = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in before:
                rows.append({"path": path, "before": "__MISSING__", "after": after[key]})
            elif key not in after:
                rows.append({"path": path, "before": before[key], "after": "__MISSING__"})
            else:
                rows.extend(_flatten_diff(before[key], after[key], path))
        return rows
    if before != after:
        return [{"path": prefix, "before": before, "after": after}]
    return []


def _build_job(
    bundle: Path,
    source_manifest: Mapping[str, Any],
    warm_state: Mapping[str, Any],
    original: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    job = deepcopy(dict(original))
    job["schema"] = "paper_i_hh_append_warm_continuation_job_v1"
    job["bundle_id"] = BUNDLE_ID
    job["job_id"] = JOB_ID
    job["controller"].update(
        {
            "energy_stop_target": TARGET_ERROR,
            "first_hit_thresholds": [TARGET_ERROR],
            "fresh_round_zero": False,
            "initial_selected_operator_count": BASE_DEPTH,
            "initial_theta_count": BASE_DEPTH,
            "initial_history_count": BASE_ITERATIONS,
            "max_adapt_iterations": MAX_TOTAL_ITERATIONS,
            "stop_policy": STOP_POLICY,
            "warm_start_state": "warm_start_state.json",
        }
    )
    job["exact_reference"]["usage"] = "post_iteration_adaptive_stop_decision"
    job["source_lock"].update(
        {
            "archive_sha256": source_manifest["archive_sha256"],
            "generic_static_adapt_variants_sha256": next(
                item["after_sha256"]
                for item in source_manifest["changed_members"]
                if item["path"] == GENERIC_REL.as_posix()
            ),
            "parent_archive_sha256": BASE_SOURCE_SHA256,
        }
    )
    job["output_contract"].update(
        {
            "continuation_accounting": "continuation_accounting.json",
            "prefix_identity": "source_prefix_identity.json",
            "transfer_archive": f"{JOB_ID}_transfer.tar.gz",
        }
    )
    job["continuation"] = {
        "mode": "warm_start_selected_theta_v1",
        "source_job_id": BASE_JOB_ID,
        "source_iterations": BASE_ITERATIONS,
        "source_depth": BASE_DEPTH,
        "source_terminal_error": BASE_TERMINAL_ERROR,
        "source_S_alg": BASE_S_ALG,
        "warm_start_state_sha256": _sha256(bundle / "warm_start_state.json"),
        "prefix_identity_sha256": warm_state["prefix_identity"]["prefix_identity_sha256"],
        "authorized_changes": [
            "warm_start_state_at_exact_validated_k50_endpoint",
            "max_total_iterations_50_to_80",
            "stop_at_first_same_cutoff_abs_error_le_2e-4",
        ],
        "incremental_ledger_required": True,
        "cumulative_S_alg_formula": "1276060 + continuation_incremental_S_alg",
    }
    allowed_paths = {
        "schema",
        "bundle_id",
        "job_id",
        "controller.energy_stop_target",
        "controller.first_hit_thresholds",
        "controller.fresh_round_zero",
        "controller.initial_selected_operator_count",
        "controller.initial_theta_count",
        "controller.initial_history_count",
        "controller.max_adapt_iterations",
        "controller.stop_policy",
        "controller.warm_start_state",
        "exact_reference.usage",
        "source_lock.archive_sha256",
        "source_lock.generic_static_adapt_variants_sha256",
        "source_lock.parent_archive_sha256",
        "output_contract.continuation_accounting",
        "output_contract.prefix_identity",
        "output_contract.transfer_archive",
        "continuation",
    }
    diff = _flatten_diff(original, job)
    unauthorized = [
        row for row in diff
        if not any(row["path"] == allowed or row["path"].startswith(allowed + ".") for allowed in allowed_paths)
    ]
    if unauthorized:
        raise RuntimeError(f"unauthorized manifest drift: {unauthorized}")
    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "created_utc": _utc_now(),
        "status": "pass",
        "source_locked": True,
        "parent_bundle_id": BASE_BUNDLE_ID,
        "parent_job_id": BASE_JOB_ID,
        "parent_source_archive_sha256": BASE_SOURCE_SHA256,
        "candidate_source_archive_sha256": source_manifest["archive_sha256"],
        "user_authorized_changes": job["continuation"]["authorized_changes"],
        "manifest_diff": diff,
        "unauthorized_manifest_diff": [],
        "scientific_settings_preserved": {
            "physics": original["physics"] == job["physics"],
            "candidate_pool": original["candidate_pool"] == job["candidate_pool"],
            "optimizer": original["optimizer"] == job["optimizer"],
            "qiskit_cost": original["qiskit_cost"] == job["qiskit_cost"],
            "seed": original["seed"] == job["seed"],
            "variant": original["variant"] == job["variant"],
            "algorithm_id": original["algorithm_id"] == job["algorithm_id"],
        },
        "source_member_diff": source_manifest["changed_members"],
        "submission_authorized": False,
        "submission_blocker": "requires review of narrow stop-policy patch and exact remote-image preflight",
    }
    if not all(audit["scientific_settings_preserved"].values()):
        raise RuntimeError("baseline scientific settings drifted")
    _write_json(bundle / "jobs" / f"{JOB_ID}.json", job)
    _write_json(bundle / "normalized_manifests" / f"{JOB_ID}.json", job)
    _write_json(bundle / "source_locked_sensitivity_audit.json", audit)
    return job, audit


def _build_queue_and_submit(bundle: Path, source_manifest: Mapping[str, Any]) -> None:
    queue_path = bundle / "queue.tsv"
    with queue_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                JOB_ID,
                f"{bundle.relative_to(_repo_root()).as_posix()}/jobs/{JOB_ID}.json",
                f"{bundle.relative_to(_repo_root()).as_posix()}/normalized_manifests/{JOB_ID}.json",
                "65536",
                "32768",
                "1",
            ]
        )
    rel = bundle.relative_to(_repo_root()).as_posix()
    submit = f'''universe = vanilla
batch_name = {BATCH_NAME}
executable = {rel}/execute_source_locked_job.sh
arguments = $(job_manifest) {rel}/warm_start_state.json {rel}/source_locked.tar.gz {source_manifest["archive_sha256"]} chtc/phase3_optuna/image.sif {EXPECTED_IMAGE_SHA256} $(job_id)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
preserve_relative_paths = True
transfer_input_files = {rel}/run_job.py, $(job_manifest), $(normalized_manifest), {rel}/warm_start_state.json, {rel}/source_prefix_identity.json, {rel}/source_locked_sensitivity_audit.json, {rel}/source_archive_manifest.json, {rel}/bundle_manifest.json, {rel}/source_locked.tar.gz, chtc/phase3_optuna/image.sif
transfer_output_files = raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz
transfer_output_remaps = "raw_outputs/{BUNDLE_ID}/$(job_id)_transfer.tar.gz = $(job_id)_transfer.tar.gz"
request_cpus = $(request_cpus)
request_memory = $(memory_mb)MB
request_disk = $(disk_mb)MB
+WantFlocking = true
log = logs/{BUNDLE_ID}.$(Cluster).$(Process).log
output = logs/{BUNDLE_ID}.$(Cluster).$(Process).out
error = logs/{BUNDLE_ID}.$(Cluster).$(Process).err
requirements = False
# LOCAL-ONLY: do not submit until review and an exact remote-image preflight pass.
queue job_id, job_manifest, normalized_manifest, memory_mb, disk_mb, request_cpus from {rel}/queue.tsv
'''
    _write_text(bundle / "submit.sub", submit)


def _artifact_inventory(bundle: Path) -> list[dict[str, Any]]:
    excluded = {"submission_artifact_hashes.json", "bundle_manifest.json", "bundle_validation_receipt.json"}
    return [
        {
            "path": path.relative_to(bundle).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name not in excluded and "__pycache__" not in path.parts
    ]


def main() -> int:
    repo = _repo_root()
    bundle = _bundle_dir()
    source_manifest = _freeze_source(repo, bundle)
    warm_state, original_manifest, original_validation = _build_warm_state(repo, bundle)
    job, audit = _build_job(bundle, source_manifest, warm_state, original_manifest)
    _build_queue_and_submit(bundle, source_manifest)
    inventory = _artifact_inventory(bundle)
    bundle_manifest = {
        "schema": "paper_i_hh_append_warm_continuation_bundle_v1",
        "created_utc": _utc_now(),
        "bundle_id": BUNDLE_ID,
        "batch_name": BATCH_NAME,
        "status": "local_preflight_only",
        "row_count": 1,
        "job_id": JOB_ID,
        "source_archive_sha256": source_manifest["archive_sha256"],
        "source_prefix_identity_sha256": warm_state["prefix_identity"]["prefix_identity_sha256"],
        "source_S_alg": BASE_S_ALG,
        "target_error": TARGET_ERROR,
        "max_total_iterations": MAX_TOTAL_ITERATIONS,
        "submission_enabled": False,
        "requirements": "False",
        "remote_image_sha256_expected": EXPECTED_IMAGE_SHA256,
        "remote_image_preflight": "pending",
        "scientific_settings_preserved": audit["scientific_settings_preserved"],
        "source_validation_receipt": original_validation,
        "artifact_inventory_without_self": inventory,
    }
    _write_json(bundle / "bundle_manifest.json", bundle_manifest)
    final_inventory = _artifact_inventory(bundle)
    hashes = {
        "schema": "paper_i_hh_append_warm_continuation_artifact_hashes_v1",
        "created_utc": _utc_now(),
        "bundle_id": BUNDLE_ID,
        "artifacts": final_inventory,
    }
    _write_json(bundle / "submission_artifact_hashes.json", hashes)
    receipt = {
        "schema": "paper_i_hh_append_warm_continuation_bundle_validation_v1",
        "created_utc": _utc_now(),
        "status": "pass_local_only",
        "bundle_id": BUNDLE_ID,
        "job_manifest_sha256": _sha256(bundle / "jobs" / f"{JOB_ID}.json"),
        "warm_start_state_sha256": _sha256(bundle / "warm_start_state.json"),
        "source_prefix_identity_sha256": warm_state["prefix_identity"]["prefix_identity_sha256"],
        "source_archive_sha256": source_manifest["archive_sha256"],
        "base_transfer_archive_sha256": BASE_TRANSFER_SHA256,
        "base_result_member_sha256": BASE_RESULT_MEMBER_SHA256,
        "base_S_alg": BASE_S_ALG,
        "target_error": TARGET_ERROR,
        "max_total_iterations": MAX_TOTAL_ITERATIONS,
        "queue_rows": 1,
        "requirements_false": True,
        "remote_upload_performed": False,
        "remote_submission_performed": False,
        "unresolved_blocker": "exact remote-image preflight and review of the narrow stop-policy patch",
    }
    _write_json(bundle / "bundle_validation_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
