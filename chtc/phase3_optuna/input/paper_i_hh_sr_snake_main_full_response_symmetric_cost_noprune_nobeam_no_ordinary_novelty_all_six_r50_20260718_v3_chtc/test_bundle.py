#!/usr/bin/env python3
"""Fail-closed tests for the frozen all-six round-50 main-route bundle."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import build_bundle as spec
import evidence_validation as evidence
import run_job as worker
import validate_fetched as fetched_validator


BUNDLE = Path(__file__).resolve().parent
EXPECTED_PHYSICS: dict[str, tuple[str, str, str, int, str]] = {
    "weak_weak": (
        "0.25", "0.25", "0.353553390593", 3, "-0.918380919994822"
    ),
    "intermediate_weak": (
        "1.25", "0.25", "0.353553390593", 3, "-0.4950053491813613"
    ),
    "strong_weak_u8": (
        "8.0", "0.25", "0.353553390593", 3, "0.5264586847939736"
    ),
    "weak_strong": (
        "0.25", "1.25", "0.790569415042", 7, "-1.1387206380749124"
    ),
    "intermediate_strong": (
        "1.25", "1.25", "0.790569415042", 7, "-0.6239396137518493"
    ),
    "strong_strong_u8": (
        "8.0", "1.25", "0.790569415042", 7, "0.5205762765682517"
    ),
}
STALE_TOKENS = (
    "target_round not in {" + "3" + "0, 50}",
    "target_controller_round: int = " + "3" + "0",
    "target_new_admissions: int = " + "3" + "0",
    "fresh_round0_full_response_runtime_repair_" + "strong_weak_strong_strong_v1",
    "strong_weak_strong_strong_repair_submission_queue",
    "retired_candidate_parent_job_v1",
)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def argv_value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


class FrozenBundleTests(unittest.TestCase):
    def test_source_freeze_is_complete_and_submission_is_fail_closed(self) -> None:
        self.assertTrue(spec.SOURCE_FREEZE_COMPLETE)
        self.assertFalse(spec.SUBMISSION_ENABLED)
        self.assertIn(
            "phase2_raw/eps_grad_suppressed_continue",
            spec.SCIENTIFIC_SUBMISSION_BLOCKER,
        )
        self.assertEqual(set(spec.SUBMISSION_REGIMES), set(EXPECTED_PHYSICS))
        self.assertEqual(
            spec.EXPECTED_HEAD,
            "8a746d244a15e2cb16099a732e78e1110a8e59f2",
        )
        self.assertEqual(
            spec.EXPECTED_TREE,
            "6cb596ab953386a9c9a3b0698e7b1489e3b0f02e",
        )
        self.assertEqual(
            spec.SOURCE_LOCK_STATE,
            "immutable_parent_archive_plus_hash_locked_overlay_v1",
        )
        archive = BUNDLE / "source_locked.tar.gz"
        self.assertTrue(archive.is_file())
        self.assertEqual(
            spec.sha256(archive),
            "fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35",
        )
        archive_manifest = load(BUNDLE / "source_archive_manifest.json")
        self.assertEqual(archive_manifest["archive_sha256"], spec.sha256(archive))
        self.assertEqual(archive_manifest["file_count"], 385)
        self.assertEqual(
            archive_manifest["immutable_parent_archive"]["sha256"],
            spec.PARENT_ARCHIVE_SHA256,
        )
        self.assertEqual(
            archive_manifest["adapt_pipeline_overlay"]["derived_file_sha256"],
            spec.DERIVED_ADAPT_PIPELINE_SHA256,
        )
        submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
        self.assertIn("requirements = False", submit)
        self.assertNotIn("TODO_FINAL_", submit)

    def test_all_six_rows_are_fresh_exact_round_50(self) -> None:
        job_names = {path.stem for path in (BUNDLE / "jobs").glob("*.json")}
        normalized_names = {
            path.stem for path in (BUNDLE / "normalized_manifests").glob("*.json")
        }
        self.assertEqual(job_names, set(EXPECTED_PHYSICS))
        self.assertEqual(normalized_names, set(EXPECTED_PHYSICS))
        queue_rows = [
            line.split("\t")
            for line in (BUNDLE / "queue.tsv").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(queue_rows), 6)
        self.assertEqual({row[0] for row in queue_rows}, set(EXPECTED_PHYSICS))

        for slug in EXPECTED_PHYSICS:
            job = load(BUNDLE / "jobs" / f"{slug}.json")
            normalized = load(BUNDLE / "normalized_manifests" / f"{slug}.json")
            for payload in (job, normalized):
                segment = payload["segment"]
                self.assertEqual(int(segment["source_controller_round"]), 0)
                self.assertEqual(int(segment["source_depth"]), 0)
                self.assertEqual(int(segment["target_controller_round"]), 50)
                self.assertEqual(int(segment["target_depth"]), 50)
                self.assertEqual(int(segment["max_new_admissions"]), 50)
                self.assertFalse(segment["future_continuation_required_after_validation"])
                self.assertIsNone(segment["future_continuation_target"])
                self.assertEqual(
                    int(segment["terminal_qiskit_sidecar_outer_iteration"]), 50
                )
            argv = [str(token) for token in job["command"]["argv"]]
            for flag in (
                "--adapt-max-depth",
                "--adapt-segment-target-controller-round",
                "--adapt-segment-target-depth",
                "--adapt-segment-max-new-admissions",
            ):
                self.assertEqual(argv_value(argv, flag), "50")

    def test_physics_and_same_cutoff_references_are_exactly_locked(self) -> None:
        physics_lock = load(BUNDLE / "physics_and_exact_reference_lock.json")
        rows = {row["regime_slug"]: row for row in physics_lock["rows"]}
        self.assertEqual(set(rows), set(EXPECTED_PHYSICS))
        for slug, (u, lam, g_ep, n_ph, exact) in EXPECTED_PHYSICS.items():
            row = rows[slug]
            job_physics = load(BUNDLE / "jobs" / f"{slug}.json")["physics"]
            self.assertEqual(str(row["u_over_t"]), u)
            self.assertEqual(str(row["lambda"]), lam)
            self.assertEqual(str(row["g_ep"]), g_ep)
            self.assertEqual(str(row["expected_exact_energy"]), exact)
            self.assertEqual(int(row["n_ph_work"]), n_ph)
            self.assertEqual(int(row["n_ph_reference"]), n_ph)
            self.assertTrue(row["same_cutoff_reference"])
            self.assertEqual(str(job_physics["g_ep_decimal_12"]), g_ep)
            self.assertEqual(str(job_physics["expected_exact_energy_decimal"]), exact)
            self.assertEqual(int(job_physics["n_ph_work"]), n_ph)
            self.assertEqual(int(job_physics["n_ph_reference"]), n_ph)
            self.assertTrue(job_physics["same_cutoff_reference"])

    def test_route_is_main_one_by_one_no_prune_no_batch_no_novelty(self) -> None:
        for slug in EXPECTED_PHYSICS:
            job = load(BUNDLE / "jobs" / f"{slug}.json")
            route = job["route_identity"]
            self.assertEqual(route["profile_request"], spec.PROFILE_REQUEST)
            self.assertEqual(route["profile_resolved"], spec.PROFILE_RESOLVED)
            self.assertEqual(
                route["profile_contract_sha256"], spec.PROFILE_CONTRACT_SHA256
            )
            execution = route["profile_contract"]["execution_settings"]
            semantics = route["profile_contract"]["semantic_invariants"]
            self.assertEqual(execution["adapt_beam_live_branches"], 1)
            self.assertEqual(execution["adapt_beam_children_per_parent"], 1)
            self.assertFalse(execution["phase1_prune_enabled"])
            self.assertFalse(execution["phase2_enable_batching"])
            self.assertFalse(execution["phase3_enable_batching"])
            self.assertEqual(execution["phase2_gram_novelty_policy"], "fallback_only_v1")
            self.assertEqual(execution["phase3_gram_novelty_policy"], "fallback_only_v1")
            self.assertEqual(
                execution["phase3_response_coordinate_scope"],
                "full_active_plus_singleton_v1",
            )
            self.assertEqual(execution["phase1_energy_model"], spec.PHASE1_ENERGY_MODEL)
            self.assertEqual(
                execution["phase2_curvature_policy"], spec.PHASE2_CURVATURE_POLICY
            )
            self.assertEqual(execution["phase2_cheap_curvature_proxy_policy"], "off")
            self.assertFalse(semantics["pruning_active"])
            self.assertFalse(semantics["ordinary_phase2_novelty_multiplier_active"])
            self.assertFalse(semantics["ordinary_phase3_novelty_multiplier_active"])
            self.assertEqual(semantics["beam_shape"], "effective_1x1_v1")

    def test_cache_and_required_evidence_are_explicit_in_both_layers(self) -> None:
        for slug in EXPECTED_PHYSICS:
            job = load(BUNDLE / "jobs" / f"{slug}.json")
            normalized = load(BUNDLE / "normalized_manifests" / f"{slug}.json")
            for payload in (job, normalized):
                self.assertEqual(payload["cache_policy"], spec.RUN_CACHE_POLICY)
                evidence = payload["evidence_requirements"]
                for gate in (
                    "exact_s_alg_ledger_closure_required",
                    "active_prefix_estimator_receipt_each_round_required",
                    "terminal_estimator_closure_receipt_required",
                    "fallback_telemetry_required",
                    "full_active_plus_singleton_response_each_round_required",
                    "full_accepted_refit_each_round_required",
                    "symmetry_and_padding_leakage_gate_required",
                    "exact_round_50_horizon_required",
                ):
                    self.assertIs(evidence[gate], True, gate)
                self.assertEqual(
                    evidence["post_run_projector_fidelity"], spec.FIDELITY_POLICY
                )
            fidelity_path = job["paths"]["ground_space_fidelity_json"]
            self.assertTrue(fidelity_path.endswith("ground_space_projector_fidelity.json"))
            self.assertIs(job["segment"]["post_run_projector_fidelity_required"], True)
            self.assertEqual(
                job["segment"]["post_run_projector_fidelity_policy"],
                spec.FIDELITY_POLICY,
            )

    def test_ledger_and_fidelity_surfaces_are_closed_before_freeze(self) -> None:
        ledger_paths = {
            "pipelines/static_adapt/estimator_call_ledger.py",
            "test/test_static_adapt_estimator_call_ledger.py",
        }
        self.assertTrue(ledger_paths.issubset(spec.CRITICAL_SOURCE_PATHS))
        self.assertTrue(ledger_paths.issubset(spec.CRITICAL_SOURCE_SHA256))
        self.assertEqual(
            spec.CRITICAL_SOURCE_SHA256[
                "pipelines/static_adapt/estimator_call_ledger.py"
            ],
            "cce95198fc5d504cd55f496b3c41a89a4bbb06490311f6d69eb999d5eb5907ce",
        )
        self.assertEqual(
            spec.CRITICAL_SOURCE_SHA256[
                "test/test_static_adapt_estimator_call_ledger.py"
            ],
            "9bfab183abd825439e902c87aa2abc09ef563da63f480a9db8293131b3dae73b",
        )
        self.assertIn(
            "test/test_static_adapt_estimator_call_ledger.py",
            spec.ARCHIVE_PATHS,
        )

        fidelity_paths = {
            "pipelines/scaffold/ground_space_fidelity.py",
            "agent_guidance/skills/paper-i-results/scripts/"
            "compute_paper_i_main_fidelities.py",
            "test/test_ground_space_fidelity.py",
        }
        self.assertEqual(
            set(spec.REQUIRED_HASH_LOCKED_FIDELITY_FILES), fidelity_paths
        )
        self.assertTrue(fidelity_paths.issubset(spec.CRITICAL_SOURCE_PATHS))
        self.assertTrue(fidelity_paths.issubset(spec.CRITICAL_SOURCE_SHA256))
        self.assertTrue(all(len(value) == 64 for value in (
            spec.REQUIRED_HASH_LOCKED_FIDELITY_FILES.values()
        )))
        external = spec.EXTERNAL_CROSS_METHOD_FIDELITY_AUDIT_EVIDENCE
        self.assertEqual(external["path"], "test/test_paper_i_main_fidelity_audit.py")
        self.assertNotIn(external["path"], spec.CRITICAL_SOURCE_PATHS)
        self.assertNotIn(external["path"], spec.REQUIRED_HASH_LOCKED_FIDELITY_FILES)
        self.assertEqual(
            worker.REQUIRED_HASH_LOCKED_FIDELITY_FILES,
            spec.REQUIRED_HASH_LOCKED_FIDELITY_FILES,
        )
        self.assertEqual(
            fetched_validator.REQUIRED_HASH_LOCKED_FIDELITY_FILES,
            spec.REQUIRED_HASH_LOCKED_FIDELITY_FILES,
        )

    def test_untracked_fidelity_overlay_is_conditional_and_duplicate_free(self) -> None:
        required = set(spec.REQUIRED_HASH_LOCKED_FIDELITY_FILES)
        self.assertEqual(
            set(spec.required_hash_locked_overlay_paths(set())), required
        )
        self.assertEqual(
            spec.required_hash_locked_overlay_paths(set(required)), ()
        )
        one_tracked = next(iter(required))
        overlays = set(
            spec.required_hash_locked_overlay_paths({one_tracked})
        )
        self.assertEqual(overlays, required - {one_tracked})
        merged = {one_tracked}
        merged.update(overlays)
        self.assertEqual(merged, required)
        self.assertEqual(len(merged), len(required))

    def test_archive_only_regressions_include_estimator_ledger(self) -> None:
        source = (BUNDLE / "build_bundle.py").read_text(encoding="utf-8")
        self.assertIn(
            '"test/test_static_adapt_estimator_call_ledger.py"', source
        )
        self.assertIn(
            "required_hash_locked_fidelity_files_hash_closed", source
        )
        self.assertIn(
            "required_untracked_fidelity_overlays_hash_closed", source
        )

    def test_successor_reuses_predecessor_scientific_archive_byte_for_byte(self) -> None:
        predecessor = (
            BUNDLE.parent / spec.PREDECESSOR_BUNDLE_ID / "source_locked.tar.gz"
        )
        successor = BUNDLE / "source_locked.tar.gz"
        self.assertTrue(predecessor.is_file())
        self.assertEqual(successor.read_bytes(), predecessor.read_bytes())
        self.assertEqual(
            spec.sha256(successor),
            "fa9014b9608ccb8a301df0429268482abf6dd10c91eb336111a15d00be256d35",
        )
        archive_manifest = load(BUNDLE / "source_archive_manifest.json")
        self.assertEqual(archive_manifest["archive_sha256"], spec.sha256(successor))
        receipt = load(BUNDLE / "operational_successor_receipt.json")
        self.assertFalse(receipt["scientific_settings_changed"])
        self.assertTrue(receipt["scientific_source_archive_unchanged"])
        self.assertEqual(
            receipt["route_contract_sha256"], spec.PROFILE_CONTRACT_SHA256
        )

    def test_stale_horizons_and_repair_identity_are_absent(self) -> None:
        surfaces = (
            "README.md",
            "build_bundle.py",
            "run_job.py",
            "evidence_validation.py",
            "validate_fetched.py",
            "execute_source_locked_job.sh",
            "submit.sub",
        )
        combined = "\n".join(
            (BUNDLE / name).read_text(encoding="utf-8") for name in surfaces
        )
        for token in STALE_TOKENS:
            self.assertNotIn(token, combined)

    def test_final_build_job_matches_worker_schema_and_argv(self) -> None:
        contract = load(BUNDLE / "jobs/weak_weak.json")["route_identity"][
            "profile_contract"
        ]
        archive = {
            "archive": (
                BUNDLE.relative_to(spec.REPO) / "source_locked.tar.gz"
            ).as_posix(),
            "archive_sha256": "a" * 64,
            "worker_source_mode": spec.SOURCE_LOCK_STATE,
            "non_scientific_archive_overlays": {},
            "required_untracked_source_modules": {},
            "required_hash_locked_fidelity_files": {},
            "required_untracked_hash_overlays": {},
        }
        with mock.patch.object(spec, "sha256", return_value="b" * 64):
            job = spec.build_job(spec.REGIMES[0], contract, archive)
        self.assertEqual(spec.FINAL_JOB_SCHEMA, worker.SCHEMA)
        self.assertEqual(job["schema"], worker.SCHEMA)
        self.assertEqual(job["bundle_id"], worker.BUNDLE_ID)
        self.assertEqual(job["run_class"], "main_route_matrix")
        self.assertEqual(job["command"]["argv"], worker.expected_argv(job))

    def test_submission_readiness_requires_every_boolean_check(self) -> None:
        ready, failed = spec.preflight_readiness({"one": True, "two": True})
        self.assertTrue(ready)
        self.assertEqual(failed, ())
        ready, failed = spec.preflight_readiness({"one": True, "two": False})
        self.assertFalse(ready)
        self.assertEqual(failed, ("two",))
        with self.assertRaisesRegex(TypeError, "must be boolean"):
            spec.preflight_readiness({"not_boolean": 1})  # type: ignore[arg-type]

    def test_local_image_is_diagnostic_but_remote_image_gates_readiness(self) -> None:
        preflight = load(BUNDLE / "preflight.json")
        checks = preflight["checks"]
        diagnostics = preflight["diagnostics"]
        for name in (
            "local_image_present",
            "local_image_hash_matches_prior_remote_digest",
        ):
            self.assertNotIn(name, checks)
            self.assertIn(name, diagnostics)
        self.assertFalse(diagnostics["local_image_fields_affect_submission_readiness"])
        for name in (
            "remote_image_sha256_rechecked",
            "remote_qiskit_import_passed",
            "remote_fake_marrakesh_instantiation_passed",
        ):
            self.assertIn(name, checks)
            self.assertTrue(checks[name])
        ready, failed = spec.preflight_readiness({
            "submission_enabled": True,
            "remote_image_sha256_rechecked": True,
            "remote_qiskit_import_passed": True,
            "remote_fake_marrakesh_instantiation_passed": True,
        })
        self.assertTrue(ready)
        self.assertEqual(failed, ())

    def test_evidence_validator_has_one_public_definition(self) -> None:
        tree = ast.parse((BUNDLE / "evidence_validation.py").read_text())
        names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        self.assertEqual(names.count("validate_parent_evidence"), 1)
        self.assertIn("_validate_retired_v4_prune_parent_evidence", names)

    def test_supported_rank_is_optional_only_for_exact_zero_query_fallback(self) -> None:
        ordinary = {
            "all_energy_models_infeasible_novelty_fallback_fired": False,
            "phase3_response_supported_rank": None,
        }
        with self.assertRaisesRegex(ValueError, "supported rank missing"):
            evidence.validate_supported_rank_or_exact_fallback(
                ordinary,
                pre_support=4,
                outer_iteration=9,
                require_supported_rank=True,
            )

        fallback = {
            "all_energy_models_infeasible_novelty_fallback_fired": True,
            "all_energy_models_infeasible_novelty_fallback_mode": (
                "collective_span_novelty_over_cost_v1"
            ),
            "all_energy_models_infeasible_novelty_fallback_reason": (
                "all_whitened_energy_models_infeasible"
            ),
            "all_energy_models_infeasible_novelty_fallback_query_charge": 0,
            "phase3_response_supported_rank": None,
        }
        receipt = evidence.validate_supported_rank_or_exact_fallback(
            fallback,
            pre_support=4,
            outer_iteration=9,
            require_supported_rank=True,
        )
        self.assertTrue(receipt["fallback_fired"])
        self.assertFalse(receipt["supported_rank_recorded"])

        for field, value in (
            ("all_energy_models_infeasible_novelty_fallback_mode", "wrong"),
            ("all_energy_models_infeasible_novelty_fallback_reason", "wrong"),
            ("all_energy_models_infeasible_novelty_fallback_query_charge", 1),
        ):
            malformed = dict(fallback)
            malformed[field] = value
            with self.assertRaisesRegex(ValueError, "fallback receipt"):
                evidence.validate_supported_rank_or_exact_fallback(
                    malformed,
                    pre_support=4,
                    outer_iteration=9,
                    require_supported_rank=True,
                )

    def test_supported_rank_bounds_still_apply_on_fallback_rounds(self) -> None:
        fallback = {
            "all_energy_models_infeasible_novelty_fallback_fired": True,
            "all_energy_models_infeasible_novelty_fallback_mode": (
                "collective_span_novelty_over_cost_v1"
            ),
            "all_energy_models_infeasible_novelty_fallback_reason": (
                "all_whitened_energy_models_infeasible"
            ),
            "all_energy_models_infeasible_novelty_fallback_query_charge": 0,
            "phase3_response_supported_rank": 5,
        }
        with self.assertRaisesRegex(ValueError, "supported rank out of bounds"):
            evidence.validate_supported_rank_or_exact_fallback(
                fallback,
                pre_support=4,
                outer_iteration=9,
                require_supported_rank=True,
            )

    def test_missing_rank_reconstructs_only_from_exact_accepted_record(self) -> None:
        row = {
            "all_energy_models_infeasible_novelty_fallback_fired": False,
            "phase3_response_supported_rank": None,
            "selected_feature_rows": [{
                "route_a_geometry_expansion_mode": (
                    "collective_span_novelty_over_cost_v1"
                ),
                "route_a_geometry_expansion_reason": (
                    "all_whitened_energy_models_infeasible"
                ),
                "route_a_geometry_expansion_novelty_query_charge": 0,
            }],
        }
        receipt = evidence.validate_supported_rank_or_exact_fallback(
            row,
            pre_support=23,
            outer_iteration=22,
            require_supported_rank=True,
            expected_fallback_fired=True,
        )
        self.assertTrue(receipt["fallback_fired"])
        self.assertEqual(
            receipt["fallback_receipt_sources"], ["selected_feature_rows[0]"]
        )
        self.assertFalse(receipt["supported_rank_recorded"])

    def test_cumulative_fallback_cannot_mask_missing_exact_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "per-round/cumulative"):
            evidence.validate_supported_rank_or_exact_fallback(
                {
                    "all_energy_models_infeasible_novelty_fallback_fired": False,
                    "phase3_response_supported_rank": None,
                },
                pre_support=23,
                outer_iteration=22,
                require_supported_rank=True,
                expected_fallback_fired=True,
            )

    def test_nested_fallback_receipt_remains_fail_closed(self) -> None:
        base = {
            "all_energy_models_infeasible_novelty_fallback_fired": False,
            "phase3_response_supported_rank": None,
            "admitted_records": [{
                "route_a_geometry_expansion_mode": (
                    "collective_span_novelty_over_cost_v1"
                ),
                "route_a_geometry_expansion_reason": (
                    "all_whitened_energy_models_infeasible"
                ),
                "route_a_geometry_expansion_novelty_query_charge": 0,
            }],
        }
        for field, value in (
            ("route_a_geometry_expansion_mode", "wrong"),
            ("route_a_geometry_expansion_reason", "wrong"),
            ("route_a_geometry_expansion_novelty_query_charge", 1),
        ):
            malformed = json.loads(json.dumps(base))
            malformed["admitted_records"][0][field] = value
            with self.assertRaisesRegex(ValueError, "fallback receipt"):
                evidence.validate_supported_rank_or_exact_fallback(
                    malformed,
                    pre_support=23,
                    outer_iteration=22,
                    require_supported_rank=True,
                    expected_fallback_fired=True,
                )

    def test_archive_only_worker_validate_only_accepts_every_manifest(self) -> None:
        relative_bundle = BUNDLE.relative_to(spec.REPO)
        with tempfile.TemporaryDirectory(prefix="sr-main-v2-archive-validate-") as td:
            root = Path(td)
            with tarfile.open(BUNDLE / "source_locked.tar.gz", "r:gz") as archive:
                archive.extractall(root, filter="data")
            staged_bundle = root / relative_bundle
            staged_bundle.mkdir(parents=True, exist_ok=True)
            for name in (
                "run_job.py",
                "evidence_validation.py",
                "source_archive_manifest.json",
                "source_revision_manifest.json",
                "physics_and_exact_reference_lock.json",
                "source_locked.tar.gz",
            ):
                shutil.copy2(BUNDLE / name, staged_bundle / name)
            shutil.copytree(BUNDLE / "jobs", staged_bundle / "jobs")
            env = {
                **os.environ,
                "PYTHONPATH": str(root),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            for path in sorted((staged_bundle / "jobs").glob("*.json")):
                completed = subprocess.run(
                    [
                        "python3",
                        str(staged_bundle / "run_job.py"),
                        "--validate-only",
                        str(path),
                    ],
                    cwd=root,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_frozen_builder_path_reaches_archive_stage(self) -> None:
        contract = load(BUNDLE / "jobs/weak_weak.json")["route_identity"][
            "profile_contract"
        ]
        sentinel = RuntimeError("archive build intercepted by test")
        with (
            mock.patch.object(
                spec,
                "verify_source_lock",
                return_value=({"schema": "test_revision"}, {
                    "contract": contract,
                    "physics_lock": {"schema": "test_physics_lock"},
                }),
            ),
            mock.patch.object(spec, "dump_json"),
            mock.patch.object(
                spec, "build_source_archive", side_effect=sentinel
            ) as archive_builder,
        ):
            with self.assertRaisesRegex(RuntimeError, "archive build intercepted"):
                spec.main()
        archive_builder.assert_called_once_with()

    def test_successor_operational_verifier_passes_without_live_derivation(self) -> None:
        result = spec.refresh_and_verify_operational_successor()
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["failed_checks"], [])


if __name__ == "__main__":
    unittest.main()
