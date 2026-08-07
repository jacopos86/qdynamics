#!/usr/bin/env python3
"""Focused contract tests for the six-regime SR-SNAKE candidate bundle.

The expected route digest is deliberately read from the frozen source revision
record.  This test suite therefore checks one immutable bundle revision without
embedding a digest from an older SR-SNAKE route.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
BUNDLE_ID = (
    "paper_i_hh_sr_snake_full_response_symmetric_cost_noprune_nobeam_"
    "no_ordinary_novelty_all_six_20260717_v1_chtc"
)
PROFILE_REQUEST = "sr_snake_no_prune_symmetric_cost_v1"
PROFILE_RESOLVED = (
    "supported_whitened_adaptive_trust_full_response_symmetric_cost_"
    "no_prune_v1"
)

EXPECTED: dict[str, dict[str, Any]] = {
    "weak_weak": {
        "u": 0.25,
        "n_ph": 3,
        "exact": -0.918380919994822,
        "target": 30,
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    "intermediate_weak": {
        "u": 1.25,
        "n_ph": 3,
        "exact": -0.4950053491813613,
        "target": 30,
        "memory_mb": 32768,
        "disk_mb": 61440,
    },
    "strong_weak_u8": {
        "u": 8.0,
        "n_ph": 3,
        "exact": 0.5264586847939736,
        "target": 50,
        "memory_mb": 40960,
        "disk_mb": 61440,
    },
    "weak_strong": {
        "u": 0.25,
        "n_ph": 7,
        "exact": -1.1387206380749124,
        "target": 50,
        "memory_mb": 49152,
        "disk_mb": 81920,
    },
    "intermediate_strong": {
        "u": 1.25,
        "n_ph": 7,
        "exact": -0.6239396137518493,
        "target": 50,
        "memory_mb": 49152,
        "disk_mb": 81920,
    },
    "strong_strong_u8": {
        "u": 8.0,
        "n_ph": 7,
        "exact": 0.5205762765682517,
        "target": 50,
        "memory_mb": 49152,
        "disk_mb": 81920,
    },
}

PHASE1_ENERGY_MODEL = "first_order_fs_trust_v1"
PHASE2_CURVATURE_POLICY = "measured_required_fail_closed_v1"
PHASE2_CHEAP_CURVATURE_PROXY_POLICY = "off"
COST_POLICY = "family_robust_symmetric_arctan_v1"
FALLBACK_POLICY = "collective_span_novelty_over_symmetric_cost_v1"

# Exact stale identities, rather than a broad ban on the text "v4" in legacy
# regression-test names or historical prose.
STALE_IDENTITY_TOKENS = (
    '"profile_request": "sr_snake_v4"',
    "paper-i-hh-sr-v4-sw-r30-phase3-service-repair-20260717-v7",
    "paper_i_hh_sr_snake_v4_candidate_all_six_20260716_v3_chtc",
    "8881960",
    "b6331521fb55f4165e177466536b4e2a5834ff09205ab5532ea70de893f156bc",
)

REQUIRED_GENERATED = (
    "source_locked.tar.gz",
    "source_archive_manifest.json",
    "source_revision_manifest.json",
    "archive_only_preflight.json",
    "scientific_settings_audit.json",
    "bundle_manifest.json",
    "preflight.json",
    "route_parity.json",
    "queue.tsv",
    "submit.sub",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generated_bundle_exists() -> bool:
    return all((BUNDLE / relative).is_file() for relative in REQUIRED_GENERATED) and all(
        (BUNDLE / "jobs" / f"{slug}.json").is_file() for slug in EXPECTED
    )


def expected_digest() -> str:
    revision = load(BUNDLE / "source_revision_manifest.json")
    if revision.get("profile_request") != PROFILE_REQUEST:
        raise AssertionError("source revision records the wrong profile request")
    if revision.get("profile_resolved") != PROFILE_RESOLVED:
        raise AssertionError("source revision records the wrong resolved profile")
    digest = str(revision.get("profile_contract_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise AssertionError("source revision has no valid route-contract digest")
    return digest


def argv_value(argv: list[str], flag: str) -> str:
    count = argv.count(flag)
    if count != 1:
        raise AssertionError(f"expected exactly one {flag}, observed {count}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise AssertionError(f"{flag} has no value")
    return str(argv[index + 1])


def python_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(value)
    raise AssertionError(f"{name} is absent from {path.name}")


def safe_archive_members(archive: Path) -> list[tarfile.TarInfo]:
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
    for member in members:
        name = PurePosixPath(member.name)
        if (
            name.is_absolute()
            or ".." in name.parts
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
            or any(
                part in {".DS_Store", "__MACOSX"} or part.startswith("._")
                for part in name.parts
            )
        ):
            raise AssertionError(f"unsafe archive member: {member.name}")
    return members


def extract_archive_only_repo(destination: Path) -> Path:
    archive = BUNDLE / "source_locked.tar.gz"
    safe_archive_members(archive)
    with tarfile.open(archive, "r:gz") as handle:
        handle.extractall(destination, filter="data")
    isolated_bundle = destination / BUNDLE.relative_to(REPO)
    isolated_bundle.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(BUNDLE, isolated_bundle)
    return isolated_bundle


def isolated_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = root / "home"
    home.mkdir(exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "PYTHONPATH": os.pathsep.join(
                (str(root), str(site.getusersitepackages()))
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


class ScaffoldContractTests(unittest.TestCase):
    def test_executable_scaffold_names_only_the_new_route_identity(self) -> None:
        builder = BUNDLE / "build_bundle.py"
        runner = BUNDLE / "run_job.py"
        self.assertEqual(python_constant(builder, "BUNDLE_ID"), BUNDLE_ID)
        self.assertEqual(python_constant(builder, "PROFILE_REQUEST"), PROFILE_REQUEST)
        self.assertEqual(python_constant(builder, "PROFILE_RESOLVED"), PROFILE_RESOLVED)
        self.assertEqual(python_constant(runner, "BUNDLE_ID"), BUNDLE_ID)
        self.assertEqual(python_constant(runner, "PROFILE_REQUEST"), PROFILE_REQUEST)
        self.assertEqual(python_constant(runner, "PROFILE"), PROFILE_RESOLVED)

        identity_surfaces = (
            "build_bundle.py",
            "run_job.py",
            "validate_fetched.py",
            "execute_source_locked_job.sh",
            "submit.sub",
        )
        text = "\n".join(
            (BUNDLE / name).read_text(encoding="utf-8")
            for name in identity_surfaces
            if (BUNDLE / name).is_file()
        )
        for stale in STALE_IDENTITY_TOKENS:
            self.assertNotIn(stale, text)

    def test_scaffold_contains_every_fail_closed_policy_gate(self) -> None:
        builder = (BUNDLE / "build_bundle.py").read_text(encoding="utf-8")
        runner = (BUNDLE / "run_job.py").read_text(encoding="utf-8")
        evidence = (BUNDLE / "evidence_validation.py").read_text(encoding="utf-8")
        combined = "\n".join((builder, runner, evidence))
        for token in (
            PHASE1_ENERGY_MODEL,
            PHASE2_CURVATURE_POLICY,
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY,
            COST_POLICY,
            FALLBACK_POLICY,
            "fallback_only_v1",
            "full_active_plus_singleton_v1",
            "supported_metric_whitened_eigh_v1",
            "displacement_calibrated_unbounded_v2",
            "supported_fs_whitened_fixed_v1",
            "expanded_runtime_projected_logical_v1",
            "sr_all_energy_models_infeasible_novelty_fallback_telemetry_v1",
            "phase2_full_candidate_occurrences",
            "validated_phase2_curvature_receipt_occurrences",
            "phase1_lambda_f_proxy_occurrences",
            "phase2_lambda_f_proxy_occurrences",
            "phase2_missing_curvature_fallback_occurrences",
        ):
            self.assertIn(token, combined)
        self.assertIn('"--adapt-max-depth"', builder)
        self.assertIn('"--adapt-max-depth"', runner)


@unittest.skipUnless(
    generated_bundle_exists(),
    "bundle has not yet been rebuilt from the final frozen source",
)
class GeneratedBundleTests(unittest.TestCase):
    def test_digest_authority_and_all_identity_records_close(self) -> None:
        digest = expected_digest()
        revision = load(BUNDLE / "source_revision_manifest.json")
        archive_only = load(BUNDLE / "archive_only_preflight.json")
        audit = load(BUNDLE / "scientific_settings_audit.json")

        self.assertEqual(revision["profile_contract_sha256"], digest)
        self.assertEqual(archive_only["source_import"]["profile_resolved"], PROFILE_RESOLVED)
        self.assertEqual(
            archive_only["source_import"]["profile_contract_sha256"], digest
        )
        self.assertEqual(audit["status"], "pass")
        self.assertEqual(audit["profile_request"], PROFILE_REQUEST)
        self.assertEqual(audit["profile_resolved"], PROFILE_RESOLVED)
        self.assertEqual(audit["profile_contract_sha256"], digest)
        self.assertEqual(audit["unexpected_executable_differences"], [])

        identity_artifacts = [
            BUNDLE / "bundle_manifest.json",
            BUNDLE / "preflight.json",
            BUNDLE / "route_parity.json",
            BUNDLE / "source_revision_manifest.json",
            BUNDLE / "scientific_settings_audit.json",
            BUNDLE / "queue.tsv",
            BUNDLE / "submit.sub",
        ] + sorted((BUNDLE / "jobs").glob("*.json")) + sorted(
            (BUNDLE / "normalized_manifests").glob("*.json")
        )
        combined = "\n".join(
            path.read_text(encoding="utf-8") for path in identity_artifacts
        )
        for stale in STALE_IDENTITY_TOKENS:
            self.assertNotIn(stale, combined)

    def test_six_fresh_same_cutoff_jobs_have_exact_horizons(self) -> None:
        digest = expected_digest()
        rows = [
            line.split("\t")
            for line in (BUNDLE / "queue.tsv").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 6)
        self.assertEqual({row[0] for row in rows}, set(EXPECTED))

        manifest = load(BUNDLE / "bundle_manifest.json")
        submission_scope = manifest["submission_scope"]
        self.assertEqual(set(submission_scope["regimes"]), set(EXPECTED))
        self.assertEqual(int(submission_scope["job_count"]), 6)

        for slug, expected in EXPECTED.items():
            job = load(BUNDLE / "jobs" / f"{slug}.json")
            normalized = load(BUNDLE / "normalized_manifests" / f"{slug}.json")
            self.assertEqual(job["bundle_id"], BUNDLE_ID)
            route = job["route_identity"]
            self.assertEqual(route["profile_request"], PROFILE_REQUEST)
            self.assertEqual(route["profile_resolved"], PROFILE_RESOLVED)
            self.assertEqual(route["profile_contract_sha256"], digest)
            self.assertEqual(
                normalized["route_identity"]["profile_contract_sha256"], digest
            )

            physics = job["physics"]
            self.assertEqual(float(physics["u_over_t"]), expected["u"])
            self.assertEqual(int(physics["n_ph_work"]), expected["n_ph"])
            self.assertEqual(int(physics["n_ph_reference"]), expected["n_ph"])
            self.assertTrue(physics["same_cutoff_reference"])
            self.assertEqual(float(physics["expected_exact_energy"]), expected["exact"])

            target = expected["target"]
            segment = job["segment"]
            self.assertEqual(int(segment["source_controller_round"]), 0)
            self.assertEqual(int(segment["source_depth"]), 0)
            self.assertEqual(int(segment["target_controller_round"]), target)
            self.assertEqual(int(segment["target_depth"]), target)
            self.assertEqual(int(segment["max_new_admissions"]), target)
            self.assertFalse(segment["future_continuation_required_after_validation"])
            self.assertIsNone(segment["future_continuation_target"])

            argv = [str(value) for value in job["command"]["argv"]]
            self.assertEqual(argv_value(argv, "--sr-route-profile"), PROFILE_REQUEST)
            self.assertEqual(int(argv_value(argv, "--adapt-max-depth")), target)
            self.assertEqual(
                int(argv_value(argv, "--adapt-segment-target-controller-round")),
                target,
            )
            self.assertEqual(
                int(argv_value(argv, "--adapt-segment-target-depth")), target
            )
            self.assertEqual(
                int(argv_value(argv, "--adapt-segment-max-new-admissions")), target
            )
            self.assertIn("--adapt-disable-hh-seed", argv)
            self.assertNotIn("--adapt-exact-gs-override", argv)
            self.assertNotIn("--adapt-exact-gs-reference-json", argv)
            self.assertEqual(
                job["command"]["explicit_method_overrides"], ["adapt_max_depth"]
            )
            resources = job["resource_request"]
            self.assertEqual(int(resources["cpus"]), 4)
            self.assertEqual(int(resources["memory_mb"]), expected["memory_mb"])
            self.assertEqual(int(resources["disk_mb"]), expected["disk_mb"])
            self.assertEqual(int(resources["max_runtime_s"]), 259200)

    def test_profile_contract_is_exactly_the_approved_scientific_route(self) -> None:
        contracts = [
            load(BUNDLE / "jobs" / f"{slug}.json")["route_identity"][
                "profile_contract"
            ]
            for slug in EXPECTED
        ]
        for contract in contracts[1:]:
            self.assertEqual(contract, contracts[0])
        contract = contracts[0]
        execution = contract["execution_settings"]
        semantics = contract["semantic_invariants"]

        required_execution = {
            "phase1_energy_model": PHASE1_ENERGY_MODEL,
            "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
            "phase2_cheap_curvature_proxy_policy": PHASE2_CHEAP_CURVATURE_PROXY_POLICY,
            "phase2_gram_novelty_policy": "fallback_only_v1",
            "phase3_gram_novelty_policy": "fallback_only_v1",
            "phase1_prune_enabled": False,
            "adapt_beam_live_branches": 1,
            "adapt_beam_children_per_parent": 1,
            "phase0_pilot_enabled": False,
            "phase2_enable_batching": False,
            "phase3_enable_batching": False,
            "phase3_shadow_damping_policy": "off",
            "historical_singleton_coordinate_solve_scope": "phase3_only_v1",
            "historical_singleton_coordinate_solve_policy": "supported_metric_whitened_eigh_v1",
            "historical_singleton_trust_region_update_policy": "displacement_calibrated_unbounded_v2",
            "phase3_response_coordinate_scope": "full_active_plus_singleton_v1",
            "adapt_accepted_refit_scope": "full_ansatz_v1",
            "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "adapt_accepted_refit_base_chart_policy": "expanded_runtime_projected_logical_v1",
            "sr_powell_coordinate_chart_policy": "expanded_runtime_projected_logical_v1",
            "adapt_full_refit_every": 0,
            "adapt_final_full_refit": "false",
            "adapt_finite_angle_fallback": False,
            "phase3_enable_rescue": False,
            "phase3_hardware_cost_normalization_mode": COST_POLICY,
            "adapt_disable_hh_seed": True,
            "adapt_seed": 7,
        }
        required_semantics = {
            "phase1_energy_model": PHASE1_ENERGY_MODEL,
            "phase1_fs_metric_role": "trust_domain_only_v1",
            "phase1_phase2_lambda_f_proxy_active": False,
            "phase2_curvature_policy": PHASE2_CURVATURE_POLICY,
            "phase2_cheap_curvature_proxy_policy": PHASE2_CHEAP_CURVATURE_PROXY_POLICY,
            "phase2_curvature_failure_policy": "abort_run_v1",
            "ordinary_phase2_novelty_multiplier_active": False,
            "ordinary_phase3_novelty_multiplier_active": False,
            "all_energy_models_infeasible_novelty_fallback_active": True,
            "all_energy_models_infeasible_novelty_fallback_policy": FALLBACK_POLICY,
            "all_energy_models_infeasible_novelty_fallback_telemetry_required": True,
            "pruning_active": False,
            "beam_shape": "effective_1x1_v1",
            "phase2_supported_whitening_active": False,
            "phase3_supported_whitening_active": True,
            "phase3_response_coordinate_scope": "full_active_plus_singleton_v1",
            "phase3_response_pre_support_invariant": "response_count_equals_active_logical_count_plus_one_v1",
            "adaptive_trust_policy": "displacement_calibrated_unbounded_v2",
            "accepted_refit_scope": "full_ansatz_v1",
            "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            "accepted_refit_base_chart_policy": "expanded_runtime_projected_logical_v1",
            "hardware_cost_policy": COST_POLICY,
            "hardware_cost_application_scope": "phase1_phase2_phase3_and_infeasible_fallback_v1",
            "phase3_shadow_damping_active": False,
            "negative_curvature_escape_active": False,
            "finite_angle_fallback_active": False,
            "periodic_full_refit_active": False,
            "terminal_full_refit_active": False,
            "terminal_prune_active": False,
            "controller_horizon_source": "per_regime_source_lock",
            "same_cutoff_reference_required": True,
        }
        for key, expected in required_execution.items():
            self.assertEqual(execution.get(key), expected, key)
        for key, expected in required_semantics.items():
            self.assertEqual(semantics.get(key), expected, key)
        self.assertNotIn("adapt_max_depth", execution)

    def test_archive_only_validate_only_passes_for_all_six_jobs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sr_symcost_bundle_") as tmp:
            root = Path(tmp)
            isolated_bundle = extract_archive_only_repo(root)
            for slug in EXPECTED:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(isolated_bundle / "run_job.py"),
                        "--validate-only",
                        str(isolated_bundle / "jobs" / f"{slug}.json"),
                    ],
                    cwd=root,
                    env=isolated_env(root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                payload = json.loads(completed.stdout)
                self.assertEqual(payload["job"], f"jobs/{slug}.json")
                self.assertNotIn("/Users/", completed.stdout)

    def test_manifest_mutations_fail_closed(self) -> None:
        baseline = load(BUNDLE / "jobs" / "weak_weak.json")
        mutations: list[tuple[str, dict[str, Any]]] = []

        def mutate(label: str, edit: Any) -> None:
            payload = copy.deepcopy(baseline)
            edit(payload)
            mutations.append((label, payload))

        mutate(
            "legacy_profile",
            lambda payload: payload["route_identity"].__setitem__(
                "profile_request", "sr_snake_v4"
            ),
        )
        mutate(
            "pruning_enabled",
            lambda payload: payload["route_identity"]["profile_contract"][
                "execution_settings"
            ].__setitem__("phase1_prune_enabled", True),
        )
        mutate(
            "ordinary_novelty_enabled",
            lambda payload: payload["route_identity"]["profile_contract"][
                "execution_settings"
            ].__setitem__("phase3_gram_novelty_policy", "ordinary_multiplier_v1"),
        )
        mutate(
            "beam_enabled",
            lambda payload: payload["route_identity"]["profile_contract"][
                "execution_settings"
            ].__setitem__("adapt_beam_live_branches", 2),
        )
        mutate(
            "fallback_disabled",
            lambda payload: payload["route_identity"]["profile_contract"][
                "semantic_invariants"
            ].__setitem__("all_energy_models_infeasible_novelty_fallback_active", False),
        )
        mutate(
            "cost_policy_drift",
            lambda payload: payload["route_identity"]["profile_contract"][
                "execution_settings"
            ].__setitem__("phase3_hardware_cost_normalization_mode", "off"),
        )
        mutate(
            "phase2_whitening_enabled",
            lambda payload: payload["route_identity"]["profile_contract"][
                "semantic_invariants"
            ].__setitem__("phase2_supported_whitening_active", True),
        )
        mutate(
            "phase3_window_coupling",
            lambda payload: payload["route_identity"]["profile_contract"][
                "execution_settings"
            ].__setitem__("phase3_response_coordinate_scope", "legacy_reopt_coupled_v1"),
        )
        mutate(
            "finite_angle_enabled",
            lambda payload: payload["route_identity"]["profile_contract"][
                "execution_settings"
            ].__setitem__("adapt_finite_angle_fallback", True),
        )
        mutate(
            "phase2_proxy_enabled",
            lambda payload: payload["route_identity"][
                "phase12_energy_model_contract"
            ].__setitem__("phase2_cheap_curvature_proxy_policy", "lambda_f_fallback_v1"),
        )
        mutate(
            "same_cutoff_broken",
            lambda payload: payload["physics"].__setitem__("n_ph_reference", 5),
        )

        no_horizon = copy.deepcopy(baseline)
        argv = list(no_horizon["command"]["argv"])
        index = argv.index("--adapt-max-depth")
        del argv[index : index + 2]
        no_horizon["command"]["argv"] = argv
        mutations.append(("missing_explicit_horizon", no_horizon))

        with tempfile.TemporaryDirectory(prefix="sr_symcost_guards_") as tmp:
            root = Path(tmp)
            isolated_bundle = extract_archive_only_repo(root)
            env = isolated_env(root)
            for label, payload in mutations:
                path = root / f"{label}.json"
                path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(isolated_bundle / "run_job.py"),
                        "--validate-only",
                        str(path),
                    ],
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0, label)

    def test_smoke_records_explicit_infeasible_fallback_telemetry(self) -> None:
        digest = expected_digest()
        smoke = load(BUNDLE / "source_lock/local_smoke_evidence.json")
        self.assertEqual(
            smoke["status"], "pass_for_bundle_construction_not_a_production_result"
        )
        self.assertEqual(smoke["exact_blockers"], [])
        self.assertEqual(len(smoke["records"]), 1)
        record = smoke["records"][0]
        self.assertTrue(record["exit_success"])
        self.assertEqual(int(record["admissions"]), 8)
        self.assertEqual(record["profile_resolved"], PROFILE_RESOLVED)
        self.assertEqual(record["profile_contract_sha256"], digest)
        telemetry = record["phase12_energy_model_telemetry"]
        self.assertEqual(telemetry["phase1_energy_model"], PHASE1_ENERGY_MODEL)
        self.assertEqual(
            telemetry["phase2_curvature_policy"], PHASE2_CURVATURE_POLICY
        )
        self.assertEqual(
            telemetry["phase2_cheap_curvature_proxy_policy"],
            PHASE2_CHEAP_CURVATURE_PROXY_POLICY,
        )
        self.assertEqual(int(telemetry["phase1_lambda_f_proxy_occurrences"]), 0)
        self.assertEqual(int(telemetry["phase2_lambda_f_proxy_occurrences"]), 0)
        self.assertEqual(
            int(telemetry["phase2_missing_curvature_fallback_occurrences"]), 0
        )
        validation = record["scientific_evidence_validation"]
        self.assertTrue(validation["infeasible_model_fallback_enabled"])
        self.assertFalse(validation["infeasible_model_fallback_fired"])
        self.assertEqual(
            int(validation["infeasible_model_fallback_activation_count"]), 0
        )
        self.assertEqual(
            validation["infeasible_model_fallback_controller_rounds"], []
        )
        self.assertIn(
            "infeasible_model_fallback_explicit_telemetry", smoke["passed_gates"]
        )

    def test_source_archive_is_safe_closed_and_archive_only_preflight_passes(self) -> None:
        archive = BUNDLE / "source_locked.tar.gz"
        archive_manifest = load(BUNDLE / "source_archive_manifest.json")
        revision = load(BUNDLE / "source_revision_manifest.json")
        digest = expected_digest()
        self.assertEqual(sha256(archive), archive_manifest["archive_sha256"])
        self.assertEqual(
            archive_manifest["worker_source_mode"],
            "hash_locked_live_worktree_snapshot_v1",
        )
        self.assertEqual(archive_manifest["git_commit"], revision["git_commit"])
        self.assertEqual(archive_manifest["git_tree"], revision["git_tree"])
        members = safe_archive_members(archive)
        self.assertTrue(members)
        for relative, record in revision["critical_source_sha256"].items():
            self.assertEqual(
                archive_manifest["files"][relative]["sha256"], record, relative
            )

        preflight = load(BUNDLE / "archive_only_preflight.json")
        self.assertEqual(preflight["status"], "pass")
        self.assertEqual(preflight["source_import"]["status"], "pass")
        self.assertEqual(preflight["source_import"]["profile_resolved"], PROFILE_RESOLVED)
        self.assertEqual(
            preflight["source_import"]["profile_contract_sha256"], digest
        )
        self.assertTrue(preflight["live_repo_import_excluded"])
        self.assertTrue(preflight["all_six_validate_only_pass"])
        self.assertEqual(len(preflight["six_validate_only_parses"]), 6)
        self.assertTrue(preflight["qiskit_helper"]["help_pass"])
        self.assertTrue(preflight["focused_source_locked_regressions"]["pass"])

    def test_preflight_and_hash_inventory_close(self) -> None:
        preflight = load(BUNDLE / "preflight.json")
        self.assertEqual(preflight["scientific_blockers"], [])
        checks = preflight["checks"]
        for key in (
            "source_archive_safe_and_closed",
            "six_job_manifests",
            "all_job_validations",
            "same_cutoff_reference_lock",
            "finite_angle_fallback_disabled",
            "phase3_rescue_disabled",
            "phase1_first_order_fs_trust_policy",
            "phase2_measured_curvature_required_fail_closed_policy",
            "phase2_cheap_curvature_proxy_off",
            "phase1_phase2_lambda_f_proxy_inactive",
            "smoke_phase2_curvature_receipt_count_closure",
            "smoke_lambda_f_proxy_occurrences_zero",
            "smoke_missing_curvature_fallback_occurrences_zero",
        ):
            self.assertTrue(checks[key], key)

        inventory_path = BUNDLE / "submission_artifact_hashes.json"
        inventory = load(inventory_path)
        actual = {
            path.relative_to(REPO).as_posix()
            for path in BUNDLE.rglob("*")
            if path.is_file()
            and path != inventory_path
            and "__pycache__" not in path.parts
            and ".pytest_cache" not in path.parts
            and path.suffix != ".pyc"
        }
        self.assertEqual(set(inventory["artifacts"]), actual)
        for relative, record in inventory["artifacts"].items():
            path = REPO / relative
            self.assertEqual(sha256(path), record["sha256"], relative)
            self.assertEqual(path.stat().st_size, int(record["size_bytes"]), relative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
