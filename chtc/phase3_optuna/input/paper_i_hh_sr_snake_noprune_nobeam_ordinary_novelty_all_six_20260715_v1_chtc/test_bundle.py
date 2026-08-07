#!/usr/bin/env python3
"""Focused non-scientific checks for the Study-B submission bundle."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import unittest
from pathlib import Path, PurePosixPath


BUNDLE = Path(__file__).resolve().parent
REPO = BUNDLE.parents[3]
SOURCE_SHA256 = (
    "94c2df6df22c6d277aefdd6559273d943e3724d476ecab6648c6dd11e1fd78c6"
)
IMAGE_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
STUDY_FLAGS = {
    "--phase2-gram-novelty-policy",
    "--phase3-gram-novelty-policy",
}
EXPECTED = {
    "weak_weak": (30, 32768, 61440, 2),
    "intermediate_weak": (30, 32768, 61440, 2),
    "strong_weak_u8": (50, 32768, 61440, 2),
    "weak_strong": (50, 40960, 61440, 4),
    "intermediate_strong": (50, 40960, 61440, 4),
    "strong_strong_u8": (50, 40960, 61440, 4),
}


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(path)
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def options(argv: list[str]) -> dict[str, object]:
    if argv[:3] != ["python3", "-m", "pipelines.static_adapt.adapt_pipeline"]:
        raise ValueError("unexpected command prefix")
    parsed: dict[str, object] = {}
    index = 3
    while index < len(argv):
        flag = argv[index]
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            parsed[flag] = argv[index + 1]
            index += 2
        else:
            parsed[flag] = True
            index += 1
    return parsed


class StudyBNoveltyOnBundleTests(unittest.TestCase):
    def test_source_archive_is_exact_and_safe(self) -> None:
        archive = BUNDLE / "source_locked.tar.gz"
        self.assertEqual(sha256(archive), SOURCE_SHA256)
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                normalized = PurePosixPath(member.name.lstrip("./"))
                self.assertFalse(normalized.is_absolute())
                self.assertNotIn("..", normalized.parts)
                self.assertTrue(member.isfile() or member.isdir())

    def test_all_six_rows_isolate_only_ordinary_novelty(self) -> None:
        manifest = load(BUNDLE / "bundle_manifest.json")
        self.assertEqual(manifest["job_count"], 6)
        self.assertEqual(
            {row["regime_slug"] for row in manifest["jobs"]}, set(EXPECTED)
        )
        for slug, (depth, _memory, _disk, cutoff) in EXPECTED.items():
            job_path = BUNDLE / "jobs" / f"{slug}.json"
            job = load(job_path)
            execution = options([str(value) for value in job["command"]["execution_argv"]])
            baseline = options(
                [str(value) for value in job["command"]["matched_baseline_argv"]]
            )
            changed = {
                key
                for key in set(execution) | set(baseline)
                if execution.get(key) != baseline.get(key)
            }
            self.assertEqual(changed, STUDY_FLAGS)
            self.assertEqual(execution["--adapt-max-depth"], str(depth))
            self.assertEqual(
                execution["--phase2-gram-novelty-policy"],
                "ordinary_multiplier_v1",
            )
            self.assertEqual(
                execution["--phase3-gram-novelty-policy"],
                "ordinary_multiplier_v1",
            )
            self.assertEqual(baseline["--phase2-gram-novelty-policy"], "fallback_only_v1")
            self.assertEqual(baseline["--phase3-gram-novelty-policy"], "fallback_only_v1")
            self.assertIs(execution["--phase1-no-prune"], True)
            self.assertIs(execution["--phase0-no-pilot"], True)
            self.assertIs(execution["--phase2-no-batching"], True)
            self.assertIs(execution["--phase3-no-batching"], True)
            self.assertEqual(execution["--adapt-beam-live-branches"], "1")
            self.assertEqual(execution["--adapt-beam-children-per-parent"], "1")
            self.assertEqual(job["physics"]["n_ph_work"], cutoff)
            self.assertEqual(job["physics"]["n_ph_reference"], cutoff)
            contract = job["scientific_contract"]
            self.assertEqual(contract["regular_energy_response_model"], "undamped")
            self.assertTrue(contract["ordinary_novelty_multipliers_enabled"])
            self.assertTrue(
                contract["all_energy_models_infeasible_novelty_fallback_retained"]
            )
            self.assertFalse(contract["pruning_enabled"])
            self.assertFalse(contract["beam_enabled"])

    def test_runner_accepts_every_manifest(self) -> None:
        runner = BUNDLE / "run_job.py"
        for slug in EXPECTED:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--validate-only",
                    str(BUNDLE / "jobs" / f"{slug}.json"),
                ],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_submit_resources_and_preflight(self) -> None:
        queue_rows = {}
        for line in (BUNDLE / "queue.tsv").read_text(encoding="utf-8").splitlines():
            slug, job_path, memory, disk = line.split("\t")
            queue_rows[slug] = (job_path, int(memory), int(disk))
        self.assertEqual(set(queue_rows), set(EXPECTED))
        for slug, (_depth, memory, disk, _cutoff) in EXPECTED.items():
            self.assertEqual(queue_rows[slug][1:], (memory, disk))
        submit = (BUNDLE / "submit.sub").read_text(encoding="utf-8")
        self.assertIn("request_cpus = 4", submit)
        self.assertIn("request_memory = $(memory_mb)MB", submit)
        self.assertIn("request_disk = $(disk_mb)MB", submit)
        self.assertIn("+MaxRuntime = 259200", submit)
        self.assertIn(IMAGE_SHA256, submit)
        self.assertEqual(submit.count("+JobBatchName"), 1)
        preflight = load(BUNDLE / "preflight.json")
        self.assertEqual(preflight["status"], "pass")
        self.assertTrue(all(preflight["checks"].values()))

    def test_submission_hash_inventory_closes(self) -> None:
        inventory = load(BUNDLE / "submission_artifact_hashes.json")
        for relative, record in inventory["artifacts"].items():
            path = REPO / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256(path), record["sha256"], relative)
            self.assertEqual(path.stat().st_size, record["size_bytes"], relative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
