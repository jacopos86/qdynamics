from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVATION_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE_DIR = ACTIVATION_ROOT / (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r100_fresh_"
    "20260808_v1_chtc"
)
V1_ACTIVATION_DIR = ACTIVATION_ROOT / (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r100_fresh_"
    "20260808_v1_chtc_activation_ordinary_v1"
)
ACTIVATION_DIR = ACTIVATION_ROOT / (
    "paper_i_ra_adapt_historical_average_singleton_plateau6_r100_fresh_"
    "20260808_v1_chtc_activation_ordinary_v2"
)


def _activation_contract():
    spec = importlib.util.spec_from_file_location(
        "paper_i_ra_r100_activation_contract",
        ACTIVATION_DIR / "activation_contract.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ra_r100_activation_validates_without_rewriting_v1() -> None:
    completed = subprocess.run(
        [sys.executable, "-B", str(ACTIVATION_DIR / "validate_activation.py")],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["status"] == "passed"
    assert receipt["ordinary_held"] is False
    assert receipt["direct_execution_count"] == 6

    v1_manifest = json.loads(
        (V1_ACTIVATION_DIR / "activation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert v1_manifest["activation_id"].endswith("activation_ordinary_v1")
    assert v1_manifest["sha256"] == (
        "5e96f34ec3cc021b72f8479fa916ff7b0c15e9cc895038016fa09fd7d4085d4e"
    )


def test_ra_r100_archive_is_explicitly_selected_and_posix_remapped() -> None:
    submit = (ACTIVATION_DIR / "submit.sub").read_text(encoding="utf-8")
    archive = "transfer/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"
    assert f"transfer_output_files = {archive}" in submit
    assert "output_destination" not in submit
    assert "osdf://" not in submit.lower()
    assert (
        f'transfer_output_remaps = "{archive}=/staging/jsstrobel/'
        "paper_i_ra_historical_average_singleton_plateau6_r100_20260809_v2/"
        'outputs/$(execution_id)__$(ClusterId)__$(ProcId).tar.gz"'
        in submit
    )


def test_ra_r100_activation_applies_only_scheduler_memory_floors() -> None:
    contract = _activation_contract()
    queue_rows = [
        line.split("\t")
        for line in (ACTIVATION_DIR / "queue.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(queue_rows) == 6
    assert [int(row[6]) for row in queue_rows[:3]] == [40_960] * 3
    assert [int(row[6]) for row in queue_rows[3:]] == [49_152] * 3

    activation = json.loads(
        (ACTIVATION_DIR / "activation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert [
        row["resources"]["request_memory_mb"]
        for row in activation["executions"]
    ] == [40_960, 40_960, 40_960, 49_152, 49_152, 49_152]
    assert activation["operational_mode"] == (
        "ordinary_unheld_posix_transfer_v2"
    )

    package_memories = []
    for job_path in sorted((PACKAGE_DIR / "jobs").glob("*.json")):
        job = json.loads(job_path.read_text(encoding="utf-8"))
        package_memories.append(job["resources"]["request_memory_mb"])
    assert sorted(package_memories) == [24_576] * 3 + [32_768] * 3
    assert contract.activation_memory_mb("row__nph3__ra", 45_000) == 45_000
    assert contract.activation_memory_mb("row__nph7__ra", 52_000) == 52_000
