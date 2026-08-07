#!/usr/bin/env python3
"""Build the Test-2 v3 successor after exact-image round-1 scope closure."""

from __future__ import annotations

import ast
import copy
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Sequence

import build_paper_i_hh_sr_material_window_fs_prune_verify_successor_20260722 as prior


BASE_ID = prior.OUTPUT_ID
BASE_BATCH = prior.OUTPUT_BATCH
BASE = prior.INPUT / BASE_ID
OUTPUT_ID = (
    "paper_i_hh_sr_snake_no_overlap_trust_material_window_fs_prune_verify_"
    "all_six_r50_20260722_v3_chtc"
)
OUTPUT_BATCH = "paper-i-hh-sr-material-window-fsprune-verify-six-r50-20260722-v3"
OUTPUT = prior.INPUT / OUTPUT_ID
BASE_SOURCE_SHA256 = "e3f05fab6964b6875c50b84613a4d7c7d7cbb43a49db5f7246025b068288bd09"
BASE_ADAPT_SHA256 = "3f150d61d0f17828ccaec36c2b6e48253800e7c45807d7edc4f0d8fe69643ba4"

OLD_SEAM = '''    phase1_prune_prefilter_policy_key = str(
        phase1_prune_prefilter_policy or PRUNE_PREFILTER_OFF
    ).strip().lower()
    if (
        sr_controller_ablation_contract_key
'''
NEW_SEAM = '''    phase1_prune_prefilter_policy_key = str(
        phase1_prune_prefilter_policy or PRUNE_PREFILTER_OFF
    ).strip().lower()
    phase1_prune_tolerance_mode_requested = str(
        phase1_prune_tolerance_mode or PRUNE_TOLERANCE_AUTO
    ).strip().lower()
    if (
        sr_controller_ablation_contract_key
'''


def _build_source(temp: Path) -> tuple[Path, dict[str, Any]]:
    archive = BASE / "source_locked.tar.gz"
    if prior.common.sha256(archive) != BASE_SOURCE_SHA256:
        raise ValueError("Test-2 v2 source archive hash drift")
    source = temp / "source"
    prior._extract(archive, source)
    adapt = source / "pipelines/static_adapt/adapt_pipeline.py"
    if prior.common.sha256(adapt) != BASE_ADAPT_SHA256:
        raise ValueError("Test-2 v2 adapt source hash drift")
    text = adapt.read_text(encoding="utf-8")
    if text.count(OLD_SEAM) != 1:
        raise ValueError("prune-tolerance normalization repair seam drift")
    text = text.replace(OLD_SEAM, NEW_SEAM, 1)
    ast.parse(text)
    adapt.write_text(text, encoding="utf-8")

    regression = source / "test/test_static_adapt_prune_tolerance_scope_order.py"
    regression.write_text(
        '''from __future__ import annotations

import ast
from pathlib import Path


def test_all_normalized_prune_receipt_keys_are_bound_before_first_read() -> None:
    tree = ast.parse(
        Path("pipelines/static_adapt/adapt_pipeline.py").read_text(encoding="utf-8")
    )
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_hardcoded_adapt_vqe"
    )
    arguments = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    events: dict[str, list[tuple[int, str]]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id.startswith("phase1_prune_"):
            events.setdefault(node.id, []).append(
                (int(node.lineno), type(node.ctx).__name__)
            )
    unbound = {}
    for name, values in events.items():
        values.sort()
        if name not in arguments and values[0][1] == "Load":
            unbound[name] = values[0]
    assert unbound == {}
''',
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "test/test_static_adapt_prune_prefilter_scope_order.py",
            regression.relative_to(source).as_posix(),
        ],
        cwd=source,
        env=env,
        check=True,
    )
    prior.common.strip_bytecode(source)
    successor = temp / "source_locked.tar.gz"
    prior.common.deterministic_archive(source, successor)
    repair = {
        "schema": "paper_i_sr_test2_prune_normalized_receipt_scope_repair_v2",
        "predecessor_bundle": BASE_ID,
        "predecessor_source_archive_sha256": BASE_SOURCE_SHA256,
        "predecessor_cluster": None,
        "detected_by": "exact_uploaded_archive_image_round1_smoke",
        "failure_class": "pre_science_unbound_normalized_prune_tolerance_mode_key",
        "changed_paths": [
            "pipelines/static_adapt/adapt_pipeline.py",
            "test/test_static_adapt_prune_tolerance_scope_order.py",
        ],
        "route_contract_sha256_unchanged": prior.ROUTE_DIGEST,
        "scientific_settings_changed": False,
        "algorithmic_query_delta": 0,
    }
    return successor, repair


def _configure_prior() -> None:
    prior.BASE_ID = BASE_ID
    prior.BASE_BATCH = BASE_BATCH
    prior.BASE = BASE
    prior.OUTPUT_ID = OUTPUT_ID
    prior.OUTPUT_BATCH = OUTPUT_BATCH
    prior.OUTPUT = OUTPUT
    prior.BASE_SOURCE_SHA256 = BASE_SOURCE_SHA256
    prior.BASE_ADAPT_SHA256 = BASE_ADAPT_SHA256
    prior._build_source = _build_source


def main(argv: Sequence[str] | None = None) -> int:
    _configure_prior()
    args = prior.parse_args(argv)
    receipt = prior.build()
    prior._patch_bundle_text(
        {
            "sr-material-window-fsprune-verify-r0-r50-20260722-v2": (
                "sr-material-window-fsprune-verify-r0-r50-20260722-v3"
            ),
            "sr-material-window-fsprune-verify-r0-r{target}-20260722-v2": (
                "sr-material-window-fsprune-verify-r0-r{target}-20260722-v3"
            ),
        }
    )
    if args.archive_preflight:
        prior.archive_preflight()
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
