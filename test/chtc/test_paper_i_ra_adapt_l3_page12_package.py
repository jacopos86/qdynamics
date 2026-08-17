from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
PACKAGE = PACKAGE_ROOT / (
    "paper_i_ra_adapt_l3_intermediate_weak_page12_r50_20260811_v2_chtc"
)
PREDECESSOR = PACKAGE_ROOT / (
    "paper_i_ra_adapt_l3_intermediate_weak_page12_r50_20260811_v1_chtc"
)
ROUTE_SHA256 = (
    "8d5f9a53d79c30abba5c26b9bba68751dea3122b2f692021a44e7db260748e83"
)
APPLICATION_SHA256 = (
    "7ef4bdc24f4dbd751bdfeebed3ab26be1dfece0a33331ba18eff38b35cfad70c"
)


def _module(path: Path, *, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_l3_successor_identity_and_inert_seal() -> None:
    contract = _module(
        PACKAGE / "package_contract.py",
        name="paper_i_l3_page12_v2_package_contract",
    )
    manifest = _json(PACKAGE / "package_manifest.json")

    assert contract.PACKAGE_ID.endswith("_20260811_v2_chtc")
    assert contract.CAMPAIGN_ID.endswith("_r50_v2")
    assert contract.BUNDLE_ID.endswith("_r50_v2")
    assert contract.BATCH_NAME.endswith("-20260811-v2")
    assert contract.TARGET_HORIZON == 50
    assert contract.TARGET_ROUTE_CONTRACT_SHA256 == ROUTE_SHA256
    assert contract.APPLICATION_SOURCE_SHA256 == APPLICATION_SHA256
    assert manifest["row_count"] == 1
    assert manifest["child_route_contract_sha256"] == ROUTE_SHA256
    assert manifest["application_source_contract_sha256"] == APPLICATION_SHA256
    assert manifest["submission_authorized"] is False
    assert manifest["submitted"] is False
    assert manifest["remote_stage"] is False


def test_l3_successor_preserves_predecessor_science() -> None:
    old_manifest = _json(PREDECESSOR / "package_manifest.json")
    new_manifest = _json(PACKAGE / "package_manifest.json")
    old_binding = old_manifest["protocols"][0]
    new_binding = new_manifest["protocols"][0]
    assert old_binding["execution_id"] == new_binding["execution_id"]
    old = _json(PREDECESSOR / old_binding["path"])
    new = _json(PACKAGE / new_binding["path"])

    for key in (
        "problem",
        "request",
        "route_contract",
        "horizon",
        "algorithm_id",
        "adapter_id",
    ):
        assert old[key] == new[key]


def test_l3_successor_uses_unique_staging_and_batch_identity() -> None:
    submit = (PACKAGE / "submit.sub.in").read_text(encoding="utf-8")
    assert (
        "/staging/jsstrobel/"
        "paper_i_ra_adapt_l3_intermediate_weak_page12_r50_20260811_v2/"
    ) in submit
    assert (
        '+JobBatchName = '
        '"paper-i-l3-intermediate-weak-page12-r50-20260811-v2"'
    ) in submit
    assert "r50_20260811_v1/" not in submit
