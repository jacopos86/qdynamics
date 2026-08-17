from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727" / (
    "paper_i_ra_adapt_pure_hubbard_page12_lowhigh4_r20_20260811_v1_chtc"
)
SOURCE_PACKAGE = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727" / (
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
    previous_contract = sys.modules.pop("package_contract", None)
    previous_cell = sys.modules.pop("run_cell", None)
    try:
        return _module(path, name=name)
    finally:
        sys.path.remove(PACKAGE.as_posix())
        sys.modules.pop("package_contract", None)
        sys.modules.pop("run_cell", None)
        if previous_contract is not None:
            sys.modules["package_contract"] = previous_contract
        if previous_cell is not None:
            sys.modules["run_cell"] = previous_cell


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _non_digest_differences(
    source: object,
    recovery: object,
    *,
    path: str = "",
) -> list[tuple[str, object, object]]:
    if type(source) is not type(recovery):
        return [(path, source, recovery)]
    if isinstance(source, dict):
        rows: list[tuple[str, object, object]] = []
        assert isinstance(recovery, dict)
        for key in sorted(set(source) | set(recovery)):
            child = f"{path}/{key}"
            if "sha256" in key:
                continue
            if key not in source or key not in recovery:
                rows.append((child, source.get(key), recovery.get(key)))
            else:
                rows.extend(
                    _non_digest_differences(
                        source[key], recovery[key], path=child
                    )
                )
        return rows
    if isinstance(source, list):
        assert isinstance(recovery, list)
        if len(source) != len(recovery):
            return [(path, source, recovery)]
        rows = []
        for index, (source_item, recovery_item) in enumerate(
            zip(source, recovery, strict=True)
        ):
            rows.extend(
                _non_digest_differences(
                    source_item,
                    recovery_item,
                    path=f"{path}/{index}",
                )
            )
        return rows
    return [] if source == recovery else [(path, source, recovery)]


def test_contract_is_exact_four_cell_low_high_r20_recovery() -> None:
    contract = _module(
        PACKAGE / "package_contract.py",
        name="paper_i_pure_hubbard_noise_r20_contract",
    )

    assert contract.TARGET_HORIZON == 20
    assert contract.SOURCE_HORIZON == 50
    assert contract.U_VALUES == (1.5, 8.0)
    assert [level for level, _noise in contract.NOISE_LEVELS] == [
        "low",
        "high",
    ]
    assert contract.CELL_COUNT == 4
    assert len(contract.expected_execution_ids()) == 4
    assert contract.INERT_PACKAGE_STATUS == "passed_inert_four_cells"
    assert contract.ACTIVATION_SCOPE == (
        "prepare_four_cell_chtc_execution_and_submission_v1"
    )
    assert contract.RESOURCE_ENVELOPE["request_cpus"] == 2
    assert contract.RESOURCE_ENVELOPE["request_memory_mb"] == 16_384
    assert contract.RESOURCE_ENVELOPE["request_disk_mb"] == 12_288
    assert contract.RESOURCE_ENVELOPE["max_runtime_seconds"] == 259_200


def test_application_sources_change_only_the_requested_horizon() -> None:
    recovery_sources = PACKAGE / "source_authority"
    source_sources = SOURCE_PACKAGE / "source_authority"
    rows = sorted(recovery_sources.glob("*.application_source_contract.json"))
    assert len(rows) == 4

    for recovery_path in rows:
        source = _json(source_sources / recovery_path.name)
        recovery = _json(recovery_path)
        assert _non_digest_differences(source, recovery) == [
            (
                "/scientific_settings/maximum_controller_rounds",
                50,
                20,
            )
        ]


def test_seal_closes_four_jobs_and_preserves_source_archive() -> None:
    manifest = _json(PACKAGE / "package_manifest.json")
    source_manifest = _json(SOURCE_PACKAGE / "package_manifest.json")
    audit = _json(PACKAGE / "source_lock_audit.json")

    assert manifest["status"] == "passed_inert_four_cells"
    assert manifest["row_count"] == 4
    assert len(manifest["jobs"]) == 4
    assert len(manifest["protocols"]) == 4
    assert manifest["source_archive"]["sha256"] == source_manifest[
        "source_archive"
    ]["sha256"]
    assert audit["scientific_changed_fields_vs_source"] == [
        "maximum_controller_rounds"
    ]
    assert audit["source_implementation_inventory_match"] is True

    queue_rows = (PACKAGE / "queue.tsv").read_text(encoding="utf-8").splitlines()
    assert len(queue_rows) == 4
    for row in queue_rows:
        fields = row.split("\t")
        assert fields[4:] == ["2", "16384", "12288", "259200"]


def test_shallow_package_validation_covers_all_four_cells() -> None:
    validator = _package_module(
        PACKAGE / "validate_package.py",
        name="paper_i_pure_hubbard_noise_r20_validator",
    )
    receipt = validator.validate_package()
    assert receipt["status"] == "passed_inert_package"
    assert receipt["shallow_worker_preflight_count"] == 4
    assert receipt["launch_ready"] is False


def test_unchanged_runtime_controls_remain_byte_identical() -> None:
    for name in (
        "execute_authorized_job.sh",
        "probe_image_runtime.py",
        "run_numerical_preflight.py",
    ):
        assert (PACKAGE / name).read_bytes() == (SOURCE_PACKAGE / name).read_bytes()
