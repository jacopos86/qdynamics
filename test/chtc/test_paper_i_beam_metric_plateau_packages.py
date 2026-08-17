from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
MACRO_PACKAGE = PACKAGE_ROOT / (
    "paper_i_ra_adapt_macro_gradient_phase0_macro_phase123_proxy_no_lanes_"
    "beam3x2_metric_prune_cap24_tau1em4_r20_20260811_v4_chtc"
)
SINGLETON_PACKAGE = PACKAGE_ROOT / (
    "paper_i_ra_adapt_global_singleton_gradient_phase0_phase123_qiskit_"
    "phase23_no_lanes_beam3x2_metric_prune_cap24_tau1em4_strong_weak_"
    "r30_20260811_v3_chtc"
)
MACRO_V3_PACKAGE = PACKAGE_ROOT / MACRO_PACKAGE.name.replace(
    "_v4_chtc", "_v3_chtc"
)
SINGLETON_V2_PACKAGE = PACKAGE_ROOT / SINGLETON_PACKAGE.name.replace(
    "_v3_chtc", "_v2_chtc"
)

MACRO_ROUTE_SHA256 = (
    "93e53e05fbcdcf23bf589c88374e0181d18e5d1abd99c68be3c80c6c37f1a9a0"
)
SINGLETON_ROUTE_SHA256 = (
    "d545fd25a162a85c4dabf09ae3fccd8ba6e095c9b794fec0fcf9a096f655c0e7"
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


def test_macro_package_contract_is_six_cell_r20_beam_metric() -> None:
    contract = _module(
        MACRO_PACKAGE / "package_contract.py",
        name="paper_i_macro_beam_metric_package_contract",
    )

    assert contract.REGIME_ROWS == (
        ("weak_weak", 3, 20),
        ("intermediate_weak", 3, 20),
        ("strong_weak_u8", 3, 20),
        ("weak_strong", 7, 20),
        ("intermediate_strong", 7, 20),
        ("strong_strong_u8", 7, 20),
    )
    assert contract.TARGET_ROUTE_CONTRACT_SHA256 == MACRO_ROUTE_SHA256
    assert contract.BEAM_LIVE_BRANCHES == 3
    assert contract.BEAM_CHILDREN_PER_PARENT == 2
    assert contract.BEAM_MAXIMUM_CHILDREN_PER_ROUND == 6
    assert contract.BEAM_S_ALG_WEIGHT == 0.005
    assert contract.PRUNING_POLICY == "metric"
    assert {
        regime: contract.RESOURCE_ENVELOPES[regime]["request_memory_mb"]
        for regime, _nph, _horizon in contract.REGIME_ROWS
    } == {
        "weak_weak": 24_576,
        "intermediate_weak": 24_576,
        "strong_weak_u8": 24_576,
        "weak_strong": 32_768,
        "intermediate_strong": 32_768,
        "strong_strong_u8": 32_768,
    }


def test_singleton_package_contract_is_one_strong_weak_r30_beam_metric(
) -> None:
    contract = _module(
        SINGLETON_PACKAGE / "package_contract.py",
        name="paper_i_singleton_beam_metric_package_contract",
    )

    assert contract.REGIME_ROWS == (("strong_weak_u8", 3, 30),)
    assert contract.TARGET_ROUTE_CONTRACT_SHA256 == SINGLETON_ROUTE_SHA256
    assert contract.BEAM_LIVE_BRANCHES == 3
    assert contract.BEAM_CHILDREN_PER_PARENT == 2
    assert contract.BEAM_MAXIMUM_CHILDREN_PER_ROUND == 6
    assert contract.BEAM_S_ALG_WEIGHT == 0.005
    assert contract.PRUNING_POLICY == "metric"
    assert contract.RESOURCE_ENVELOPES["strong_weak_u8"] == {
        "request_cpus": 4,
        "request_memory_mb": 65_536,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
        "basis": "page12_singleton_beam_metric_nph3_r30_v1",
    }


def test_sealed_packages_bind_typed_routes_resources_and_plain_queues(
) -> None:
    expected = (
        (MACRO_PACKAGE, 6, 20, MACRO_ROUTE_SHA256),
        (SINGLETON_PACKAGE, 1, 30, SINGLETON_ROUTE_SHA256),
    )
    for package, row_count, horizon, route_sha256 in expected:
        manifest = _json(package / "package_manifest.json")
        assert manifest["row_count"] == row_count
        assert manifest["child_route_contract_sha256"] == route_sha256
        queue_rows = [
            line.split("\t")
            for line in (package / "queue.tsv").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert len(queue_rows) == row_count
        assert all(len(row) == 8 for row in queue_rows)
        assert "max_materialize" not in (
            package / "submit.sub.in"
        ).read_text(encoding="utf-8")
        for binding in manifest["protocols"]:
            protocol = _json(package / binding["path"])
            assert protocol["horizon"] == horizon
            method = protocol["request"]["method"]
            assert method["insertion"] == {"kind": "plateau_commutation"}
            assert method["pruning"] == {"kind": "metric"}
            assert method["beam"] == {
                "kind": "fork_local",
                "live_parent_branches": 3,
                "admission_children_per_parent": 2,
                "maximum_admission_children_per_round": 6,
                "s_alg_weight": 0.005,
                "calibration_status": "uncalibrated_default",
            }
            route = protocol["route_contract"]
            assert route["sha256"] == route_sha256
            assert route["execution_settings"]["phase1_prune_enabled"] is True
            assert route["semantic_invariants"]["beam_shape"] == (
                "three_live_two_children_per_parent_v1"
            )


def test_submit_templates_use_home_logs_and_unique_staging_outputs() -> None:
    roots = {
        MACRO_PACKAGE: (
            "/home/jsstrobel/"
            "paper_i_page13_macro_beam_metric_r20_20260811_v4_runtime",
            "/staging/jsstrobel/"
            "paper_i_page13_macro_beam_metric_r20_20260811_v4_runtime",
        ),
        SINGLETON_PACKAGE: (
            "/home/jsstrobel/"
            "paper_i_page12_sw_singleton_beam_metric_r30_20260811_v3_runtime",
            "/staging/jsstrobel/"
            "paper_i_page12_sw_singleton_beam_metric_r30_20260811_v3_runtime",
        ),
    }
    for package, (log_root, transfer_root) in roots.items():
        submit = (package / "submit.sub.in").read_text(encoding="utf-8")
        assert f"log = {log_root}/logs/" in submit
        assert f"output = {log_root}/logs/" in submit
        assert f"error = {log_root}/logs/" in submit
        assert f"={transfer_root}/transfer/" in submit
        assert "log = /staging/" not in submit


def test_current_packages_preserve_prior_scientific_protocol_contracts() -> None:
    for prior, current in (
        (MACRO_V3_PACKAGE, MACRO_PACKAGE),
        (SINGLETON_V2_PACKAGE, SINGLETON_PACKAGE),
    ):
        old_manifest = _json(prior / "package_manifest.json")
        new_manifest = _json(current / "package_manifest.json")
        old_protocols = {
            row["execution_id"]: _json(prior / row["path"])
            for row in old_manifest["protocols"]
        }
        new_protocols = {
            row["execution_id"]: _json(current / row["path"])
            for row in new_manifest["protocols"]
        }
        assert old_protocols.keys() == new_protocols.keys()
        for execution_id in old_protocols:
            old = old_protocols[execution_id]
            new = new_protocols[execution_id]
            assert old["problem"] == new["problem"]
            assert old["request"] == new["request"]
            assert old["route_contract"] == new["route_contract"]
            assert old["horizon"] == new["horizon"]
            assert old["algorithm_id"] == new["algorithm_id"]
