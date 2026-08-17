from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page12_insertion_comparators_20260812.py"
)
PAGE16_RUNNER_PATH = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "run_local_page16_insertion_comparators_20260812.py"
)

PAGE16_CONDITIONAL_EXECUTION_IDS = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_"
    "append_only",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_weak__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_append_only",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_strong__nph7__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_"
    "always_commutation_reduced",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_append_only",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_strong__nph7__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_"
    "append_only",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_strong__"
    "nph7__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_strong_u8__"
    "nph7__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_strong__"
    "nph7__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_append_only",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_strong_u8__"
    "nph7__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_append_only",
)
PAGE16_TERMINAL_CHTC_EXECUTION_IDS = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_"
    "always_commutation_reduced",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_weak__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__strong_weak_u8__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced",
)


def _load_runner():
    name = "paper_i_page12_local_insertion_fallback_test"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_at(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _macro_terminal_receipt(
    worker, *, eligible: tuple[str, ...] = (),
) -> dict:
    eligible_set = set(eligible)
    stopped = tuple(
        execution_id
        for execution_id in PAGE16_CONDITIONAL_EXECUTION_IDS
        if execution_id not in eligible_set
    )
    return worker.digested(
        {
            "schema": (
                "paper_i_page16_insertion_comparator_macro_k30_k50_"
                "terminal_clearance_v1"
            ),
            "status": "passed_all_required_macro_k30_k50_work_terminal",
            "adapter_sha256": (
                "56c50f046759d4299d768cb609f08fce8c79e3190aaadb6609afdde4f5452e07"
            ),
            "activation_manifest_sha256": "a" * 64,
            "runtime_manifest_sha256": "b" * 64,
            "k30_runtime_manifest_sha256": "c" * 64,
            "decision_status_sha256": "d" * 64,
            "terminal_chtc_status_sha256": "e" * 64,
            "conditional_execution_ids": list(PAGE16_CONDITIONAL_EXECUTION_IDS),
            "terminal_chtc_k50_execution_ids": list(
                PAGE16_TERMINAL_CHTC_EXECUTION_IDS
            ),
            "eligible_k50_continuation_execution_ids": list(eligible),
            "stop_at_k30_execution_ids": list(stopped),
            "closed_k50_continuation_execution_ids": list(eligible),
            "all_k30_cells_closed": True,
            "all_extension_required_cells_closed_at_k50": True,
            "remaining_macro_execution_ids": [],
            "active_macro_execution_ids": [],
            "scientific_execution_performed_by_receipt": False,
        }
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _redigest(worker, value: dict) -> dict:
    return worker.digested(
        {key: item for key, item in value.items() if key != "sha256"}
    )


@pytest.mark.parametrize(
    "order",
    (("page12", "page16"), ("page16", "page12")),
    ids=("page12_then_page16", "page16_then_page12"),
)
def test_sealed_worker_loaders_are_isolated_in_both_orders(order) -> None:
    runner_names = {
        "page12": f"paper_i_page12_loader_order_{order[0]}",
        "page16": f"paper_i_page16_loader_order_{order[0]}",
    }
    worker_names = (
        "paper_i_page12_insertion_comparator_sealed_worker",
        "paper_i_page16_insertion_comparator_sealed_worker",
    )
    tracked_modules = (
        "package_contract",
        *worker_names,
        *runner_names.values(),
    )
    missing = object()
    saved_modules = {
        name: sys.modules.get(name, missing) for name in tracked_modules
    }
    saved_path = list(sys.path)
    try:
        for name in tracked_modules:
            sys.modules.pop(name, None)
        page12 = _load_runner_at(RUNNER_PATH, runner_names["page12"])
        page16 = _load_runner_at(PAGE16_RUNNER_PATH, runner_names["page16"])
        package_paths = {
            page12.PACKAGE_DIR.as_posix(),
            page16.PACKAGE_DIR.as_posix(),
        }
        sys.path[:] = [entry for entry in sys.path if entry not in package_paths]
        if order[0] == "page12":
            # Reproduce the combined-suite state: Page-16's package directory
            # is already present but Page-12 is loaded next.
            sys.path.insert(0, page16.PACKAGE_DIR.as_posix())

        runners = {"page12": page12, "page16": page16}
        expected_package_ids = {
            "page12": (
                "paper_i_ra_adapt_page12_insertion_comparators_"
                "r50_20260812_v1_chtc"
            ),
            "page16": (
                "paper_i_ra_adapt_page16_insertion_comparators_"
                "weak50_strong30_20260812_v1_chtc"
            ),
        }
        loaded = {}
        for label in order:
            before_path = list(sys.path)
            before_contract = sys.modules.get("package_contract", missing)
            loaded[label] = runners[label]._load_worker()
            assert loaded[label].PACKAGE_ID == expected_package_ids[label]
            if label == "page12":
                assert sys.path == before_path
                assert sys.modules.get("package_contract", missing) is before_contract

        assert {
            label: worker.PACKAGE_ID for label, worker in loaded.items()
        } == {
            label: expected_package_ids[label] for label in order
        }
    finally:
        sys.path[:] = saved_path
        for name, module in saved_modules.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_page12_fallback_is_exact_dormant_and_current_mac_ineligible(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    manifest, rows = runner._closed_package(worker)

    assert manifest["sha256"] == runner.PACKAGE_MANIFEST_CANONICAL_SHA256
    assert tuple(row["execution_id"] for row in rows) == runner.EXECUTION_IDS
    assert len(runner.EXECUTION_IDS) == 12
    assert runner.EXECUTION_IDS == tuple(
        runner._execution_id(regime, nph, policy)
        for policy in ("always_commutation_reduced", "append_only")
        for regime, nph in runner.REGIMES
    )
    assert [
        worker._load_closed_job(runner.PACKAGE_DIR / row["job_path"])[0][
            "comparator_policy"
        ]
        for row in rows
    ] == ["always_commutation_reduced"] * 6 + ["append_only"] * 6

    monkeypatch.setattr(
        runner, "_physical_memory_bytes", lambda: 16 * 1024**3
    )
    monkeypatch.setattr(runner, "_memory_pressure_percent", lambda: 50)
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=40 * 1024**3),
    )
    capacity = runner._capacity_receipt(worker, tmp_path)
    assert capacity["status"] == "blocked"
    assert capacity["all_rows_local_capable"] is False
    assert sum(
        row["locally_eligible"] for row in capacity["row_assessments"]
    ) == 0

    activation_dir = tmp_path / "activation"
    activation = runner.prepare_activation(
        activation_dir=activation_dir,
        output_parent=tmp_path / "outputs",
    )
    assert activation["execution_authorized"] is False
    assert activation["submission_authorized"] is False
    assert activation["execution_entrypoint_present"] is False
    assert activation["authorization_count"] == 12
    for row in activation["authorizations"]:
        authority = json.loads(
            (activation_dir / row["path"]).read_text(encoding="utf-8")
        )
        assert authority["execution_authorized"] is False

    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert 'mode.add_argument("--run"' not in source
    assert 'mode.add_argument("--run-cell"' not in source
    assert 'mode.add_argument("--run-wave"' not in source


def test_macro_terminal_gate_requires_exact_hybrid_inventory_and_partition(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    receipt_path = tmp_path / "macro-terminal.json"
    receipt = _macro_terminal_receipt(
        worker,
        eligible=(PAGE16_CONDITIONAL_EXECUTION_IDS[2],),
    )
    _write_json(receipt_path, receipt)
    monkeypatch.setattr(
        runner,
        "_trusted_macro_terminal_replay",
        lambda _path: receipt,
        raising=False,
    )

    passed, value, blocker = runner._external_gate(
        worker,
        receipt_path,
        kind="macro_terminal",
        now=datetime.now(timezone.utc),
    )
    assert passed is True
    assert value == receipt
    assert blocker is None

    mutations = (
        {
            "conditional_execution_ids": list(
                PAGE16_CONDITIONAL_EXECUTION_IDS[1:]
            )
        },
        {
            "terminal_chtc_k50_execution_ids": list(
                reversed(PAGE16_TERMINAL_CHTC_EXECUTION_IDS)
            )
        },
        {
            "eligible_k50_continuation_execution_ids": [
                PAGE16_CONDITIONAL_EXECUTION_IDS[2]
            ],
            "stop_at_k30_execution_ids": list(
                PAGE16_CONDITIONAL_EXECUTION_IDS
            ),
        },
        {
            "eligible_k50_continuation_execution_ids": [],
            "stop_at_k30_execution_ids": list(
                PAGE16_CONDITIONAL_EXECUTION_IDS[1:]
            ),
            "closed_k50_continuation_execution_ids": [],
        },
        {"closed_k50_continuation_execution_ids": []},
    )
    for mutation in mutations:
        invalid = _redigest(worker, {**receipt, **mutation})
        _write_json(receipt_path, invalid)
        passed, value, blocker = runner._external_gate(
            worker,
            receipt_path,
            kind="macro_terminal",
            now=datetime.now(timezone.utc),
        )
        assert passed is False
        assert value is None
        assert blocker is not None
        assert blocker.startswith("macro_terminal_receipt_invalid:")


def test_macro_terminal_gate_requires_pinned_producer_and_terminal_hashes(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    receipt_path = tmp_path / "macro-terminal.json"
    receipt = _macro_terminal_receipt(worker)
    _write_json(receipt_path, receipt)
    monkeypatch.setattr(
        runner,
        "_trusted_macro_terminal_replay",
        lambda _path: receipt,
        raising=False,
    )

    assert runner.EXPECTED_MACRO_TERMINAL_PRODUCER_SHA256 == (
        "0e3a342fa21d925c941a4c3b8e0476c23907ba52d46d459310527a6e0123d761"
    )
    assert runner.EXPECTED_MACRO_TERMINAL_ADAPTER_SHA256 == (
        "56c50f046759d4299d768cb609f08fce8c79e3190aaadb6609afdde4f5452e07"
    )

    invalid_fields = (
        ("adapter_sha256", "f" * 64),
        ("activation_manifest_sha256", "not-hex"),
        ("runtime_manifest_sha256", "1" * 63),
        ("k30_runtime_manifest_sha256", "g" * 64),
        ("decision_status_sha256", None),
        ("terminal_chtc_status_sha256", ""),
        ("all_k30_cells_closed", 1),
        ("all_extension_required_cells_closed_at_k50", 1),
        ("remaining_macro_execution_ids", ["unfinished"]),
        ("active_macro_execution_ids", ["active"]),
        ("scientific_execution_performed_by_receipt", 0),
    )
    for field, invalid_value in invalid_fields:
        invalid = _redigest(worker, {**receipt, field: invalid_value})
        _write_json(receipt_path, invalid)
        passed, value, blocker = runner._external_gate(
            worker,
            receipt_path,
            kind="macro_terminal",
            now=datetime.now(timezone.utc),
        )
        assert passed is False
        assert value is None
        assert blocker is not None
        assert blocker.startswith("macro_terminal_receipt_invalid:")

    _write_json(receipt_path, receipt)
    for source_attribute in (
        "MACRO_TERMINAL_PRODUCER_PATH",
        "MACRO_TERMINAL_ADAPTER_PATH",
    ):
        drifted_source = tmp_path / f"{source_attribute}.py"
        drifted_source.write_text("# drifted\n", encoding="utf-8")
        with monkeypatch.context() as scoped:
            scoped.setattr(runner, source_attribute, drifted_source)
            passed, value, blocker = runner._external_gate(
                worker,
                receipt_path,
                kind="macro_terminal",
                now=datetime.now(timezone.utc),
            )
        assert passed is False
        assert value is None
        assert blocker is not None
        assert blocker.startswith("macro_terminal_receipt_invalid:")


def test_macro_terminal_gate_requires_exact_trusted_replay(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    receipt_path = tmp_path / "macro-terminal.json"
    stored = _macro_terminal_receipt(worker)
    trusted = _redigest(
        worker,
        {**stored, "decision_status_sha256": "f" * 64},
    )
    _write_json(receipt_path, stored)
    monkeypatch.setattr(
        runner,
        "_trusted_macro_terminal_replay",
        lambda _path: trusted,
        raising=False,
    )

    passed, value, blocker = runner._external_gate(
        worker,
        receipt_path,
        kind="macro_terminal",
        now=datetime.now(timezone.utc),
    )

    assert passed is False
    assert value is None
    assert blocker is not None
    assert "trusted replay" in blocker


def test_macro_terminal_gate_fails_closed_on_isolated_unclosed_replay(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    receipt_path = tmp_path / "macro-terminal.json"
    receipt = _macro_terminal_receipt(
        worker,
        eligible=(PAGE16_CONDITIONAL_EXECUTION_IDS[2],),
    )
    _write_json(receipt_path, receipt)
    calls = []

    def reject_replay(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            returncode=2,
            stdout="",
            stderr="eligible k50 continuation remains unclosed",
        )

    monkeypatch.setattr(runner.subprocess, "run", reject_replay)

    passed, value, blocker = runner._external_gate(
        worker,
        receipt_path,
        kind="macro_terminal",
        now=datetime.now(timezone.utc),
    )

    assert passed is False
    assert value is None
    assert blocker is not None
    assert "eligible k50 continuation remains unclosed" in blocker
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[:3] == [sys.executable, "-B", "-c"]
    assert kwargs["cwd"] == runner.REPO_ROOT
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True


def test_macro_terminal_gate_accepts_only_exact_isolated_recomputation(
    tmp_path: Path, monkeypatch,
) -> None:
    runner = _load_runner()
    worker = runner._load_worker()
    receipt_path = tmp_path / "macro-terminal.json"
    receipt = _macro_terminal_receipt(worker)
    _write_json(receipt_path, receipt)

    def replaying(value):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(value, sort_keys=True, separators=(",", ":")),
            stderr="",
        )

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda _command, **_kwargs: replaying(receipt),
    )
    passed, value, blocker = runner._external_gate(
        worker,
        receipt_path,
        kind="macro_terminal",
        now=datetime.now(timezone.utc),
    )
    assert passed is True
    assert value == receipt
    assert blocker is None

    drifted = _redigest(
        worker,
        {**receipt, "terminal_chtc_status_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda _command, **_kwargs: replaying(drifted),
    )
    passed, value, blocker = runner._external_gate(
        worker,
        receipt_path,
        kind="macro_terminal",
        now=datetime.now(timezone.utc),
    )
    assert passed is False
    assert value is None
    assert blocker is not None
    assert "trusted replay" in blocker
