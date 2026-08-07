from __future__ import annotations

import ast
import hashlib
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import pytest

import pipelines.static_adapt.current_checkpoint as current_checkpoint


def _checkpoint_payload(
    *,
    depth: int,
    route_family: str = "singleton_response_snake",
) -> dict[str, Any]:
    history = [
        {
            "depth": index,
            "pool_index": index - 1,
            "selected_batch_labels": [f"X{index}"],
            "selected_pool_indices": [index - 1],
            "selected_logical_size": 1,
            "selected_feature_rows": [
                {
                    "controller_snapshot": {
                        "controller_round": index,
                    }
                }
            ],
        }
        for index in range(1, depth + 1)
    ]
    if route_family in {
        "greedy_batch_response_snake",
        "combinatorial_batch_response_snake",
    }:
        admission_field = (
            "greedy_batch_admission"
            if route_family == "greedy_batch_response_snake"
            else "combinatorial_batch_admission"
        )
        for index, row in enumerate(history, start=1):
            row["active_prefix_checkpoint"] = {
                "schema": "test_active_prefix_checkpoint_v1",
                "depth": index,
            }
            row[admission_field] = {
                "composition_identity": f"batch-{index}",
                "selected_record_ids": [f"record-{index}"],
                "selected_generator_ids": [f"generator-{index}"],
                "selected_original_positions": [index - 1],
                "selected_effective_positions": [index - 1],
            }
    return {
        "adapt_vqe": {
            "route_family": route_family,
            "history": history,
            "history_count": depth,
            "pool_size": 8,
            "operators": [f"X{index}" for index in range(1, depth + 1)],
            "estimator_call_accounting": {
                "S_alg": depth,
                "S_unique": depth,
            },
            "terminal_active_prefix_checkpoint": {
                "schema": "test_active_prefix_checkpoint_v1",
                "depth": depth,
            },
        }
    }


def _ledger_payload(*, depth: int) -> dict[str, Any]:
    return {
        "schema": "estimator_call_ledger_v1",
        "ledger_fingerprint": f"ledger-at-depth-{depth}",
        "summary": {
            "unique_primitive_count": depth,
            "S_unique": depth,
        },
        "occurrence_summary": {
            "total_call_occurrences": depth,
        },
    }


def _published_sidecar_paths(
    current_path: Path,
    *,
    route_pointer_field: str = "verified_singleton_resume_sidecar",
) -> set[Path]:
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    adapt = payload["adapt_vqe"]
    return {
        current_path.with_name(adapt[pointer_name]["path"])
        for pointer_name in (
            "estimator_call_ledger_checkpoint",
            route_pointer_field,
        )
    }


def test_neutral_checkpoint_helpers_are_owned_by_current_checkpoint() -> None:
    assert (
        current_checkpoint._publish_active_cli_current_checkpoint.__module__
        == "pipelines.static_adapt.current_checkpoint"
    )
    assert (
        current_checkpoint._stable_json_digest.__module__
        == "pipelines.static_adapt.current_checkpoint"
    )


def test_current_checkpoint_defines_its_neutral_helpers() -> None:
    owner_path = Path(current_checkpoint.__file__)
    tree = ast.parse(owner_path.read_text(encoding="utf-8"))
    defined_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_publish_active_cli_current_checkpoint",
        "_stable_json_digest",
    }.issubset(defined_functions)


def test_publishing_successor_retires_unreferenced_predecessor_sidecars(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_sidecars = _published_sidecar_paths(current_path)
    assert all(path.is_file() for path in predecessor_sidecars)

    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=2),
        ledger_payload=_ledger_payload(depth=2),
        path=current_path,
        keep_history_tail=1,
    )

    successor_sidecars = _published_sidecar_paths(current_path)
    assert predecessor_sidecars.isdisjoint(successor_sidecars)
    assert all(path.is_file() for path in successor_sidecars)
    assert all(not path.exists() for path in predecessor_sidecars)


@pytest.mark.parametrize(
    ("route_family", "route_pointer_field"),
    (
        (
            "greedy_batch_response_snake",
            "greedy_batch_checkpoint_sidecar",
        ),
        (
            "combinatorial_batch_response_snake",
            "combinatorial_batch_checkpoint_sidecar",
        ),
    ),
)
def test_publishing_successor_retires_predecessor_batch_route_sidecar(
    tmp_path: Path,
    route_family: str,
    route_pointer_field: str,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1, route_family=route_family),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_sidecars = _published_sidecar_paths(
        current_path,
        route_pointer_field=route_pointer_field,
    )

    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=2, route_family=route_family),
        ledger_payload=_ledger_payload(depth=2),
        path=current_path,
        keep_history_tail=1,
    )

    successor_sidecars = _published_sidecar_paths(
        current_path,
        route_pointer_field=route_pointer_field,
    )
    assert predecessor_sidecars.isdisjoint(successor_sidecars)
    assert all(path.is_file() for path in successor_sidecars)
    assert all(not path.exists() for path in predecessor_sidecars)


def test_sidecar_retention_preserves_unreferenced_and_tampered_evidence(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_sidecars = _published_sidecar_paths(current_path)
    tampered_sidecar = next(
        path
        for path in predecessor_sidecars
        if ".estimator_call_ledger_checkpoint." in path.name
    )
    tampered_sidecar.write_text("tampered failure evidence\n", encoding="utf-8")
    unrelated_sidecar = (
        tmp_path / "current.verified_singleton_resume.0000000000000000.json"
    )
    unrelated_sidecar.write_text("unreferenced evidence\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with caplog.at_level(
            logging.WARNING,
            logger=current_checkpoint.__name__,
        ):
            current_checkpoint._publish_active_cli_current_checkpoint(
                _checkpoint_payload(depth=2),
                ledger_payload=_ledger_payload(depth=2),
                path=current_path,
                keep_history_tail=1,
            )

    assert tampered_sidecar.read_text(encoding="utf-8") == (
        "tampered failure evidence\n"
    )
    assert unrelated_sidecar.read_text(encoding="utf-8") == (
        "unreferenced evidence\n"
    )
    assert all(
        not path.exists()
        for path in predecessor_sidecars
        if path != tampered_sidecar
    )
    assert any(
        "digest mismatch" in message for message in caplog.messages
    )


def test_disagreeing_ledger_pointer_owners_preserve_both_sidecars(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor = json.loads(current_path.read_text(encoding="utf-8"))
    adapt_ledger_path = current_path.with_name(
        predecessor["adapt_vqe"]["estimator_call_ledger_checkpoint"]["path"]
    )
    disagreement_bytes = b"disagreeing pointer failure evidence\n"
    disagreement_sha256 = hashlib.sha256(disagreement_bytes).hexdigest()
    disagreement_path = current_path.with_name(
        "current.estimator_call_ledger_checkpoint."
        f"{disagreement_sha256[:16]}.json"
    )
    disagreement_path.write_bytes(disagreement_bytes)
    predecessor["checkpoint"]["estimator_call_ledger_checkpoint"].update(
        {
            "path": disagreement_path.name,
            "sha256": disagreement_sha256,
        }
    )
    current_path.write_text(
        json.dumps(predecessor, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with caplog.at_level(
        logging.WARNING,
        logger=current_checkpoint.__name__,
    ):
        current_checkpoint._publish_active_cli_current_checkpoint(
            _checkpoint_payload(depth=2),
            ledger_payload=_ledger_payload(depth=2),
            path=current_path,
            keep_history_tail=1,
        )

    assert adapt_ledger_path.is_file()
    assert disagreement_path.read_bytes() == disagreement_bytes
    assert any(
        "checkpoint owners disagree" in message
        for message in caplog.messages
    )


def test_failed_successor_publication_preserves_predecessor_and_new_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_path = tmp_path / "current.json"
    current_checkpoint._publish_active_cli_current_checkpoint(
        _checkpoint_payload(depth=1),
        ledger_payload=_ledger_payload(depth=1),
        path=current_path,
        keep_history_tail=1,
    )
    predecessor_bytes = current_path.read_bytes()
    predecessor_sidecars = _published_sidecar_paths(current_path)
    original_replace = current_checkpoint.os.replace

    def _fail_current_publication(source: str, destination: str) -> None:
        if Path(destination) == current_path:
            raise OSError("test current-checkpoint publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(
        current_checkpoint.os,
        "replace",
        _fail_current_publication,
    )

    with pytest.raises(
        OSError,
        match="test current-checkpoint publication failure",
    ):
        current_checkpoint._publish_active_cli_current_checkpoint(
            _checkpoint_payload(depth=2),
            ledger_payload=_ledger_payload(depth=2),
            path=current_path,
            keep_history_tail=1,
        )

    assert current_path.read_bytes() == predecessor_bytes
    assert all(path.is_file() for path in predecessor_sidecars)
    assert len(
        tuple(
            tmp_path.glob(
                "current.estimator_call_ledger_checkpoint.*.json"
            )
        )
    ) == 2
    assert len(
        tuple(tmp_path.glob("current.verified_singleton_resume.*.json"))
    ) == 2
