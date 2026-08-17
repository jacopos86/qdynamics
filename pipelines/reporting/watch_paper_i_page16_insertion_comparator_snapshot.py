#!/usr/bin/env python3
"""Refresh Page 17 after each authenticated comparator evidence revision.

The watcher is reporting-only.  It never launches, resumes, stops, or adopts a
scientific run.  A local curve becomes renderable only after its execution
manifest, worker receipt, plateau gate, summary, and complete artifact inventory
close against the fixed activation.  Closed continuation-eligible k=30 cells
remain right-censored until a source-locked authenticated k=50 closure replaces
that same execution ID in the report.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    append_paper_i_insertion_comparator_snapshot_pages as snapshot,
)


UPDATER_PATH = Path(snapshot.__file__).resolve()
WATCH_STATUS_PATH = snapshot.REPORT_DIR / f"{snapshot.STEM}_watch_status.json"
LOCK_PATH = snapshot.REPORT_DIR / f"{snapshot.STEM}_watch.lock"
STATUS_SCHEMA = "paper_i_page16_insertion_comparator_page17_auto_refresh_status_v1"
MIN_POLL_SECONDS = 30.0
DEFAULT_MAX_POLL_SECONDS = 300.0
TERMINAL_WAVE_STATES = {"failed", "interrupted"}
RUNNER_NAME = "run_local_page16_insertion_comparators_20260812.py"
SUPERVISOR_NAME = "supervise_local_page16_insertion_comparator_waves_20260812.py"
EXPECTED_CHTC_EXECUTION_IDS = (
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__weak_weak__nph3__"
    "ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_no_lanes_"
    "always_commutation_reduced",
    "page16_macro_gradient_phase0_phase23_qiskit_no_lanes__intermediate_weak__"
    "nph3__ra_page16_macro_gradient_phase0_macro_phase123_qiskit_phase23_"
    "no_lanes_always_commutation_reduced",
    snapshot.SW_ALWAYS_EXECUTION_ID,
)


class WatchError(ValueError):
    pass


def _same_exact_id_set(actual: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    return len(actual) == len(expected) and set(actual) == set(expected)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return snapshot._canonical_sha256(value)


def _write_status(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(dict(value))
    unsigned["updated_at_utc"] = _utc_now()
    payload = {**unsigned, "sha256": _canonical_sha256(unsigned)}
    WATCH_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot._atomic_json(WATCH_STATUS_PATH, payload)
    artifact_root = os.environ.get("REMOTE_ARTIFACT_DIR")
    if artifact_root:
        artifact_path = Path(artifact_root) / "page17_auto_refresh_status.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot._atomic_json(artifact_path, payload)
    return payload


def _load_previous_status() -> dict[str, Any] | None:
    if not WATCH_STATUS_PATH.exists() and not WATCH_STATUS_PATH.is_symlink():
        return None
    value = snapshot._load_digested_file(
        WATCH_STATUS_PATH, label="Page-17 watcher status"
    )
    if value.get("schema") != STATUS_SCHEMA:
        raise WatchError("Page-17 watcher status schema drifted")
    return value


def _load_source_status(path: Path, *, label: str) -> dict[str, Any] | None:
    if not path.exists() and not path.is_symlink():
        return None
    return snapshot._load_digested_file(path, label=label)


def _active_campaign_processes(runtime_dir: Path) -> list[dict[str, Any]]:
    try:
        output = subprocess.run(
            ["ps", "-axo", "pid=,command=", "-ww"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WatchError("cannot inspect active local campaign processes") from exc
    own_pid = os.getpid()
    matches: list[dict[str, Any]] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        pid_text, _, command = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == own_pid:
            continue
        kind = None
        if RUNNER_NAME in command and runtime_dir.as_posix() in command:
            kind = "runner"
        elif SUPERVISOR_NAME in command:
            kind = "supervisor"
        if kind is not None:
            matches.append({"pid": pid, "kind": kind})
    return matches


def _campaign_snapshot(
    *,
    runtime_dir: Path,
    activation_dir: Path,
    expected_adapter_sha256: str,
) -> dict[str, Any]:
    inventory = snapshot.authenticated_effective_local_comparator_inventory(
        runtime_dir=runtime_dir,
        activation_dir=activation_dir,
        expected_adapter_sha256=expected_adapter_sha256,
        compile_costs=False,
    )
    wave_statuses: dict[str, Any] = {}
    explicit_failures: list[dict[str, Any]] = []
    for wave_number in range(1, 6):
        path = runtime_dir / "status" / f"wave_{wave_number}.json"
        status = _load_source_status(path, label=f"local wave {wave_number} status")
        if status is None:
            continue
        if status.get("wave") != wave_number:
            raise WatchError(f"local wave {wave_number} status identity drifted")
        state = str(status.get("status", ""))
        wave_statuses[str(wave_number)] = {
            "status": state,
            "canonical_sha256": status["sha256"],
            "completed_execution_ids": copy.deepcopy(
                status.get("completed_execution_ids", [])
            ),
        }
        if state in TERMINAL_WAVE_STATES:
            explicit_failures.append(
                {
                    "source": f"wave_{wave_number}",
                    "status": state,
                    "failure": copy.deepcopy(status.get("failure")),
                }
            )
    supervisor_path = runtime_dir / "status" / "waves_2_5_supervisor.json"
    supervisor = _load_source_status(
        supervisor_path, label="local waves 2--5 supervisor status"
    )
    supervisor_status = None
    if supervisor is not None:
        supervisor_status = {
            "status": str(supervisor.get("status", "")),
            "canonical_sha256": supervisor["sha256"],
            "completed_waves": copy.deepcopy(supervisor.get("completed_waves", [])),
            "next_wave": supervisor.get("next_wave"),
        }
        if str(supervisor.get("status", "")) in TERMINAL_WAVE_STATES:
            explicit_failures.append(
                {
                    "source": "waves_2_5_supervisor",
                    "status": str(supervisor["status"]),
                }
            )
    active = _active_campaign_processes(runtime_dir)
    closed_ids = tuple(
        execution_id
        for execution_id in inventory["execution_ids"]
        if execution_id in inventory["completed"]
    )
    inventory_ids = tuple(str(row) for row in inventory["execution_ids"])
    expected_local_ids = tuple(
        execution_id
        for execution_id in inventory_ids
        if execution_id != snapshot.SW_ALWAYS_EXECUTION_ID
    )
    if (
        len(inventory_ids) != 10
        or len(set(inventory_ids)) != 10
        or snapshot.SW_ALWAYS_EXECUTION_ID not in inventory_ids
        or len(expected_local_ids) != 9
    ):
        raise WatchError("hybrid local/CHTC execution inventory drifted")
    partial_ids = tuple(
        execution_id
        for execution_id in inventory["execution_ids"]
        if str(inventory["cell_states"].get(execution_id, "")).startswith(
            "published_partial_unclosed"
        )
    )
    terminal_failed = bool(explicit_failures) and not active
    all_closed = _same_exact_id_set(closed_ids, expected_local_ids) and not partial_ids
    return {
        "inventory": inventory,
        "closed_execution_ids": closed_ids,
        "unclosed_published_execution_ids": partial_ids,
        "wave_statuses": wave_statuses,
        "supervisor_status": supervisor_status,
        "active_campaign_processes": active,
        "explicit_failures": explicit_failures,
        "terminal_failed_no_active_supervisor_path": terminal_failed,
        "expected_local_execution_ids": expected_local_ids,
        "all_nine_local_cells_closed": all_closed,
        "all_required_continuations_closed": inventory[
            "all_required_continuations_closed"
        ],
        "macro_terminal_authenticated": inventory[
            "macro_terminal_authenticated"
        ],
        "continuation_evidence_revision": inventory[
            "continuation_evidence_revision"
        ],
    }


def _reported_local_execution_ids() -> tuple[str, ...]:
    provenance = snapshot.load(snapshot.TARGET_PROVENANCE)
    report = provenance.get("phase0_insertion_comparator_snapshot")
    if not isinstance(report, Mapping):
        return ()
    completed = report.get("completed_comparators")
    if not isinstance(completed, Mapping):
        return ()
    result: list[str] = []
    for regime in snapshot.REGIME_ORDER:
        policies = completed.get(regime)
        if not isinstance(policies, Mapping):
            continue
        for policy in snapshot.EXPECTED_POLICIES:
            row = policies.get(policy)
            if (
                isinstance(row, Mapping)
                and row.get("execution_origin") == "local"
                and isinstance(row.get("execution_id"), str)
            ):
                result.append(str(row["execution_id"]))
    return tuple(result)


def _reported_local_evidence_revisions() -> dict[str, str]:
    provenance = snapshot.load(snapshot.TARGET_PROVENANCE)
    report = provenance.get("phase0_insertion_comparator_snapshot")
    if not isinstance(report, Mapping):
        return {}
    completed = report.get("completed_comparators")
    if not isinstance(completed, Mapping):
        return {}
    revisions: dict[str, str] = {}
    for regime in snapshot.REGIME_ORDER:
        policies = completed.get(regime)
        if not isinstance(policies, Mapping):
            continue
        for policy in snapshot.EXPECTED_POLICIES:
            row = policies.get(policy)
            if not isinstance(row, Mapping) or row.get("execution_origin") != "local":
                continue
            execution_id = row.get("execution_id")
            revision = row.get("evidence_revision")
            if not isinstance(execution_id, str) or revision is None:
                continue
            if (
                not isinstance(revision, str)
                or len(revision) != 64
                or any(
                    character not in "0123456789abcdef" for character in revision
                )
                or execution_id in revisions
            ):
                raise WatchError("reported local evidence revision is malformed")
            revisions[execution_id] = revision
    return revisions


def _reported_continuation_evidence_revision() -> str | None:
    provenance = snapshot.load(snapshot.TARGET_PROVENANCE)
    report = provenance.get("phase0_insertion_comparator_snapshot")
    if not isinstance(report, Mapping):
        return None
    campaign = report.get("campaign_execution_state")
    if not isinstance(campaign, Mapping):
        return None
    revision = campaign.get("continuation_evidence_revision")
    if revision is None:
        return None
    if (
        not isinstance(revision, str)
        or len(revision) != 64
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise WatchError("reported continuation evidence revision is malformed")
    return revision


def _reported_chtc_execution_ids() -> tuple[str, ...]:
    provenance = snapshot.load(snapshot.TARGET_PROVENANCE)
    report = provenance.get("phase0_insertion_comparator_snapshot")
    if not isinstance(report, Mapping):
        return ()
    completed = report.get("completed_comparators")
    if not isinstance(completed, Mapping):
        return ()
    result: list[str] = []
    for regime in snapshot.REGIME_ORDER:
        policies = completed.get(regime)
        if not isinstance(policies, Mapping):
            continue
        for policy in snapshot.EXPECTED_POLICIES:
            row = policies.get(policy)
            if (
                isinstance(row, Mapping)
                and row.get("execution_origin") == "CHTC"
                and isinstance(row.get("execution_id"), str)
            ):
                result.append(str(row["execution_id"]))
    return tuple(result)


def _candidate_chtc_execution_ids() -> tuple[str, ...]:
    receipt = snapshot.SW_ALWAYS_CLOSURE_RECEIPT
    if receipt.exists() or receipt.is_symlink():
        return EXPECTED_CHTC_EXECUTION_IDS
    return EXPECTED_CHTC_EXECUTION_IDS[:2]


def _valid_update_page_shape(page_count: Any, preserved_page_count: Any) -> bool:
    """Accept Page 17 alone or Page 17 with the dense Page 18 preserved."""

    return (page_count, preserved_page_count) in {(17, 16), (18, 17)}


def _run_updater(
    *,
    runtime_dir: Path,
    activation_dir: Path,
    expected_adapter_sha256: str,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            UPDATER_PATH.as_posix(),
            "--runtime-dir",
            runtime_dir.as_posix(),
            "--activation-dir",
            activation_dir.as_posix(),
            "--expected-local-adapter-sha256",
            expected_adapter_sha256,
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise WatchError(
            "Page-17 updater failed: "
            + (completed.stderr.strip() or f"exit {completed.returncode}")
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise WatchError("Page-17 updater returned non-JSON output") from exc
    pdf = result.get("pdf")
    if (
        result.get("status") != "updated_existing_report_in_place"
        or not _valid_update_page_shape(
            result.get("page_count"), result.get("preserved_page_count")
        )
        or not isinstance(pdf, Mapping)
        or not isinstance(pdf.get("sha256"), str)
        or len(pdf["sha256"]) != 64
        or int(pdf.get("size_bytes", -1)) <= 0
    ):
        raise WatchError("Page-17 updater returned an unexpected result")
    result["reported_local_execution_ids"] = list(_reported_local_execution_ids())
    result["reported_local_evidence_revisions"] = (
        _reported_local_evidence_revisions()
    )
    result["reported_continuation_evidence_revision"] = (
        _reported_continuation_evidence_revision()
    )
    result["reported_chtc_execution_ids"] = list(_reported_chtc_execution_ids())
    return result


def _state_fingerprint(
    campaign: Mapping[str, Any],
    *,
    candidate_chtc_execution_ids: tuple[str, ...] = (),
    reported_chtc_execution_ids: tuple[str, ...] = (),
) -> str:
    return _canonical_sha256(
        {
            "closed_execution_ids": list(campaign["closed_execution_ids"]),
            "evidence_revisions": copy.deepcopy(
                campaign["inventory"]["evidence_revisions"]
            ),
            "all_required_continuations_closed": campaign[
                "all_required_continuations_closed"
            ],
            "macro_terminal_authenticated": campaign[
                "macro_terminal_authenticated"
            ],
            "continuation_evidence_revision": campaign[
                "continuation_evidence_revision"
            ],
            "unclosed_published_execution_ids": list(
                campaign["unclosed_published_execution_ids"]
            ),
            "wave_statuses": campaign["wave_statuses"],
            "supervisor_status": campaign["supervisor_status"],
            "active_process_kinds": [
                row["kind"] for row in campaign["active_campaign_processes"]
            ],
            "explicit_failures": campaign["explicit_failures"],
            "candidate_chtc_execution_ids": list(candidate_chtc_execution_ids),
            "reported_chtc_execution_ids": list(reported_chtc_execution_ids),
        }
    )


def _next_poll_seconds(
    *,
    previous: Mapping[str, Any] | None,
    fingerprint: str,
    base: float,
    maximum: float,
) -> float:
    if previous is None or previous.get("source_state_fingerprint") != fingerprint:
        return base
    prior = float(previous.get("next_poll_seconds", base))
    return min(maximum, max(base, prior * 1.5))


def _status_payload(
    *,
    status: str,
    campaign: Mapping[str, Any] | None,
    refreshed: tuple[str, ...],
    reported: tuple[str, ...],
    reported_revisions: Mapping[str, str],
    reported_continuation_revision: str | None,
    candidate_chtc: tuple[str, ...],
    reported_chtc: tuple[str, ...],
    fingerprint: str | None,
    next_poll_seconds: float | None,
    refresh_result: Mapping[str, Any] | None,
    last_error: str | None,
    runtime_dir: Path,
    activation_dir: Path,
    expected_adapter_sha256: str,
) -> dict[str, Any]:
    all_three_chtc_cells_refreshed = _same_exact_id_set(
        candidate_chtc, EXPECTED_CHTC_EXECUTION_IDS
    ) and _same_exact_id_set(reported_chtc, EXPECTED_CHTC_EXECUTION_IDS)
    expected_local = (
        ()
        if campaign is None
        else tuple(campaign.get("expected_local_execution_ids", ()))
    )
    all_nine_local_cells_refreshed = (
        campaign is not None
        and campaign["all_nine_local_cells_closed"]
        and len(expected_local) == 9
        and _same_exact_id_set(reported, expected_local)
        and dict(reported_revisions)
        == {
            execution_id: campaign["inventory"]["evidence_revisions"][execution_id]
            for execution_id in expected_local
        }
    )
    all_twelve_hybrid_cells_refreshed = (
        all_nine_local_cells_refreshed
        and all_three_chtc_cells_refreshed
        and campaign is not None
        and campaign["all_required_continuations_closed"]
        and campaign["macro_terminal_authenticated"]
        and reported_continuation_revision
        == campaign["continuation_evidence_revision"]
    )
    return {
        "schema": STATUS_SCHEMA,
        "status": status,
        "reporting_only": True,
        "scientific_execution_performed": False,
        "submission_performed": False,
        "paper_evidence_adopted": False,
        "runtime_dir": runtime_dir.as_posix(),
        "activation_dir": activation_dir.as_posix(),
        "expected_local_adapter_sha256": expected_adapter_sha256,
        "validated_closed_local_execution_ids": (
            [] if campaign is None else list(campaign["closed_execution_ids"])
        ),
        "unclosed_published_execution_ids": (
            []
            if campaign is None
            else list(campaign["unclosed_published_execution_ids"])
        ),
        "refreshed_local_execution_ids": list(refreshed),
        "reported_local_execution_ids": list(reported),
        "validated_local_evidence_revisions": (
            {}
            if campaign is None
            else copy.deepcopy(campaign["inventory"]["evidence_revisions"])
        ),
        "reported_local_evidence_revisions": dict(reported_revisions),
        "validated_continuation_evidence_revision": (
            None
            if campaign is None
            else campaign["continuation_evidence_revision"]
        ),
        "reported_continuation_evidence_revision": (
            reported_continuation_revision
        ),
        "candidate_chtc_execution_ids": list(candidate_chtc),
        "reported_chtc_execution_ids": list(reported_chtc),
        "source_campaign": (
            None
            if campaign is None
            else {
                "campaign_state": campaign["inventory"]["campaign_state"],
                "wave_statuses": copy.deepcopy(campaign["wave_statuses"]),
                "supervisor_status": copy.deepcopy(campaign["supervisor_status"]),
                "active_campaign_processes": copy.deepcopy(
                    campaign["active_campaign_processes"]
                ),
                "explicit_failures": copy.deepcopy(campaign["explicit_failures"]),
                "terminal_failed_no_active_supervisor_path": campaign[
                    "terminal_failed_no_active_supervisor_path"
                ],
                "expected_local_execution_ids": list(
                    campaign.get("expected_local_execution_ids", ())
                ),
                "all_nine_local_cells_closed": campaign[
                    "all_nine_local_cells_closed"
                ],
                "all_required_continuations_closed": campaign[
                    "all_required_continuations_closed"
                ],
                "macro_terminal_authenticated": campaign[
                    "macro_terminal_authenticated"
                ],
                "continuation_evidence_revision": campaign[
                    "continuation_evidence_revision"
                ],
                "all_three_chtc_cells_refreshed": (
                    all_three_chtc_cells_refreshed
                ),
                "all_twelve_hybrid_cells_refreshed": (
                    all_twelve_hybrid_cells_refreshed
                ),
            }
        ),
        "source_state_fingerprint": fingerprint,
        "next_poll_seconds": next_poll_seconds,
        "page17_pdf": (
            snapshot.binding(snapshot.TARGET_PDF)
            if snapshot.TARGET_PDF.is_file()
            else None
        ),
        "page17_provenance": (
            snapshot.binding(snapshot.TARGET_PROVENANCE)
            if snapshot.TARGET_PROVENANCE.is_file()
            else None
        ),
        "last_refresh_result": (
            None if refresh_result is None else copy.deepcopy(dict(refresh_result))
        ),
        "last_error": last_error,
    }


def watch(
    *,
    runtime_dir: Path,
    activation_dir: Path,
    expected_adapter_sha256: str,
    poll_seconds: float,
    max_poll_seconds: float,
    once: bool,
) -> int:
    previous = _load_previous_status()
    prior_refreshed = tuple(
        str(row)
        for row in (
            []
            if previous is None
            else previous.get("refreshed_local_execution_ids", [])
        )
    )
    while True:
        campaign: dict[str, Any] | None = None
        reported: tuple[str, ...] = ()
        reported_revisions: dict[str, str] = {}
        reported_continuation_revision: str | None = None
        candidate_chtc: tuple[str, ...] = ()
        reported_chtc: tuple[str, ...] = ()
        refreshed = prior_refreshed
        refresh_result: dict[str, Any] | None = None
        last_error: str | None = None
        try:
            campaign = _campaign_snapshot(
                runtime_dir=runtime_dir,
                activation_dir=activation_dir,
                expected_adapter_sha256=expected_adapter_sha256,
            )
            expected_ids = tuple(campaign["inventory"]["execution_ids"])
            expected_local_ids = tuple(campaign["expected_local_execution_ids"])
            closed = tuple(campaign["closed_execution_ids"])
            current_revisions = {
                execution_id: str(
                    campaign["inventory"]["evidence_revisions"][execution_id]
                )
                for execution_id in closed
            }
            current_continuation_revision = str(
                campaign["continuation_evidence_revision"]
            )
            reported = _reported_local_execution_ids()
            reported_revisions = _reported_local_evidence_revisions()
            reported_continuation_revision = (
                _reported_continuation_evidence_revision()
            )
            candidate_chtc = _candidate_chtc_execution_ids()
            reported_chtc = _reported_chtc_execution_ids()
            for label, ids in (
                ("prior refreshed", prior_refreshed),
                ("reported", reported),
            ):
                if len(ids) != len(set(ids)) or not set(ids).issubset(expected_ids):
                    raise WatchError(f"{label} local completion set is unauthorized")
            if not set(prior_refreshed).issubset(closed):
                raise WatchError(
                    "authenticated completion set regressed behind watcher status"
                )
            if not set(reported).issubset(closed):
                raise WatchError(
                    "PDF reports local evidence that no longer authenticates"
                )
            if (
                not set(reported_revisions).issubset(reported)
                or any(
                    len(revision) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in revision
                    )
                    for revision in reported_revisions.values()
                )
            ):
                raise WatchError("reported local evidence revisions are unauthorized")
            if (
                len(candidate_chtc) != len(set(candidate_chtc))
                or len(reported_chtc) != len(set(reported_chtc))
                or not set(reported_chtc).issubset(candidate_chtc)
            ):
                raise WatchError("reported CHTC completion set is unauthorized")
            needs_refresh = (
                set(reported) != set(closed)
                or reported_revisions != current_revisions
                or reported_continuation_revision
                != current_continuation_revision
                or set(reported_chtc) != set(candidate_chtc)
            )
            if needs_refresh:
                try:
                    refresh_result = _run_updater(
                        runtime_dir=runtime_dir,
                        activation_dir=activation_dir,
                        expected_adapter_sha256=expected_adapter_sha256,
                    )
                    reported = tuple(
                        str(row)
                        for row in refresh_result["reported_local_execution_ids"]
                    )
                    reported_revisions = {
                        str(key): str(value)
                        for key, value in refresh_result[
                            "reported_local_evidence_revisions"
                        ].items()
                    }
                    reported_continuation_revision = refresh_result[
                        "reported_continuation_evidence_revision"
                    ]
                    reported_chtc = tuple(
                        str(row)
                        for row in refresh_result["reported_chtc_execution_ids"]
                    )
                    if (
                        not _same_exact_id_set(reported, closed)
                        or reported_revisions != current_revisions
                        or reported_continuation_revision
                        != current_continuation_revision
                        or not _same_exact_id_set(reported_chtc, candidate_chtc)
                    ):
                        raise WatchError(
                            "Page-17 updater did not publish the authenticated revision"
                        )
                    refreshed = closed
                    print(
                        json.dumps(
                            {
                                "event": "page17_refreshed",
                                "closed_local_count": len(closed),
                                "closed_local_execution_ids": list(closed),
                                "pdf_sha256": refresh_result["pdf"]["sha256"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                except (OSError, ValueError) as exc:
                    last_error = str(exc)
                    print(
                        json.dumps(
                            {"event": "page17_refresh_failed", "error": last_error},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            fingerprint = _state_fingerprint(
                campaign,
                candidate_chtc_execution_ids=candidate_chtc,
                reported_chtc_execution_ids=reported_chtc,
            )
            next_poll = _next_poll_seconds(
                previous=previous,
                fingerprint=fingerprint,
                base=poll_seconds,
                maximum=max_poll_seconds,
            )
            if campaign["terminal_failed_no_active_supervisor_path"]:
                state = "source_campaign_terminal_failed_no_active_supervisor_path"
                next_poll = None
            elif (
                campaign["all_nine_local_cells_closed"]
                and len(expected_local_ids) == 9
                and _same_exact_id_set(closed, expected_local_ids)
                and _same_exact_id_set(reported, expected_local_ids)
                and reported_revisions == current_revisions
                and reported_continuation_revision
                == current_continuation_revision
                and _same_exact_id_set(
                    candidate_chtc, EXPECTED_CHTC_EXECUTION_IDS
                )
                and _same_exact_id_set(reported_chtc, EXPECTED_CHTC_EXECUTION_IDS)
                and campaign["all_required_continuations_closed"]
                and campaign["macro_terminal_authenticated"]
            ):
                state = "passed_all_twelve_hybrid_cells_refreshed"
                next_poll = None
            elif last_error is not None:
                state = "refresh_retry_pending"
            elif (
                campaign["all_nine_local_cells_closed"]
                and _same_exact_id_set(
                    candidate_chtc, EXPECTED_CHTC_EXECUTION_IDS
                )
                and _same_exact_id_set(reported_chtc, EXPECTED_CHTC_EXECUTION_IDS)
                and campaign["all_required_continuations_closed"]
                and not campaign["macro_terminal_authenticated"]
            ):
                state = "watching_for_authenticated_macro_terminal_receipt"
            elif (
                campaign["all_nine_local_cells_closed"]
                and _same_exact_id_set(
                    candidate_chtc, EXPECTED_CHTC_EXECUTION_IDS
                )
                and _same_exact_id_set(reported_chtc, EXPECTED_CHTC_EXECUTION_IDS)
                and (
                    not campaign["all_required_continuations_closed"]
                    or not campaign["macro_terminal_authenticated"]
                )
            ):
                state = "watching_for_required_k50_continuations"
            else:
                state = "watching_for_next_authenticated_local_closure"
            payload = _status_payload(
                status=state,
                campaign=campaign,
                refreshed=refreshed,
                reported=reported,
                reported_revisions=reported_revisions,
                reported_continuation_revision=(
                    reported_continuation_revision
                ),
                candidate_chtc=candidate_chtc,
                reported_chtc=reported_chtc,
                fingerprint=fingerprint,
                next_poll_seconds=next_poll,
                refresh_result=refresh_result,
                last_error=last_error,
                runtime_dir=runtime_dir,
                activation_dir=activation_dir,
                expected_adapter_sha256=expected_adapter_sha256,
            )
            previous = _write_status(payload)
            prior_refreshed = refreshed
            if state == "source_campaign_terminal_failed_no_active_supervisor_path":
                return 2
            if state == "passed_all_twelve_hybrid_cells_refreshed":
                return 0
            if once:
                return 0 if last_error is None else 1
            assert next_poll is not None
            time.sleep(next_poll)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            payload = _status_payload(
                status="watcher_authentication_failed",
                campaign=campaign,
                refreshed=refreshed,
                reported=reported,
                reported_revisions=reported_revisions,
                reported_continuation_revision=(
                    reported_continuation_revision
                ),
                candidate_chtc=candidate_chtc,
                reported_chtc=reported_chtc,
                fingerprint=None,
                next_poll_seconds=None,
                refresh_result=refresh_result,
                last_error=error,
                runtime_dir=runtime_dir,
                activation_dir=activation_dir,
                expected_adapter_sha256=expected_adapter_sha256,
            )
            _write_status(payload)
            print(error, file=sys.stderr, flush=True)
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=snapshot.LOCAL_RUNTIME_DIR)
    parser.add_argument(
        "--activation-dir", type=Path, default=snapshot.LOCAL_ACTIVATION_DIR
    )
    parser.add_argument(
        "--expected-local-adapter-sha256",
        default=snapshot.EXPECTED_LOCAL_ADAPTER_SHA256,
    )
    parser.add_argument("--poll-seconds", type=float, default=MIN_POLL_SECONDS)
    parser.add_argument(
        "--max-poll-seconds", type=float, default=DEFAULT_MAX_POLL_SECONDS
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < MIN_POLL_SECONDS:
        raise SystemExit("--poll-seconds must be at least 30")
    if args.max_poll_seconds < args.poll_seconds:
        raise SystemExit("--max-poll-seconds must be at least --poll-seconds")
    if len(args.expected_local_adapter_sha256) != 64:
        raise SystemExit("--expected-local-adapter-sha256 must be a pinned SHA-256")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Page-17 completion watcher is already running.", file=sys.stderr)
            return 2
        return watch(
            runtime_dir=args.runtime_dir.resolve(),
            activation_dir=args.activation_dir.resolve(),
            expected_adapter_sha256=args.expected_local_adapter_sha256,
            poll_seconds=args.poll_seconds,
            max_poll_seconds=args.max_poll_seconds,
            once=args.once,
        )


if __name__ == "__main__":
    raise SystemExit(main())
