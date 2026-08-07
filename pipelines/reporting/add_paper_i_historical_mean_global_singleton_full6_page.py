#!/usr/bin/env python3
"""Add or refresh one six-panel global-singleton RA comparison page.

The page is deliberately additive.  It preserves the six existing report
pages byte-for-byte at the PDF page-content level, plots the authenticated
Append-ADAPT singleton trajectory through round 50 in the weak-Holstein
(``nph=3``) panels and through round 70 in the strong-Holstein (``nph=7``)
panels, and adds a historical-mean global-singleton RA trajectory only after a
complete worker attempt archive closes every publication and scientific
receipt through round 50.  The first update may contain the three ``nph=3``
cells and explicit pending panels for ``nph=7``; a later strict completion
superset replaces page 7 in place.

Large attempt members are never extracted.  Archive authentication and result
projection are streaming operations so checkpoint, ledger, and result payloads
do not consume additional persistent disk.  A separately sealed compact
projection may be produced beside a remotely preserved archive and consumed
locally; it binds the full-scan validation, exact source code, trajectory, and
the only two RA prefix-cost observations used by this page.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.reporting import (  # noqa: E402
    add_paper_i_append_r70_singleton_progress_page as legacy_page,
)
from pipelines.reporting import (  # noqa: E402
    build_paper_i_ra_adapt_stationary_core_master_pdf as master,
)
from pipelines.reporting import (  # noqa: E402
    build_paper_i_ra_vs_adapt_common_accuracy_cost_pdf as common_cost,
)


LEGACY_ADAPTER_SCHEMA = (
    "paper_i_historical_mean_global_singleton_vs_append_r70_full6_adapter_v1"
)
ADAPTER_SCHEMA = (
    "paper_i_historical_mean_global_singleton_vs_append_mixed_horizon_"
    "full6_adapter_v2"
)
RA_PROJECTION_SCHEMA = (
    "paper_i_historical_mean_global_singleton_page7_ra_compact_projection_v1"
)
APPEND_ADAPTER_SCHEMA = "paper_i_append_adapt_singleton_r70_progress_adapter_v1"
LEGACY_PAGE_ID = "historical_mean_global_singleton_vs_append_r70_six_regime_v1"
PAGE_ID = (
    "historical_mean_global_singleton_vs_append_mixed_horizon_six_regime_v2"
)
LEGACY_REPORT_KEY = "historical_mean_global_singleton_vs_append_r70"
REPORT_KEY = "historical_mean_global_singleton_vs_append_mixed_horizon"
EXPECTED_BASE_PAGE_6 = "ra_historical_average_vs_append_singleton_r70_costs_v3"

REGIME_ORDER = (
    "weak_weak",
    "intermediate_weak",
    "strong_weak_u8",
    "weak_strong",
    "intermediate_strong",
    "strong_strong_u8",
)
REGIME_LABELS = {
    "weak_weak": "Weak--weak",
    "intermediate_weak": "Intermediate--weak",
    "strong_weak_u8": "Strong--weak",
    "weak_strong": "Weak--strong",
    "intermediate_strong": "Intermediate--strong",
    "strong_strong_u8": "Strong--strong",
}
REGIME_ABBREVIATIONS = {
    "weak_weak": "WW",
    "intermediate_weak": "IW",
    "strong_weak_u8": "SW",
    "weak_strong": "WS",
    "intermediate_strong": "IS",
    "strong_strong_u8": "SS",
}
NPH_BY_REGIME = {
    "weak_weak": 3,
    "intermediate_weak": 3,
    "strong_weak_u8": 3,
    "weak_strong": 7,
    "intermediate_strong": 7,
    "strong_strong_u8": 7,
}
NPH3_REGIMES = frozenset(REGIME_ORDER[:3])
NPH7_REGIMES = frozenset(REGIME_ORDER[3:])
APPEND_TERMINAL_ROUND_BY_REGIME = {
    regime: (50 if regime in NPH3_REGIMES else 70) for regime in REGIME_ORDER
}
QISKIT_FIELDS = ("N2q", "D2q", "Dc", "W1q")
COST_FIELDS = (*QISKIT_FIELDS, "S_alg")
ROUTE_CONTRACT_SHA256 = (
    "69af64db5bbaf5b811685b8353b82b748dc13d16306e4c08ddfe5ffde07f301b"
)
SOURCE_ARCHIVE_SHA256 = (
    "7e7fa374f629ce684035d318176f354b24cfdf7cf4ac9548be921c790bf57d01"
)
ASSET_STEM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CAPTURE_LIMIT_BYTES = 128 * 1024 * 1024
NPH3_ARCHIVE_VALIDATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph3_r50_"
    "attempt_archive_validation_v1"
)
NPH3_ARCHIVE_VALIDATOR = REPO_ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "validate_nph3_v3_attempt_archive.py"
)
# Clusters 9401087/9401094 exposed cross-receipt ED-reference offsets up to
# 6.4171e-14; this tolerance is confined to compact-projection validation.
COMPACT_PROJECTION_DELTA_E_ABS_TOL = 7.0e-14
PRIOR_COMPACT_PROJECTION_UPDATER_BINDING = {
    "path": "pipelines/reporting/add_paper_i_historical_mean_global_singleton_full6_page.py",
    "sha256": "f3a353fc7a1341ae0b456ea83f1cc4d17b1b0a6009d7d75a89563a6feb38306b",
    "size_bytes": 142314,
}
LEGACY_PAGE7_ADAPTER_BINDING = {
    "canonical_sha256": (
        "7176d45c1167bd17b81b2955846c971d8ebab8deda4b54951138e5bdd0bfe63d"
    ),
    "sha256": (
        "8e2a0b70f041cb18d7805e31865ee41092e4079510191d16186a40ca4fe2902a"
    ),
    "size_bytes": 132093,
}

ROUTE_DESCRIPTION = (
    "Fresh historical-mean global-singleton RA: global guarded singleton "
    "Phase-I pool; shortlists 24 to 12; one Phase-III admission; stationary "
    "source gradients; all-phase resource weighting; commutation-reduced "
    "plateau insertion; prior-mean ratio threshold 1e-4; Powell-200, seed 7."
)
LEGACY_LIMITATION = (
    "Page 7 is a supplemental diagnostic comparison of the fresh "
    "historical-mean global-singleton RA plateau route against the preserved "
    "fresh Append-ADAPT singleton R70 comparator; it is not adopted Paper-I "
    "evidence. Pending panels contain no inferred RA values."
)
LIMITATION = (
    "Page 7 is a supplemental diagnostic comparison of the fresh "
    "historical-mean global-singleton RA plateau route against the preserved "
    "fresh Append-ADAPT singleton comparator at k=50 for weak-Holstein "
    "(nph=3) and k=70 for strong-Holstein (nph=7); it is not adopted Paper-I "
    "evidence. Pending panels contain no inferred RA values."
)


class Page7InputError(ValueError):
    """Raised when inputs cannot support the guarded page-7 update."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    if "sha256" in result:
        raise Page7InputError("self-digest input already contains sha256")
    result["sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    observed = value.get("sha256")
    unsigned = copy.deepcopy(dict(value))
    unsigned.pop("sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if observed != expected:
        raise Page7InputError(f"{label} self-digest drifted")
    return str(observed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Page7InputError(f"{label} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise Page7InputError(f"{label} must be a JSON object")
    return value


def mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Page7InputError(f"{label} must be an object")
    return value


def sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise Page7InputError(f"{label} must be an array")
    return value


def integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Page7InputError(f"{label} must be an integer >= {minimum}")
    return value


def finite(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Page7InputError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise Page7InputError(f"{label} must be finite{suffix}")
    return result


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise Page7InputError(f"{label} must be text")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise Page7InputError(f"{label} must be a normalized relative path")
    return path


PACKAGE_SPECS: dict[int, dict[str, Any]] = {
    3: {
        "package_dir": REPO_ROOT
        / (
            "chtc/paper_i_ra_adapt_repair_20260727/"
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_20260802_v3_chtc"
        ),
        "package_id": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_20260802_v3_chtc"
        ),
        "campaign_id": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_v3"
        ),
        "manifest_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_package_manifest_v2"
        ),
        "manifest_sha256": (
            "dd756ffa8fa0b1d9b21f906d2587a664ff49743f4eb80c4f1c787c0989cf4f23"
        ),
        "job_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_job_v2"
        ),
        "authorization_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_execution_authorization_v2"
        ),
        "activation_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_ordinary_activation_v2"
        ),
        "attempt_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_worker_attempt_v2"
        ),
        "worker_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_worker_receipt_v2"
        ),
        "execution_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph3_r50_execution_manifest_v2"
        ),
        "regimes": NPH3_REGIMES,
    },
    7: {
        "package_dir": REPO_ROOT
        / (
            "chtc/paper_i_ra_adapt_repair_20260727/"
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_20260802_v2_chtc"
        ),
        "package_id": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_20260802_v2_chtc"
        ),
        "campaign_id": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_v2"
        ),
        "manifest_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_package_manifest_v2"
        ),
        "manifest_sha256": (
            "fe7fd6f5f572c3ca90dbf43ec43c69f35282d4c699cd271d8cd6564555bb495f"
        ),
        "job_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_job_v2"
        ),
        "authorization_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_execution_authorization_v2"
        ),
        "activation_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_ordinary_activation_v2"
        ),
        "attempt_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_worker_attempt_v2"
        ),
        "worker_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_worker_receipt_v2"
        ),
        "execution_schema": (
            "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
            "nph7_r50_execution_manifest_v2"
        ),
        "regimes": NPH7_REGIMES,
    },
}

RESUME_SPEC: dict[str, Any] = {
    "package_dir": REPO_ROOT
    / (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
        "nph7_r50_20260802_v3_resume128gb_chtc"
    ),
    "activation_dir": REPO_ROOT
    / (
        "chtc/paper_i_ra_adapt_repair_20260727/"
        "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
        "nph7_r50_20260802_v3_resume128gb_chtc_activation_ordinary_v1"
    ),
    "package_id": (
        "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
        "nph7_r50_20260802_v3_resume128gb_chtc"
    ),
    "campaign_id": (
        "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_"
        "nph7_r50_resume128gb_v3"
    ),
    "manifest_schema": (
        "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
        "package_manifest_v1"
    ),
    "manifest_sha256": (
        "f34dfa4e7157ef6e009c5a78547c116989392a46d31d4c0448dc1fc87d7968b0"
    ),
    "manifest_file_sha256": (
        "b5d6e0fa93e701354a06a381594a0e28380919648c600723e7c548fd7cff018d"
    ),
    "activation_sha256": (
        "5fa7899f3fbc7ef7e5878216e70a5e00bbfd827f3f939ed0689de7c943dcabdf"
    ),
    "job_schema": (
        "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_job_v1"
    ),
    "authorization_schema": (
        "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
        "execution_authorization_v1"
    ),
    "activation_schema": (
        "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
        "activation_manifest_v1"
    ),
    "attempt_schema": (
        "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
        "worker_attempt_v1"
    ),
    "worker_schema": (
        "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
        "worker_receipt_v1"
    ),
    "execution_schema": (
        "paper_i_ra_adapt_historical_mean_global_singleton_resume128gb_"
        "execution_manifest_v1"
    ),
    "route_profile": (
        "paper_i_ra_adapt__single_pauli_word_v1__"
        "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
        "identity_phase_ii__stationary_source_response_v1__"
        "all_phase_resource_weighting_v1"
    ),
    "resume_rounds": {
        "weak_strong": 35,
        "intermediate_strong": 31,
        "strong_strong_u8": 17,
    },
}


def validate_package_manifest(nph: int) -> dict[str, Any]:
    spec = PACKAGE_SPECS[nph]
    path = Path(spec["package_dir"]) / "package_manifest.json"
    manifest = load_json(path, label=f"nph={nph} package manifest")
    canonical = verify_self_digest(
        manifest, label=f"nph={nph} package manifest"
    )
    source = mapping(manifest.get("source_archive"), label="source archive")
    if (
        manifest.get("schema") != spec["manifest_schema"]
        or manifest.get("package_id") != spec["package_id"]
        or manifest.get("campaign_id") != spec["campaign_id"]
        or canonical != spec["manifest_sha256"]
        or source.get("sha256") != SOURCE_ARCHIVE_SHA256
        or set(manifest.get("execution_ids", ()))
        != {
            expected_execution_id(regime)
            for regime in spec["regimes"]
        }
    ):
        raise Page7InputError(f"nph={nph} package authority drifted")
    return {
        **file_binding(path),
        "canonical_sha256": canonical,
        "package_id": spec["package_id"],
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }


def expected_execution_id(regime: str) -> str:
    nph = NPH_BY_REGIME[regime]
    version = "v3" if nph == 3 else "v2"
    return (
        f"historical_mean_global_singleton_{version}_nph{nph}_r50__"
        f"{regime}__nph{nph}__ra_global_singleton_plateau"
    )


def expected_resume_execution_id(regime: str) -> str:
    if regime not in RESUME_SPEC["resume_rounds"]:
        raise Page7InputError(f"{regime}: no authenticated resume route exists")
    return (
        f"{expected_execution_id(regime)}__resume_from_d"
        f"{RESUME_SPEC['resume_rounds'][regime]}_to_r50_v1"
    )


def validate_resume_package_manifest() -> dict[str, Any]:
    spec = RESUME_SPEC
    path = Path(spec["package_dir"]) / "package_manifest.json"
    manifest = load_json(path, label="nph=7 resume package manifest")
    canonical = verify_self_digest(
        manifest, label="nph=7 resume package manifest"
    )
    source_package = mapping(
        manifest.get("source_package"), label="resume source package"
    )
    expected_ids = {
        expected_resume_execution_id(regime) for regime in NPH7_REGIMES
    }
    if (
        manifest.get("schema") != spec["manifest_schema"]
        or manifest.get("status")
        != "passed_inert_three_authenticated_resumes"
        or manifest.get("package_id") != spec["package_id"]
        or manifest.get("campaign_id") != spec["campaign_id"]
        or canonical != spec["manifest_sha256"]
        or sha256_file(path) != spec["manifest_file_sha256"]
        or manifest.get("row_count") != 3
        or set(manifest.get("execution_ids", ())) != expected_ids
        or manifest.get("scientific_protocol_changed") is not False
        or manifest.get("scientific_settings_changed") != []
        or manifest.get("source_held_jobs_preserved") is not True
        or manifest.get("execution_authorized") is not False
        or manifest.get("submission_authorized") is not False
        or manifest.get("submitted") is not False
        or source_package.get("manifest_sha256")
        != PACKAGE_SPECS[7]["manifest_sha256"]
        or source_package.get("manifest_file_sha256")
        != sha256_file(Path(PACKAGE_SPECS[7]["package_dir"]) / "package_manifest.json")
    ):
        raise Page7InputError("nph=7 resume package authority drifted")
    return {
        **file_binding(path),
        "canonical_sha256": canonical,
        "package_id": spec["package_id"],
        "source_package_manifest_sha256": PACKAGE_SPECS[7]["manifest_sha256"],
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
    }


def validate_append_adapter(path: Path) -> dict[str, Any]:
    adapter = load_json(path, label="Append singleton R70 adapter")
    canonical = verify_self_digest(adapter, label="Append singleton R70 adapter")
    if (
        adapter.get("schema") != APPEND_ADAPTER_SCHEMA
        or adapter.get("status") != "passed"
        or tuple(adapter.get("regime_order", ())) != REGIME_ORDER
        or tuple(adapter.get("completed_regimes", ())) != REGIME_ORDER
        or tuple(adapter.get("pending_regimes", ())) != ()
    ):
        raise Page7InputError("Append adapter is not the complete six-regime set")
    raw_cells = sequence(adapter.get("cells"), label="Append cells")
    cells = {
        str(mapping(raw, label="Append cell").get("regime_id")): raw
        for raw in raw_cells
    }
    if len(raw_cells) != 6 or set(cells) != set(REGIME_ORDER):
        raise Page7InputError("Append adapter regime closure drifted")
    for regime in REGIME_ORDER:
        cell = mapping(cells[regime], label=f"Append {regime}")
        if cell.get("nph") != NPH_BY_REGIME[regime]:
            raise Page7InputError(f"Append {regime} cutoff drifted")
        exact = finite(
            cell.get("exact_same_cutoff_energy"),
            label=f"Append {regime} exact energy",
        )
        points = sequence(cell.get("points"), label=f"Append {regime} points")
        if len(points) != 71:
            raise Page7InputError(f"Append {regime} must contain rounds 0..70")
        for expected_round, raw_point in enumerate(points):
            point = mapping(raw_point, label=f"Append {regime} point")
            energy = finite(point.get("energy"), label="Append energy")
            error = finite(point.get("delta_e"), label="Append error", minimum=0.0)
            if (
                point.get("round") != expected_round
                or not math.isclose(
                    error,
                    abs(energy - exact),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-14,
                )
            ):
                raise Page7InputError(f"Append {regime} trajectory drifted")
        endpoints = mapping(cell.get("endpoints"), label="Append endpoints")
        for endpoint_round in (50, 70):
            endpoint = mapping(
                endpoints.get(f"round_{endpoint_round}"),
                label=f"Append {regime} endpoint {endpoint_round}",
            )
            costs = mapping(endpoint.get("costs"), label="Append endpoint costs")
            if endpoint.get("round") != endpoint_round or set(costs) != set(
                COST_FIELDS
            ):
                raise Page7InputError(f"Append {regime} endpoint costs drifted")
            for field in COST_FIELDS:
                integer(costs.get(field), label=f"Append {regime} {field}")
    return {
        **copy.deepcopy(adapter),
        "sha256": canonical,
        "file_binding": file_binding(path),
    }


def _read_archive_members(
    path: Path,
    *,
    capture: frozenset[str],
) -> tuple[dict[str, tuple[str, int]], dict[str, bytes]]:
    """Authenticate every regular member in one streaming gzip pass."""

    bindings: dict[str, tuple[str, int]] = {}
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                name = member.name
                pure = PurePosixPath(name)
                if (
                    name in bindings
                    or pure.is_absolute()
                    or "." in pure.parts
                    or ".." in pure.parts
                    or pure.as_posix() != name
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise Page7InputError("attempt archive has an unsafe member")
                stream = archive.extractfile(member)
                if stream is None:
                    raise Page7InputError("attempt archive member is unreadable")
                if name in capture and member.size > CAPTURE_LIMIT_BYTES:
                    raise Page7InputError(
                        f"captured attempt sidecar is unexpectedly large: {name}"
                    )
                digest = hashlib.sha256()
                size = 0
                kept = bytearray() if name in capture else None
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    if kept is not None:
                        kept.extend(block)
                if size != member.size:
                    raise Page7InputError("attempt archive member size drifted")
                bindings[name] = (digest.hexdigest(), size)
                if kept is not None:
                    payloads[name] = bytes(kept)
    except (OSError, EOFError, tarfile.TarError) as exc:
        raise Page7InputError(f"attempt archive is unreadable: {exc}") from exc
    return bindings, payloads


def _json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Page7InputError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise Page7InputError(f"{label} must be a JSON object")
    return value


def _work(value: Any, *, label: str) -> dict[str, Any]:
    row = mapping(value, label=label)
    components = mapping(row.get("components"), label=f"{label} components")
    normalized = {
        "N_H_outer": integer(
            components.get("n_h_outer", components.get("N_H_outer")),
            label=f"{label} N_H_outer",
        ),
        "N_H_refit": integer(
            components.get("n_h_refit", components.get("N_H_refit")),
            label=f"{label} N_H_refit",
        ),
        "N_grad": integer(
            components.get("n_grad", components.get("N_grad")),
            label=f"{label} N_grad",
        ),
        "N_metric": integer(
            components.get("n_metric", components.get("N_metric")),
            label=f"{label} N_metric",
        ),
    }
    s_alg = integer(
        row.get("s_alg", row.get("S_alg")), label=f"{label} S_alg"
    )
    if sum(normalized.values()) != s_alg:
        raise Page7InputError(f"{label} components do not close")
    return {"components": normalized, "S_alg": s_alg}


def _stream_result_projection(
    path: Path, *, result_member: str, execution_id: str
) -> dict[str, Any]:
    """Project scalar scientific closure from a large result member."""

    try:
        import ijson
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise Page7InputError("streaming result validation requires ijson") from exc

    result_seen = False
    projection: dict[str, Any] = {
        "transition_rounds": [],
        "transition_before": [],
        "transition_after": [],
        "transition_cumulative_s_alg": [],
        "trajectory_rounds": [],
        "trajectory_energies": [],
        "prefix_s_alg": [],
        "prefix_components": [],
    }
    component_rows: dict[int, dict[str, int]] = {}
    scalar_events = {"string", "number", "boolean", "null"}
    try:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if member.name != result_member:
                    continue
                if result_seen or not member.isfile():
                    raise Page7InputError(
                        f"{execution_id}: result member is duplicated or unsafe"
                    )
                result_seen = True
                stream = archive.extractfile(member)
                if stream is None:
                    raise Page7InputError(f"{execution_id}: result is unreadable")
                prefix_index = -1
                for prefix, event, value in ijson.parse(stream, use_float=True):
                    if event not in scalar_events:
                        if (
                            prefix
                            == "run.canonical_reporting.accepted_prefix_work.item"
                            and event == "start_map"
                        ):
                            prefix_index += 1
                            component_rows[prefix_index] = {}
                        continue
                    if prefix == "schema":
                        projection["schema"] = value
                    elif prefix == "run.stop.completed_controller_rounds":
                        projection["completed_controller_rounds"] = value
                    elif prefix == "run.route.profile":
                        projection["route_profile"] = value
                    elif prefix == "run.route.contract_sha256":
                        projection["route_contract_sha256"] = value
                    elif prefix == "run.problem.problem_request_sha256":
                        projection["problem_request_sha256"] = value
                    elif (
                        prefix
                        == "run.canonical_reporting.exact_same_cutoff_energy"
                    ):
                        projection["exact_same_cutoff_energy"] = value
                    elif (
                        prefix
                        == "run.canonical_reporting.candidate_representation"
                    ):
                        projection["candidate_representation"] = value
                    elif prefix == "run.accepted_transitions.item.controller_round":
                        projection["transition_rounds"].append(value)
                    elif prefix == "run.accepted_transitions.item.energy_before":
                        projection["transition_before"].append(value)
                    elif prefix == "run.accepted_transitions.item.energy_after":
                        projection["transition_after"].append(value)
                    elif (
                        prefix
                        == "run.accepted_transitions.item.cumulative_s_alg"
                    ):
                        projection["transition_cumulative_s_alg"].append(value)
                    elif prefix == "run.accepted_trajectory.item.controller_round":
                        projection["trajectory_rounds"].append(value)
                    elif prefix == "run.accepted_trajectory.item.energy":
                        projection["trajectory_energies"].append(value)
                    elif (
                        prefix
                        == "run.canonical_reporting.accepted_prefix_work.item.s_alg"
                    ):
                        projection["prefix_s_alg"].append(value)
                    elif prefix.startswith(
                        "run.canonical_reporting.accepted_prefix_work.item.components."
                    ):
                        field = prefix.rsplit(".", 1)[-1]
                        component_rows[prefix_index][field] = value
                    elif prefix == "run.estimator_accounting.all_work.s_alg":
                        projection["all_work_s_alg"] = value
                    elif prefix.startswith(
                        "run.estimator_accounting.all_work.components."
                    ):
                        projection.setdefault("all_work_components", {})[
                            prefix.rsplit(".", 1)[-1]
                        ] = value
                break
    except (OSError, EOFError, tarfile.TarError, ijson.JSONError) as exc:
        raise Page7InputError(
            f"{execution_id}: streamed result projection failed"
        ) from exc
    if not result_seen:
        raise Page7InputError(f"{execution_id}: result member is missing")
    projection["prefix_components"] = [
        component_rows[index] for index in sorted(component_rows)
    ]
    return projection


def _stream_ledger_projection(
    path: Path, *, ledger_member: str, execution_id: str
) -> dict[str, Any]:
    """Project only terminal accounting fields from the large ledger."""

    try:
        import ijson
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise Page7InputError("streaming ledger validation requires ijson") from exc

    result: dict[str, Any] = {"components": {}}
    seen = False
    scalar_events = {"string", "number", "boolean", "null"}
    try:
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                if member.name != ledger_member:
                    continue
                if seen or not member.isfile():
                    raise Page7InputError(
                        f"{execution_id}: ledger member is duplicated or unsafe"
                    )
                seen = True
                stream = archive.extractfile(member)
                if stream is None:
                    raise Page7InputError(f"{execution_id}: ledger is unreadable")
                for prefix, event, value in ijson.parse(stream, use_float=True):
                    if event not in scalar_events:
                        continue
                    if prefix == "schema":
                        result["schema"] = value
                    elif prefix == "adapt_success":
                        result["adapt_success"] = value
                    elif prefix == "adapt_error":
                        result["adapt_error"] = value
                    elif prefix == "accounting.complete":
                        result["complete"] = value
                    elif prefix == "accounting.status":
                        result["status"] = value
                    elif prefix == "accounting.S_alg":
                        result["S_alg"] = value
                    elif prefix.startswith("accounting.components."):
                        result["components"][prefix.rsplit(".", 1)[-1]] = value
                    elif prefix == "ledger.schema":
                        result["ledger_schema"] = value
                    elif prefix == "ledger.occurrence_summary.S_alg":
                        result["occurrence_S_alg"] = value
                break
    except (OSError, EOFError, tarfile.TarError, ijson.JSONError) as exc:
        raise Page7InputError(
            f"{execution_id}: streamed ledger projection failed"
        ) from exc
    if not seen:
        raise Page7InputError(f"{execution_id}: estimator ledger is missing")
    return result


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    execution_id: str,
    exact_energy: float,
) -> dict[str, Any]:
    if (
        summary.get("schema") != "paper_i_run_summary_v1"
        or summary.get("available_controller_rounds") != 50
    ):
        raise Page7InputError(f"{execution_id}: Paper-I summary is incomplete")
    raw_trace = sequence(
        summary.get("accepted_error_trace"), label=f"{execution_id} trace"
    )
    if len(raw_trace) != 50:
        raise Page7InputError(f"{execution_id}: trace is not rounds 1..50")
    points: list[dict[str, Any]] = []
    previous_energy: float | None = None
    for expected_round, raw in enumerate(raw_trace, start=1):
        row = mapping(raw, label=f"{execution_id} trace row")
        energy = finite(row.get("accepted_energy"), label="accepted energy")
        exact = finite(
            row.get("exact_same_cutoff_energy"), label="trace exact energy"
        )
        error = finite(
            row.get("absolute_energy_error"),
            label="accepted absolute error",
            minimum=0.0,
        )
        if (
            row.get("controller_round") != expected_round
            or not math.isclose(exact, exact_energy, rel_tol=0.0, abs_tol=1.0e-12)
            or not math.isclose(
                error,
                abs(energy - exact),
                rel_tol=1.0e-12,
                abs_tol=1.0e-14,
            )
            or (previous_energy is not None and energy > previous_energy + 1.0e-10)
        ):
            raise Page7InputError(f"{execution_id}: trace closure drifted")
        previous_energy = energy
        points.append(
            {"round": expected_round, "energy": energy, "delta_e": error}
        )

    from pipelines.reporting.paper_i_run_summary import (
        PaperIErrorTracePoint,
        select_paper_i_effective_plateau,
    )

    selected = select_paper_i_effective_plateau(
        tuple(
            PaperIErrorTracePoint(
                controller_round=point["round"],
                absolute_energy_error=point["delta_e"],
            )
            for point in points
        )
    )
    plateau = mapping(
        summary.get("effective_plateau"), label=f"{execution_id} plateau"
    )
    if (
        plateau.get("policy") != selected.policy
        or plateau.get("controller_round") != selected.controller_round
        or plateau.get("available_horizon_controller_rounds") != 50
        or not math.isclose(
            finite(plateau.get("absolute_energy_error"), label="plateau error"),
            selected.absolute_energy_error,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    ):
        raise Page7InputError(f"{execution_id}: effective plateau drifted")
    requested = sequence(
        summary.get("requested_rounds"), label=f"{execution_id} requested rounds"
    )
    if len(requested) != 1:
        raise Page7InputError(f"{execution_id}: requested round-50 row is missing")
    terminal = mapping(requested[0], label=f"{execution_id} requested round 50")
    if terminal.get("controller_round") != 50 or terminal.get("status") != "available":
        raise Page7InputError(f"{execution_id}: round-50 observation unavailable")
    terminal_work = _work(
        terminal.get("algorithmic_work"), label=f"{execution_id} terminal work"
    )
    all_work = _work(
        summary.get("canonical_all_work"), label=f"{execution_id} all work"
    )
    if terminal_work != all_work:
        raise Page7InputError(f"{execution_id}: terminal/all-work closure drifted")
    provenance = mapping(
        summary.get("provenance"), label=f"{execution_id} summary provenance"
    )
    if (
        provenance.get("candidate_representation") != "single_pauli_word_v1"
        or provenance.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or not math.isclose(
            finite(
                provenance.get("exact_same_cutoff_energy"),
                label="summary exact energy",
            ),
            exact_energy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise Page7InputError(f"{execution_id}: summary provenance drifted")
    return {
        "points": points,
        "effective_plateau": {
            "policy": selected.policy,
            "round": selected.controller_round,
            "delta_e": selected.absolute_energy_error,
            "best_observed_delta_e": selected.best_observed_error,
            "selection_threshold": selected.selection_threshold,
        },
        "terminal_work": terminal_work,
    }


def _artifact_rows(value: Any, *, label: str) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(sequence(value, label=label)):
        row = mapping(raw, label=f"{label} row {index}")
        relative = row.get("path")
        if not isinstance(relative, str) or relative in rows:
            raise Page7InputError(f"{label} paths are invalid")
        rows[relative] = row
    return rows


def _validate_binding_rows(
    rows: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, tuple[str, int]],
    *,
    prefix: str,
    label: str,
) -> None:
    expected_names = {f"{prefix}{relative}" for relative in rows}
    if expected_names != set(observed):
        raise Page7InputError(f"{label} member closure drifted")
    for relative, row in rows.items():
        digest, size = observed[f"{prefix}{relative}"]
        if row.get("sha256") != digest or row.get("size_bytes") != size:
            raise Page7InputError(f"{label} byte binding drifted for {relative}")


def _validate_scientific_payloads(
    path: Path,
    *,
    regime: str,
    execution_id: str,
    exact_energy: float,
    summary: Mapping[str, Any],
    artifact_prefix: str,
) -> dict[str, Any]:
    """Close the full k=1..50 result, summary, and estimator ledger."""

    summary_projection = _validate_summary(
        summary, execution_id=execution_id, exact_energy=exact_energy
    )
    result_member = f"{artifact_prefix}result.json"
    result = _stream_result_projection(
        path, result_member=result_member, execution_id=execution_id
    )
    expected_rounds = list(range(1, 51))
    result_energies = [
        finite(value, label=f"{regime} result energy")
        for value in result["trajectory_energies"]
    ]
    trace_energies = [point["energy"] for point in summary_projection["points"]]
    prefix_s_alg = [
        integer(value, label=f"{regime} prefix S_alg")
        for value in result["prefix_s_alg"]
    ]
    transition_s_alg = [
        integer(value, label=f"{regime} transition S_alg")
        for value in result["transition_cumulative_s_alg"]
    ]
    terminal_s_alg = summary_projection["terminal_work"]["S_alg"]
    if (
        result.get("schema") != "paper_i_ra_adapt_result_v1"
        or result.get("completed_controller_rounds") != 50
        or result.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or result.get("candidate_representation") != "single_pauli_word_v1"
        or result["transition_rounds"] != expected_rounds
        or result["trajectory_rounds"] != expected_rounds
        or len(result["transition_before"]) != 50
        or len(result["transition_after"]) != 50
        or len(result_energies) != 50
        or len(prefix_s_alg) != 50
        or len(transition_s_alg) != 50
        or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-12)
            for left, right in zip(result_energies, trace_energies, strict=True)
        )
        or any(
            not math.isclose(
                finite(left, label="transition after"),
                right,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for left, right in zip(
                result["transition_after"], trace_energies, strict=True
            )
        )
        or prefix_s_alg != transition_s_alg
        or prefix_s_alg[-1] != terminal_s_alg
        or result.get("all_work_s_alg") != terminal_s_alg
        or not math.isclose(
            finite(result.get("exact_same_cutoff_energy"), label="result exact"),
            exact_energy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise Page7InputError(f"{regime}: result/summary closure drifted")

    ledger_member = f"{artifact_prefix}estimator_ledger.json"
    ledger = _stream_ledger_projection(
        path, ledger_member=ledger_member, execution_id=execution_id
    )
    ledger_components = {
        field: integer(
            mapping(ledger.get("components"), label="ledger components").get(field),
            label=f"ledger {field}",
        )
        for field in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    }
    if (
        ledger.get("schema") != "paper_i_estimator_call_ledger_sidecar_v2"
        or ledger.get("ledger_schema") != "estimator_call_ledger_v1"
        or ledger.get("adapt_success") is not True
        or ledger.get("adapt_error") is not None
        or ledger.get("complete") is not True
        or ledger.get("S_alg") != terminal_s_alg
        or ledger.get("occurrence_S_alg") != terminal_s_alg
        or ledger_components
        != summary_projection["terminal_work"]["components"]
    ):
        raise Page7InputError(f"{regime}: estimator-ledger closure drifted")

    initial_energy = finite(
        result["transition_before"][0], label=f"{regime} initial energy"
    )
    return {
        "points": [
            {
                "round": 0,
                "energy": initial_energy,
                "delta_e": abs(initial_energy - exact_energy),
            },
            *summary_projection["points"],
        ],
        "effective_plateau": summary_projection["effective_plateau"],
        "terminal_work": summary_projection["terminal_work"],
        "result_member": result_member,
    }


def _validate_resume_attempt(
    path: Path,
    *,
    regime: str,
    bindings: Mapping[str, tuple[str, int]],
    raw: Mapping[str, bytes],
    job: Mapping[str, Any],
    authorization: Mapping[str, Any],
    activation: Mapping[str, Any],
    outer: Mapping[str, Any],
    worker: Mapping[str, Any],
    execution: Mapping[str, Any],
    summary: Mapping[str, Any],
    artifact_prefix: str,
) -> dict[str, Any]:
    """Validate one completed accepted-state nph=7 memory-repair resume."""

    spec = RESUME_SPEC
    if regime not in NPH7_REGIMES:
        raise Page7InputError(f"{regime}: resume package is nph=7 only")
    execution_id = expected_resume_execution_id(regime)
    source_execution_id = expected_execution_id(regime)
    package_manifest = validate_resume_package_manifest()
    local_manifest = load_json(
        Path(spec["package_dir"]) / "package_manifest.json",
        label=f"{regime} local resume package manifest",
    )
    job_rows = {
        str(mapping(row, label="resume package job").get("execution_id")): mapping(
            row, label="resume package job"
        )
        for row in sequence(local_manifest.get("jobs"), label="resume package jobs")
    }
    local_job_row = mapping(
        job_rows.get(execution_id), label=f"{regime} resume package job binding"
    )
    job_digest, job_size = bindings["authority/job.json"]

    resume = mapping(job.get("resume_input"), label=f"{regime} resume input")
    resume_archive = mapping(
        resume.get("archive"), label=f"{regime} resume archive"
    )
    resume_members = sequence(
        resume.get("members"), label=f"{regime} resume members"
    )
    resume_roles = {
        str(mapping(row, label="resume member").get("role"))
        for row in resume_members
    }
    checkpoint_rows = [
        mapping(row, label="resume checkpoint member")
        for row in resume_members
        if mapping(row, label="resume member").get("role") == "checkpoint"
    ]
    resume_round = RESUME_SPEC["resume_rounds"][regime]
    source_job_binding = mapping(
        job.get("source_job"), label=f"{regime} source job binding"
    )
    source_job_path = _append_source_path(
        source_job_binding.get("path"), label=f"{regime} source job"
    )
    source_job = load_json(source_job_path, label=f"{regime} source job")
    source_job_canonical = verify_self_digest(
        source_job, label=f"{regime} source job"
    )
    source_package = mapping(
        job.get("source_package"), label=f"{regime} source package"
    )
    if (
        job.get("schema") != spec["job_schema"]
        or job.get("execution_id") != execution_id
        or job.get("package_id") != spec["package_id"]
        or job.get("campaign_id") != spec["campaign_id"]
        or job.get("regime_id") != regime
        or job.get("nph") != 7
        or job.get("target_horizon") != 50
        or job.get("execution_mode")
        != "authenticated_accepted_state_resume_to_50"
        or job.get("route_profile") != spec["route_profile"]
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("scientific_protocol_changed") is not False
        or job.get("scientific_settings_changed") != []
        or job.get("source_job_preserved_held") is not True
        or job.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or job.get("source_execution_id") != source_execution_id
        or job.get("source_checkpoint") is not None
        or mapping(job.get("resources"), label="resume resources").get(
            "request_memory_mb"
        )
        != 131_072
        or mapping(job.get("resources"), label="resume resources").get(
            "request_disk_mb"
        )
        != 81_920
        or resume.get("validation_status") != "passed"
        or resume.get("pointer_closed") is not True
        or resume.get("resume_controller_round") != resume_round
        or resume.get("member_count") != 3
        or len(resume_members) != 3
        or resume_roles
        != {
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        }
        or len(checkpoint_rows) != 1
        or checkpoint_rows[0].get("path") != resume.get("checkpoint_path")
        or checkpoint_rows[0].get("sha256") != resume.get("checkpoint_sha256")
        or source_package.get("manifest_sha256")
        != PACKAGE_SPECS[7]["manifest_sha256"]
        or source_package.get("manifest_file_sha256")
        != sha256_file(Path(PACKAGE_SPECS[7]["package_dir"]) / "package_manifest.json")
        or local_job_row.get("canonical_sha256") != job.get("sha256")
        or local_job_row.get("sha256") != job_digest
        or local_job_row.get("size_bytes") != job_size
        or source_job_path.stat().st_size != source_job_binding.get("size_bytes")
        or sha256_file(source_job_path) != source_job_binding.get("sha256")
        or source_job_canonical != source_job_binding.get("canonical_sha256")
    ):
        raise Page7InputError(f"{regime}: resume job authority drifted")

    if (
        source_job.get("schema") != PACKAGE_SPECS[7]["job_schema"]
        or source_job.get("execution_id") != source_execution_id
        or source_job.get("package_id") != PACKAGE_SPECS[7]["package_id"]
        or source_job.get("campaign_id") != PACKAGE_SPECS[7]["campaign_id"]
        or source_job.get("regime_id") != regime
        or source_job.get("nph") != 7
        or source_job.get("target_horizon") != 50
        or source_job.get("execution_mode") != "fresh_0_to_50"
        or source_job.get("active_gradient_policy")
        != "stationary_source_response_v1"
        or source_job.get("resource_weighting_scope")
        != "all_phase_resource_weighting_v1"
        or source_job.get("candidate_representation") != "single_pauli_word_v1"
        or source_job.get("phase_i_candidate_supply")
        != "global_guarded_singleton_pool_v1"
        or source_job.get("phase_i_candidate_visibility")
        != "all_executable_candidates_v1"
        or source_job.get("phase_ii_candidate_exposure")
        != "identity_on_retained_singletons_v1"
        or source_job.get("phase_i_shortlist_size") != 24
        or source_job.get("phase_ii_shortlist_size") != 12
        or source_job.get("phase_iii_admission_cardinality") != 1
        or source_job.get("insertion_policy") != "plateau_commutation"
        or source_job.get("plateau_prior_mean_decrease_ratio_threshold")
        != 1.0e-4
        or source_job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or source_job.get("fresh_start_contract")
        != {"kind": "fresh_start", "resume_archive": None, "source_checkpoint": None}
        or source_job.get("protocol_sha256")
        != job.get("scientific_protocol_sha256")
    ):
        raise Page7InputError(f"{regime}: resumed scientific route drifted")
    exact_energy = finite(
        source_job.get("exact_same_cutoff_energy"), label="source exact energy"
    )

    authorization_digest, _authorization_size = bindings[
        "authority/execution_authorization.json"
    ]
    if (
        authorization.get("schema") != spec["authorization_schema"]
        or authorization.get("execution_id") != execution_id
        or authorization.get("package_id") != spec["package_id"]
        or authorization.get("campaign_id") != spec["campaign_id"]
        or authorization.get("job_sha256") != job.get("sha256")
        or authorization.get("package_manifest_sha256")
        != spec["manifest_sha256"]
        or authorization.get("scientific_protocol_sha256")
        != job.get("scientific_protocol_sha256")
        or authorization.get("checkpoint_sha256")
        != resume.get("checkpoint_sha256")
        or authorization.get("resume_controller_round") != resume_round
        or authorization.get("target_horizon") != 50
        or authorization.get("resources") != job.get("resources")
        or authorization.get("source_execution_id") != source_execution_id
        or authorization.get("source_held_job_removal_authorized") is not False
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise Page7InputError(f"{regime}: resume authorization drifted")

    activation_path = Path(spec["activation_dir"]) / "activation_manifest.json"
    activation_file_sha256 = sha256_file(activation_path)
    activation_executions = {
        str(mapping(row, label="resume activation execution").get("execution_id")):
        mapping(row, label="resume activation execution")
        for row in sequence(activation.get("executions"), label="resume executions")
    }
    activation_authorizations = {
        str(mapping(row, label="resume activation authorization").get("execution_id")):
        mapping(row, label="resume activation authorization")
        for row in sequence(
            activation.get("execution_authorizations"),
            label="resume activation authorizations",
        )
    }
    activation_execution = mapping(
        activation_executions.get(execution_id),
        label="resume activation execution row",
    )
    activation_job = mapping(
        activation_execution.get("job"), label="resume activation job binding"
    )
    activation_auth = mapping(
        activation_authorizations.get(execution_id),
        label="resume activation authorization binding",
    )
    if (
        activation.get("schema") != spec["activation_schema"]
        or activation.get("sha256") != spec["activation_sha256"]
        or bindings["authority/activation_manifest.json"][0]
        != activation_file_sha256
        or activation.get("package_id") != spec["package_id"]
        or activation.get("campaign_id") != spec["campaign_id"]
        or activation.get("execution_authorized") is not True
        or activation.get("submission_authorized") is not True
        or activation.get("paper_evidence_adopted") is not False
        or activation.get("source_held_jobs_preserved") is not True
        or activation_job.get("canonical_sha256") != job.get("sha256")
        or activation_job.get("sha256") != job_digest
        or activation_auth.get("canonical_sha256") != authorization.get("sha256")
        or activation_auth.get("sha256") != authorization_digest
        or activation_execution.get("resume_archive") != resume_archive
    ):
        raise Page7InputError(f"{regime}: resume activation binding drifted")

    if (
        outer.get("schema") != spec["attempt_schema"]
        or outer.get("execution_id") != execution_id
        or outer.get("worker_exit_status") != 0
        or outer.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or outer.get("resume_archive_sha256") != resume_archive.get("sha256")
        or outer.get("image_sha256")
        != mapping(activation.get("remote_image"), label="resume image").get("sha256")
        or outer.get("job_file_sha256") != job_digest
        or outer.get("authorization_file_sha256") != authorization_digest
        or outer.get("activation_manifest_file_sha256")
        != bindings["authority/activation_manifest.json"][0]
    ):
        raise Page7InputError(f"{regime}: resume attempt receipt drifted")
    cluster_id = integer(outer.get("cluster_id"), label="resume cluster id")
    proc_id = integer(outer.get("proc_id"), label="resume proc id")
    attempt_ordinal = integer(
        outer.get("attempt_ordinal"), label="resume attempt ordinal", minimum=1
    )
    expected_identity = (
        f"{execution_id}\t{cluster_id}\t{proc_id}\t{attempt_ordinal}\n"
    ).encode("ascii")
    if (
        raw["worker_outputs/attempt_identity.tsv"] != expected_identity
        or raw["worker_outputs/worker_exit_status.txt"].strip() != b"0"
        or activation_execution.get("queue_index") != proc_id
    ):
        raise Page7InputError(f"{regime}: resume attempt identity drifted")
    worker_rows = _artifact_rows(
        outer.get("worker_files"), label=f"{regime} resume worker files"
    )
    observed_worker = {
        name: binding
        for name, binding in bindings.items()
        if name.startswith("worker_outputs/")
    }
    _validate_binding_rows(
        worker_rows,
        observed_worker,
        prefix="worker_outputs/",
        label=f"{regime} resume outer worker inventory",
    )
    triplets = sequence(
        outer.get("resumable_checkpoint_triplets"),
        label=f"{regime} resumable checkpoint triplets",
    )
    if (
        outer.get("resumable_checkpoint_triplet_count") != len(triplets)
        or not triplets
        or outer.get("failure_safe_checkpoint_transfer") is not True
    ):
        raise Page7InputError(f"{regime}: failure-safe checkpoint closure drifted")
    for raw_triplet in triplets:
        triplet = mapping(raw_triplet, label="resumable checkpoint triplet")
        if triplet.get("pointer_closed_by_sibling_identity") is not True:
            raise Page7InputError(f"{regime}: checkpoint pointer is not closed")
        for key in (
            "checkpoint",
            "estimator_ledger_checkpoint",
            "verified_resume_sidecar",
        ):
            row = mapping(triplet.get(key), label=f"checkpoint triplet {key}")
            relative = safe_relative_path(
                row.get("path"), label=f"checkpoint triplet {key} path"
            ).as_posix()
            observed = bindings.get(f"worker_outputs/{relative}")
            if observed != (row.get("sha256"), row.get("size_bytes")):
                raise Page7InputError(f"{regime}: checkpoint triplet binding drifted")

    if (
        worker.get("schema") != spec["worker_schema"]
        or worker.get("status") != "passed"
        or worker.get("package_id") != spec["package_id"]
        or worker.get("campaign_id") != spec["campaign_id"]
        or worker.get("execution_id") != execution_id
        or worker.get("job_sha256") != job.get("sha256")
        or worker.get("authorization_sha256") != authorization.get("sha256")
        or worker.get("controller_rounds_completed") != 50
        or worker.get("source_checkpoint_consumed") is not True
        or worker.get("source_checkpoint_sha256")
        != resume.get("checkpoint_sha256")
        or worker.get("source_held_job_preserved") is not True
    ):
        raise Page7InputError(f"{regime}: resume worker receipt drifted")
    artifact_rows = _artifact_rows(
        worker.get("artifacts"), label=f"{regime} resume artifacts"
    )
    required_artifacts = {
        "checkpoint.json",
        "estimator_ledger.json",
        "execution_manifest.json",
        "paper_i_summary.json",
        "result.json",
    }
    if not required_artifacts.issubset(artifact_rows):
        raise Page7InputError(f"{regime}: resume artifact closure drifted")
    observed_artifacts = {
        name: binding
        for name, binding in bindings.items()
        if name.startswith(artifact_prefix)
    }
    _validate_binding_rows(
        artifact_rows,
        observed_artifacts,
        prefix=artifact_prefix,
        label=f"{regime} resume artifact inventory",
    )

    if (
        execution.get("schema") != spec["execution_schema"]
        or execution.get("status") != "passed"
        or execution.get("package_id") != spec["package_id"]
        or execution.get("campaign_id") != spec["campaign_id"]
        or execution.get("execution_id") != execution_id
        or execution.get("source_execution_id") != source_execution_id
        or execution.get("job_sha256") != job.get("sha256")
        or execution.get("authorization_sha256") != authorization.get("sha256")
        or execution.get("scientific_protocol_sha256")
        != job.get("scientific_protocol_sha256")
        or execution.get("scientific_protocol_changed") is not False
        or execution.get("scientific_settings_changed") != []
        or execution.get("source_checkpoint_sha256")
        != resume.get("checkpoint_sha256")
        or execution.get("resume_controller_round") != resume_round
        or execution.get("target_horizon") != 50
        or execution.get("controller_rounds_completed") != 50
        or execution.get("source_held_job_preserved") is not True
        or worker.get("execution_manifest_sha256") != execution.get("sha256")
    ):
        raise Page7InputError(f"{regime}: resume execution manifest drifted")
    output_rows = {
        str(relative): mapping(row, label=f"{regime} resume output payload")
        for relative, row in mapping(
            execution.get("output_payloads"), label="resume output payloads"
        ).items()
    }
    if set(output_rows) != set(artifact_rows) - {"execution_manifest.json"}:
        raise Page7InputError(f"{regime}: resume output-payload closure drifted")
    for relative, row in output_rows.items():
        observed_digest, observed_size = bindings[f"{artifact_prefix}{relative}"]
        if (
            row.get("sha256") != observed_digest
            or row.get("size_bytes") != observed_size
        ):
            raise Page7InputError(f"{regime}: resume output binding drifted")

    scientific = _validate_scientific_payloads(
        path,
        regime=regime,
        execution_id=execution_id,
        exact_energy=exact_energy,
        summary=summary,
        artifact_prefix=artifact_prefix,
    )
    return {
        "regime_id": regime,
        "display_name": REGIME_LABELS[regime],
        "nph": 7,
        "status": "complete",
        "execution_id": execution_id,
        "exact_same_cutoff_energy": exact_energy,
        "points": scientific["points"],
        "effective_plateau": scientific["effective_plateau"],
        "terminal_work": scientific["terminal_work"],
        "source": {
            "archive": file_binding(path),
            "cluster_id": cluster_id,
            "proc_id": proc_id,
            "attempt_ordinal": attempt_ordinal,
            "attempt_receipt_canonical_sha256": outer["sha256"],
            "worker_receipt_canonical_sha256": worker["sha256"],
            "execution_manifest_canonical_sha256": execution["sha256"],
            "job_canonical_sha256": job["sha256"],
            "authorization_canonical_sha256": authorization["sha256"],
            "package_manifest": package_manifest,
            "source_package_manifest_sha256": PACKAGE_SPECS[7][
                "manifest_sha256"
            ],
            "source_job_canonical_sha256": source_job["sha256"],
            "source_execution_id": source_execution_id,
            "resume_controller_round": resume_round,
            "source_checkpoint_sha256": resume["checkpoint_sha256"],
            "resume_archive_sha256": resume_archive["sha256"],
            "result_member": scientific["result_member"],
            "full_worker_inventory_streamed": True,
            "large_members_extracted": False,
        },
    }


def validate_attempt_archive(path: Path, *, regime: str) -> dict[str, Any]:
    """Validate one completed worker attempt without extracting its payloads."""

    if regime not in REGIME_ORDER or not path.is_file() or path.is_symlink():
        raise Page7InputError(f"{regime}: attempt archive is unavailable")
    nph = NPH_BY_REGIME[regime]
    artifact_prefix = "worker_outputs/artifacts/"
    small_members = frozenset(
        {
            "worker_outputs/attempt_identity.tsv",
            "worker_outputs/worker_exit_status.txt",
            "worker_outputs/worker_receipt.json",
            f"{artifact_prefix}execution_manifest.json",
            f"{artifact_prefix}paper_i_summary.json",
            "authority/job.json",
            "authority/execution_authorization.json",
            "authority/activation_manifest.json",
            "worker_attempt_receipt.json",
        }
    )
    bindings, raw = _read_archive_members(path, capture=small_members)
    if not small_members.issubset(raw):
        missing = sorted(small_members - set(raw))
        raise Page7InputError(
            f"{regime}: completed archive members missing: {', '.join(missing)}"
        )

    job = _json_bytes(raw["authority/job.json"], label=f"{regime} job")
    authorization = _json_bytes(
        raw["authority/execution_authorization.json"],
        label=f"{regime} authorization",
    )
    activation = _json_bytes(
        raw["authority/activation_manifest.json"],
        label=f"{regime} activation manifest",
    )
    outer = _json_bytes(
        raw["worker_attempt_receipt.json"], label=f"{regime} attempt receipt"
    )
    worker = _json_bytes(
        raw["worker_outputs/worker_receipt.json"],
        label=f"{regime} worker receipt",
    )
    execution = _json_bytes(
        raw[f"{artifact_prefix}execution_manifest.json"],
        label=f"{regime} execution manifest",
    )
    summary = _json_bytes(
        raw[f"{artifact_prefix}paper_i_summary.json"],
        label=f"{regime} Paper-I summary",
    )
    for value, label in (
        (job, "job"),
        (authorization, "authorization"),
        (activation, "activation manifest"),
        (outer, "attempt receipt"),
        (worker, "worker receipt"),
        (execution, "execution manifest"),
    ):
        verify_self_digest(value, label=f"{regime} {label}")

    if job.get("package_id") == RESUME_SPEC["package_id"]:
        return _validate_resume_attempt(
            path,
            regime=regime,
            bindings=bindings,
            raw=raw,
            job=job,
            authorization=authorization,
            activation=activation,
            outer=outer,
            worker=worker,
            execution=execution,
            summary=summary,
            artifact_prefix=artifact_prefix,
        )

    spec = PACKAGE_SPECS[nph]
    execution_id = expected_execution_id(regime)

    package_manifest = validate_package_manifest(nph)
    local_manifest = load_json(
        Path(spec["package_dir"]) / "package_manifest.json",
        label=f"{regime} local package manifest",
    )
    job_rows = {
        str(mapping(row, label="package job").get("execution_id")): mapping(
            row, label="package job"
        )
        for row in sequence(local_manifest.get("jobs"), label="package jobs")
    }
    local_job_row = mapping(
        job_rows.get(execution_id), label=f"{regime} package job binding"
    )
    job_digest, job_size = bindings["authority/job.json"]
    if (
        job.get("schema") != spec["job_schema"]
        or job.get("execution_id") != execution_id
        or job.get("package_id") != spec["package_id"]
        or job.get("campaign_id") != spec["campaign_id"]
        or job.get("regime_id") != regime
        or job.get("nph") != nph
        or job.get("target_horizon") != 50
        or job.get("execution_mode") != "fresh_0_to_50"
        or job.get("active_gradient_policy") != "stationary_source_response_v1"
        or job.get("resource_weighting_scope")
        != "all_phase_resource_weighting_v1"
        or job.get("candidate_representation") != "single_pauli_word_v1"
        or job.get("phase_i_candidate_supply")
        != "global_guarded_singleton_pool_v1"
        or job.get("phase_i_candidate_visibility")
        != "all_executable_candidates_v1"
        or job.get("phase_ii_candidate_exposure")
        != "identity_on_retained_singletons_v1"
        or job.get("phase_i_shortlist_size") != 24
        or job.get("phase_ii_shortlist_size") != 12
        or job.get("phase_iii_admission_cardinality") != 1
        or job.get("insertion_policy") != "plateau_commutation"
        or job.get("plateau_prior_mean_decrease_ratio_threshold") != 1.0e-4
        or job.get("route_contract_sha256") != ROUTE_CONTRACT_SHA256
        or job.get("fresh_start_contract")
        != {"kind": "fresh_start", "resume_archive": None, "source_checkpoint": None}
        or local_job_row.get("canonical_sha256") != job.get("sha256")
        or local_job_row.get("sha256") != job_digest
        or local_job_row.get("size_bytes") != job_size
    ):
        raise Page7InputError(f"{regime}: job route authority drifted")
    exact_energy = finite(job.get("exact_same_cutoff_energy"), label="exact energy")

    authorization_digest, _authorization_size = bindings[
        "authority/execution_authorization.json"
    ]
    if (
        authorization.get("schema") != spec["authorization_schema"]
        or authorization.get("execution_id") != execution_id
        or authorization.get("package_id") != spec["package_id"]
        or authorization.get("campaign_id") != spec["campaign_id"]
        or authorization.get("job_spec_sha256") != job.get("sha256")
        or authorization.get("package_manifest_sha256")
        != spec["manifest_sha256"]
        or authorization.get("source_archive_sha256")
        != SOURCE_ARCHIVE_SHA256
        or authorization.get("execution_authorized") is not True
        or authorization.get("submission_authorized") is not True
        or authorization.get("paper_evidence_adoption_authorized") is not False
    ):
        raise Page7InputError(f"{regime}: authorization drifted")
    activation_executions = {
        str(mapping(row, label="activation execution").get("execution_id")): mapping(
            row, label="activation execution"
        )
        for row in sequence(activation.get("executions"), label="activation executions")
    }
    activation_authorizations = {
        str(mapping(row, label="activation authorization").get("execution_id")): mapping(
            row, label="activation authorization"
        )
        for row in sequence(
            activation.get("execution_authorizations"),
            label="activation authorizations",
        )
    }
    activation_execution = mapping(
        activation_executions.get(execution_id), label="activation execution row"
    )
    activation_job = mapping(
        activation_execution.get("job"), label="activation job binding"
    )
    activation_auth = mapping(
        activation_authorizations.get(execution_id),
        label="activation authorization binding",
    )
    if (
        activation.get("schema") != spec["activation_schema"]
        or activation.get("package_id") != spec["package_id"]
        or activation.get("execution_authorized") is not True
        or activation.get("paper_evidence_adopted") is not False
        or activation_job.get("canonical_sha256") != job.get("sha256")
        or activation_job.get("sha256") != job_digest
        or activation_auth.get("canonical_sha256") != authorization.get("sha256")
        or activation_auth.get("sha256") != authorization_digest
    ):
        raise Page7InputError(f"{regime}: activation binding drifted")

    if (
        outer.get("schema") != spec["attempt_schema"]
        or outer.get("execution_id") != execution_id
        or outer.get("worker_exit_status") != 0
        or outer.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
        or outer.get("image_sha256") != authorization.get("remote_image_sha256")
        or outer.get("job_file_sha256") != job_digest
        or outer.get("authorization_file_sha256") != authorization_digest
        or outer.get("activation_manifest_file_sha256")
        != bindings["authority/activation_manifest.json"][0]
    ):
        raise Page7InputError(f"{regime}: attempt receipt drifted")
    cluster_id = integer(outer.get("cluster_id"), label="cluster id")
    proc_id = integer(outer.get("proc_id"), label="proc id")
    attempt_ordinal = integer(
        outer.get("attempt_ordinal"), label="attempt ordinal", minimum=1
    )
    expected_identity = (
        f"{execution_id}\t{cluster_id}\t{proc_id}\t{attempt_ordinal}\n"
    ).encode("ascii")
    if (
        raw["worker_outputs/attempt_identity.tsv"] != expected_identity
        or raw["worker_outputs/worker_exit_status.txt"].strip() != b"0"
        or activation_execution.get("queue_index") != proc_id
    ):
        raise Page7InputError(f"{regime}: attempt identity marker drifted")
    worker_rows = _artifact_rows(
        outer.get("worker_files"), label=f"{regime} worker files"
    )
    observed_worker = {
        name: binding
        for name, binding in bindings.items()
        if name.startswith("worker_outputs/")
    }
    _validate_binding_rows(
        worker_rows,
        observed_worker,
        prefix="worker_outputs/",
        label=f"{regime} outer worker inventory",
    )

    if (
        worker.get("schema") != spec["worker_schema"]
        or worker.get("status") != "passed"
        or worker.get("package_id") != spec["package_id"]
        or worker.get("campaign_id") != spec["campaign_id"]
        or worker.get("execution_id") != execution_id
        or worker.get("job_spec_sha256") != job.get("sha256")
        or worker.get("authorization_sha256") != authorization.get("sha256")
        or worker.get("controller_rounds_completed") != 50
        or worker.get("fresh_start") is not True
    ):
        raise Page7InputError(f"{regime}: worker completion receipt drifted")
    artifact_rows = _artifact_rows(
        worker.get("artifacts"), label=f"{regime} artifacts"
    )
    required_artifacts = {
        "checkpoint.json",
        "estimator_ledger.json",
        "execution_manifest.json",
        "paper_i_summary.json",
        "result.json",
    }
    if not required_artifacts.issubset(artifact_rows):
        raise Page7InputError(f"{regime}: artifact closure drifted")
    observed_artifacts = {
        name: binding
        for name, binding in bindings.items()
        if name.startswith(artifact_prefix)
    }
    _validate_binding_rows(
        artifact_rows,
        observed_artifacts,
        prefix=artifact_prefix,
        label=f"{regime} artifact inventory",
    )

    if (
        execution.get("schema") != spec["execution_schema"]
        or execution.get("status") != "passed"
        or execution.get("package_id") != spec["package_id"]
        or execution.get("campaign_id") != spec["campaign_id"]
        or execution.get("execution_id") != execution_id
        or execution.get("job_spec_sha256") != job.get("sha256")
        or execution.get("authorization_sha256") != authorization.get("sha256")
        or execution.get("target_horizon") != 50
        or execution.get("controller_rounds_completed") != 50
        or execution.get("fresh_start") is not True
        or execution.get("source_checkpoint_consumed") is not False
        or worker.get("execution_manifest_sha256") != execution.get("sha256")
    ):
        raise Page7InputError(f"{regime}: execution manifest drifted")
    output_rows = {
        str(relative): mapping(row, label=f"{regime} output payload")
        for relative, row in mapping(
            execution.get("output_payloads"), label="output payloads"
        ).items()
    }
    expected_preliminary = set(artifact_rows) - {"execution_manifest.json"}
    if set(output_rows) != expected_preliminary:
        raise Page7InputError(f"{regime}: output-payload closure drifted")
    for relative, row in output_rows.items():
        observed_digest, observed_size = bindings[f"{artifact_prefix}{relative}"]
        if (
            row.get("sha256") != observed_digest
            or row.get("size_bytes") != observed_size
        ):
            raise Page7InputError(f"{regime}: output payload binding drifted")

    scientific = _validate_scientific_payloads(
        path,
        regime=regime,
        execution_id=execution_id,
        exact_energy=exact_energy,
        summary=summary,
        artifact_prefix=artifact_prefix,
    )
    archive = file_binding(path)
    return {
        "regime_id": regime,
        "display_name": REGIME_LABELS[regime],
        "nph": nph,
        "status": "complete",
        "execution_id": execution_id,
        "exact_same_cutoff_energy": exact_energy,
        "points": scientific["points"],
        "effective_plateau": scientific["effective_plateau"],
        "terminal_work": scientific["terminal_work"],
        "source": {
            "archive": archive,
            "cluster_id": cluster_id,
            "proc_id": proc_id,
            "attempt_ordinal": attempt_ordinal,
            "attempt_receipt_canonical_sha256": outer["sha256"],
            "worker_receipt_canonical_sha256": worker["sha256"],
            "execution_manifest_canonical_sha256": execution["sha256"],
            "job_canonical_sha256": job["sha256"],
            "authorization_canonical_sha256": authorization["sha256"],
            "package_manifest": package_manifest,
            "result_member": scientific["result_member"],
            "full_worker_inventory_streamed": True,
            "large_members_extracted": False,
        },
    }


def _effective_plateau(points: Sequence[Any], *, label: str) -> dict[str, Any]:
    from pipelines.reporting.paper_i_run_summary import (
        PaperIErrorTracePoint,
        select_paper_i_effective_plateau,
    )

    positive: list[PaperIErrorTracePoint] = []
    by_round: dict[int, Mapping[str, Any]] = {}
    for expected_round, raw in enumerate(points):
        point = mapping(raw, label=f"{label} point")
        round_index = integer(point.get("round"), label=f"{label} round")
        if round_index != expected_round:
            raise Page7InputError(f"{label} rounds are not complete and ordered")
        by_round[round_index] = point
        if round_index > 0:
            positive.append(
                PaperIErrorTracePoint(
                    controller_round=round_index,
                    absolute_energy_error=finite(
                        point.get("delta_e"),
                        label=f"{label} error",
                        minimum=0.0,
                    ),
                )
            )
    selected = select_paper_i_effective_plateau(tuple(positive))
    point = by_round[selected.controller_round]
    return {
        "policy": selected.policy,
        "round": selected.controller_round,
        "delta_e": finite(point.get("delta_e"), label=f"{label} marker error"),
        "best_observed_delta_e": selected.best_observed_error,
        "selection_threshold": selected.selection_threshold,
    }


def _normalize_cost_observation(
    value: Any,
    *,
    label: str,
    expected_round: int,
    expected_error: float,
) -> dict[str, Any]:
    row = mapping(value, label=label)
    costs = mapping(row.get("costs"), label=f"{label} costs")
    if set(costs) != set(COST_FIELDS):
        raise Page7InputError(f"{label} cost tuple drifted")
    normalized = {
        field: integer(costs.get(field), label=f"{label} {field}")
        for field in COST_FIELDS
    }
    error = finite(row.get("delta_e"), label=f"{label} error", minimum=0.0)
    if row.get("round") != expected_round or not math.isclose(
        error, expected_error, rel_tol=0.0, abs_tol=1.0e-15
    ):
        raise Page7InputError(f"{label} selected prefix drifted")
    return {
        "round": expected_round,
        "delta_e": error,
        "costs": normalized,
        "checkpoint_sha256": row.get("checkpoint_sha256"),
        "compile": copy.deepcopy(row.get("compile", {})),
    }


def _append_source_path(raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str):
        raise Page7InputError(f"{label} path is unavailable")
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else REPO_ROOT / candidate
    if path.is_symlink() or not path.is_file():
        raise Page7InputError(f"{label} is unavailable or is a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise Page7InputError(f"{label} escaped the active repository") from exc
    if not resolved.is_file():
        raise Page7InputError(f"{label} is unavailable")
    return resolved


def _compile_prefix_cost(
    method: str,
    cell: Mapping[str, Any],
    *,
    controller_round: int,
    error: float,
    compiler: Callable[[Any], Any] | None,
) -> dict[str, Any]:
    if method == "ra":
        source = mapping(cell.get("source"), label="RA source")
        archive = _append_source_path(
            mapping(source.get("archive"), label="RA archive").get("path"),
            label="RA attempt archive",
        )
        prefix = common_cost._ra_prefix_from_archive(
            {
                "method_family": "ra",
                "execution_id": cell["execution_id"],
                "attempt_path": str(archive),
                "result_member": source["result_member"],
            },
            controller_round=controller_round,
        )
    elif method == "append":
        source = mapping(cell.get("source"), label="Append source")
        archive_binding = mapping(source.get("archive"), label="Append archive")
        archive = _append_source_path(
            archive_binding.get("path"), label="Append attempt archive"
        )
        if (
            archive.stat().st_size != archive_binding.get("size_bytes")
            or sha256_file(archive) != archive_binding.get("sha256")
        ):
            raise Page7InputError("Append attempt archive byte binding drifted")
        job_binding = mapping(source.get("job"), label="Append job")
        job_path = _append_source_path(job_binding.get("path"), label="Append job")
        job = load_json(job_path, label="Append job")
        if (
            sha256_file(job_path) != job_binding.get("sha256")
            or verify_self_digest(job, label="Append job")
            != job_binding.get("canonical_sha256")
        ):
            raise Page7InputError("Append job byte binding drifted")
        manifest_binding = mapping(
            source.get("package_manifest"), label="Append package manifest"
        )
        manifest_path = _append_source_path(
            manifest_binding.get("path"), label="Append package manifest"
        )
        if (
            sha256_file(manifest_path) != manifest_binding.get("sha256")
            or verify_self_digest(
                load_json(manifest_path, label="Append package manifest"),
                label="Append package manifest",
            )
            != manifest_binding.get("canonical_sha256")
        ):
            raise Page7InputError("Append package-manifest binding drifted")
        package_dir = manifest_path.parent
        master._configure_package_dir(package_dir)
        prefix = common_cost._append_prefix_from_archive(
            {
                "method_family": "append",
                "execution_id": cell["execution_id"],
                "attempt_path": str(archive),
                "result_member": "worker_outputs/payload/result.json",
                "package_dir": str(package_dir),
                "job": job,
            },
            controller_round=controller_round,
        )
    else:  # pragma: no cover - internal guard
        raise Page7InputError(f"unknown prefix method: {method}")
    observation = master._fixed_prefix_qiskit_observation(
        prefix, error=error, compiler=compiler
    )
    return {
        "round": int(observation["k"]),
        "delta_e": float(observation["error"]),
        "costs": {
            "N2q": int(observation["N2q"]),
            "D2q": int(observation["D2q"]),
            "Dc": int(observation["Dc"]),
            "W1q": int(observation["W1q"]),
            "S_alg": int(observation["S_alg"]),
        },
        "checkpoint_sha256": observation["checkpoint_sha256"],
        "compile": {
            "compile_convention": observation["compile_convention"],
            "qiskit_version": observation.get("qiskit_version"),
            "source": "authenticated_prefix_shared_locked_compiler_v1",
        },
    }


PrefixCostProvider = Callable[
    [str, Mapping[str, Any], int, float], Mapping[str, Any]
]


def _common_accuracy_selection(
    ra_points: Sequence[Any],
    append_points: Sequence[Any],
    *,
    regime: str,
    policy: str = "full_horizon_equal_attainable_error_v1",
) -> dict[str, Any]:
    ra = [
        mapping(point, label=f"{regime} RA point")
        for point in ra_points
        if mapping(point, label=f"{regime} RA point").get("round", 0) > 0
    ]
    append = [
        mapping(point, label=f"{regime} Append point")
        for point in append_points
        if mapping(point, label=f"{regime} Append point").get("round", 0) > 0
    ]
    ra_min = min(finite(point.get("delta_e"), label="RA error") for point in ra)
    append_min = min(
        finite(point.get("delta_e"), label="Append error") for point in append
    )
    target = max(ra_min, append_min)
    ra_cross = next(
        point
        for point in ra
        if finite(point.get("delta_e"), label="RA crossing") <= target
    )
    append_cross = next(
        point
        for point in append
        if finite(point.get("delta_e"), label="Append crossing") <= target
    )
    ra_horizon = max(integer(point.get("round"), label="RA round") for point in ra)
    append_horizon = max(
        integer(point.get("round"), label="Append round") for point in append
    )
    if policy == "full_horizon_equal_attainable_error_v1":
        if ra_horizon != 50 or append_horizon != 70:
            raise Page7InputError(
                "legacy full-horizon common accuracy requires RA k=50 and "
                "Append k=70"
            )
        minimum_fields = {
            "ra_minimum_delta_e_k1_50": ra_min,
            "append_minimum_delta_e_k1_70": append_min,
        }
    elif policy == "display_horizon_equal_attainable_error_v2":
        minimum_fields = {
            "ra_minimum_delta_e": ra_min,
            "append_minimum_delta_e": append_min,
            "ra_horizon": ra_horizon,
            "append_horizon": append_horizon,
        }
    else:
        raise Page7InputError(f"unsupported common-accuracy policy: {policy}")
    return {
        "policy": policy,
        "target_delta_e": target,
        **minimum_fields,
        "ra_round": int(ra_cross["round"]),
        "ra_delta_e": float(ra_cross["delta_e"]),
        "append_round": int(append_cross["round"]),
        "append_delta_e": float(append_cross["delta_e"]),
        "round_zero_excluded": True,
        "earliest_crossing": True,
    }


def _portable_repo_file_binding(path: Path, *, label: str) -> dict[str, Any]:
    """Bind source bytes without embedding a machine-specific checkout root."""

    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise Page7InputError(f"{label} is unavailable or is a symlink")
    try:
        relative = resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise Page7InputError(f"{label} escaped the active repository") from exc
    return {
        "path": relative,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _portable_package_manifest(nph: int) -> dict[str, Any]:
    binding = validate_package_manifest(nph)
    path = Path(PACKAGE_SPECS[nph]["package_dir"]) / "package_manifest.json"
    return {
        **binding,
        "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
    }


def _remote_archive_binding(
    *, path: str, sha256: str, size_bytes: int
) -> dict[str, Any]:
    if not isinstance(path, str) or "\x00" in path:
        raise Page7InputError("remote archive path is invalid")
    pure = PurePosixPath(path)
    if (
        not pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != path
    ):
        raise Page7InputError("remote archive path must be normalized and absolute")
    if not isinstance(sha256, str) or not SHA256_RE.fullmatch(sha256):
        raise Page7InputError("remote archive SHA-256 is invalid")
    size = integer(size_bytes, label="remote archive size", minimum=1)
    return {
        "path": path,
        "sha256": sha256,
        "size_bytes": size,
        "state": "preserved_remote_not_fetched",
    }


def _portable_append_binding(append: Mapping[str, Any]) -> dict[str, Any]:
    binding = mapping(append.get("file_binding"), label="Append file binding")
    return {
        "sha256": binding.get("sha256"),
        "size_bytes": binding.get("size_bytes"),
        "canonical_sha256": append.get("sha256"),
    }


def _validate_projection_source_code(
    value: Any, *, expected_path: Path, label: str
) -> dict[str, Any]:
    observed = mapping(value, label=label)
    expected = _portable_repo_file_binding(expected_path, label=label)
    if dict(observed) != expected:
        raise Page7InputError(f"{label} byte binding drifted")
    return expected


def _validate_projection_updater_source_code(value: Any) -> dict[str, Any]:
    """Accept current bytes or the exact updater that sealed cluster 9401087."""

    label = "compact-projection updater"
    observed = dict(mapping(value, label=label))
    current = _portable_repo_file_binding(Path(__file__), label=label)
    if observed == current or observed == PRIOR_COMPACT_PROJECTION_UPDATER_BINDING:
        return observed
    raise Page7InputError(f"{label} byte binding drifted")


def _validate_projection_archive_validation(
    value: Any,
    *,
    regime: str,
    source_archive: Mapping[str, Any],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the embedded full-scan receipt used by an nph=3 projection."""

    if regime not in NPH3_REGIMES:
        raise Page7InputError(
            "compact projection v1 requires the authenticated nph=3 validator"
        )
    validation = mapping(value, label=f"{regime} archive validation")
    canonical = verify_self_digest(
        validation, label=f"{regime} archive validation"
    )
    archive = mapping(
        validation.get("archive"), label=f"{regime} validated archive"
    )
    member_validation = mapping(
        validation.get("member_validation"),
        label=f"{regime} member validation",
    )
    source = mapping(cell.get("source"), label=f"{regime} projected source")
    worker_attempt = mapping(
        validation.get("worker_attempt_receipt"),
        label=f"{regime} validated attempt receipt",
    )
    worker = mapping(
        validation.get("worker_receipt"),
        label=f"{regime} validated worker receipt",
    )
    execution = mapping(
        validation.get("execution_manifest"),
        label=f"{regime} validated execution manifest",
    )
    bindings = mapping(
        validation.get("bindings"), label=f"{regime} validation bindings"
    )
    job = mapping(bindings.get("job"), label=f"{regime} validated job")
    authorization = mapping(
        bindings.get("authorization"),
        label=f"{regime} validated authorization",
    )
    required_closure = (
        "gzip_and_full_tar_scan_passed",
        "compressed_hash_size_stream_closure_passed",
        "safe_unique_regular_only_member_closure_passed",
        "worker_inventory_hash_size_closure_passed",
        "nested_artifact_inventory_closure_passed",
        "authority_byte_identity_passed",
        "fifty_round_success_closure_passed",
    )
    if (
        validation.get("schema") != NPH3_ARCHIVE_VALIDATION_SCHEMA
        or validation.get("status") != "passed"
        or validation.get("execution_id") != cell.get("execution_id")
        or validation.get("cluster_id") != source.get("cluster_id")
        or validation.get("proc_id") != source.get("proc_id")
        or validation.get("attempt_ordinal") != source.get("attempt_ordinal")
        or validation.get("controller_rounds_completed") != 50
        or archive.get("sha256") != source_archive.get("sha256")
        or archive.get("size_bytes") != source_archive.get("size_bytes")
        or any(member_validation.get(field) is not True for field in required_closure)
        or integer(
            member_validation.get("member_count"),
            label=f"{regime} validated member count",
            minimum=1,
        )
        < 1
        or integer(
            member_validation.get("worker_file_count"),
            label=f"{regime} validated worker file count",
            minimum=1,
        )
        < 1
        or worker_attempt.get("canonical_sha256")
        != source.get("attempt_receipt_canonical_sha256")
        or worker_attempt.get("worker_exit_status") != 0
        or worker.get("canonical_sha256")
        != source.get("worker_receipt_canonical_sha256")
        or worker.get("controller_rounds_completed") != 50
        or execution.get("canonical_sha256")
        != source.get("execution_manifest_canonical_sha256")
        or execution.get("controller_rounds_completed") != 50
        or job.get("canonical_sha256") != source.get("job_canonical_sha256")
        or authorization.get("canonical_sha256")
        != source.get("authorization_canonical_sha256")
        or bindings.get("source_archive_sha256") != SOURCE_ARCHIVE_SHA256
    ):
        raise Page7InputError(f"{regime}: compact archive validation drifted")
    return {**copy.deepcopy(dict(validation)), "sha256": canonical}


def _validate_projected_ra_cell(
    value: Any, *, regime: str, source_archive: Mapping[str, Any]
) -> dict[str, Any]:
    cell = copy.deepcopy(dict(mapping(value, label=f"{regime} projected RA cell")))
    exact = finite(
        cell.get("exact_same_cutoff_energy"),
        label=f"{regime} projected exact energy",
    )
    points = sequence(cell.get("points"), label=f"{regime} projected points")
    if (
        cell.get("regime_id") != regime
        or cell.get("display_name") != REGIME_LABELS[regime]
        or cell.get("nph") != NPH_BY_REGIME[regime]
        or cell.get("status") != "complete"
        or cell.get("execution_id") != expected_execution_id(regime)
        or len(points) != 51
    ):
        raise Page7InputError(f"{regime}: compact RA cell identity drifted")
    previous_energy: float | None = None
    for expected_round, raw in enumerate(points):
        point = mapping(raw, label=f"{regime} projected point")
        energy = finite(point.get("energy"), label=f"{regime} projected energy")
        error = finite(
            point.get("delta_e"),
            label=f"{regime} projected error",
            minimum=0.0,
        )
        if (
            point.get("round") != expected_round
            or not math.isclose(
                error,
                abs(energy - exact),
                rel_tol=1.0e-12,
                abs_tol=COMPACT_PROJECTION_DELTA_E_ABS_TOL,
            )
            or (previous_energy is not None and energy > previous_energy + 1.0e-10)
        ):
            raise Page7InputError(f"{regime}: compact RA trajectory drifted")
        previous_energy = energy
    marker = _effective_plateau(points, label=f"{regime} projected RA")
    if cell.get("effective_plateau") != marker:
        raise Page7InputError(f"{regime}: compact RA plateau drifted")
    terminal_work = _work(
        cell.get("terminal_work"), label=f"{regime} projected terminal work"
    )
    source = mapping(cell.get("source"), label=f"{regime} projected source")
    if (
        mapping(source.get("archive"), label=f"{regime} projected archive")
        != source_archive
        or source.get("package_manifest") != _portable_package_manifest(3)
        or source.get("result_member")
        != "worker_outputs/artifacts/result.json"
        or source.get("full_worker_inventory_streamed") is not True
        or source.get("large_members_extracted") is not False
        or any(
            not isinstance(source.get(field), str)
            or not SHA256_RE.fullmatch(str(source.get(field)))
            for field in (
                "attempt_receipt_canonical_sha256",
                "worker_receipt_canonical_sha256",
                "execution_manifest_canonical_sha256",
                "job_canonical_sha256",
                "authorization_canonical_sha256",
            )
        )
    ):
        raise Page7InputError(f"{regime}: compact RA source drifted")
    cell["terminal_work"] = terminal_work
    return cell


def _projection_observations(
    value: Any,
    *,
    regime: str,
    cell: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    raw = mapping(value, label=f"{regime} projected RA observations")
    needed = {50, integer(selection.get("ra_round"), label="common RA round")}
    try:
        observed_rounds = {int(round_index) for round_index in raw}
    except (TypeError, ValueError) as exc:
        raise Page7InputError(
            f"{regime}: compact RA observation key drifted"
        ) from exc
    if (
        not needed.issubset(observed_rounds)
        or any(str(round_index) not in raw for round_index in observed_rounds)
        or any(round_index < 1 or round_index > 50 for round_index in observed_rounds)
    ):
        raise Page7InputError(f"{regime}: compact RA observation closure drifted")
    points = {
        int(mapping(point, label="projected RA point")["round"]): mapping(
            point, label="projected RA point"
        )
        for point in sequence(cell.get("points"), label="projected RA points")
    }
    normalized: dict[int, dict[str, Any]] = {}
    for round_index in sorted(observed_rounds):
        observation = _normalize_cost_observation(
            raw[str(round_index)],
            label=f"{regime} projected RA k={round_index}",
            expected_round=round_index,
            expected_error=finite(
                points[round_index].get("delta_e"),
                label=f"{regime} projected RA error",
            ),
        )
        compile_row = mapping(
            observation.get("compile"),
            label=f"{regime} projected RA compile receipt",
        )
        if (
            compile_row.get("compile_convention")
            != "table_i_basis_gate_transpile_v1"
            or compile_row.get("source")
            != "authenticated_prefix_shared_locked_compiler_v1"
            or not isinstance(observation.get("checkpoint_sha256"), str)
            or not SHA256_RE.fullmatch(str(observation.get("checkpoint_sha256")))
        ):
            raise Page7InputError(f"{regime}: compact RA compile receipt drifted")
        normalized[round_index] = observation
    if normalized[50]["costs"]["S_alg"] != mapping(
        cell.get("terminal_work"), label="projected terminal work"
    ).get("S_alg"):
        raise Page7InputError(f"{regime}: compact RA terminal S_alg drifted")
    return normalized


def validate_compact_ra_projection(
    path: Path, *, regime: str, append: Mapping[str, Any]
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Page7InputError(f"{regime}: compact RA projection is unavailable")
    projection = load_json(path, label=f"{regime} compact RA projection")
    canonical = verify_self_digest(
        projection, label=f"{regime} compact RA projection"
    )
    source_archive = mapping(
        projection.get("source_archive"),
        label=f"{regime} compact source archive",
    )
    normalized_archive = _remote_archive_binding(
        path=str(source_archive.get("path", "")),
        sha256=str(source_archive.get("sha256", "")),
        size_bytes=source_archive.get("size_bytes"),
    )
    if dict(source_archive) != normalized_archive:
        raise Page7InputError(f"{regime}: compact remote archive binding drifted")
    if (
        projection.get("schema") != RA_PROJECTION_SCHEMA
        or projection.get("status") != "passed"
        or projection.get("regime_id") != regime
        or projection.get("execution_id") != expected_execution_id(regime)
        or projection.get("paper_evidence_adopted") is not False
        or projection.get("append_adapter") != _portable_append_binding(append)
    ):
        raise Page7InputError(f"{regime}: compact RA projection identity drifted")
    source_code = mapping(
        projection.get("source_code"), label=f"{regime} projection source code"
    )
    _validate_projection_updater_source_code(source_code.get("updater"))
    _validate_projection_source_code(
        source_code.get("archive_validator"),
        expected_path=NPH3_ARCHIVE_VALIDATOR,
        label="compact-projection archive validator",
    )
    cell = _validate_projected_ra_cell(
        projection.get("cell"), regime=regime, source_archive=source_archive
    )
    validation = _validate_projection_archive_validation(
        projection.get("archive_validation"),
        regime=regime,
        source_archive=source_archive,
        cell=cell,
    )
    validation_file = mapping(
        projection.get("archive_validation_file"),
        label=f"{regime} archive-validation file binding",
    )
    canonical_validation_file = canonical_json_bytes(validation) + b"\n"
    if (
        validation_file.get("canonical_sha256") != validation["sha256"]
        or validation_file.get("sha256")
        != hashlib.sha256(canonical_validation_file).hexdigest()
        or validation_file.get("size_bytes") != len(canonical_validation_file)
    ):
        raise Page7InputError(f"{regime}: archive-validation binding drifted")
    append_cells = {
        str(raw["regime_id"]): mapping(raw, label="Append cell")
        for raw in sequence(append.get("cells"), label="Append cells")
    }
    selection = _common_accuracy_selection(
        sequence(cell.get("points"), label="projected RA points"),
        sequence(
            append_cells[regime].get("points"), label="projected Append points"
        ),
        regime=regime,
    )
    if projection.get("common_accuracy_selection") != selection:
        raise Page7InputError(f"{regime}: compact common-accuracy selection drifted")
    observations = _projection_observations(
        projection.get("ra_cost_observations"),
        regime=regime,
        cell=cell,
        selection=selection,
    )
    cell_source = dict(mapping(cell.get("source"), label="projected RA source"))
    cell_source["compact_projection"] = {
        **file_binding(path),
        "canonical_sha256": canonical,
        "archive_validation_canonical_sha256": validation["sha256"],
        "full_archive_preserved_remote": True,
    }
    cell["source"] = cell_source
    return {
        "cell": cell,
        "observations": observations,
        "canonical_sha256": canonical,
    }


def build_compact_ra_projection(
    *,
    append_adapter_path: Path,
    regime: str,
    archive_path: Path,
    archive_validation_path: Path,
    remote_archive_path: str,
    remote_archive_sha256: str,
    remote_archive_size_bytes: int,
    output: Path,
    prefix_cost_provider: PrefixCostProvider | None = None,
    compiler: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Seal a portable RA cell after full validation beside the remote archive."""

    if regime not in NPH3_REGIMES:
        raise Page7InputError(
            "compact projection v1 currently supports completed nph=3 cells"
        )
    append = validate_append_adapter(append_adapter_path)
    cell = validate_attempt_archive(archive_path, regime=regime)
    compile_cell = copy.deepcopy(cell)
    local_archive = mapping(
        mapping(cell.get("source"), label="RA source").get("archive"),
        label="RA archive",
    )
    source_archive = _remote_archive_binding(
        path=remote_archive_path,
        sha256=remote_archive_sha256,
        size_bytes=remote_archive_size_bytes,
    )
    if (
        local_archive.get("sha256") != source_archive["sha256"]
        or local_archive.get("size_bytes") != source_archive["size_bytes"]
    ):
        raise Page7InputError("remote/local archive byte bindings differ")
    cell = copy.deepcopy(cell)
    cell_source = dict(mapping(cell.get("source"), label="RA source"))
    cell_source["archive"] = source_archive
    cell_source["package_manifest"] = _portable_package_manifest(3)
    cell["source"] = cell_source

    if archive_validation_path.is_symlink() or not archive_validation_path.is_file():
        raise Page7InputError("full archive validation file is unavailable")
    validation = load_json(
        archive_validation_path, label=f"{regime} full archive validation"
    )
    if archive_validation_path.read_bytes() != canonical_json_bytes(validation) + b"\n":
        raise Page7InputError("full archive validation file is not canonical JSON")
    validated = _validate_projection_archive_validation(
        validation,
        regime=regime,
        source_archive=source_archive,
        cell=cell,
    )
    append_cells = {
        str(raw["regime_id"]): mapping(raw, label="Append cell")
        for raw in sequence(append.get("cells"), label="Append cells")
    }
    selection = _common_accuracy_selection(
        sequence(cell.get("points"), label="RA points"),
        sequence(append_cells[regime].get("points"), label="Append points"),
        regime=regime,
    )
    displayed_append_points = sequence(
        append_cells[regime].get("points"), label="displayed Append points"
    )[: APPEND_TERMINAL_ROUND_BY_REGIME[regime] + 1]
    displayed_selection = _common_accuracy_selection(
        sequence(cell.get("points"), label="RA points"),
        displayed_append_points,
        regime=regime,
        policy="display_horizon_equal_attainable_error_v2",
    )

    def observe(round_index: int) -> Mapping[str, Any]:
        point = mapping(
            sequence(cell.get("points"), label="RA points")[round_index],
            label="RA point",
        )
        error = finite(point.get("delta_e"), label="RA error")
        if prefix_cost_provider is not None:
            return prefix_cost_provider("ra", cell, round_index, error)
        return _compile_prefix_cost(
            "ra",
            compile_cell,
            controller_round=round_index,
            error=error,
            compiler=compiler,
        )

    needed = {
        50,
        int(selection["ra_round"]),
        int(displayed_selection["ra_round"]),
    }
    observations = {
        str(round_index): _normalize_cost_observation(
            observe(round_index),
            label=f"{regime} projected RA k={round_index}",
            expected_round=round_index,
            expected_error=float(cell["points"][round_index]["delta_e"]),
        )
        for round_index in sorted(needed)
    }
    projection = digested(
        {
            "schema": RA_PROJECTION_SCHEMA,
            "status": "passed",
            "classification": "supplemental_diagnostic_not_adopted_evidence",
            "paper_evidence_adopted": False,
            "regime_id": regime,
            "execution_id": cell["execution_id"],
            "source_archive": source_archive,
            "archive_validation": validated,
            "archive_validation_file": {
                "sha256": sha256_file(archive_validation_path),
                "size_bytes": archive_validation_path.stat().st_size,
                "canonical_sha256": validated["sha256"],
            },
            "source_code": {
                "updater": _portable_repo_file_binding(
                    Path(__file__), label="compact-projection updater"
                ),
                "archive_validator": _portable_repo_file_binding(
                    NPH3_ARCHIVE_VALIDATOR,
                    label="compact-projection archive validator",
                ),
            },
            "append_adapter": _portable_append_binding(append),
            "cell": cell,
            "common_accuracy_selection": selection,
            "ra_cost_observations": observations,
        }
    )
    if output.exists() or output.is_symlink():
        existing = load_json(output, label="existing compact RA projection")
        verify_self_digest(existing, label="existing compact RA projection")
        if canonical_json_bytes(existing) != canonical_json_bytes(projection):
            raise Page7InputError("compact RA projection output already differs")
        return existing
    legacy_page._atomic_write_json(output, projection)
    return projection


def _cell_digest(cell: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(cell)).hexdigest()


def _write_monotone_adapter(path: Path, adapter: Mapping[str, Any]) -> dict[str, Any]:
    if not path.exists() and not path.is_symlink():
        legacy_page._atomic_write_json(path, adapter)
        return copy.deepcopy(dict(adapter))
    existing = load_json(path, label="existing page-7 adapter")
    verify_self_digest(existing, label="existing page-7 adapter")
    if existing.get("schema") != ADAPTER_SCHEMA:
        raise Page7InputError("existing page-7 adapter schema drifted")
    if canonical_json_bytes(existing) == canonical_json_bytes(adapter):
        return existing
    old_cells = {
        str(mapping(cell, label="existing cell").get("regime_id")): mapping(
            cell, label="existing cell"
        )
        for cell in sequence(existing.get("cells"), label="existing cells")
    }
    new_cells = {
        str(mapping(cell, label="new cell").get("regime_id")): mapping(
            cell, label="new cell"
        )
        for cell in sequence(adapter.get("cells"), label="new cells")
    }
    old_completed = {
        regime for regime, cell in old_cells.items() if cell.get("status") == "complete"
    }
    new_completed = {
        regime for regime, cell in new_cells.items() if cell.get("status") == "complete"
    }
    if not old_completed < new_completed:
        raise Page7InputError(
            "adapter replacement must be a strict completion superset"
        )
    if existing.get("append_adapter") != adapter.get("append_adapter"):
        raise Page7InputError("Append adapter binding drifted during completion")
    for regime in REGIME_ORDER:
        old_cell = old_cells[regime]
        new_cell = new_cells[regime]
        if canonical_json_bytes(old_cell.get("append")) != canonical_json_bytes(
            new_cell.get("append")
        ):
            raise Page7InputError(f"Append cell drifted: {regime}")
        if regime in old_completed and canonical_json_bytes(
            old_cell
        ) != canonical_json_bytes(new_cell):
            raise Page7InputError(f"completed cell drifted: {regime}")
    legacy_page._atomic_write_json(path, adapter)
    return copy.deepcopy(dict(adapter))


def build_adapter(
    *,
    append_adapter_path: Path,
    ra_attempts: Mapping[str, Path],
    ra_projections: Mapping[str, Path] | None = None,
    output: Path,
    prefix_cost_provider: PrefixCostProvider | None = None,
    compiler: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    projection_paths = {} if ra_projections is None else dict(ra_projections)
    supplied = set(ra_attempts) | set(projection_paths)
    if not supplied.issubset(set(REGIME_ORDER)):
        raise Page7InputError("an unknown RA regime was supplied")
    overlap = set(ra_attempts) & set(projection_paths)
    if overlap:
        raise Page7InputError(
            "RA attempt and compact projection duplicate regimes: "
            + ", ".join(sorted(overlap))
        )
    append = validate_append_adapter(append_adapter_path)
    append_cells = {
        str(cell["regime_id"]): mapping(cell, label="Append cell")
        for cell in append["cells"]
    }
    ra_cells = {
        regime: validate_attempt_archive(Path(ra_attempts[regime]), regime=regime)
        for regime in REGIME_ORDER
        if regime in ra_attempts
    }
    projected = {
        regime: validate_compact_ra_projection(
            Path(projection_paths[regime]), regime=regime, append=append
        )
        for regime in REGIME_ORDER
        if regime in projection_paths
    }
    ra_cells.update(
        {regime: projection["cell"] for regime, projection in projected.items()}
    )

    def observe(
        method: str,
        cell: Mapping[str, Any],
        controller_round: int,
        error: float,
    ) -> Mapping[str, Any]:
        regime = str(cell["regime_id"])
        if method == "ra" and regime in projected:
            observations = mapping(
                projected[regime].get("observations"),
                label=f"{regime} compact RA observations",
            )
            observation = observations.get(controller_round)
            if observation is None:
                raise Page7InputError(
                    f"{regime}: compact RA observation k={controller_round} "
                    "is unavailable"
                )
            return mapping(
                observation,
                label=f"{regime} compact RA observation k={controller_round}",
            )
        if prefix_cost_provider is not None:
            return prefix_cost_provider(method, cell, controller_round, error)
        return _compile_prefix_cost(
            method,
            cell,
            controller_round=controller_round,
            error=error,
            compiler=compiler,
        )

    cost_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def cost(
        method: str,
        cell: Mapping[str, Any],
        round_index: int,
        error: float,
    ) -> dict[str, Any]:
        key = (method, str(cell["regime_id"]), round_index)
        cached = cost_cache.get(key)
        if cached is not None:
            return copy.deepcopy(cached)
        value = _normalize_cost_observation(
            observe(method, cell, round_index, error),
            label=f"{cell['regime_id']} {method} k={round_index}",
            expected_round=round_index,
            expected_error=error,
        )
        cost_cache[key] = value
        return copy.deepcopy(value)

    cells: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        append_cell = append_cells[regime]
        append_source_points = sequence(
            append_cell.get("points"), label=f"Append {regime} points"
        )
        append_terminal_round = APPEND_TERMINAL_ROUND_BY_REGIME[regime]
        append_points = copy.deepcopy(
            append_source_points[: append_terminal_round + 1]
        )
        if [
            mapping(point, label=f"Append {regime} displayed point").get("round")
            for point in append_points
        ] != list(range(append_terminal_round + 1)):
            raise Page7InputError(
                f"Append {regime} displayed trajectory is not rounds "
                f"0..{append_terminal_round}"
            )
        append_marker = _effective_plateau(
            append_points, label=f"Append {regime}"
        )
        append_terminal = copy.deepcopy(
            mapping(
                append_cell["endpoints"][f"round_{append_terminal_round}"],
                label=f"Append {regime} terminal k={append_terminal_round}",
            )
        )
        if regime not in ra_cells:
            cells.append(
                {
                    "regime_id": regime,
                    "display_name": REGIME_LABELS[regime],
                    "nph": NPH_BY_REGIME[regime],
                    "status": "pending",
                    "pending_reason": "validated_complete_RA_attempt_unavailable",
                    "append": {
                        "execution_id": append_cell["execution_id"],
                        "exact_same_cutoff_energy": append_cell[
                            "exact_same_cutoff_energy"
                        ],
                        "points": copy.deepcopy(append_points),
                        "effective_plateau": append_marker,
                        "terminal": append_terminal,
                        "display_terminal_round": append_terminal_round,
                        "source": copy.deepcopy(append_cell.get("source", {})),
                    },
                    "ra": None,
                    "common_accuracy": None,
                }
            )
            continue
        ra_cell = ra_cells[regime]
        exact = float(ra_cell["exact_same_cutoff_energy"])
        if not math.isclose(
            exact,
            finite(
                append_cell.get("exact_same_cutoff_energy"),
                label=f"Append {regime} exact energy",
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise Page7InputError(f"{regime}: RA/Append exact reference drifted")
        ra_points = sequence(ra_cell.get("points"), label=f"RA {regime} points")
        ra_marker = _effective_plateau(ra_points, label=f"RA {regime}")
        if ra_marker != ra_cell["effective_plateau"]:
            raise Page7InputError(f"{regime}: recomputed RA plateau drifted")
        selection = _common_accuracy_selection(
            ra_points,
            append_points,
            regime=regime,
            policy="display_horizon_equal_attainable_error_v2",
        )
        ra_terminal_error = float(mapping(ra_points[-1], label="RA terminal")["delta_e"])
        ra_terminal = cost("ra", ra_cell, 50, ra_terminal_error)
        if ra_terminal["costs"]["S_alg"] != ra_cell["terminal_work"]["S_alg"]:
            raise Page7InputError(f"{regime}: compiled terminal S_alg drifted")
        ra_common = cost(
            "ra",
            ra_cell,
            int(selection["ra_round"]),
            float(selection["ra_delta_e"]),
        )
        append_round = int(selection["append_round"])
        endpoint_key = f"round_{append_round}"
        if endpoint_key in append_cell["endpoints"]:
            endpoint = mapping(
                append_cell["endpoints"][endpoint_key],
                label=f"Append {regime} endpoint",
            )
            append_common = _normalize_cost_observation(
                endpoint,
                label=f"Append {regime} k={append_round}",
                expected_round=append_round,
                expected_error=float(selection["append_delta_e"]),
            )
        else:
            append_common = cost(
                "append",
                append_cell,
                append_round,
                float(selection["append_delta_e"]),
            )
        common = {
            **selection,
            "ra": ra_common,
            "append": append_common,
        }
        cells.append(
            {
                "regime_id": regime,
                "display_name": REGIME_LABELS[regime],
                "nph": NPH_BY_REGIME[regime],
                "status": "complete",
                "append": {
                    "execution_id": append_cell["execution_id"],
                    "exact_same_cutoff_energy": exact,
                    "points": copy.deepcopy(append_points),
                    "effective_plateau": append_marker,
                    "terminal": append_terminal,
                    "display_terminal_round": append_terminal_round,
                    "source": copy.deepcopy(append_cell.get("source", {})),
                },
                "ra": {
                    **copy.deepcopy(ra_cell),
                    "terminal": ra_terminal,
                },
                "common_accuracy": common,
            }
        )
    completed = [cell["regime_id"] for cell in cells if cell["status"] == "complete"]
    pending = [cell["regime_id"] for cell in cells if cell["status"] == "pending"]
    package_bindings: dict[str, Any] = {}
    for regime in REGIME_ORDER:
        if regime not in ra_cells:
            continue
        binding = copy.deepcopy(
            mapping(
                mapping(ra_cells[regime]["source"], label="RA source").get(
                    "package_manifest"
                ),
                label="RA package manifest",
            )
        )
        package_id = str(binding.get("package_id"))
        if package_id in package_bindings and package_bindings[package_id] != binding:
            raise Page7InputError(f"{regime}: RA package binding drifted")
        package_bindings[package_id] = binding
    adapter = digested(
        {
            "schema": ADAPTER_SCHEMA,
            "status": (
                "passed_complete"
                if not pending
                else ("passed_partial" if completed else "passed_pending")
            ),
            "classification": "supplemental_diagnostic_not_adopted_evidence",
            "paper_evidence_adopted": False,
            "regime_order": list(REGIME_ORDER),
            "completed_regimes": completed,
            "pending_regimes": pending,
            "append_adapter": {
                **append["file_binding"],
                "canonical_sha256": append["sha256"],
            },
            "ra_packages": package_bindings,
            "same_cutoff_reference": copy.deepcopy(append["same_cutoff_reference"]),
            "route_description": ROUTE_DESCRIPTION,
            "display_rounds_by_regime": {
                regime: {
                    "minimum": 0,
                    "maximum": APPEND_TERMINAL_ROUND_BY_REGIME[regime],
                }
                for regime in REGIME_ORDER
            },
            "ra_rounds": {"minimum": 0, "maximum": 50},
            "append_rounds_by_regime": {
                regime: {
                    "minimum": 0,
                    "maximum": APPEND_TERMINAL_ROUND_BY_REGIME[regime],
                }
                for regime in REGIME_ORDER
            },
            "marker_policy": "paper_i_effective_plateau_v1",
            "cost_policy": {
                "tuple_fields": list(COST_FIELDS),
                "terminal": {
                    "ra_round": 50,
                    "append_round_by_regime": copy.deepcopy(
                        APPEND_TERMINAL_ROUND_BY_REGIME
                    ),
                },
                "matched": "display_horizon_equal_attainable_error_v2",
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "optimization_level": 0,
                "seed_transpiler": 7,
                "reference_state_included": True,
            },
            "limitations": [LIMITATION],
            "cells": cells,
        }
    )
    return _write_monotone_adapter(output, adapter)


def validate_adapter(path: Path) -> dict[str, Any]:
    adapter = load_json(path, label="page-7 adapter")
    canonical = verify_self_digest(adapter, label="page-7 adapter")
    append_binding = mapping(
        adapter.get("append_adapter"), label="page-7 Append adapter binding"
    )
    append_source_path = Path(
        str(append_binding.get("path", ""))
    ).expanduser().resolve()
    append_source = validate_append_adapter(append_source_path)
    if (
        append_binding.get("canonical_sha256") != append_source["sha256"]
        or append_binding.get("sha256")
        != append_source["file_binding"]["sha256"]
        or append_binding.get("size_bytes")
        != append_source["file_binding"]["size_bytes"]
        or Path(str(append_binding.get("path", ""))).expanduser().resolve()
        != Path(append_source["file_binding"]["path"])
        or adapter.get("same_cutoff_reference")
        != append_source.get("same_cutoff_reference")
    ):
        raise Page7InputError("page-7 Append adapter binding drifted")
    append_source_cells = {
        str(mapping(cell, label="source Append cell").get("regime_id")): mapping(
            cell, label="source Append cell"
        )
        for cell in sequence(append_source.get("cells"), label="source Append cells")
    }
    completed = tuple(adapter.get("completed_regimes", ()))
    pending = tuple(adapter.get("pending_regimes", ()))
    if (
        adapter.get("schema") != ADAPTER_SCHEMA
        or adapter.get("paper_evidence_adopted") is not False
        or tuple(adapter.get("regime_order", ())) != REGIME_ORDER
        or set(completed).union(pending) != set(REGIME_ORDER)
        or set(completed).intersection(pending)
        or adapter.get("status")
        != (
            "passed_complete"
            if not pending
            else ("passed_partial" if completed else "passed_pending")
        )
    ):
        raise Page7InputError("page-7 adapter identity drifted")
    expected_rounds_by_regime = {
        regime: {
            "minimum": 0,
            "maximum": APPEND_TERMINAL_ROUND_BY_REGIME[regime],
        }
        for regime in REGIME_ORDER
    }
    expected_cost_policy = {
        "tuple_fields": list(COST_FIELDS),
        "terminal": {
            "ra_round": 50,
            "append_round_by_regime": copy.deepcopy(
                APPEND_TERMINAL_ROUND_BY_REGIME
            ),
        },
        "matched": "display_horizon_equal_attainable_error_v2",
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "optimization_level": 0,
        "seed_transpiler": 7,
        "reference_state_included": True,
    }
    if (
        adapter.get("display_rounds_by_regime") != expected_rounds_by_regime
        or adapter.get("append_rounds_by_regime") != expected_rounds_by_regime
        or adapter.get("ra_rounds") != {"minimum": 0, "maximum": 50}
        or adapter.get("marker_policy") != "paper_i_effective_plateau_v1"
        or adapter.get("route_description") != ROUTE_DESCRIPTION
        or adapter.get("limitations") != [LIMITATION]
        or canonical_json_bytes(adapter.get("cost_policy"))
        != canonical_json_bytes(expected_cost_policy)
    ):
        raise Page7InputError("page-7 reporting policy drifted")
    raw_cells = sequence(adapter.get("cells"), label="page-7 cells")
    cells = {
        str(mapping(cell, label="page-7 cell").get("regime_id")): mapping(
            cell, label="page-7 cell"
        )
        for cell in raw_cells
    }
    if len(raw_cells) != 6 or set(cells) != set(REGIME_ORDER):
        raise Page7InputError("page-7 adapter regime closure drifted")
    for regime in REGIME_ORDER:
        cell = cells[regime]
        expected_status = "complete" if regime in completed else "pending"
        append_terminal_round = APPEND_TERMINAL_ROUND_BY_REGIME[regime]
        append = mapping(cell.get("append"), label=f"{regime} Append")
        append_points = sequence(
            append.get("points"), label=f"{regime} Append points"
        )
        source_append = append_source_cells[regime]
        source_points = sequence(
            source_append.get("points"), label=f"{regime} source Append points"
        )
        source_terminal = mapping(
            mapping(
                source_append.get("endpoints"),
                label=f"{regime} source Append endpoints",
            ).get(f"round_{append_terminal_round}"),
            label=f"{regime} source Append terminal",
        )
        if (
            cell.get("status") != expected_status
            or cell.get("nph") != NPH_BY_REGIME[regime]
            or append.get("display_terminal_round") != append_terminal_round
            or len(append_points) != append_terminal_round + 1
            or [mapping(point, label="Append point").get("round") for point in append_points]
            != list(range(append_terminal_round + 1))
            or canonical_json_bytes(append_points)
            != canonical_json_bytes(source_points[: append_terminal_round + 1])
            or append.get("execution_id") != source_append.get("execution_id")
            or append.get("exact_same_cutoff_energy")
            != source_append.get("exact_same_cutoff_energy")
            or canonical_json_bytes(append.get("source"))
            != canonical_json_bytes(source_append.get("source"))
        ):
            raise Page7InputError(f"{regime}: page-7 cell drifted")
        if append.get("effective_plateau") != _effective_plateau(
            append_points, label=f"{regime} Append"
        ):
            raise Page7InputError(f"{regime}: Append marker drifted")
        append_terminal = mapping(
            append.get("terminal"), label=f"{regime} Append terminal"
        )
        if canonical_json_bytes(append_terminal) != canonical_json_bytes(
            source_terminal
        ):
            raise Page7InputError(f"{regime}: Append terminal source drifted")
        _normalize_cost_observation(
            append_terminal,
            label=f"{regime} Append terminal",
            expected_round=append_terminal_round,
            expected_error=finite(
                mapping(append_points[-1], label="Append terminal point").get(
                    "delta_e"
                ),
                label="Append terminal error",
            ),
        )
        if expected_status == "pending":
            if cell.get("ra") is not None or cell.get("common_accuracy") is not None:
                raise Page7InputError(f"{regime}: pending RA values were inferred")
            continue
        ra = mapping(cell.get("ra"), label=f"{regime} RA")
        ra_points = sequence(ra.get("points"), label=f"{regime} RA points")
        if (
            len(ra_points) != 51
            or [mapping(point, label="RA point").get("round") for point in ra_points]
            != list(range(51))
        ):
            raise Page7InputError(f"{regime}: RA trajectory is not rounds 0..50")
        if ra.get("effective_plateau") != _effective_plateau(
            ra_points, label=f"{regime} RA"
        ):
            raise Page7InputError(f"{regime}: RA marker drifted")
        _normalize_cost_observation(
            ra.get("terminal"),
            label=f"{regime} RA terminal",
            expected_round=50,
            expected_error=finite(
                mapping(ra_points[-1], label="RA terminal point").get("delta_e"),
                label="RA terminal error",
            ),
        )
        common = mapping(
            cell.get("common_accuracy"), label=f"{regime} common accuracy"
        )
        selection = _common_accuracy_selection(
            ra_points,
            append_points,
            regime=regime,
            policy="display_horizon_equal_attainable_error_v2",
        )
        if (
            set(common) != set(selection).union({"ra", "append"})
            or any(common.get(key) != value for key, value in selection.items())
        ):
            raise Page7InputError(f"{regime}: common-accuracy policy drifted")
        ra_by_round = {
            int(mapping(point, label="RA point")["round"]): mapping(
                point, label="RA point"
            )
            for point in ra_points
        }
        append_by_round = {
            int(mapping(point, label="Append point")["round"]): mapping(
                point, label="Append point"
            )
            for point in append_points
        }
        _normalize_cost_observation(
            common.get("ra"),
            label=f"{regime} common RA",
            expected_round=integer(common.get("ra_round"), label="common RA round"),
            expected_error=finite(
                ra_by_round[int(common["ra_round"])].get("delta_e"),
                label="common RA error",
            ),
        )
        _normalize_cost_observation(
            common.get("append"),
            label=f"{regime} common Append",
            expected_round=integer(
                common.get("append_round"), label="common Append round"
            ),
            expected_error=finite(
                append_by_round[int(common["append_round"])].get("delta_e"),
                label="common Append error",
            ),
        )
    return {
        **copy.deepcopy(adapter),
        "sha256": canonical,
        "file_binding": file_binding(path),
    }


def format_delta_e(value: float) -> str:
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def format_costs(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return r"$\text{pending}$"
    costs = {field: integer(value.get(field), label=field) for field in COST_FIELDS}
    mantissa, exponent = f"{costs['S_alg']:.1e}".split("e")
    return (
        f"$({costs['N2q']:,},{costs['D2q']:,},{costs['Dc']:,},"
        f"{costs['W1q']:,},{mantissa}\\mathrm{{e}}{int(exponent)})$"
    )


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def render_plot(adapter: Mapping[str, Any], *, png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter

    cells = {str(cell["regime_id"]): cell for cell in adapter["cells"]}
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.1,
            "axes.labelsize": 8.3,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.1, 4.05), constrained_layout=True)
    for index, regime in enumerate(REGIME_ORDER):
        ax = axes.flat[index]
        cell = cells[regime]
        append = cell["append"]
        append_points = append["points"]
        ax.plot(
            [point["round"] for point in append_points],
            [point["delta_e"] for point in append_points],
            color="#4C78A8",
            linewidth=1.55,
        )
        append_marker = append["effective_plateau"]
        ax.scatter(
            [append_marker["round"]],
            [append_marker["delta_e"]],
            marker="o",
            color="#4C78A8",
            s=27,
            zorder=5,
        )
        if cell["status"] == "complete":
            ra = cell["ra"]
            ra_points = ra["points"]
            ax.plot(
                [point["round"] for point in ra_points],
                [point["delta_e"] for point in ra_points],
                color="#E45756",
                linewidth=1.65,
            )
            ra_marker = ra["effective_plateau"]
            ax.scatter(
                [ra_marker["round"]],
                [ra_marker["delta_e"]],
                marker="D",
                color="#E45756",
                s=27,
                zorder=5,
            )
        else:
            ax.text(
                0.98,
                0.96,
                "RA pending\nvalidated completion",
                transform=ax.transAxes,
                ha="right",
                va="top",
                color="#A33A35",
                fontsize=7.2,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#D9D9D9",
                    "alpha": 0.88,
                    "boxstyle": "round,pad=0.24",
                },
            )
        ax.set_title(REGIME_LABELS[regime])
        ax.set_xlim(0, int(append["display_terminal_round"]))
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", linewidth=0.45, alpha=0.34)
        ax.grid(True, which="minor", linewidth=0.25, alpha=0.14)
        if index // 3 == 1:
            ax.set_xlabel("ADAPT iteration")
        if index % 3 == 0:
            ax.set_ylabel(r"Same-cutoff $\Delta E$")
    fig.suptitle(
        "Global-singleton RA plateau vs fresh Append-ADAPT singleton",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.legend(
        handles=(
            Line2D(
                [0],
                [0],
                color="#4C78A8",
                marker="o",
                label=(
                    "Fresh Append-ADAPT singleton "
                    "(k=0..50 weak Holstein; k=0..70 strong Holstein)"
                ),
            ),
            Line2D(
                [0],
                [0],
                color="#E45756",
                marker="D",
                label="Historical-mean global-singleton RA plateau (k=0..50)",
            ),
        ),
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8.0,
        title="Marker denotes first effective plateau prefix",
        title_fontsize=7.4,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_page_tex(
    adapter: Mapping[str, Any], *, plot_pdf: Path, tex_path: Path
) -> None:
    endpoint_rows: list[str] = []
    matched_rows: list[str] = []
    for cell in adapter["cells"]:
        regime = str(cell["regime_id"])
        append_terminal = cell["append"]["terminal"]
        append_terminal_round = int(cell["append"]["display_terminal_round"])
        if cell["status"] == "complete":
            ra_terminal = cell["ra"]["terminal"]
            endpoint_rows.append(
                " & ".join(
                    (
                        str(cell["display_name"]),
                        format_delta_e(float(ra_terminal["delta_e"])),
                        format_costs(ra_terminal["costs"]),
                        str(append_terminal_round),
                        format_delta_e(float(append_terminal["delta_e"])),
                        format_costs(append_terminal["costs"]),
                    )
                )
                + r" \\"
            )
            common = cell["common_accuracy"]
            matched_rows.append(
                " & ".join(
                    (
                        REGIME_ABBREVIATIONS[regime],
                        format_delta_e(float(common["target_delta_e"])),
                        str(common["ra"]["round"]),
                        format_costs(common["ra"]["costs"]),
                        str(common["append"]["round"]),
                        format_costs(common["append"]["costs"]),
                    )
                )
                + r" \\"
            )
        else:
            endpoint_rows.append(
                " & ".join(
                    (
                        str(cell["display_name"]),
                        r"$\text{pending}$",
                        r"$\text{pending}$",
                        str(append_terminal_round),
                        format_delta_e(float(append_terminal["delta_e"])),
                        format_costs(append_terminal["costs"]),
                    )
                )
                + r" \\"
            )
            matched_rows.append(
                " & ".join(
                    (
                        REGIME_ABBREVIATIONS[regime],
                        r"$\text{pending}$",
                        "--",
                        r"$\text{pending}$",
                        "--",
                        r"$\text{pending}$",
                    )
                )
                + r" \\"
            )
    plot_argument = latex_escape(plot_pdf.resolve().as_posix())
    route = latex_escape(str(adapter["route_description"]))
    tex = rf"""\documentclass[10pt,letterpaper]{{article}}
\usepackage[landscape,margin=0.16in]{{geometry}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage[T1]{{fontenc}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\begin{{document}}
\begin{{center}}
\includegraphics[width=0.96\textwidth,height=3.42in,keepaspectratio]{{{plot_argument}}}
\vspace{{0.25em}}

\scriptsize
\setlength{{\tabcolsep}}{{3.1pt}}
\begin{{tabular}}{{@{{}}lrrrrr@{{}}}}
\toprule
Regime & $\Delta E_{{50}}^{{\rm RA}}$ & $C_{{50}}^{{\rm RA}}$ &
$k_A$ & $\Delta E_{{k_A}}^{{\rm Append}}$ &
$C_{{k_A}}^{{\rm Append}}$ \\
\midrule
{chr(10).join(endpoint_rows)}
\bottomrule
\end{{tabular}}
\vspace{{0.10em}}

{{\scriptsize\bfseries Equal-attainable-error costs over each complete trajectory}}
\vspace{{-0.12em}}

\fontsize{{6.4}}{{6.8}}\selectfont
\setlength{{\tabcolsep}}{{1.9pt}}
\renewcommand{{\arraystretch}}{{0.76}}
\begin{{tabular}}{{@{{}}ccrrrr@{{}}}}
\toprule
Reg. & $\Delta E_\cap$ & $k_\cap^{{\rm RA}}$ & $C_\cap^{{\rm RA}}$ &
$k_\cap^{{\rm Append}}$ & $C_\cap^{{\rm Append}}$ \\
\midrule
{chr(10).join(matched_rows)}
\bottomrule
\end{{tabular}}
\end{{center}}
\vspace{{-0.48em}}
\tiny
$C=(N_{{2q}},D_{{2q}},D_c,W_{{1q}},S_{{\rm alg}})$; Qiskit fields use the
source-locked Table-I compiler (optimization level 0, seed 7, reference state
included). $\Delta E_\cap$ is the larger of the two displayed-trajectory minima;
each method is costed at its earliest crossing over the displayed trajectory.
$k_A=50$ for weak-Holstein ($n_{{\rm ph}}=3$) and $k_A=70$ for
strong-Holstein ($n_{{\rm ph}}=7$). {route}
\end{{document}}
"""
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(tex, encoding="utf-8")


def build_assets(
    adapter: Mapping[str, Any], *, asset_dir: Path, asset_stem: str
) -> dict[str, Path]:
    if not ASSET_STEM_RE.fullmatch(asset_stem) or asset_stem in {".", ".."}:
        raise Page7InputError("asset_stem must be a safe filename component")
    assets = {
        "plot_png": asset_dir / f"{asset_stem}_plot.png",
        "plot_pdf": asset_dir / f"{asset_stem}_plot.pdf",
        "page_tex": asset_dir / f"{asset_stem}.tex",
        "page_pdf": asset_dir / f"{asset_stem}.pdf",
    }
    render_plot(adapter, png_path=assets["plot_png"], pdf_path=assets["plot_pdf"])
    write_page_tex(adapter, plot_pdf=assets["plot_pdf"], tex_path=assets["page_tex"])
    legacy_page._compile_page(assets["page_tex"], assets["page_pdf"])
    return assets


def _load_bound_legacy_adapter(value: Any) -> dict[str, Any]:
    """Authenticate the one supported all-R70 page-7 predecessor."""

    binding = mapping(value, label="legacy page-7 adapter binding")
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    legacy = load_json(path, label="legacy page-7 adapter")
    canonical = verify_self_digest(legacy, label="legacy page-7 adapter")
    observed_file = file_binding(path)
    if (
        binding.get("canonical_sha256") != canonical
        or binding.get("sha256") != observed_file["sha256"]
        or binding.get("size_bytes") != observed_file["size_bytes"]
        or Path(str(binding.get("path", ""))).expanduser().resolve()
        != Path(observed_file["path"])
    ):
        raise Page7InputError("legacy page-7 adapter byte binding drifted")
    if any(
        binding.get(field) != expected
        for field, expected in LEGACY_PAGE7_ADAPTER_BINDING.items()
    ):
        raise Page7InputError("legacy page-7 adapter is not the pinned predecessor")
    completed = tuple(legacy.get("completed_regimes", ()))
    pending = tuple(legacy.get("pending_regimes", ()))
    if (
        legacy.get("schema") != LEGACY_ADAPTER_SCHEMA
        or tuple(legacy.get("regime_order", ())) != REGIME_ORDER
        or set(completed).union(pending) != set(REGIME_ORDER)
        or set(completed).intersection(pending)
        or legacy.get("display_rounds") != {"minimum": 0, "maximum": 70}
        or legacy.get("ra_rounds") != {"minimum": 0, "maximum": 50}
        or legacy.get("append_rounds") != {"minimum": 0, "maximum": 70}
        or legacy.get("marker_policy") != "paper_i_effective_plateau_v1"
        or legacy.get("route_description") != ROUTE_DESCRIPTION
        or legacy.get("limitations") != [LEGACY_LIMITATION]
    ):
        raise Page7InputError("legacy page-7 adapter identity drifted")
    append_binding = mapping(
        legacy.get("append_adapter"), label="legacy Append adapter binding"
    )
    append_source = validate_append_adapter(
        Path(str(append_binding.get("path", ""))).expanduser().resolve()
    )
    if (
        append_binding.get("canonical_sha256") != append_source["sha256"]
        or append_binding.get("sha256")
        != append_source["file_binding"]["sha256"]
        or append_binding.get("size_bytes")
        != append_source["file_binding"]["size_bytes"]
    ):
        raise Page7InputError("legacy Append adapter binding drifted")
    source_cells = {
        str(mapping(cell, label="legacy source Append cell").get("regime_id")): mapping(
            cell, label="legacy source Append cell"
        )
        for cell in sequence(append_source.get("cells"), label="legacy source cells")
    }
    raw_cells = sequence(legacy.get("cells"), label="legacy page-7 cells")
    cells = {
        str(mapping(cell, label="legacy page-7 cell").get("regime_id")): mapping(
            cell, label="legacy page-7 cell"
        )
        for cell in raw_cells
    }
    if len(raw_cells) != 6 or set(cells) != set(REGIME_ORDER):
        raise Page7InputError("legacy page-7 adapter regime closure drifted")
    for regime in REGIME_ORDER:
        cell = cells[regime]
        append = mapping(cell.get("append"), label=f"legacy {regime} Append")
        points = sequence(append.get("points"), label=f"legacy {regime} points")
        source = source_cells[regime]
        if (
            len(points) != 71
            or [mapping(point, label="legacy Append point").get("round") for point in points]
            != list(range(71))
            or canonical_json_bytes(points)
            != canonical_json_bytes(source.get("points"))
            or append.get("execution_id") != source.get("execution_id")
            or append.get("exact_same_cutoff_energy")
            != source.get("exact_same_cutoff_energy")
            or canonical_json_bytes(append.get("source"))
            != canonical_json_bytes(source.get("source"))
            or canonical_json_bytes(append.get("terminal"))
            != canonical_json_bytes(
                mapping(source.get("endpoints"), label="legacy endpoints").get(
                    "round_70"
                )
            )
        ):
            raise Page7InputError(f"legacy {regime} Append cell drifted")
        expected_status = "complete" if regime in completed else "pending"
        if cell.get("status") != expected_status:
            raise Page7InputError(f"legacy {regime} status drifted")
        if expected_status == "pending":
            if cell.get("ra") is not None or cell.get("common_accuracy") is not None:
                raise Page7InputError(f"legacy {regime} inferred pending RA")
            continue
        ra = mapping(cell.get("ra"), label=f"legacy {regime} RA")
        ra_points = sequence(ra.get("points"), label=f"legacy {regime} RA points")
        if (
            len(ra_points) != 51
            or [mapping(point, label="legacy RA point").get("round") for point in ra_points]
            != list(range(51))
        ):
            raise Page7InputError(f"legacy {regime} RA trajectory drifted")
    return {
        **copy.deepcopy(legacy),
        "sha256": canonical,
        "file_binding": observed_file,
    }


def _validate_legacy_policy_migration(
    legacy: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Permit only the requested all-R70 to weak-50/strong-70 revision."""

    if (
        legacy.get("completed_regimes") != current.get("completed_regimes")
        or legacy.get("pending_regimes") != current.get("pending_regimes")
        or legacy.get("append_adapter") != current.get("append_adapter")
        or legacy.get("ra_packages") != current.get("ra_packages")
        or legacy.get("same_cutoff_reference")
        != current.get("same_cutoff_reference")
        or legacy.get("route_description") != current.get("route_description")
    ):
        raise Page7InputError("legacy page-7 migration authority drifted")
    old_cells = {
        str(mapping(cell, label="legacy migration cell").get("regime_id")): mapping(
            cell, label="legacy migration cell"
        )
        for cell in sequence(legacy.get("cells"), label="legacy migration cells")
    }
    new_cells = {
        str(mapping(cell, label="mixed migration cell").get("regime_id")): mapping(
            cell, label="mixed migration cell"
        )
        for cell in sequence(current.get("cells"), label="mixed migration cells")
    }
    for regime in REGIME_ORDER:
        old = old_cells[regime]
        new = new_cells[regime]
        for field in ("regime_id", "display_name", "nph", "status", "pending_reason"):
            if old.get(field) != new.get(field):
                raise Page7InputError(f"{regime}: migration changed {field}")
        if canonical_json_bytes(old.get("ra")) != canonical_json_bytes(new.get("ra")):
            raise Page7InputError(f"{regime}: migration changed RA evidence")
        old_append = mapping(old.get("append"), label=f"legacy {regime} Append")
        new_append = mapping(new.get("append"), label=f"mixed {regime} Append")
        for field in (
            "execution_id",
            "exact_same_cutoff_energy",
            "source",
        ):
            if canonical_json_bytes(old_append.get(field)) != canonical_json_bytes(
                new_append.get(field)
            ):
                raise Page7InputError(f"{regime}: migration changed Append {field}")
        horizon = APPEND_TERMINAL_ROUND_BY_REGIME[regime]
        if canonical_json_bytes(new_append.get("points")) != canonical_json_bytes(
            sequence(old_append.get("points"), label="legacy Append points")[
                : horizon + 1
            ]
        ):
            raise Page7InputError(f"{regime}: migration changed Append prefix")
        if old.get("status") == "pending":
            if new.get("common_accuracy") is not None:
                raise Page7InputError(f"{regime}: migration inferred common accuracy")
            continue
        old_common = mapping(
            old.get("common_accuracy"), label=f"legacy {regime} common accuracy"
        )
        new_common = mapping(
            new.get("common_accuracy"), label=f"mixed {regime} common accuracy"
        )
        if (
            old_common.get("ra_round") != new_common.get("ra_round")
            or canonical_json_bytes(old_common.get("ra"))
            != canonical_json_bytes(new_common.get("ra"))
        ):
            raise Page7InputError(
                f"{regime}: migration lacks an authenticated RA crossing cost"
            )


def _report_cell_projection(cell: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "regime_id": cell["regime_id"],
        "status": cell["status"],
        "cell_sha256": _cell_digest(cell),
        "append_terminal_round": cell["append"]["display_terminal_round"],
        "append_terminal": copy.deepcopy(cell["append"]["terminal"]),
    }
    if cell["status"] == "complete":
        result.update(
            {
                "ra_round_50": copy.deepcopy(cell["ra"]["terminal"]),
                "ra_effective_plateau": copy.deepcopy(
                    cell["ra"]["effective_plateau"]
                ),
                "append_effective_plateau": copy.deepcopy(
                    cell["append"]["effective_plateau"]
                ),
                "common_accuracy": copy.deepcopy(cell["common_accuracy"]),
                "ra_source": copy.deepcopy(cell["ra"]["source"]),
            }
        )
    else:
        result["pending_reason"] = cell["pending_reason"]
    return result


def update_page7(
    *,
    target_pdf: Path,
    target_provenance: Path,
    adapter_path: Path,
    asset_dir: Path,
    asset_stem: str,
) -> dict[str, Any]:
    adapter = validate_adapter(adapter_path)
    provenance = load_json(target_provenance, label="target provenance")
    outputs = mapping(provenance.get("outputs"), label="target outputs")
    pdf_binding = mapping(
        outputs.get("partial_progress_pdf"), label="target PDF binding"
    )
    if (
        not target_pdf.is_file()
        or target_pdf.is_symlink()
        or target_provenance.is_symlink()
        or pdf_binding.get("sha256") != sha256_file(target_pdf)
        or pdf_binding.get("size_bytes") != target_pdf.stat().st_size
    ):
        raise Page7InputError("target PDF/provenance byte binding drifted")
    layout = mapping(provenance.get("layout"), label="target layout")
    before_hashes = legacy_page._page_content_hashes(target_pdf)
    page_count = len(before_hashes)
    if (
        layout.get("page_count") != page_count
        or layout.get("page_6") != EXPECTED_BASE_PAGE_6
        or page_count not in {6, 7}
    ):
        raise Page7InputError("target layout is not the supported six/page-7 report")

    replace_existing = page_count == 7
    legacy_policy_migration = False
    if replace_existing:
        existing_page_id = layout.get("page_7")
        if existing_page_id == LEGACY_PAGE_ID:
            if provenance.get(REPORT_KEY) is not None:
                raise Page7InputError(
                    "legacy page 7 already carries mixed-horizon provenance"
                )
            report = mapping(
                provenance.get(LEGACY_REPORT_KEY),
                label="legacy page-7 report",
            )
            if (
                report.get("schema") != LEGACY_PAGE_ID
                or report.get("page_id") != LEGACY_PAGE_ID
                or report.get("paper_evidence_adopted") is not False
            ):
                raise Page7InputError("legacy page-7 report identity drifted")
            legacy_adapter = _load_bound_legacy_adapter(report.get("adapter"))
            if (
                report.get("completed_regimes")
                != legacy_adapter.get("completed_regimes")
                or report.get("pending_regimes")
                != legacy_adapter.get("pending_regimes")
            ):
                raise Page7InputError("legacy page-7 report completion drifted")
            _validate_legacy_policy_migration(legacy_adapter, adapter)
            legacy_policy_migration = True
        elif existing_page_id == PAGE_ID:
            report = mapping(
                provenance.get(REPORT_KEY), label="existing page-7 report"
            )
        else:
            raise Page7InputError("existing page 7 has an unsupported identity")
        if not legacy_policy_migration:
            old_adapter = mapping(
                report.get("adapter"), label="existing adapter binding"
            )
            if old_adapter.get("canonical_sha256") == adapter["sha256"]:
                return {
                    "status": "already_current",
                    "output_pdf": str(target_pdf),
                    "output_provenance": str(target_provenance),
                    "sha256": sha256_file(target_pdf),
                    "pages": 7,
                    "preserved_pages_1_6": True,
                }
            old_completed = set(report.get("completed_regimes", ()))
            new_completed = set(adapter["completed_regimes"])
            if not old_completed < new_completed:
                raise Page7InputError(
                    "page-7 replacement must be a strict completion superset"
                )
            old_digests = mapping(
                report.get("completed_cell_sha256"),
                label="existing completed-cell digests",
            )
            new_cells = {
                str(cell["regime_id"]): mapping(cell, label="new page-7 cell")
                for cell in adapter["cells"]
            }
            if any(
                old_digests.get(regime) != _cell_digest(new_cells[regime])
                for regime in old_completed
            ):
                raise Page7InputError("a previously completed page-7 cell drifted")
            old_append_digests = mapping(
                report.get("append_cell_sha256"),
                label="existing Append-cell digests",
            )
            if any(
                old_append_digests.get(regime)
                != hashlib.sha256(
                    canonical_json_bytes(new_cells[regime].get("append"))
                ).hexdigest()
                for regime in REGIME_ORDER
            ):
                raise Page7InputError("an existing page-7 Append cell drifted")
    elif (
        layout.get("page_7") is not None
        or provenance.get(REPORT_KEY) is not None
        or provenance.get(LEGACY_REPORT_KEY) is not None
    ):
        raise Page7InputError("six-page target carries stale page-7 provenance")

    assets = build_assets(adapter, asset_dir=asset_dir, asset_stem=asset_stem)
    from pypdf import PdfReader, PdfWriter

    page_reader = PdfReader(str(assets["page_pdf"]), strict=False)
    if len(page_reader.pages) != 1:
        raise Page7InputError("page-7 asset is not exactly one page")
    existing_pages = PdfReader(str(target_pdf), strict=False).pages
    writer = PdfWriter()
    for page in existing_pages[:6]:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])
    temporary_pdf = target_pdf.with_name(f".{target_pdf.name}.page7.tmp")
    try:
        with temporary_pdf.open("wb") as stream:
            writer.write(stream)
        after_hashes = legacy_page._page_content_hashes(temporary_pdf)
        if len(after_hashes) != 7 or after_hashes[:6] != before_hashes[:6]:
            raise Page7InputError("page-7 update altered an existing report page")
        new_pdf_binding = file_binding(temporary_pdf)
        new_pdf_binding["path"] = str(target_pdf.resolve())
        updated = copy.deepcopy(provenance)
        updated["layout"]["page_count"] = 7
        updated["layout"]["page_7"] = PAGE_ID
        updated["outputs"]["partial_progress_pdf"] = new_pdf_binding
        output_keys: list[str] = []
        for output_key, asset_key in (
            ("historical_mean_global_singleton_full6_plot_png", "plot_png"),
            ("historical_mean_global_singleton_full6_plot_pdf", "plot_pdf"),
            ("historical_mean_global_singleton_full6_page_tex", "page_tex"),
            ("historical_mean_global_singleton_full6_page_pdf", "page_pdf"),
        ):
            updated["outputs"][output_key] = file_binding(assets[asset_key])
            output_keys.append(output_key)
        completed_cell_sha256 = {
            str(cell["regime_id"]): _cell_digest(cell)
            for cell in adapter["cells"]
            if cell["status"] == "complete"
        }
        append_cell_sha256 = {
            str(cell["regime_id"]): hashlib.sha256(
                canonical_json_bytes(cell["append"])
            ).hexdigest()
            for cell in adapter["cells"]
        }
        updated[REPORT_KEY] = {
            "schema": PAGE_ID,
            "classification": "supplemental_diagnostic_not_adopted_evidence",
            "paper_evidence_adopted": False,
            "page_id": PAGE_ID,
            "adapter": {
                **adapter["file_binding"],
                "canonical_sha256": adapter["sha256"],
            },
            "completed_regimes": copy.deepcopy(adapter["completed_regimes"]),
            "pending_regimes": copy.deepcopy(adapter["pending_regimes"]),
            "completed_cell_sha256": completed_cell_sha256,
            "append_cell_sha256": append_cell_sha256,
            "route_description": adapter["route_description"],
            "marker_policy": adapter["marker_policy"],
            "cost_policy": copy.deepcopy(adapter["cost_policy"]),
            "limitations": copy.deepcopy(adapter["limitations"]),
            "cells": [
                _report_cell_projection(mapping(cell, label="adapter cell"))
                for cell in adapter["cells"]
            ],
            "structural_validation": {
                "pages": 7,
                "preserved_pages_1_6_content_sha256": before_hashes[:6],
                "prior_page_7_content_sha256": (
                    before_hashes[6] if replace_existing else None
                ),
                "new_page_7_content_sha256": after_hashes[6],
            },
            "outputs": {
                key: copy.deepcopy(updated["outputs"][key]) for key in output_keys
            },
        }
        if legacy_policy_migration:
            updated[REPORT_KEY]["reporting_policy_migration"] = {
                "schema": "paper_i_page7_append_horizon_policy_migration_v1",
                "from_page_id": LEGACY_PAGE_ID,
                "to_page_id": PAGE_ID,
                "from_adapter": copy.deepcopy(report["adapter"]),
                "to_adapter": copy.deepcopy(updated[REPORT_KEY]["adapter"]),
                "scientific_ra_evidence_changed": False,
                "append_source_changed": False,
                "requested_display_change": {
                    "weak_holstein_append_terminal_round": 50,
                    "strong_holstein_append_terminal_round": 70,
                },
            }
        limitations = list(updated.get("limitations", ()))
        if legacy_policy_migration:
            limitations = [
                limitation
                for limitation in limitations
                if limitation != LEGACY_LIMITATION
            ]
        if LIMITATION not in limitations:
            limitations.append(LIMITATION)
        updated["limitations"] = limitations
        os.replace(temporary_pdf, target_pdf)
        legacy_page._atomic_write_json(target_provenance, updated)
    finally:
        temporary_pdf.unlink(missing_ok=True)
    return {
        "status": (
            "migrated_page_7_append_horizon_policy"
            if legacy_policy_migration
            else ("replaced_page_7" if replace_existing else "appended_page_7")
        ),
        "output_pdf": str(target_pdf),
        "output_provenance": str(target_provenance),
        "sha256": sha256_file(target_pdf),
        "pages": 7,
        "completed_regimes": copy.deepcopy(adapter["completed_regimes"]),
        "pending_regimes": copy.deepcopy(adapter["pending_regimes"]),
        "preserved_pages_1_6": True,
    }


def _regime_path_args(
    values: Sequence[str], *, option: str, noun: str
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for raw in values:
        regime, separator, path_text = raw.partition("=")
        if not separator or regime not in REGIME_ORDER or not path_text:
            raise Page7InputError(
                f"{option} must be REGIME=/path/to/{noun}"
            )
        if regime in paths:
            raise Page7InputError(f"duplicate {option} regime: {regime}")
        paths[regime] = Path(path_text).expanduser().resolve()
    return paths


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--append-adapter", type=Path, required=True)
    result.add_argument(
        "--ra-attempt",
        action="append",
        default=[],
        metavar="REGIME=ARCHIVE",
        help="Explicit completed attempt archive; repeat for every available cell.",
    )
    result.add_argument(
        "--ra-projection",
        action="append",
        default=[],
        metavar="REGIME=PROJECTION",
        help=(
            "Sealed compact RA projection made beside a remotely preserved "
            "archive; repeat for every projected cell."
        ),
    )
    result.add_argument("--adapter", type=Path, required=True)
    result.add_argument("--target-pdf", type=Path, required=True)
    result.add_argument("--target-provenance", type=Path, required=True)
    result.add_argument("--asset-dir", type=Path, required=True)
    result.add_argument("--asset-stem", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        attempts = _regime_path_args(
            args.ra_attempt,
            option="--ra-attempt",
            noun="completed.tar.gz",
        )
        projections = _regime_path_args(
            args.ra_projection,
            option="--ra-projection",
            noun="projection.json",
        )
        build_adapter(
            append_adapter_path=args.append_adapter.resolve(),
            ra_attempts=attempts,
            ra_projections=projections,
            output=args.adapter.resolve(),
        )
        result = update_page7(
            target_pdf=args.target_pdf.resolve(),
            target_provenance=args.target_provenance.resolve(),
            adapter_path=args.adapter.resolve(),
            asset_dir=args.asset_dir.resolve(),
            asset_stem=args.asset_stem,
        )
    except (OSError, Page7InputError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
