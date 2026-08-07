#!/usr/bin/env python3
"""Materialize a source-locked weak--weak plateau-threshold sensitivity.

The immutable page-8 v3 package is the sole executable predecessor.  The
source-value replay (1e-4) is materialized first.  Fan-out to 1e-5 and 1e-6 is
refused until the replay reproduces both the page-8 terminal error and the
complete accepted generator/insertion-position sequence.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from typing import Any, Mapping, Sequence


SWEEP_DIR = Path(__file__).resolve().parent
REPO_ROOT = SWEEP_DIR.parents[2]
SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc"
)
SOURCE_PACKAGE = REPO_ROOT / SOURCE_PACKAGE_RELATIVE
SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "37092457bf337ce14bcb472fdcdb1d34227363ada5765434db09da2bff770ec0"
)
SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "311b2327e2ad254156eac6fb7f5f2d6f91a391913fce93edeae99b2d52ba031b"
)
SOURCE_PROTOCOL_BUNDLE_FILE_SHA256 = (
    "1b472a3d82fbeae0ff681476fb9b01ced61e02bbd7ac5d0268eb0665cc91c7c6"
)
SOURCE_PROTOCOL_BUNDLE_CANONICAL_SHA256 = (
    "e5cb1974760df4ab041a8e3a6451310f164a5b76af2ce336d753830a1023d231"
)
SOURCE_LOCKS_FILE_SHA256 = (
    "26858f9752c2adeb814e9b5b266378fea57ea41a94db9d34cda2610857cf28c8"
)
SOURCE_LOCKS_CANONICAL_SHA256 = (
    "2f2853bd0cc0cf9abfdceff86314556b0ad12f85eeff13e78ee9147508042a39"
)
SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "20acd89a7b6747d3f93960fbf1c4a5e7c680679631fb09422026030ba2dc3be6"
)
SOURCE_ARCHIVE_FILE_SHA256 = (
    "bd94a87b632646e051bf99fe760639275de04f1e21cfb660fc5e8ef21f56d4bd"
)
SOURCE_ARCHIVE_MANIFEST_FILE_SHA256 = (
    "e1952eef464976c088a2aa84ca1aada1c6f0ec34cf5f089a2de7e33ce64c5ded"
)
SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256 = (
    "0363bc046caa8a5b8088fb09d04210590f1735716269e5d081cc3af7522f416c"
)
SOURCE_PROTOCOL_RELATIVE = Path(
    "protocols/phase3_on_plateau_r50__weak_weak__nph3__"
    "ra_singleton_plateau.json"
)
SOURCE_PROTOCOL_FILE_SHA256 = (
    "a2d0307fdc3dd681520b7903f579c003778e6529aea029deeca0c6a14f5533ba"
)
SOURCE_PROTOCOL_CANONICAL_SHA256 = (
    "2244babcd8ccaeeb27db380f04ca130ab26f3fc66b7d45c80806789b9bb73620"
)
SOURCE_RESULT_ARCHIVE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "retrieved_chtc_20260804_phase3_on_plateau_singleton_r50_v3/"
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau__"
    "cluster_9463747__proc_0.tar.gz"
)
SOURCE_RESULT_ARCHIVE_SHA256 = (
    "2a521d4a99ee7146c6decce98914e2996db81dbbba8ca6ada7d31f0362366998"
)
SOURCE_RESULT_MEMBER = "worker_outputs/artifacts/result.json"
SOURCE_RESULT_FILE_SHA256 = (
    "ad54d8cca79d2cf89537a92a08241d851f37d6f3b700bc7a28d3039e26cee911"
)

SOURCE_THRESHOLD = 1.0e-4
FANOUT_THRESHOLDS = (1.0e-5, 1.0e-6)
APPEND_TARGET_ABSOLUTE_ENERGY_ERROR = 9.416688540042628e-10
TARGET_HORIZON = 50
SOURCE_EXECUTION_ID = (
    "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau"
)
SOURCE_FILE_RELATIVE = Path("pipelines/static_adapt/sr_snake_route_profile.py")
SOURCE_FILE_BEFORE_SHA256 = (
    "8ef372d55a955dee10bed280fee399760f2fc67e36c9fda2b5114cf9a897216b"
)
SOURCE_DECLARATION = (
    b"INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD = "
    b"1.0e-4"
)
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_phase3_population_on_insertion_plateau_v1"
)
PLATEAU_COMPARISON = "marginal_to_prior_mean_strictly_below_v2"
PLATEAU_TRIGGER = (
    "immediately_preceding_marginal_over_prior_mean_"
    "accepted_post_full_refit_energy_decrease_v2"
)
PLATEAU_CALIBRATION = "source_locked_counterfactual_trigger_replay_v2"
ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "late_resource_weighting_v1__phase3_population_on_insertion_plateau_v1"
)
PACKAGE_SCHEMA_PREFIX = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_sweep_"
    "weak_weak_r50"
)
PLAN_PATH = SWEEP_DIR / "sensitivity_plan.json"
ANCHOR_COMPARISON_PATH = SWEEP_DIR / "anchor_comparison.json"
FANOUT_MANIFEST_PATH = SWEEP_DIR / "fanout_manifest.json"
LOCAL_ANCHOR_FAILURE_PATH = SWEEP_DIR / "local_anchor_failure.json"
REMOTE_IMAGE = {
    "path": "chtc/phase3_optuna/image.sif",
    "sha256": "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f",
    "byte_verification_required_before_submit": True,
}
EXPECTED_DERIVATIVE_HASHES = {
    1.0e-4: {
        "source_member_sha256": SOURCE_FILE_BEFORE_SHA256,
        "implementation_inventory_sha256": SOURCE_IMPLEMENTATION_INVENTORY_SHA256,
        "source_locks_canonical_sha256": SOURCE_LOCKS_CANONICAL_SHA256,
        "source_locks_file_sha256": SOURCE_LOCKS_FILE_SHA256,
        "parent_route_contract_sha256": (
            "aa669d7f0c3621d9ddf7f8595f96333c56b536c8fc79547607e76d8d91d4b6ff"
        ),
        "route_contract_sha256": (
            "ac868db4dab4f8446ff06e768c5ea77512ef70764efd5699621bd95ad341599d"
        ),
    },
    1.0e-5: {
        "source_member_sha256": (
            "d39471909003913dfddaf7519e8a1690082631f393b6090e2b11098a74ba812a"
        ),
        "implementation_inventory_sha256": (
            "059a74e5d26408b12767421e1049929cd6fe643053b95b6bd472530c65c2ef65"
        ),
        "source_locks_canonical_sha256": (
            "80ee7ad1afd7d7b48a6396f216b266011b11618447e723ddec1a3d5719e24c97"
        ),
        "source_locks_file_sha256": (
            "b120115ede440825362c37018f099e939b2810ab52c8d06c512a3807d14a3656"
        ),
        "parent_route_contract_sha256": (
            "5d8aa8b6cefa7891b5bfd3647bde17f6d7255f59b9a5885d58c47e1c83e40e82"
        ),
        "route_contract_sha256": (
            "2f992c0439512f550bfd595880e65fd6083e70a9dfd6b343734f15e0e5815133"
        ),
    },
    1.0e-6: {
        "source_member_sha256": (
            "676dc81c77ced1c965f0b395d9f03fe95f6c356dbf78f27fa4dc0d9375ab4cd9"
        ),
        "implementation_inventory_sha256": (
            "3d15608d890fdeddbdef61c46d6e1700ff311d3700d816396e3dffe5b612fdba"
        ),
        "source_locks_canonical_sha256": (
            "2b08b9421cb6c66db33f6fbc0c2908c4ae72735603fd6552b1cdd18d1bba28ec"
        ),
        "source_locks_file_sha256": (
            "16012f807708e72c4d20956cc6541b8b33e1589a931a45cbbe86d143ccfc88c6"
        ),
        "parent_route_contract_sha256": (
            "c63b30d3204281fcf8c5688860cde45b337e0dcf9e52ad55627e932e765e551a"
        ),
        "route_contract_sha256": (
            "1579b15c50eaa4daa5b7c8ab9343488634452775fb04f13bc52b5cd0dd5e2ff2"
        ),
    },
}


class SweepError(RuntimeError):
    """Fail-closed sweep materialization error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    observed = canonical_sha256(unsigned)
    if value.get("sha256") != observed:
        raise SweepError(f"{label} self-digest drifted.")
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SweepError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise SweepError(f"{label} must be a JSON object.")
    return payload


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SweepError(f"Unsafe binding target: {path}")
    row: dict[str, Any] = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        payload = load_json(path, label=row["path"])
        row["canonical_sha256"] = verify_self_digest(
            payload, label=row["path"]
        )
    return row


def threshold_token(value: float) -> str:
    if value == 1.0e-4:
        return "1em4_anchor"
    if value == 1.0e-5:
        return "1em5"
    if value == 1.0e-6:
        return "1em6"
    raise SweepError(f"Undeclared threshold: {value!r}")


def package_dir(value: float) -> Path:
    return SWEEP_DIR / f"threshold_{threshold_token(value)}"


def _load_bound_source() -> dict[str, Any]:
    manifest_path = SOURCE_PACKAGE / "package_manifest.json"
    if (
        sha256_file(manifest_path) != SOURCE_PACKAGE_MANIFEST_FILE_SHA256
    ):
        raise SweepError("Immutable v3 package manifest bytes drifted.")
    manifest = load_json(manifest_path, label="immutable v3 package manifest")
    if (
        verify_self_digest(manifest, label="immutable v3 package manifest")
        != SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256
    ):
        raise SweepError("Immutable v3 package manifest digest drifted.")
    exact = (
        (
            SOURCE_PACKAGE / "protocol_bundle_manifest.json",
            SOURCE_PROTOCOL_BUNDLE_FILE_SHA256,
            SOURCE_PROTOCOL_BUNDLE_CANONICAL_SHA256,
            "protocol bundle",
        ),
        (
            SOURCE_PACKAGE / "source_locks_snapshot.json",
            SOURCE_LOCKS_FILE_SHA256,
            SOURCE_LOCKS_CANONICAL_SHA256,
            "source locks",
        ),
        (
            SOURCE_PACKAGE / "source/source_archive_manifest.json",
            SOURCE_ARCHIVE_MANIFEST_FILE_SHA256,
            SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256,
            "source archive manifest",
        ),
        (
            SOURCE_PACKAGE / SOURCE_PROTOCOL_RELATIVE,
            SOURCE_PROTOCOL_FILE_SHA256,
            SOURCE_PROTOCOL_CANONICAL_SHA256,
            "weak--weak source protocol",
        ),
    )
    loaded: dict[str, Any] = {"manifest": manifest}
    for path, file_sha, canonical_sha, label in exact:
        if sha256_file(path) != file_sha:
            raise SweepError(f"Immutable v3 {label} bytes drifted.")
        payload = load_json(path, label=f"immutable v3 {label}")
        if verify_self_digest(payload, label=label) != canonical_sha:
            raise SweepError(f"Immutable v3 {label} digest drifted.")
        loaded[label] = payload
    archive = SOURCE_PACKAGE / "source/source_locked.tar.gz"
    if sha256_file(archive) != SOURCE_ARCHIVE_FILE_SHA256:
        raise SweepError("Immutable v3 source archive bytes drifted.")
    if (
        loaded["source locks"].get("implementation_sources", {}).get("sha256")
        != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
    ):
        raise SweepError("Immutable v3 implementation inventory drifted.")
    loaded["archive"] = archive
    return loaded


def _extract_source(source: Mapping[str, Any], destination: Path) -> list[dict[str, Any]]:
    rows = source["source archive manifest"].get("members")
    if not isinstance(rows, list) or not rows:
        raise SweepError("Immutable v3 source member closure is absent.")
    declared = {
        str(row["path"]): dict(row)
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(declared) != len(rows):
        raise SweepError("Immutable v3 source member closure is malformed.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(source["archive"], "r:gz") as archive:
        for member in archive:
            row = declared.get(member.name)
            if (
                row is None
                or member.name in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
                or member.size != int(row.get("size_bytes", -1))
            ):
                raise SweepError(f"Unsafe v3 source member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SweepError(f"Unreadable v3 source member: {member.name}")
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            size = 0
            with target.open("xb") as output:
                for block in iter(lambda: extracted.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
            if size != member.size or digest.hexdigest() != row.get("sha256"):
                raise SweepError(f"Extracted v3 member drifted: {member.name}")
            observed.add(member.name)
    if observed != set(declared):
        raise SweepError("Immutable v3 source extraction is incomplete.")
    return [declared[path] for path in sorted(declared)]


def _patch_threshold(source_root: Path, threshold: float) -> dict[str, Any] | None:
    path = source_root / SOURCE_FILE_RELATIVE
    if sha256_file(path) != SOURCE_FILE_BEFORE_SHA256:
        raise SweepError("Page-8 threshold source bytes drifted.")
    if threshold == SOURCE_THRESHOLD:
        return None
    before = path.read_bytes()
    if before.count(SOURCE_DECLARATION) != 1:
        raise SweepError("Page-8 threshold declaration is not unique.")
    replacement = SOURCE_DECLARATION.rsplit(b" ", 1)[0] + (
        b" 1.0e-5" if threshold == 1.0e-5 else b" 1.0e-6"
    )
    after = before.replace(SOURCE_DECLARATION, replacement, 1)
    path.write_bytes(after)
    return {
        "path": SOURCE_FILE_RELATIVE.as_posix(),
        "before_sha256": SOURCE_FILE_BEFORE_SHA256,
        "after_sha256": sha256_file(path),
        "before_value": SOURCE_THRESHOLD,
        "after_value": threshold,
        "replacement_count": 1,
    }


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            del sys.modules[name]
    sys.path[:] = [
        item
        for item in sys.path
        if not (
            (Path(item or ".").resolve() / "pipelines").exists()
            or (Path(item or ".").resolve() / "src").exists()
        )
    ]
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()
    module = importlib.import_module("pipelines.static_adapt.ra_adapt")
    try:
        Path(str(module.__file__)).resolve().relative_to(root)
    except ValueError as exc:
        raise SweepError("Materializer escaped the sealed source root.") from exc


def _problem_from_receipt(receipt: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )

    return resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )


def _write_source_archive(
    source_root: Path,
    destination: Path,
    members: Sequence[Mapping[str, Any]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for row in members:
                    path = source_root / str(row["path"])
                    if (
                        not path.is_file()
                        or path.is_symlink()
                        or path.stat().st_size != int(row["size_bytes"])
                        or sha256_file(path) != row["sha256"]
                    ):
                        raise SweepError(f"Source member drifted: {row['path']}")
                    info = archive.gettarinfo(str(path), arcname=str(row["path"]))
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())


def _scalar_differences(
    before: Any,
    after: Any,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    rows: list[tuple[tuple[str | int, ...], Any, Any]] = []
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after)):
            if key not in before:
                rows.append(((*path, str(key)), "<missing>", after[key]))
            elif key not in after:
                rows.append(((*path, str(key)), before[key], "<missing>"))
            else:
                rows.extend(
                    _scalar_differences(
                        before[key], after[key], (*path, str(key))
                    )
                )
        return rows
    if (
        isinstance(before, Sequence)
        and not isinstance(before, (str, bytes))
        and isinstance(after, Sequence)
        and not isinstance(after, (str, bytes))
        and len(before) == len(after)
    ):
        for index, (left, right) in enumerate(zip(before, after)):
            rows.extend(_scalar_differences(left, right, (*path, index)))
        return rows
    if before != after:
        rows.append((path, before, after))
    return rows


def _scientific_audit(
    source_payload: Mapping[str, Any],
    target_payload: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    preserved = (
        "problem",
        "parent_inventory",
        "executable_pool",
        "optimizer",
        "optimizer_maxiter",
        "seeds",
        "candidate_representation",
        "adapter_id",
        "active_gradient_policy",
        "resource_weighting_scope",
        "accepted_refit_scope",
        "accepted_refit_coordinate_chart",
        "accepted_refit_base_chart_policy",
        "phase3_solver_id",
        "phase3_multiplier_contract",
        "estimator_accounting_convention",
        "compile_identity",
        "algorithm_id",
    )
    drift = [key for key in preserved if source_payload[key] != target_payload[key]]
    if drift:
        raise SweepError(f"Non-threshold scientific fields drifted: {drift}")
    source_request = source_payload["request"]
    target_request = target_payload["request"]
    for key in ("adapter", "method", "execution", "kind"):
        if source_request[key] != target_request[key]:
            raise SweepError(f"Non-threshold request field drifted: {key}")
    observation_differences = _scalar_differences(
        source_request["observation"], target_request["observation"]
    )
    allowed_observation_paths = {
        ("checkpoint", "path"),
        ("estimator_ledger", "path"),
    }
    if {
        path for path, _before, _after in observation_differences
    } - allowed_observation_paths:
        raise SweepError("Non-path observation settings drifted.")
    for path, before, after in observation_differences:
        if (
            path not in allowed_observation_paths
            or not isinstance(before, str)
            or not isinstance(after, str)
            or before.replace(SOURCE_EXECUTION_ID, str(target_payload["bundle_materialization"]["cell_id"]))
            != after
        ):
            raise SweepError("Observation path changed beyond execution identity.")
    source_route = copy.deepcopy(source_payload["route_contract"])
    target_route = copy.deepcopy(target_payload["route_contract"])
    observed_threshold = target_route["semantic_invariants"][
        "plateau_prior_mean_decrease_ratio_threshold"
    ]
    if observed_threshold != threshold:
        raise SweepError("Resolved route did not serialize the requested threshold.")
    source_parent = source_route["lineage_authority"]["parent_contract_sha256"]
    target_parent = target_route["lineage_authority"]["parent_contract_sha256"]
    source_route.pop("sha256", None)
    target_route.pop("sha256", None)
    target_route["semantic_invariants"][
        "plateau_prior_mean_decrease_ratio_threshold"
    ] = SOURCE_THRESHOLD
    target_route["lineage_authority"]["parent_contract_sha256"] = source_parent
    if source_route != target_route:
        paths = [
            list(path)
            for path, _left, _right in _scalar_differences(
                source_route, target_route
            )
        ]
        raise SweepError(f"Route changed beyond threshold lineage: {paths}")
    return {
        "changed_fields_vs_source": (
            []
            if threshold == SOURCE_THRESHOLD
            else [
                "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD"
            ]
        ),
        "non_swept_settings_diff": [],
        "source_route_contract_sha256": source_payload["route_contract"]["sha256"],
        "target_route_contract_sha256": target_payload["route_contract"]["sha256"],
        "source_parent_route_contract_sha256": source_parent,
        "target_parent_route_contract_sha256": target_parent,
        "resolved_threshold": observed_threshold,
        "observation_path_differences": [
            {"path": list(path), "before": before, "after": after}
            for path, before, after in observation_differences
        ],
    }


def _exact_energy(source_locks: Mapping[str, Any], source_lock_id: str) -> float:
    cell = source_locks["cell_locks"][source_lock_id]
    reference = cell["resolver_trace"]["same_cutoff_ed_reference"]
    if int(reference["nph"]) != 3:
        raise SweepError("Weak--weak same-cutoff reference drifted.")
    return float(reference["E_ED"])


def materialize_one(threshold: float) -> dict[str, Any]:
    destination = package_dir(threshold)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite package: {destination}")
    source = _load_bound_source()
    temporary = tempfile.TemporaryDirectory(
        prefix=f"paper-i-threshold-{threshold_token(threshold)}-"
    )
    try:
        source_root = (
            Path(temporary.name) / "source_locked_checkout"
        ).resolve()
        predecessor_members = _extract_source(source, source_root)
        patch = _patch_threshold(source_root, threshold)
        _activate_source_root(source_root)

        from pipelines.static_adapt.ra_adapt.bundles import (
            _build_request,
            _bundle_protocol_materialization_authority,
            _cell_from_manifest_row,
            _decorate_protocol_payload,
            _implementation_source_inventory,
            _source_lock_refs,
            _validate_protocol_payload,
        )
        from pipelines.static_adapt.ra_adapt.contracts import (
            canonical_sha256 as protocol_canonical_sha256,
            resolved_ra_adapt_protocol_from_mapping,
        )
        from pipelines.static_adapt.ra_adapt.engine import (
            build_resolved_ra_protocol,
        )

        source_payload = source["weak--weak source protocol"]
        source_protocol = resolved_ra_adapt_protocol_from_mapping(source_payload)
        source_bundle_rows = source["protocol bundle"].get("cells")
        if not isinstance(source_bundle_rows, list):
            raise SweepError("Source bundle cells are absent.")
        matching = [
            row
            for row in source_bundle_rows
            if isinstance(row, Mapping)
            and row.get("cell_id")
            == "phase3_on_plateau_r50__weak_weak__nph3__ra_singleton_plateau"
        ]
        if len(matching) != 1:
            raise SweepError("Weak--weak source bundle row is not unique.")
        source_cell = _cell_from_manifest_row(matching[0])
        token = threshold_token(threshold)
        execution_id = SOURCE_EXECUTION_ID
        target_cell = replace(
            source_cell,
            cell_id=execution_id,
            horizon=TARGET_HORIZON,
        )
        package_id = (
            "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_"
            f"{token}_weak_weak_r50_20260804_v1_chtc"
        )
        campaign_id = (
            "paper_i_ra_adapt_singleton_phase3_on_plateau_threshold_"
            "sensitivity_weak_weak_r50_20260804_v1"
        )
        implementation_inventory = _implementation_source_inventory(source_root)
        source_inventory = source["source locks"].get("implementation_sources")
        if (
            not isinstance(source_inventory, Mapping)
            or source_inventory.get("sha256")
            != SOURCE_IMPLEMENTATION_INVENTORY_SHA256
        ):
            raise SweepError("Source implementation inventory authority drifted.")
        source_locks = copy.deepcopy(source["source locks"])
        source_locks["implementation_sources"] = copy.deepcopy(
            implementation_inventory
        )
        source_locks.pop("sha256", None)
        source_locks["sha256"] = protocol_canonical_sha256(source_locks)
        expected_hashes = EXPECTED_DERIVATIVE_HASHES[threshold]
        source_locks_file_sha256 = hashlib.sha256(
            canonical_json_bytes(source_locks) + b"\n"
        ).hexdigest()
        if (
            implementation_inventory.get("sha256")
            != expected_hashes["implementation_inventory_sha256"]
            or source_locks["sha256"]
            != expected_hashes["source_locks_canonical_sha256"]
            or source_locks_file_sha256
            != expected_hashes["source_locks_file_sha256"]
        ):
            raise SweepError("Threshold implementation/source-lock hashes drifted.")

        destination.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(
            SWEEP_DIR / "package_contract_template.py",
            destination / "package_contract.py",
        )
        shutil.copyfile(
            SWEEP_DIR / "run_cell_template.py",
            destination / "run_cell.py",
        )
        shutil.copyfile(
            SOURCE_PACKAGE / "execute_authorized_job.sh",
            destination / "execute_authorized_job.sh",
        )
        shutil.copyfile(
            SOURCE_PACKAGE / "submit.sub.in",
            destination / "submit.sub.in",
        )
        os.chmod(destination / "execute_authorized_job.sh", 0o755)

        bundle_manifest = digested(
            {
                "schema": f"{PACKAGE_SCHEMA_PREFIX}_protocol_bundle_v1",
                "package_id": package_id,
                "campaign_id": campaign_id,
                "source_package": {
                    "path": SOURCE_PACKAGE_RELATIVE.as_posix(),
                    "package_manifest_sha256": SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
                    "protocol_bundle_manifest_sha256": SOURCE_PROTOCOL_BUNDLE_CANONICAL_SHA256,
                    "source_locks_sha256": SOURCE_LOCKS_CANONICAL_SHA256,
                    "source_archive_sha256": SOURCE_ARCHIVE_FILE_SHA256,
                },
                "source_patch": [] if patch is None else [patch],
                "source_locks_snapshot_sha256": source_locks["sha256"],
                "implementation_source_inventory_sha256": implementation_inventory[
                    "sha256"
                ],
                "threshold": threshold,
                "execution_target": "chtc",
                "remote_image": dict(REMOTE_IMAGE),
                "fanout_execution_blocked_pending_chtc_anchor": (
                    threshold != SOURCE_THRESHOLD
                ),
                "cells": [target_cell.to_dict()],
                "execution_authorized": False,
                "submission_authorized": False,
            }
        )
        write_json(destination / "protocol_bundle_manifest.json", bundle_manifest)

        problem = _problem_from_receipt(source_protocol.problem)
        refs = _source_lock_refs(source_locks, cell=target_cell)
        authority = _bundle_protocol_materialization_authority(
            cell=target_cell,
            bundle_id=package_id,
            bundle_manifest_sha256=bundle_manifest["sha256"],
            source_locks_sha256=source_locks["sha256"],
            source_lock_refs=refs,
            active_gradient_policy=ACTIVE_GRADIENT_POLICY,
            resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
        )
        request = _build_request(target_cell, bundle_dir=destination)
        resolved = build_resolved_ra_protocol(
            problem, request, materialization_authority=authority
        )
        target_payload = _decorate_protocol_payload(
            resolved.to_dict(),
            cell=target_cell,
            request=request,
            cell_source_lock=source_locks["cell_locks"][
                target_cell.source_lock_id
            ],
            materialization_authority=authority,
        )
        _validate_protocol_payload(
            target_payload,
            cell=target_cell,
            bundle_id=package_id,
            bundle_manifest_sha256=bundle_manifest["sha256"],
            active_gradient_policy=ACTIVE_GRADIENT_POLICY,
            resource_weighting_scope=RESOURCE_WEIGHTING_SCOPE,
            source_lock_refs=refs,
            cell_source_lock=source_locks["cell_locks"][
                target_cell.source_lock_id
            ],
            source_locks_sha256=source_locks["sha256"],
        )
        target_protocol = resolved_ra_adapt_protocol_from_mapping(target_payload)
        science_audit = _scientific_audit(
            source_payload, target_payload, threshold=threshold
        )
        route = target_payload["route_contract"]
        route_parent = route["lineage_authority"]["parent_contract_sha256"]
        if (
            route["sha256"] != expected_hashes["route_contract_sha256"]
            or route_parent
            != expected_hashes["parent_route_contract_sha256"]
        ):
            raise SweepError("Threshold route-contract hashes drifted.")

        control = digested(
            {
                "schema": f"{PACKAGE_SCHEMA_PREFIX}_package_control_v1",
                "package_id": package_id,
                "package_status": "passed_inert_one_row",
                "campaign_id": campaign_id,
                "algorithm_id": ALGORITHM_ID,
                "route_contract_sha256": route["sha256"],
                "parent_route_contract_sha256": route_parent,
                "route_profile": ROUTE_PROFILE,
                "threshold": threshold,
                "plateau_comparison": PLATEAU_COMPARISON,
                "plateau_trigger": PLATEAU_TRIGGER,
                "plateau_calibration": PLATEAU_CALIBRATION,
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "target_horizon": TARGET_HORIZON,
                "execution_target": "chtc",
                "execution_ids": [execution_id],
                "package_manifest_schema": f"{PACKAGE_SCHEMA_PREFIX}_package_manifest_v1",
                "job_schema": f"{PACKAGE_SCHEMA_PREFIX}_job_v1",
                "authorization_schema": f"{PACKAGE_SCHEMA_PREFIX}_execution_authorization_v1",
            }
        )
        write_json(destination / "package_control.json", control)
        write_json(destination / "source_locks_snapshot.json", source_locks)
        protocol_path = destination / "protocols" / f"{execution_id}.json"
        write_json(protocol_path, target_payload)
        protocol_binding = binding(protocol_path, root=destination, canonical=True)

        predecessor_by_path = {
            str(row["path"]): dict(row) for row in predecessor_members
        }
        source_inventory_by_path = {
            str(row["path"]): dict(row)
            for row in source_inventory.get("files", [])
            if isinstance(row, Mapping)
        }
        target_inventory_by_path = {
            str(row["path"]): dict(row)
            for row in implementation_inventory.get("files", [])
            if isinstance(row, Mapping)
        }
        implementation_paths = {
            str(row["path"])
            for row in implementation_inventory.get("files", [])
        }
        expected_changed_paths = (
            set()
            if threshold == SOURCE_THRESHOLD
            else {SOURCE_FILE_RELATIVE.as_posix()}
        )
        inventory_changed_paths = {
            path
            for path in source_inventory_by_path
            if source_inventory_by_path[path] != target_inventory_by_path.get(path)
        }
        if (
            set(target_inventory_by_path) != set(source_inventory_by_path)
            or len(target_inventory_by_path) != 160
            or implementation_paths - set(predecessor_by_path)
            or inventory_changed_paths != expected_changed_paths
        ):
            raise SweepError("Threshold implementation member closure drifted.")
        members: list[dict[str, Any]] = []
        for relative in sorted(predecessor_by_path):
            path = source_root / relative
            row = {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            if (
                relative != SOURCE_FILE_RELATIVE.as_posix()
                and row != predecessor_by_path[relative]
            ):
                raise SweepError(f"Non-threshold source member drifted: {relative}")
            if threshold == SOURCE_THRESHOLD and row != predecessor_by_path[relative]:
                raise SweepError(f"Anchor source member drifted: {relative}")
            members.append(row)
        archive_changed_paths = {
            row["path"]
            for row in members
            if row != predecessor_by_path[row["path"]]
        }
        if (
            len(members) != 165
            or {str(row["path"]) for row in members} != set(predecessor_by_path)
            or archive_changed_paths != expected_changed_paths
            or sha256_file(source_root / SOURCE_FILE_RELATIVE)
            != expected_hashes["source_member_sha256"]
        ):
            raise SweepError("Threshold source archive delta closure drifted.")
        source_archive = destination / "source/source_locked.tar.gz"
        _write_source_archive(source_root, source_archive, members)
        source_archive_manifest = digested(
            {
                "schema": f"{PACKAGE_SCHEMA_PREFIX}_source_archive_manifest_v1",
                "status": "passed",
                "implementation_source_inventory_sha256": implementation_inventory[
                    "sha256"
                ],
                "archive": binding(source_archive, root=destination),
                "member_count": len(members),
                "members": members,
                "global_source_paths": sorted(
                    str(row["path"])
                    for row in source_locks["global_sources"].values()
                ),
                "runtime_path_dependencies": ["requirements.txt"],
                "no_ambient_repo_imports": True,
                "predecessor_archive_sha256": SOURCE_ARCHIVE_FILE_SHA256,
                "source_patch": [] if patch is None else [patch],
            }
        )
        write_json(
            destination / "source/source_archive_manifest.json",
            source_archive_manifest,
        )

        cell_lock = source_locks["cell_locks"][target_cell.source_lock_id]
        job = digested(
            {
                "schema": control["job_schema"],
                "package_id": package_id,
                "campaign_id": campaign_id,
                "execution_id": execution_id,
                "source_cell_id": source_cell.cell_id,
                "source_lock_id": target_cell.source_lock_id,
                "source_lock_sha256": cell_lock["sha256"],
                "regime_id": "weak_weak",
                "nph": 3,
                "run_class": "diagnostic",
                "execution_target": "chtc",
                "execution_mode": "fresh_0_to_50",
                "source_horizon": TARGET_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "protocol_path": protocol_binding["path"],
                "protocol_sha256": target_protocol.sha256,
                "protocol_file_sha256": protocol_binding["sha256"],
                "protocol_bundle_manifest_sha256": bundle_manifest["sha256"],
                "source_locks_snapshot_sha256": source_locks["sha256"],
                "implementation_source_inventory_sha256": implementation_inventory[
                    "sha256"
                ],
                "active_gradient_policy": ACTIVE_GRADIENT_POLICY,
                "resource_weighting_scope": RESOURCE_WEIGHTING_SCOPE,
                "candidate_representation": CANDIDATE_REPRESENTATION,
                "insertion_policy": "plateau_commutation",
                "plateau_prior_mean_decrease_ratio_threshold": threshold,
                "plateau_threshold_comparison": PLATEAU_COMPARISON,
                "plateau_trigger_source": PLATEAU_TRIGGER,
                "route_contract_sha256": route["sha256"],
                "exact_same_cutoff_energy": _exact_energy(
                    source_locks, target_cell.source_lock_id
                ),
                "resources": {
                    "request_cpus": 4,
                    "request_memory_mb": 24_576,
                    "request_disk_mb": 40_960,
                    "max_runtime_seconds": 259_200,
                },
                "fresh_start_contract": {
                    "kind": "fresh_start",
                    "source_checkpoint": None,
                    "resume_archive": None,
                },
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        job_path = destination / "jobs" / f"{execution_id}.json"
        write_json(job_path, job)
        queue_path = destination / "queue.tsv"
        resources = job["resources"]
        with queue_path.open("xb") as stream:
            stream.write(
                (
                    "\t".join(
                        (
                            execution_id,
                            job_path.relative_to(destination).as_posix(),
                            protocol_path.relative_to(destination).as_posix(),
                            job["sha256"],
                            str(resources["request_cpus"]),
                            str(resources["request_memory_mb"]),
                            str(resources["request_disk_mb"]),
                            str(resources["max_runtime_seconds"]),
                        )
                    )
                    + "\n"
                ).encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())

        source_lock_audit = digested(
            {
                "schema": "source_locked_sensitivity_row_audit_v1",
                "status": "passed",
                "run_class": "diagnostic",
                "variable": "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD",
                "source_value": SOURCE_THRESHOLD,
                "row_value": threshold,
                "source_package_manifest_sha256": SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
                "source_protocol_sha256": SOURCE_PROTOCOL_CANONICAL_SHA256,
                "source_archive_sha256": SOURCE_ARCHIVE_FILE_SHA256,
                "source_implementation_inventory_sha256": SOURCE_IMPLEMENTATION_INVENTORY_SHA256,
                "target_implementation_inventory_sha256": implementation_inventory[
                    "sha256"
                ],
                "source_patch": [] if patch is None else [patch],
                "source_archive_member_count": len(members),
                "source_archive_changed_paths": sorted(archive_changed_paths),
                "implementation_inventory_member_count": len(
                    target_inventory_by_path
                ),
                "implementation_inventory_changed_paths": sorted(
                    inventory_changed_paths
                ),
                **science_audit,
                "wrapper_used": False,
                "runner_mode": "direct_source_locked_run_ra_adapt",
                "execution_target": "chtc",
                "remote_image": dict(REMOTE_IMAGE),
                "unresolved_source_fields": [],
                "fields_added_by_current_defaults": [],
                "external_sensitivity_calibration": {
                    "status": "weak_weak_training_diagnostic_only",
                    "route_serialized_calibration_string_preserved": PLATEAU_CALIBRATION,
                    "scientific_risk": (
                        "the_single_scalar_jointly_controls_insertion_domain_"
                        "opening_and_phase3_competitive_population_activation"
                    ),
                    "mechanism_effects_separately_identifiable": False,
                },
            }
        )
        write_json(destination / "source_lock_audit.json", source_lock_audit)

        execution_plan = digested(
            {
                "schema": f"{PACKAGE_SCHEMA_PREFIX}_chtc_execution_plan_v1",
                "package_id": package_id,
                "campaign_id": campaign_id,
                "run_class": "diagnostic",
                "execution_target": "chtc",
                "execution_mode": "fresh_0_to_50",
                "execution_ids": [execution_id],
                "row_count": 1,
                "threshold": threshold,
                "remote_image": dict(REMOTE_IMAGE),
                "queue_sha256": sha256_file(queue_path),
                "fanout_execution_blocked_pending_chtc_anchor": (
                    threshold != SOURCE_THRESHOLD
                ),
                "source_archive_manifest_sha256": source_archive_manifest[
                    "sha256"
                ],
                "source_lock_audit_sha256": source_lock_audit["sha256"],
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            }
        )
        write_json(destination / "execution_plan.json", execution_plan)

        manifest = digested(
            {
                "schema": control["package_manifest_schema"],
                "status": "passed_inert_one_row",
                "package_id": package_id,
                "campaign_id": campaign_id,
                "row_count": 1,
                "execution_ids": [execution_id],
                "source_cell_ids": [source_cell.cell_id],
                "source_horizon": TARGET_HORIZON,
                "target_horizon": TARGET_HORIZON,
                "threshold": threshold,
                "execution_target": "chtc",
                "remote_image": dict(REMOTE_IMAGE),
                "fanout_execution_blocked_pending_chtc_anchor": (
                    threshold != SOURCE_THRESHOLD
                ),
                "control_files": [
                    binding(destination / name, root=destination)
                    for name in (
                        "package_control.json",
                        "package_contract.py",
                        "run_cell.py",
                        "execute_authorized_job.sh",
                        "submit.sub.in",
                    )
                ],
                "protocol_bundle_manifest": binding(
                    destination / "protocol_bundle_manifest.json",
                    root=destination,
                    canonical=True,
                ),
                "source_locks_snapshot": binding(
                    destination / "source_locks_snapshot.json",
                    root=destination,
                    canonical=True,
                ),
                "source_archive": binding(source_archive, root=destination),
                "source_archive_manifest": binding(
                    destination / "source/source_archive_manifest.json",
                    root=destination,
                    canonical=True,
                ),
                "source_lock_audit": binding(
                    destination / "source_lock_audit.json",
                    root=destination,
                    canonical=True,
                ),
                "execution_plan": binding(
                    destination / "execution_plan.json",
                    root=destination,
                    canonical=True,
                ),
                "queue": binding(queue_path, root=destination),
                "protocols": [
                    {
                        "execution_id": execution_id,
                        "source_cell_id": source_cell.cell_id,
                        **protocol_binding,
                    }
                ],
                "jobs": [
                    {
                        "execution_id": execution_id,
                        **binding(job_path, root=destination, canonical=True),
                    }
                ],
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_ready": False,
                "submit_descriptor_present": False,
                "submitted": False,
            }
        )
        write_json(destination / "package_manifest.json", manifest)
        return {
            "threshold": threshold,
            "package_dir": destination.relative_to(REPO_ROOT).as_posix(),
            "package_id": package_id,
            "execution_id": execution_id,
            "package_manifest_sha256": manifest["sha256"],
            "source_archive_sha256": sha256_file(source_archive),
            "route_contract_sha256": route["sha256"],
            "parent_route_contract_sha256": route_parent,
            "implementation_source_inventory_sha256": implementation_inventory[
                "sha256"
            ],
            "source_patch": [] if patch is None else [patch],
            "source_lock_audit_sha256": source_lock_audit["sha256"],
            "execution_target": "chtc",
            "remote_image": dict(REMOTE_IMAGE),
            "fanout_execution_blocked_pending_chtc_anchor": (
                threshold != SOURCE_THRESHOLD
            ),
            "status": "passed_inert_one_row",
        }
    except BaseException:
        if destination.exists() and not (destination / "package_manifest.json").exists():
            shutil.rmtree(destination)
        raise
    finally:
        temporary.cleanup()


def _source_result_projection() -> dict[str, Any]:
    archive_path = REPO_ROOT / SOURCE_RESULT_ARCHIVE_RELATIVE
    if sha256_file(archive_path) != SOURCE_RESULT_ARCHIVE_SHA256:
        raise SweepError("Page-8 source result archive drifted.")
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(SOURCE_RESULT_MEMBER)
        stream = archive.extractfile(member)
        if stream is None:
            raise SweepError("Page-8 source result is unreadable.")
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_RESULT_FILE_SHA256:
        raise SweepError("Page-8 source result bytes drifted.")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SweepError("Page-8 source result is malformed.")
    return _result_projection(payload)


def _result_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise SweepError("Result has no typed run receipt.")
    trajectory = run.get("accepted_trajectory")
    summary = run.get("paper_i_summary")
    if not isinstance(trajectory, list) or not isinstance(summary, Mapping):
        raise SweepError("Result lacks trajectory or Paper-I summary.")
    trace = summary.get("accepted_error_trace")
    if not isinstance(trace, list) or len(trace) != TARGET_HORIZON:
        raise SweepError("Result lacks the complete R50 error trace.")
    identity = [
        {
            "controller_round": row["controller_round"],
            "generator_ids": row["generator_ids"],
            "insertion_positions": row["insertion_positions"],
            "operators": row["operators"],
        }
        for row in trajectory
    ]
    return {
        "controller_rounds": len(trajectory),
        "terminal_absolute_energy_error": float(
            trace[-1]["absolute_energy_error"]
        ),
        "terminal_active_ansatz_depth": int(trace[-1]["active_ansatz_depth"]),
        "accepted_identity_sha256": canonical_sha256(identity),
    }


def materialize_anchor() -> dict[str, Any]:
    if PLAN_PATH.exists() or PLAN_PATH.is_symlink():
        raise FileExistsError(f"Refusing to overwrite sweep plan: {PLAN_PATH}")
    source_projection = _source_result_projection()
    row = materialize_one(SOURCE_THRESHOLD)
    plan = digested(
        {
            "schema": "source_locked_sensitivity_audit_v1",
            "status": "anchor_pending",
            "source": {
                "table_label": "page_8_weak_weak_singleton",
                "method": "RA-ADAPT singleton, Phase-III on insertion plateau",
                "regime_or_case": "weak_weak__nph3",
                "source_json": (
                    f"{SOURCE_RESULT_ARCHIVE_RELATIVE.as_posix()}::"
                    f"{SOURCE_RESULT_MEMBER}"
                ),
                "source_sha256": SOURCE_RESULT_FILE_SHA256,
                "source_command_or_manifest": (
                    SOURCE_PACKAGE_RELATIVE / "package_manifest.json"
                ).as_posix(),
                "source_command_or_manifest_sha256": (
                    SOURCE_PACKAGE_MANIFEST_FILE_SHA256
                ),
                "runner_mode": "direct_source_locked_run_ra_adapt",
                "execution_target": "chtc",
                "remote_image": dict(REMOTE_IMAGE),
                "route_or_profile_id": ROUTE_PROFILE,
                "source_variable_value": SOURCE_THRESHOLD,
                "source_result_projection": source_projection,
            },
            "sweep": {
                "run_class": "diagnostic",
                "variable": "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD",
                "grid": [SOURCE_THRESHOLD, *FANOUT_THRESHOLDS],
                "runner_mode": "direct_source_locked_run_ra_adapt",
                "execution_target": "chtc",
                "wrapper_used": False,
                "wrapper_kind": None,
                "baseline_materialization_status": "complete",
                "unresolved_source_fields": [],
                "fields_added_by_current_defaults": [],
                "settings_changed": [
                    "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD"
                ],
            },
            "planned_rows": [row],
            "anchor": {
                "value": SOURCE_THRESHOLD,
                "anchor_result_json": None,
                "anchor_reproduces_source": False,
                "status": "pending_chtc_execution",
                "non_swept_settings_diff": [],
            },
            "local_replay": (
                binding(LOCAL_ANCHOR_FAILURE_PATH, root=SWEEP_DIR, canonical=True)
                if LOCAL_ANCHOR_FAILURE_PATH.is_file()
                else None
            ),
        }
    )
    write_json(PLAN_PATH, plan)
    return {"status": "anchor_materialized", "plan": plan, "row": row}


def verify_anchor(result_path: Path) -> dict[str, Any]:
    if ANCHOR_COMPARISON_PATH.exists() or ANCHOR_COMPARISON_PATH.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite anchor comparison: {ANCHOR_COMPARISON_PATH}"
        )
    plan = load_json(PLAN_PATH, label="sensitivity plan")
    verify_self_digest(plan, label="sensitivity plan")
    if plan.get("status") != "anchor_pending":
        raise SweepError("Sensitivity plan is not waiting for its anchor.")
    observed = _result_projection(load_json(result_path, label="anchor result"))
    expected = plan["source"]["source_result_projection"]
    error_diff = abs(
        observed["terminal_absolute_energy_error"]
        - expected["terminal_absolute_energy_error"]
    )
    sequence_match = (
        observed["accepted_identity_sha256"]
        == expected["accepted_identity_sha256"]
    )
    depth_match = (
        observed["terminal_active_ansatz_depth"]
        == expected["terminal_active_ansatz_depth"]
    )
    passed = error_diff <= 1.0e-12 and sequence_match and depth_match
    comparison = digested(
        {
            "schema": "source_locked_sensitivity_anchor_comparison_v1",
            "status": "passed" if passed else "diagnostic_invalid",
            "source_plan_sha256": plan["sha256"],
            "value": SOURCE_THRESHOLD,
            "anchor_result_json": result_path.resolve().relative_to(
                REPO_ROOT.resolve()
            ).as_posix(),
            "anchor_result_file_sha256": sha256_file(result_path),
            "anchor_reproduces_source": passed,
            "metric_abs_diff": error_diff,
            "operator_sequence_match": sequence_match,
            "insertion_position_sequence_match": sequence_match,
            "terminal_depth_match": depth_match,
            "expected": expected,
            "observed": observed,
            "non_swept_settings_diff": [],
        }
    )
    write_json(ANCHOR_COMPARISON_PATH, comparison)
    if not passed:
        raise SweepError(
            "The source-value anchor did not reproduce page 8; fan-out is blocked."
        )
    return comparison


def materialize_fanout() -> dict[str, Any]:
    if FANOUT_MANIFEST_PATH.exists() or FANOUT_MANIFEST_PATH.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite fan-out manifest: {FANOUT_MANIFEST_PATH}"
        )
    plan = load_json(PLAN_PATH, label="sensitivity plan")
    verify_self_digest(plan, label="sensitivity plan")
    if plan.get("status") != "anchor_pending":
        raise SweepError("Variant materialization requires a sealed anchor plan.")
    rows = [materialize_one(value) for value in FANOUT_THRESHOLDS]
    manifest = digested(
        {
            "schema": "source_locked_sensitivity_fanout_manifest_v1",
            "status": "passed_inert_two_rows_awaiting_chtc_anchor",
            "sensitivity_plan_sha256": plan["sha256"],
            "anchor_comparison_required_before_activation": True,
            "anchor_comparison_sha256": None,
            "selection_rule": (
                "minimum_terminal_r50_same_cutoff_absolute_energy_error_"
                "requiring_strictly_below_append"
            ),
            "append_target_absolute_energy_error": APPEND_TARGET_ABSOLUTE_ENERGY_ERROR,
            "rows": rows,
            "execution_authorized": False,
            "execution_target": "chtc",
            "submission_authorized": False,
        }
    )
    write_json(FANOUT_MANIFEST_PATH, manifest)
    return manifest


def record_local_anchor_failure() -> dict[str, Any]:
    if LOCAL_ANCHOR_FAILURE_PATH.exists() or LOCAL_ANCHOR_FAILURE_PATH.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite local-anchor failure: {LOCAL_ANCHOR_FAILURE_PATH}"
        )
    payload = digested(
        {
            "schema": "source_locked_sensitivity_local_anchor_failure_v1",
            "status": "diagnostic_invalid_environment_divergence",
            "execution_target": "local_macos",
            "source_threshold": SOURCE_THRESHOLD,
            "source_package_manifest_sha256": SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_FILE_SHA256,
            "source_result_sha256": SOURCE_RESULT_FILE_SHA256,
            "first_symmetry_tied_operator_divergence_round": 7,
            "first_material_decision_divergence_round": 15,
            "stopped_after_controller_round": 22,
            "example_source_round_15_operator": "guarded_singleton::eeeyezee",
            "example_local_round_15_operator": "guarded_singleton::yxeeeeze",
            "local_variants_launched": False,
            "conclusion": (
                "macos_local_replay_is_not_a_valid_source_value_anchor;_"
                "calibration_must_use_the_original_chtc_image"
            ),
            "paper_evidence_adopted": False,
        }
    )
    write_json(LOCAL_ANCHOR_FAILURE_PATH, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--materialize-anchor", action="store_true")
    mode.add_argument("--verify-anchor", type=Path)
    mode.add_argument("--materialize-fanout", action="store_true")
    mode.add_argument("--record-local-anchor-failure", action="store_true")
    args = parser.parse_args()
    try:
        if args.record_local_anchor_failure:
            payload = record_local_anchor_failure()
        elif args.materialize_anchor:
            payload = materialize_anchor()
        elif args.verify_anchor is not None:
            payload = verify_anchor(args.verify_anchor.resolve())
        else:
            payload = materialize_fanout()
    except (FileExistsError, OSError, SweepError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
