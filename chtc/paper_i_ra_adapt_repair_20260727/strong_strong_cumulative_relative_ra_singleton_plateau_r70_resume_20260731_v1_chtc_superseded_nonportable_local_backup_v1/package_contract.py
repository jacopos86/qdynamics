#!/usr/bin/env python3
"""Contract helpers for the cumulative-relative strong--strong r70 row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


PACKAGE_ID = (
    "paper_i_ra_adapt_cumulative_relative_ss_singleton_plateau_"
    "r70_resume_20260731_v1_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_cumulative_relative_ss_singleton_plateau_"
    "r70_resume_v1"
)
EXECUTION_ID = (
    "core__strong_strong_u8__nph7__ra_singleton_plateau__r70"
)
SOURCE_CELL_ID = (
    "core__strong_strong_u8__nph7__ra_singleton_plateau"
)
SOURCE_HORIZON = 50
TARGET_HORIZON = 70
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
PLATEAU_RATIO_THRESHOLD = 1.0e-4
PLATEAU_COMPARISON = (
    "marginal_to_prior_cumulative_strictly_below_v1"
)
PLATEAU_TRIGGER = (
    "immediately_preceding_marginal_over_prior_cumulative_"
    "accepted_post_full_refit_energy_decrease_v1"
)
SOURCE_PROTOCOL_CANONICAL_SHA256 = (
    "ee0c304698ba4a6532e4271d57db2bd06f091f3e7f7b2e9947c7d01b0e2f2ae0"
)
SOURCE_PROTOCOL_FILE_SHA256 = (
    "38dfb80bbef62ecc3e148f7bc429fe0ee5ca615b9b1c99e52e6108153cbe7687"
)
ROUTE_CONTRACT_SHA256 = (
    "7ae0d6294ccee8ff8d79d3ed308d50659b6a7c810423835b0c2cdfa849efa4fd"
)
BASE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "1121f0dcadcf9ba9dd46896fb91f36342fa1f9fcc72e196ea79ffa2dadcd9e0f"
)
IMAGE_SIF_SHA256 = (
    "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
)
IMAGE_SIF_SIZE_BYTES = 216_371_200

SOURCE_BUNDLE_ROOT = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v13/ra_repair_stationary_late_core_v1"
)
MATERIALIZATION_ROOT = Path(
    "output/local_runs/"
    "paper_i_ra_adapt_cumulative_plateau_pair_r20_local_20260731_v1/"
    "materialization"
)
SOURCE_PROTOCOL = (
    MATERIALIZATION_ROOT
    / "protocols"
    / f"{SOURCE_CELL_ID}.json"
)
R20_RUN_ROOT = Path(
    "output/local_runs/"
    "paper_i_ra_adapt_cumulative_plateau_pair_r20_local_20260731_v1/"
    "runs"
) / SOURCE_CELL_ID
R30_ROOT = Path(
    "output/local_runs/"
    "paper_i_ra_adapt_cumulative_plateau_singleton_r30_finalized_local_"
    "20260731_v5"
)
R50_ROOT = Path(
    "output/local_runs/"
    "paper_i_ra_adapt_cumulative_plateau_singleton_r50_resume_local_"
    "20260731_v1"
)
R50_CHECKPOINT_ROOT = R50_ROOT / "canonical_resume_checkpoint"
R50_CHECKPOINT = R50_CHECKPOINT_ROOT / "checkpoint.json"
R50_LEDGER_SIDECAR = (
    R50_CHECKPOINT_ROOT
    / "checkpoint.estimator_call_ledger_checkpoint.de29984c4b344c8b.json"
)
R50_VERIFIED_SIDECAR = (
    R50_CHECKPOINT_ROOT
    / "checkpoint.verified_singleton_resume.bd328ecc06734905.json"
)

R50_CHECKPOINT_BINDING = {
    "path": "checkpoint.json",
    "sha256": (
        "b8186aabb56c8fee9ff71d5a6a9c6f5a7c18ea42e36431b65d54fca245386811"
    ),
    "size_bytes": 780_674_671,
}
R50_LEDGER_BINDING = {
    "path": R50_LEDGER_SIDECAR.name,
    "sha256": (
        "de29984c4b344c8b239c7a597800384255f45acf7f80c5ad3c767e71012426ab"
    ),
    "size_bytes": 1_524_419_419,
}
R50_VERIFIED_BINDING = {
    "path": R50_VERIFIED_SIDECAR.name,
    "sha256": (
        "bd328ecc06734905001e23b8b016d1062b2740c9045133871dc2988deec9db13"
    ),
    "size_bytes": 5_185,
}

# Each tuple is (path, whole-file SHA-256, canonical self-digest).
KNOWN_JSON_BINDINGS: dict[str, tuple[Path, str, str]] = {
    "v13_bundle_manifest": (
        SOURCE_BUNDLE_ROOT / "bundle_manifest.json",
        "d49f4a03d6b74d830c9f1aecbb4f191bfeb8afd4b67d25dab75aca727ef9936e",
        "7a4518c160ba5270c52c98f9885e839cdb1c9f22e5e4e8d525a57d4a8625304a",
    ),
    "v13_source_locks": (
        SOURCE_BUNDLE_ROOT / "source_locks.json",
        "4b65d21bde548345a4dc47995974271f110c23bb6ead2237b1bccea18e0ae8cd",
        "0a5e58daa29f133d3e3aa1406672053cb8e80f6182baed9f89690b4aeec16c17",
    ),
    "materialization_plan": (
        MATERIALIZATION_ROOT / "materialization_plan.json",
        "2abe5310f31e2cedaa4f08ddc912a272e6157fa94479be5f41fd34a1ed4420ff",
        "f8c7437e921b270ff49f04d886e839470f24a5671ff732200ba5096975e14346",
    ),
    "materialization_receipt": (
        MATERIALIZATION_ROOT / "materialization_receipt.json",
        "34aff8762ba131e77e7e9a16d11c2343acba0bfca43324253db1f5b7ad88829b",
        "4bbcc5f2da757d029a810c16ecb17987b6cedebaa653d855665cc11754704ff1",
    ),
    "materialization_validation": (
        MATERIALIZATION_ROOT / "validation_report.json",
        "cf694db3447e10d2354336e4abbce84f66d8a818e7f288a3f478fcb79d8c14fd",
        "fceb3581021993020395e68a3d46808bd32d8577c1f1ee04d639eecf4b37f930",
    ),
    "materialization_source_locks": (
        MATERIALIZATION_ROOT / "source_locks_snapshot.json",
        "07ad1133b337bb816bbc8cf39a3dad73eb7daafe818a87d6f4ef4bbded97717c",
        "f16fb043f8ac49e013001525b62343173a3b420bdcb0059c1ca6b752d8297fca",
    ),
    "source_protocol": (
        SOURCE_PROTOCOL,
        SOURCE_PROTOCOL_FILE_SHA256,
        SOURCE_PROTOCOL_CANONICAL_SHA256,
    ),
    "r20_terminal": (
        R20_RUN_ROOT / "terminal_receipt.json",
        "7d83f8eb6465344d8392378276d1aac9c55987bf1ce800a7c57496ba0baf418d",
        "65070e55091160b48517695447f7ddf57fd6edd977c52dd76c5bebddc5d3826c",
    ),
    "r20_manifest": (
        R20_RUN_ROOT / "run_manifest.json",
        "059324e347eb086112a37f578d414e8afe95e1283614178377aa3062b497f4dc",
        "b2b73c0493aa86c5a674b03f4ce933ebd88ae4eba3e0d01f605419d789f8df90",
    ),
    "r20_authorization": (
        R20_RUN_ROOT / "execution_authorization.json",
        "7dba9392308a681c4000289c015990db174589b88464f983c6b7326594a72edc",
        "32428d0ff5d7d8463f9a33cde4098fc1be9ae1ed5aef0008f316769208df7767",
    ),
    "r30_terminal": (
        R30_ROOT / "terminal_receipt.json",
        "10da6ebddb78aada30a3bef7272e613b37b747bf87f897c18bdf039c36f910bb",
        "8ad3cbedc7263f3314c3829608a08251bc97bf946803a3d01f24e773939d702c",
    ),
    "r30_repair": (
        R30_ROOT / "canonical_resume_checkpoint" / "repair_receipt.json",
        "f48cafa14e56c8d85c9f7bb4fc4edd2750aa06f0c2966a3b8f3f4ab0cc3b50fe",
        "c5b499654aa2be71a963f87b2709670a060ff30c66ea9b382cb16d0755d0af57",
    ),
    "r50_manifest": (
        R50_ROOT / "run_manifest.json",
        "2d8259cbebb83941d6bcfaa5d4bad79e301e4f9bc70ddee6f44f86bed24af4f5",
        "f2a7a615a881d66da05f589ea563e8ee9000084ff0cae3e751bd53048659515f",
    ),
    "r50_authorization": (
        R50_ROOT / "execution_authorization.json",
        "935c12954ddb20133d6cd7995cbab3b274e4af819e3252aa9db707e7f2e026a5",
        "524031fed17149e335b3dc10cad38e8d0653f53e28183471f220bf9cb50b36c4",
    ),
    "r50_terminal": (
        R50_ROOT / "terminal_receipt.json",
        "5f9b2a5f8524099d8af3a236837036c394a440c7022ea67b3567fbf1457bee25",
        "446999c1d184defdcd246387ca2dc74ae311230a35d7a39252f47f3e6d224754",
    ),
    "r50_repair": (
        R50_CHECKPOINT_ROOT / "repair_receipt.json",
        "100af564640811666e4c18915e1994f001b5eb848779f6dd0174b1cc413ddf69",
        "c4d1fb06a3b08fc1b974d8bd020e1322077a7c5b04efa8df3b6606e70a7c9d22",
    ),
}

R50_RESULT_BINDING = {
    "path": R50_ROOT / "result.json",
    "sha256": (
        "5f6a67b37f67f7e9dede28083d373443c68009fa0738a21bb0bdce650a5e1c8c"
    ),
    "size_bytes": 55_711_685,
}
R50_SUMMARY_BINDING = {
    "path": R50_ROOT / "paper_i_summary.json",
    "sha256": (
        "bb0bc9b2c1bd58e994d7f798492158167f24442fe8cab1ef2857b7073dd67e3f"
    ),
    "size_bytes": 72_534,
}

DERIVED_PROTOCOL_CHANGED_PATHS = (
    "horizon",
    "request.execution.stop.maximum_controller_rounds",
    "sha256",
    "stopping_rule.maximum_controller_rounds",
)

# These files may differ from the materialization inventory only through the
# explicitly reviewed, execution-plumbing repairs named here.  The builder
# rejects every other source drift.
ALLOWED_OPERATIONAL_SOURCE_DELTAS: dict[str, str] = {
    "pipelines/static_adapt/estimator_call_ledger.py": (
        "linear_time_consumer_key_validation_without_receipt_or_"
        "scientific_accounting_change_v1"
    ),
    "pipelines/static_adapt/adapt_pipeline.py": (
        "occurrence_stable_checkpoint_writer_plumbing_without_"
        "scientific_state_or_controller_change_v1"
    ),
}


class PackageContractError(ValueError):
    """Raised when a package input or sealed artifact drifts."""


def repo_root_from_script(script: str | Path) -> Path:
    path = Path(script).resolve()
    for parent in path.parents:
        if (parent / "AGENTS.md").is_file() and (parent / "pipelines").is_dir():
            return parent
    raise PackageContractError("Cannot resolve the active repository root.")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PackageContractError(
            f"Cannot load {label or path.as_posix()}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise PackageContractError(
            f"{label or path.as_posix()} must be a JSON object."
        )
    return value


def verify_self_digest(
    value: Mapping[str, Any], *, label: str
) -> str:
    body = dict(value)
    claimed = body.pop("sha256", None)
    observed = canonical_sha256(body)
    if claimed != observed:
        raise PackageContractError(f"{label} self-digest drifted.")
    return observed


def safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, (str, Path)):
        raise PackageContractError(f"{label} must be a relative path.")
    path = PurePosixPath(str(value))
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PackageContractError(f"Unsafe {label}: {value}")
    return path


def file_binding(
    path: Path,
    *,
    relative_to: Path,
    canonical: bool = False,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        payload = load_json(path)
        binding["canonical_sha256"] = verify_self_digest(
            payload, label=path.name
        )
    return binding


def scalar_differences(
    before: Any,
    after: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            return [(path, before, after)]
        rows: list[tuple[tuple[str | int, ...], Any, Any]] = []
        for key in sorted(before):
            rows.extend(
                scalar_differences(
                    before[key], after[key], path=(*path, str(key))
                )
            )
        return rows
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [(path, before, after)]
        rows = []
        for index, (left, right) in enumerate(zip(before, after)):
            rows.extend(
                scalar_differences(
                    left, right, path=(*path, index)
                )
            )
        return rows
    return [] if before == after else [(path, before, after)]


__all__ = [name for name in globals() if name.isupper()] + [
    "PackageContractError",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "file_binding",
    "load_json",
    "repo_root_from_script",
    "safe_relative_path",
    "scalar_differences",
    "sha256_file",
    "verify_self_digest",
]
