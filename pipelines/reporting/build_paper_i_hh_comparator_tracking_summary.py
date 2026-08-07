#!/usr/bin/env python3
"""Build a bounded-memory tracker summary from one comparator transfer archive.

The large ``result.json`` member is never materialized.  System ``jq --stream``
receives the member bytes directly and emits only the small identity, terminal,
and 50-round trajectory projection needed by the Paper-I tracker.  The embedded
worker validation receipt remains the authority for optimizer, ledger, leakage,
and compiled-resource gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "paper_i_hh_comparator_tracking_summary_v1"
SUPPORTED_RECEIPT_SCHEMAS = {
    "paper_i_hh_append_completion_validation_receipt_v1",
    "paper_i_hh_geo_completion_validation_receipt_v1",
}

_JQ_STREAM_FILTER = r"""
if length != 2 then empty
elif .[0] == ["status"] then
  {kind:"scalar", field:"status", value:.[1]}
elif .[0] == ["method_id"] then
  {kind:"scalar", field:"method_id", value:.[1]}
elif .[0] == ["n_ph_work"] then
  {kind:"scalar", field:"n_ph_work", value:.[1]}
elif .[0] == ["n_ph_reference"] then
  {kind:"scalar", field:"n_ph_reference", value:.[1]}
elif .[0] == ["same_cutoff_reference"] then
  {kind:"scalar", field:"same_cutoff_reference", value:.[1]}
elif .[0] == ["result", "abs_delta_e_same_cutoff"] then
  {kind:"scalar", field:"terminal_error", value:.[1]}
elif .[0] == ["result", "adapt_depth_reached"] then
  {kind:"scalar", field:"active_depth", value:.[1]}
elif .[0] == ["result", "S_alg"] then
  {kind:"scalar", field:"S_alg", value:.[1]}
elif ((.[0] | length) == 4
      and .[0][0] == "result"
      and .[0][1] == "adapt_history"
      and (.[0][3] == "outer_iteration"
           or .[0][3] == "iteration"
           or .[0][3] == "abs_delta_e_same_cutoff_after"
           or .[0][3] == "abs_delta_e_after")) then
  {kind:"history", index:.[0][2], field:.[0][3], value:.[1]}
else empty
end
"""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _finite(value: Any, *, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is nonfinite: {value!r}")
    return parsed


def _integer(value: Any, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not an integer: {value!r}") from exc
    return parsed


def _iter_named_json_array(handle: BinaryIO, key: str) -> Any:
    """Yield objects from one named JSON array without loading the document."""

    pattern = json.dumps(str(key)).encode("utf-8")
    carry = b""
    initial = b""
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            raise ValueError(f"JSON member lacks array {key!r}")
        data = carry + chunk
        position = data.find(pattern)
        if position < 0:
            carry = data[-(len(pattern) + 32) :]
            continue
        remainder = data[position + len(pattern) :]
        while b"[" not in remainder:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise ValueError(f"JSON array {key!r} has no opening bracket")
            remainder += chunk
        initial = remainder.split(b"[", 1)[1]
        break

    buffer = bytearray()
    started = False
    depth = 0
    in_string = False
    escaped = False

    def chunks() -> Any:
        yield initial
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return
            yield chunk

    for chunk in chunks():
        for byte in chunk:
            if not started:
                if byte in b" \t\r\n,":
                    continue
                if byte == ord("]"):
                    return
                if byte not in (ord("{"), ord("[")):
                    raise ValueError(f"unsupported scalar item in JSON array {key!r}")
                started = True
                depth = 1
                buffer.append(byte)
                continue

            buffer.append(byte)
            if in_string:
                if escaped:
                    escaped = False
                elif byte == ord("\\"):
                    escaped = True
                elif byte == ord('"'):
                    in_string = False
                continue
            if byte == ord('"'):
                in_string = True
            elif byte in (ord("{"), ord("[")):
                depth += 1
            elif byte in (ord("}"), ord("]")):
                depth -= 1
                if depth == 0:
                    item = json.loads(buffer)
                    if not isinstance(item, Mapping):
                        raise TypeError(f"JSON array {key!r} contains a non-object item")
                    yield dict(item)
                    buffer.clear()
                    started = False
    raise ValueError(f"JSON array {key!r} ended before an item closed")


def _tar_array_item(
    archive_path: Path,
    *,
    member_name: str,
    array_key: str,
    zero_index: int,
) -> dict[str, Any]:
    """Return one array item from a compressed JSON member with bounded memory."""

    with tarfile.open(archive_path, "r|gz") as archive:
        for info in archive:
            if info.name != member_name:
                archive.members.clear()
                continue
            handle = archive.extractfile(info)
            if handle is None:
                raise RuntimeError(f"cannot extract {member_name} from {archive_path}")
            for index, item in enumerate(_iter_named_json_array(handle, array_key)):
                if index == zero_index:
                    return item
            raise IndexError(
                f"{member_name}:{array_key} does not contain index {zero_index}"
            )
    raise RuntimeError(f"missing result member {member_name} in {archive_path}")


def _tar_json_member(archive_path: Path, *, member_name: str) -> dict[str, Any]:
    """Read one known-small JSON member from a compressed transfer archive."""

    with tarfile.open(archive_path, "r|gz") as archive:
        for info in archive:
            if info.name != member_name:
                archive.members.clear()
                continue
            handle = archive.extractfile(info)
            if handle is None:
                raise RuntimeError(f"cannot extract {member_name} from {archive_path}")
            payload = json.load(handle)
            if not isinstance(payload, Mapping):
                raise TypeError(f"JSON member is not an object: {archive_path}:{member_name}")
            return dict(payload)
    raise RuntimeError(f"missing JSON member {member_name} in {archive_path}")


def _stream_result_projection(handle: BinaryIO) -> tuple[dict[str, Any], str, int, str]:
    """Feed one result member through jq and return its compact projection."""

    try:
        jq_version = subprocess.check_output(
            ["jq", "--version"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("system jq is required for bounded comparator summaries") from exc

    process = subprocess.Popen(
        ["jq", "--stream", "--compact-output", _JQ_STREAM_FILTER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        raise RuntimeError("jq stdin pipe was not created")
    digest = hashlib.sha256()
    size = 0
    try:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            process.stdin.write(chunk)
        process.stdin.close()
        process.stdin = None
        stdout, stderr = process.communicate()
    except BaseException:
        process.kill()
        process.wait()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            "jq --stream failed while projecting comparator result: "
            + stderr.decode("utf-8", errors="replace")[-4000:]
        )

    scalars: dict[str, Any] = {}
    history: dict[int, dict[str, Any]] = {}
    for raw_line in stdout.splitlines():
        event = json.loads(raw_line)
        if not isinstance(event, Mapping):
            raise TypeError("jq projection emitted a non-object event")
        kind = event.get("kind")
        field = str(event.get("field"))
        if kind == "scalar":
            if field in scalars:
                raise ValueError(f"duplicate streamed scalar field: {field}")
            scalars[field] = event.get("value")
        elif kind == "history":
            index = _integer(event.get("index"), label="history index")
            row = history.setdefault(index, {})
            if field in row:
                raise ValueError(f"duplicate history field {field!r} at index {index}")
            row[field] = event.get("value")
        else:
            raise ValueError(f"unrecognized jq projection event: {event!r}")
    return {"scalars": scalars, "history": history}, digest.hexdigest(), size, jq_version


def _trajectory(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_history = projection.get("history")
    if not isinstance(raw_history, Mapping) or set(raw_history) != set(range(50)):
        raise ValueError("streamed comparator result must contain exactly 50 history rows")
    points: list[dict[str, Any]] = []
    for index in range(50):
        row = raw_history[index]
        if not isinstance(row, Mapping):
            raise TypeError(f"history row {index} is not an object")
        outer = row.get("outer_iteration")
        if outer is None:
            iteration = row.get("iteration")
            outer = index + 1 if iteration is None else _integer(
                iteration, label=f"history row {index} iteration"
            ) + 1
        round_id = _integer(outer, label=f"history row {index} round")
        if round_id != index + 1:
            raise ValueError(
                f"history round order drift at index {index}: {round_id}!={index + 1}"
            )
        error = row.get("abs_delta_e_same_cutoff_after")
        if error is None:
            error = row.get("abs_delta_e_after")
        points.append(
            {
                "round": round_id,
                "error": abs(_finite(error, label=f"history row {index} error")),
            }
        )
    return points


def _validate_receipt(receipt: Mapping[str, Any], *, job_id: str) -> None:
    if receipt.get("schema") not in SUPPORTED_RECEIPT_SCHEMAS:
        raise ValueError(f"unsupported validation receipt schema: {receipt.get('schema')!r}")
    required = {
        "status": "pass",
        "job_id": job_id,
        "adapt_iterations": 50,
        "ledger_closure": "pass",
        "sector_leak_flag": False,
        "boson_truncation_leak_flag": False,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"validation receipt {field} drift: {receipt.get(field)!r}!={expected!r}"
            )
    for field in (
        "active_depth",
        "same_cutoff_abs_error",
        "S_alg",
        "compiled_count_2q_total",
        "compiled_depth_2q_total",
        "compiled_depth_total",
    ):
        if receipt.get(field) is None:
            raise ValueError(f"validation receipt lacks {field}")
    if not str(receipt.get("variant") or ""):
        raise ValueError("validation receipt lacks variant identity")


def build_tracking_summary(
    *, archive_path: Path, job_id: str, output_json: Path | None = None
) -> dict[str, Any]:
    """Build and optionally persist one source-bound compact summary."""

    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    result_suffix = f"{job_id}/result.json"
    receipt_suffix = f"{job_id}/validation_receipt.json"
    projection: dict[str, Any] | None = None
    result_name: str | None = None
    result_sha: str | None = None
    result_size: int | None = None
    receipt: dict[str, Any] | None = None
    receipt_name: str | None = None
    receipt_sha: str | None = None
    receipt_size: int | None = None
    jq_version: str | None = None
    with tarfile.open(archive_path, "r|gz") as archive:
        for info in archive:
            if info.name.endswith(result_suffix):
                if projection is not None:
                    raise RuntimeError(f"duplicate result member matching {result_suffix}")
                handle = archive.extractfile(info)
                if handle is None:
                    raise RuntimeError(f"cannot extract {info.name} from {archive_path}")
                projection, result_sha, result_size, jq_version = _stream_result_projection(
                    handle
                )
                result_name = info.name
            elif info.name.endswith(receipt_suffix):
                if receipt is not None:
                    raise RuntimeError(f"duplicate receipt member matching {receipt_suffix}")
                handle = archive.extractfile(info)
                if handle is None:
                    raise RuntimeError(f"cannot extract {info.name} from {archive_path}")
                raw = handle.read()
                parsed = json.loads(raw)
                if not isinstance(parsed, Mapping):
                    raise TypeError(f"validation receipt is not an object: {info.name}")
                receipt = dict(parsed)
                receipt_name = info.name
                receipt_sha = hashlib.sha256(raw).hexdigest()
                receipt_size = len(raw)
            archive.members.clear()
    if projection is None or result_name is None or result_sha is None or result_size is None:
        raise RuntimeError(f"missing result member ending with {result_suffix}")
    if receipt is None or receipt_name is None or receipt_sha is None or receipt_size is None:
        raise RuntimeError(f"missing validation receipt ending with {receipt_suffix}")
    _validate_receipt(receipt, job_id=job_id)

    scalars = projection.get("scalars")
    if not isinstance(scalars, Mapping):
        raise TypeError("streamed comparator projection lacks scalar fields")
    required_scalars = {
        "status",
        "method_id",
        "n_ph_work",
        "n_ph_reference",
        "same_cutoff_reference",
        "terminal_error",
        "active_depth",
        "S_alg",
    }
    missing = sorted(required_scalars - set(scalars))
    if missing:
        raise ValueError(f"streamed comparator result lacks fields: {missing}")
    if scalars.get("status") != "completed":
        raise ValueError(f"comparator result status drift: {scalars.get('status')!r}")
    if scalars.get("same_cutoff_reference") is not True:
        raise ValueError("comparator result is not same-cutoff")
    n_ph_work = _integer(scalars.get("n_ph_work"), label="n_ph_work")
    n_ph_reference = _integer(scalars.get("n_ph_reference"), label="n_ph_reference")
    if n_ph_work != n_ph_reference:
        raise ValueError(f"working/reference cutoff drift: {n_ph_work}!={n_ph_reference}")
    trajectory = _trajectory(projection)
    terminal_error = abs(_finite(scalars.get("terminal_error"), label="terminal error"))
    active_depth = _integer(scalars.get("active_depth"), label="active depth")
    s_alg = _integer(scalars.get("S_alg"), label="S_alg")
    if not math.isclose(
        terminal_error,
        _finite(receipt.get("same_cutoff_abs_error"), label="receipt terminal error"),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("streamed terminal error disagrees with validation receipt")
    if active_depth != _integer(receipt.get("active_depth"), label="receipt active depth"):
        raise ValueError("streamed active depth disagrees with validation receipt")
    if s_alg != _integer(receipt.get("S_alg"), label="receipt S_alg"):
        raise ValueError("streamed S_alg disagrees with validation receipt")

    archive_sha = _sha256_path(archive_path)
    summary = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "archive": {
            "path": _display_path(archive_path),
            "sha256": archive_sha,
            "size_bytes": archive_path.stat().st_size,
        },
        "result_member": {
            "name": result_name,
            "sha256": result_sha,
            "size_bytes": result_size,
        },
        "validation_receipt_member": {
            "name": receipt_name,
            "sha256": receipt_sha,
            "size_bytes": receipt_size,
        },
        "projection": {
            "mode": "system_jq_stream_v1",
            "jq_version": jq_version,
            "bounded_memory": True,
        },
        "identity": {
            "job_id": job_id,
            "method_id": str(scalars.get("method_id")),
            "variant": str(receipt.get("variant")),
            "n_ph_work": n_ph_work,
            "n_ph_reference": n_ph_reference,
            "same_cutoff_reference": True,
        },
        "validation": {
            **receipt,
            "optimizer_gate_authority": (
                "embedded worker receipt status=pass; receipt is emitted only after "
                "the source-locked optimizer checks pass"
            ),
        },
        "result": {
            "status": "complete",
            "n_ph": n_ph_work,
            "rounds": 50,
            "active_depth": active_depth,
            "terminal_error": terminal_error,
            "s_alg": s_alg,
            "s_alg_scope": "validated comparator ledger",
            "trajectory": trajectory,
        },
        "qiskit": {
            "N2q": _integer(
                receipt.get("compiled_count_2q_total"), label="compiled N2q"
            ),
            "D2q": _integer(
                receipt.get("compiled_depth_2q_total"), label="compiled D2q"
            ),
            "Dc": _integer(receipt.get("compiled_depth_total"), label="compiled depth"),
        },
    }
    if output_json is not None:
        output_json = output_json.resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_json.with_name(output_json.name + ".tmp")
        temporary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output_json)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    output_json = args.output_json
    if output_json is None:
        stem = args.archive.name.removesuffix("_transfer.tar.gz")
        output_json = args.archive.with_name(f"{stem}_tracking_summary.json")
    summary = build_tracking_summary(
        archive_path=args.archive,
        job_id=args.job_id,
        output_json=output_json,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "output_json": str(output_json.resolve()),
                "archive_sha256": summary["archive"]["sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
