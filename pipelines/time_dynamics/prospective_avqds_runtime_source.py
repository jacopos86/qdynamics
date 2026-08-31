"""Exact-blind runtime-source archives for prospective AVQDS executors.

The scientific and bootstrap AVQDS jobs must not receive the repository-wide
runtime archive used by exact-reference jobs.  This module derives the local
Python import closure from the committed seed epoch plus the declared executor
overlay, writes only that closure, and verifies every archived byte against the
commit that owns it.  Campaign JSON, exact-reference data, receipts, reports,
and lifecycle evidence are therefore outside the runtime archive by
construction.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCOUT_PROFILE = "avqds_scout"
TRAJECTORY_PROFILE = "avqds_trajectory"

_PROFILE_ROOTS = {
    SCOUT_PROFILE: (
        "pipelines/time_dynamics/prospective_avqds_scout_worker.py",
        "pipelines/time_dynamics/runners/run_prospective_avqds_scout.py",
    ),
    TRAJECTORY_PROFILE: (
        "pipelines/time_dynamics/prospective_avqds_trajectory_worker.py",
        "pipelines/time_dynamics/runners/run_prospective_avqds_trajectory.py",
    ),
}
_LOCAL_ROOTS = frozenset({"docs", "pipelines", "src"})
_REVISION = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_PATH_PARTS = frozenset(
    {
        "action_registry",
        "campaigns",
        "exact_references",
        "lifecycle",
        "manifests",
        "output",
        "receipts",
    }
)
_DYNAMIC_IMPORT_EXPANSIONS_V1: Mapping[str, tuple[str, ...]] = {
    # These control-plane resolvers are imported for shared DTOs but are never
    # called by the AVQDS runtime action path.  Their empty expansion is an
    # explicit fail-closed declaration, rather than an ignored dynamic import.
    "pipelines/time_dynamics/acquisition_registry_contract.py": (),
    "pipelines/time_dynamics/acquisition_supervisor.py": (),
    "pipelines/time_dynamics/prospective_execution_bindings.py": (),
    "pipelines/time_dynamics/benchmarks/registry.py": (),
    # Lazy namespaces that can be exercised by imported scientific code.
    "pipelines/time_dynamics/ap_mclachlan/__init__.py": tuple(
        f"pipelines.time_dynamics.ap_mclachlan.{name}"
        for name in (
            "adaptive_trajectory",
            "fixed_step",
            "geometry",
            "geometry_eval",
            "hamiltonian",
            "integrators",
            "inverse",
            "prune_cost",
            "state",
            "support_atoms",
            "support_decision",
            "trajectory",
        )
    ),
    "pipelines/time_dynamics/adapters/hh.py": (
        "pipelines.time_dynamics.legacy.checkpoint_controller",
        "pipelines.time_dynamics.legacy.checkpoint_compile_audit",
        "pipelines.time_dynamics.legacy.checkpoint_exact_audit",
        "pipelines.time_dynamics.runners.hh_from_adapt_artifact",
    ),
    "pipelines/qse_spectra/__init__.py": tuple(
        f"pipelines.qse_spectra.{name}"
        for name in (
            "conductivity",
            "core",
            "green_functions",
            "hh_current_observables",
            "hh_response_observables",
            "io",
            "record_selection",
            "response_functions",
            "spectral_functions",
        )
    ),
    "src/quantum/chemistry/__init__.py": (
        # The base revision advertises molecular_adapt_core lazily even though
        # that module is absent from the commit. It is not used by the AVQDS
        # action path; every extant local target remains explicitly sealed.
        "src.quantum.chemistry.molecular_hamiltonian",
        "src.quantum.chemistry.molecular_uccsd",
        "src.quantum.chemistry.psi4_adapter",
        "src.quantum.chemistry.vibronic_h2",
    ),
}


class AVQDSRuntimeSourceError(ValueError):
    """An AVQDS runtime archive is incomplete, stale, or not exact-blind."""


@dataclass(frozen=True)
class AVQDSRuntimeSourceInventory:
    profile: str
    base_revision: str
    base_full_archive_sha256: str
    overlay_revision: str
    archived_files: tuple[str, ...]
    effective_python_files: tuple[str, ...]
    archived_files_sha256: str
    archive_sha256: str
    archive_bytes: int

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema": "paper_ii_avqds_exact_blind_runtime_source_inventory_v1",
            "profile": self.profile,
            "base_revision": self.base_revision,
            "base_full_archive_sha256": self.base_full_archive_sha256,
            "overlay_revision": self.overlay_revision,
            "archived_files": list(self.archived_files),
            "effective_python_files": list(self.effective_python_files),
            "archived_files_sha256": self.archived_files_sha256,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "exact_reference_blind": True,
            "campaign_data_files": [],
            "exact_reference_runtime_artifacts": [],
            "qiskit_compilation_requested": False,
        }


def _canonical_sha(value: object) -> str:
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str, allow_missing: bool = False) -> bytes | None:
    completed = subprocess.run(
        ("git", *arguments), cwd=root, check=False, capture_output=True
    )
    if completed.returncode == 0:
        return completed.stdout
    if allow_missing:
        return None
    detail = completed.stderr.decode("utf-8", errors="replace").strip()
    raise AVQDSRuntimeSourceError(f"Git source identity failed: {detail[:500]}")


def _git_tree(root: Path, revision: str) -> frozenset[str]:
    raw = _git(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        revision,
        "--",
        "docs",
        "pipelines",
        "src",
    )
    assert raw is not None
    return frozenset(
        line for line in raw.decode("utf-8").splitlines() if line
    )


def _git_file(root: Path, revision: str, path: str) -> bytes:
    raw = _git(root, "show", f"{revision}:{path}", allow_missing=True)
    if raw is None:
        raise AVQDSRuntimeSourceError(
            f"runtime import dependency is absent at committed revision: {path}"
        )
    return raw


_FULL_ARCHIVE_SHA_CACHE: dict[tuple[str, str], str] = {}


def _full_archive_sha(root: Path, revision: str) -> str:
    """Digest the whole committed tree, memoized per repository/revision.

    A full commit ID is content-addressed, so its ``git archive`` stream is
    immutable.  The Paper-II base tree is ~1.5 GiB, and every package build
    reproduces this digest several times, so recomputing it dominated
    packaging time without adding any tamper evidence.
    """

    key = (str(Path(root).resolve()), str(revision))
    cached = _FULL_ARCHIVE_SHA_CACHE.get(key)
    if cached is not None:
        return cached
    process = subprocess.Popen(
        ("git", "archive", "--format=tar", revision),
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    for chunk in iter(lambda: process.stdout.read(1 << 20), b""):
        digest.update(chunk)
    stderr = b"" if process.stderr is None else process.stderr.read()
    if process.wait() != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise AVQDSRuntimeSourceError(
            f"cannot reproduce full source archive: {detail[:500]}"
        )
    result = digest.hexdigest()
    _FULL_ARCHIVE_SHA_CACHE[key] = result
    return result


def _validated_overlay(
    root: Path,
    *,
    base_revision: str,
    overlay_revision: str,
    overlay_files: Mapping[str, Mapping[str, Any]],
    base_tree: frozenset[str],
    overlay_tree: frozenset[str],
    require_current_files: bool,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    for path, raw_row in sorted(overlay_files.items()):
        _assert_safe_archive_path(f"source/{path}")
        if not isinstance(raw_row, Mapping) or path not in overlay_tree:
            raise AVQDSRuntimeSourceError(
                f"declared executor overlay path is not committed: {path}"
            )
        overlay = _git_file(root, overlay_revision, path)
        overlay_sha = hashlib.sha256(overlay).hexdigest()
        base_present = path in base_tree
        base = None if not base_present else _git_file(root, base_revision, path)
        base_sha = None if base is None else hashlib.sha256(base).hexdigest()
        if (
            bool(raw_row.get("base_present")) != base_present
            or raw_row.get("base_sha256") != base_sha
            or raw_row.get("overlay_sha256") != overlay_sha
        ):
            raise AVQDSRuntimeSourceError(
                f"declared executor overlay digest changed: {path}"
            )
        if require_current_files:
            current = root / path
            if not current.is_file() or _file_sha(current) != overlay_sha:
                raise AVQDSRuntimeSourceError(
                    f"executor overlay worktree bytes differ from committed HEAD: {path}"
                )
        result[path] = overlay
    return result


def _module_path(module: Sequence[str], available: frozenset[str]) -> str | None:
    if not module or module[0] not in _LOCAL_ROOTS:
        return None
    stem = "/".join(module)
    for candidate in (f"{stem}.py", f"{stem}/__init__.py"):
        if candidate in available:
            return candidate
    return None


def _module_parts(path: str) -> tuple[str, ...]:
    parts = list(PurePosixPath(path).parts)
    if parts[-1] == "__init__.py":
        return tuple(parts[:-1])
    parts[-1] = parts[-1][:-3]
    return tuple(parts)


def _import_targets(path: str, content: bytes, available: frozenset[str]) -> set[str]:
    try:
        tree = ast.parse(content.decode("utf-8"), filename=path)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise AVQDSRuntimeSourceError(f"cannot parse runtime dependency: {path}") from exc
    current = _module_parts(path)
    package = current if path.endswith("/__init__.py") else current[:-1]
    result: set[str] = set()
    for node in ast.walk(tree):
        modules: list[tuple[str, ...]] = []
        required = False
        if isinstance(node, ast.Import):
            modules = [tuple(alias.name.split(".")) for alias in node.names]
            required = True
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                trim = node.level - 1
                if trim > len(package):
                    raise AVQDSRuntimeSourceError(
                        f"invalid relative runtime import in {path}"
                    )
                prefix = package[: len(package) - trim]
                module = tuple((node.module or "").split(".")) if node.module else ()
                base = prefix + tuple(part for part in module if part)
            else:
                base = tuple((node.module or "").split("."))
            modules = [base]
            modules.extend(base + tuple(alias.name.split(".")) for alias in node.names)
            required = True
        if not required:
            continue
        required_base = modules[0] if modules else ()
        if required_base and required_base[0] in _LOCAL_ROOTS:
            base_path = _module_path(required_base, available)
            if base_path is None:
                raise AVQDSRuntimeSourceError(
                    f"runtime import dependency is missing for {path}: "
                    + ".".join(required_base)
                )
            result.add(base_path)
        for module in modules[1:] if len(modules) > 1 else modules:
            target = _module_path(module, available)
            if target is not None:
                result.add(target)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def callable_name(node: ast.expr) -> str:
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = callable_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    dynamic_calls = {
        "__import__",
        "import_module",
        "importlib.import_module",
        "pkgutil.resolve_name",
        "pkgutil.iter_modules",
        "importlib.resources.files",
        "resources.files",
    }
    saw_nonliteral_dynamic = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = callable_name(node.func)
        if name not in dynamic_calls and not name.endswith(".import_module"):
            continue
        argument = node.args[0] if node.args else None
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            module = tuple(argument.value.split("."))
            if module and module[0] in _LOCAL_ROOTS:
                target = _module_path(module, available)
                if target is None:
                    raise AVQDSRuntimeSourceError(
                        f"literal dynamic runtime import is missing for {path}: "
                        f"{argument.value}"
                    )
                result.add(target)
            continue
        saw_nonliteral_dynamic = True
    if saw_nonliteral_dynamic:
        expansions = _DYNAMIC_IMPORT_EXPANSIONS_V1.get(path)
        if expansions is None:
            raise AVQDSRuntimeSourceError(
                f"unmapped dynamic runtime dependency in {path}"
            )
        for name in expansions:
            target = _module_path(tuple(name.split(".")), available)
            if target is None:
                raise AVQDSRuntimeSourceError(
                    f"mapped dynamic runtime dependency is missing for {path}: {name}"
                )
            result.add(target)
    return result


def _parent_initializers(path: str, available: frozenset[str]) -> set[str]:
    parts = PurePosixPath(path).parts[:-1]
    result: set[str] = set()
    for index in range(1, len(parts) + 1):
        candidate = PurePosixPath(*parts[:index], "__init__.py").as_posix()
        if candidate in available:
            result.add(candidate)
    return result


def _effective_closure(
    root: Path,
    *,
    profile: str,
    base_revision: str,
    base_tree: frozenset[str],
    overlay_revision: str,
    overlay_bytes: Mapping[str, bytes],
) -> tuple[tuple[str, ...], dict[str, bytes]]:
    roots = _PROFILE_ROOTS.get(profile)
    if roots is None:
        raise AVQDSRuntimeSourceError(f"unknown AVQDS runtime profile: {profile}")
    available = frozenset(set(base_tree).union(overlay_bytes))
    pending = list(roots)
    selected: set[str] = set()
    contents: dict[str, bytes] = {}
    while pending:
        path = pending.pop()
        if path in selected:
            continue
        if path not in available or not path.endswith(".py"):
            raise AVQDSRuntimeSourceError(
                f"runtime entry/import is not available in the declared composition: {path}"
            )
        content = (
            overlay_bytes[path]
            if path in overlay_bytes
            else _git_file(root, base_revision, path)
        )
        selected.add(path)
        contents[path] = content
        pending.extend(_import_targets(path, content, available).difference(selected))
        pending.extend(_parent_initializers(path, available).difference(selected))
    return tuple(sorted(selected)), contents


def _assert_safe_archive_path(path: str) -> None:
    pure = PurePosixPath(path)
    if (
        not path.startswith("source/")
        or pure.is_absolute()
        or ".." in pure.parts
        or any(part.lower() in _FORBIDDEN_PATH_PARTS for part in pure.parts)
        or not path.endswith(".py")
    ):
        raise AVQDSRuntimeSourceError(
            f"runtime source archive exposes a forbidden path: {path!r}"
        )


def _contract(
    root: Path,
    *,
    profile: str,
    base_revision: str,
    base_full_archive_sha256: str,
    overlay_revision: str,
    overlay_files: Mapping[str, Mapping[str, Any]],
    require_current_files: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, bytes], dict[str, bytes]]:
    if not _REVISION.fullmatch(base_revision) or not _REVISION.fullmatch(overlay_revision):
        raise AVQDSRuntimeSourceError("runtime source revisions are not full commit IDs")
    if not _SHA256.fullmatch(base_full_archive_sha256):
        raise AVQDSRuntimeSourceError("base source archive digest is invalid")
    if _full_archive_sha(root, base_revision) != base_full_archive_sha256:
        raise AVQDSRuntimeSourceError("base source epoch full archive digest changed")
    base_tree = _git_tree(root, base_revision)
    overlay_tree = _git_tree(root, overlay_revision)
    overlay = _validated_overlay(
        root,
        base_revision=base_revision,
        overlay_revision=overlay_revision,
        overlay_files=overlay_files,
        base_tree=base_tree,
        overlay_tree=overlay_tree,
        require_current_files=require_current_files,
    )
    closure, effective = _effective_closure(
        root,
        profile=profile,
        base_revision=base_revision,
        base_tree=base_tree,
        overlay_revision=overlay_revision,
        overlay_bytes=overlay,
    )
    archived = tuple(path for path in closure if path in base_tree)
    archived_contents = {
        f"source/{path}": _git_file(root, base_revision, path) for path in archived
    }
    for member in archived_contents:
        _assert_safe_archive_path(member)
    return archived, closure, archived_contents, overlay


def _archive_bytes(contents: Mapping[str, bytes]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
    ) as compressed:
        with tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT
        ) as archive:
            for name in sorted(contents):
                value = contents[name]
                info = tarfile.TarInfo(name)
                info.size = len(value)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(value))
    return raw.getvalue()


def _write_archive(path: Path, contents: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = _archive_bytes(contents)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as raw:
            raw.write(expected)
            raw.flush()
            os.fsync(raw.fileno())
        if path.exists():
            if path.read_bytes() != Path(temporary).read_bytes():
                raise AVQDSRuntimeSourceError(
                    f"refusing to overwrite immutable runtime archive: {path}"
                )
            os.unlink(temporary)
            temporary = ""
        else:
            os.replace(temporary, path)
            temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _read_archive(path: Path) -> dict[str, bytes]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            result: dict[str, bytes] = {}
            observed_order: list[str] = []
            for member in archive.getmembers():
                name = PurePosixPath(member.name).as_posix()
                _assert_safe_archive_path(name)
                if (
                    member.name != name
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                    or name in result
                    or member.mode != 0o644
                    or member.mtime != 0
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or bool(member.pax_headers)
                ):
                    raise AVQDSRuntimeSourceError(
                        f"runtime source archive has an unsafe member: {name!r}"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise AVQDSRuntimeSourceError(
                        f"runtime source member cannot be read: {name!r}"
                    )
                result[name] = handle.read()
                observed_order.append(name)
            if observed_order != sorted(observed_order):
                raise AVQDSRuntimeSourceError(
                    "runtime source archive member order is not canonical"
                )
            return result
    except (OSError, tarfile.TarError) as exc:
        raise AVQDSRuntimeSourceError("runtime source archive is not valid gzip/tar") from exc


def _import_smoke(
    archive_contents: Mapping[str, bytes],
    *,
    overlay_contents: Mapping[str, bytes],
    profile: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="paper-ii-avqds-source-smoke-") as temporary:
        root = Path(temporary)
        for member, content in archive_contents.items():
            target = root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        source_root = root / "source"
        for path, content in overlay_contents.items():
            target = source_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        modules = [path[:-3].replace("/", ".") for path in _PROFILE_ROOTS[profile]]
        script = (
            "import importlib, pathlib\n"
            + "\n".join(f"importlib.import_module({module!r})" for module in modules)
            + "\nassert not (pathlib.Path('pipelines/time_dynamics/campaigns') / "
            "'paper_ii_exact_response_scales_v1.json').exists()\n"
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        environment["PYTHONPATH"] = str(source_root)
        completed = subprocess.run(
            (sys.executable, "-c", script),
            cwd=source_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise AVQDSRuntimeSourceError(
                f"extracted exact-blind runtime import smoke failed: {detail[:4000]}"
            )


def smoke_avqds_runtime_work_order(
    archive_path: str | Path,
    *,
    repository_root: str | Path,
    profile: str,
    overlay_revision: str,
    overlay_files: Mapping[str, Mapping[str, Any]],
    work_order_path: str | Path,
    dependency_paths: Sequence[str | Path],
    composite_source_identity_sha256: str,
) -> None:
    """Exercise the real extracted runner validation path without execution."""

    root = Path(repository_root).resolve(strict=True)
    archive = Path(archive_path)
    if not archive.is_absolute():
        archive = root / archive
    contents = _read_archive(archive.resolve(strict=True))
    overlay = {
        path: _git_file(root, overlay_revision, path) for path in overlay_files
    }
    with tempfile.TemporaryDirectory(prefix="paper-ii-avqds-action-smoke-") as temporary:
        scratch = Path(temporary)
        for member, content in contents.items():
            target = scratch / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        source_root = scratch / "source"
        for path, content in overlay.items():
            target = source_root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        copied: set[str] = set()
        for value in (work_order_path, *dependency_paths):
            source = Path(value)
            if not source.is_absolute():
                source = root / source
            source = source.resolve(strict=True)
            relative = source.relative_to(root).as_posix()
            if relative in copied:
                continue
            copied.add(relative)
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        order_relative = Path(work_order_path)
        if order_relative.is_absolute():
            order_relative = order_relative.resolve(strict=True).relative_to(root)
        runner = (
            "pipelines.time_dynamics.runners.run_prospective_avqds_scout"
            if profile == SCOUT_PROFILE
            else "pipelines.time_dynamics.runners.run_prospective_avqds_trajectory"
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME"}
        }
        environment["PYTHONPATH"] = str(source_root)
        environment["PAPER_II_VERIFIED_PACKAGED_SOURCE_SHA256"] = (
            composite_source_identity_sha256
        )
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                runner,
                "validate",
                "--repo-root",
                ".",
                "--work-order",
                order_relative.as_posix(),
            ),
            cwd=source_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise AVQDSRuntimeSourceError(
                f"extracted runtime work-order replay failed: {detail[:4000]}"
            )


def _validate_avqds_runtime_source_archive(
    archive_path: str | Path,
    *,
    repository_root: str | Path,
    profile: str,
    base_revision: str,
    base_full_archive_sha256: str,
    overlay_revision: str,
    overlay_files: Mapping[str, Mapping[str, Any]],
    run_import_smoke: bool = True,
    require_current_files: bool,
) -> AVQDSRuntimeSourceInventory:
    root = Path(repository_root).resolve(strict=True)
    path = Path(archive_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AVQDSRuntimeSourceError("runtime archive is outside repository") from exc
    archived, closure, expected, overlay = _contract(
        root,
        profile=profile,
        base_revision=base_revision,
        base_full_archive_sha256=base_full_archive_sha256,
        overlay_revision=overlay_revision,
        overlay_files=overlay_files,
        require_current_files=require_current_files,
    )
    observed = _read_archive(path)
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        changed = sorted(
            name for name in set(expected).intersection(observed) if expected[name] != observed[name]
        )
        raise AVQDSRuntimeSourceError(
            "runtime archive diverges from exact committed allowlist "
            f"(missing={missing}, extra={extra}, changed={changed})"
        )
    if path.read_bytes() != _archive_bytes(expected):
        raise AVQDSRuntimeSourceError(
            "runtime source archive bytes are not the canonical deterministic encoding"
        )
    if run_import_smoke:
        _import_smoke(observed, overlay_contents=overlay, profile=profile)
    rows = {
        path: hashlib.sha256(expected[f"source/{path}"]).hexdigest()
        for path in archived
    }
    return AVQDSRuntimeSourceInventory(
        profile=profile,
        base_revision=base_revision,
        base_full_archive_sha256=base_full_archive_sha256,
        overlay_revision=overlay_revision,
        archived_files=archived,
        effective_python_files=closure,
        archived_files_sha256=_canonical_sha(rows),
        archive_sha256=_file_sha(path),
        archive_bytes=path.stat().st_size,
    )


def validate_avqds_runtime_source_archive(
    archive_path: str | Path,
    *,
    repository_root: str | Path,
    profile: str,
    base_revision: str,
    base_full_archive_sha256: str,
    overlay_revision: str,
    overlay_files: Mapping[str, Mapping[str, Any]],
    run_import_smoke: bool = True,
) -> AVQDSRuntimeSourceInventory:
    """Validate a package against both its commits and current overlay bytes."""

    return _validate_avqds_runtime_source_archive(
        archive_path,
        repository_root=repository_root,
        profile=profile,
        base_revision=base_revision,
        base_full_archive_sha256=base_full_archive_sha256,
        overlay_revision=overlay_revision,
        overlay_files=overlay_files,
        run_import_smoke=run_import_smoke,
        require_current_files=True,
    )


def validate_frozen_avqds_runtime_source_archive(
    archive_path: str | Path,
    *,
    repository_root: str | Path,
    profile: str,
    base_revision: str,
    base_full_archive_sha256: str,
    overlay_revision: str,
    overlay_files: Mapping[str, Mapping[str, Any]],
    run_import_smoke: bool = True,
) -> AVQDSRuntimeSourceInventory:
    """Validate an already-submitted archive from its immutable Git identities."""

    return _validate_avqds_runtime_source_archive(
        archive_path,
        repository_root=repository_root,
        profile=profile,
        base_revision=base_revision,
        base_full_archive_sha256=base_full_archive_sha256,
        overlay_revision=overlay_revision,
        overlay_files=overlay_files,
        run_import_smoke=run_import_smoke,
        require_current_files=False,
    )


def materialize_avqds_runtime_source_archive(
    archive_path: str | Path,
    *,
    repository_root: str | Path,
    profile: str,
    base_revision: str,
    base_full_archive_sha256: str,
    overlay_revision: str,
    overlay_files: Mapping[str, Mapping[str, Any]],
    run_import_smoke: bool = True,
) -> AVQDSRuntimeSourceInventory:
    root = Path(repository_root).resolve(strict=True)
    path = Path(archive_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AVQDSRuntimeSourceError("runtime archive is outside repository") from exc
    _archived, _closure, contents, _overlay = _contract(
        root,
        profile=profile,
        base_revision=base_revision,
        base_full_archive_sha256=base_full_archive_sha256,
        overlay_revision=overlay_revision,
        overlay_files=overlay_files,
        require_current_files=True,
    )
    _write_archive(path, contents)
    return validate_avqds_runtime_source_archive(
        path,
        repository_root=root,
        profile=profile,
        base_revision=base_revision,
        base_full_archive_sha256=base_full_archive_sha256,
        overlay_revision=overlay_revision,
        overlay_files=overlay_files,
        run_import_smoke=run_import_smoke,
    )


__all__ = [
    "AVQDSRuntimeSourceError",
    "AVQDSRuntimeSourceInventory",
    "SCOUT_PROFILE",
    "TRAJECTORY_PROFILE",
    "materialize_avqds_runtime_source_archive",
    "smoke_avqds_runtime_work_order",
    "validate_frozen_avqds_runtime_source_archive",
    "validate_avqds_runtime_source_archive",
]
