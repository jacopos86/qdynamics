#!/usr/bin/env python3
"""Build the lane's upload bundle from a commit plus the lane working tree.

Everything the bundle carries that is not the lane itself -- ``paper_5/src``,
``paper_5/pyproject.toml``, ``chtc/validate_condor_submit_lifecycle.py`` --
comes from ``git archive HEAD``, so the digest names a commit and cannot pick
up another agent's in-flight edits in this shared worktree.  The lane
directory is taken from the working tree, because it is this key's own
uncommitted work; the script prints ``git status --short`` for both halves so
the report can say exactly which files that is.

The archive is written with normalized metadata -- names sorted, every mtime
the HEAD commit's own author time, uid/gid 0 and no owner names, modes
reduced to 0644/0755, and a gzip header with no timestamp -- so rebuilding
from the same commit and the same lane bytes reproduces the digest.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import subprocess
import sys
import tarfile
from pathlib import Path

LANE = "chtc/paper_v_cwrmf_event_guard_accel_20260915"
WORKTREE = Path(
    subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        cwd=Path(__file__).resolve().parent,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
)
COMMITTED_PATHS = (
    "paper_5/src",
    "paper_5/pyproject.toml",
    "chtc/validate_condor_submit_lifecycle.py",
)
OUT_DIR = WORKTREE / "output/local_runs/paper_v_cwrmf_event_guard_accel_20260915"
BUNDLE = OUT_DIR / "paper_v_cwrmf_event_guard_accel_20260915.tgz"
EXCLUDED = {"__pycache__", ".DS_Store"}


def git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=WORKTREE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main() -> int:
    commit = git("rev-parse", "HEAD").strip()
    commit_time = int(git("show", "-s", "--format=%at", commit).strip())
    print(f"HEAD {commit}")
    print("--- working tree, what the bundle takes from the commit ---")
    print(git("status", "--short", "--", *COMMITTED_PATHS), end="")
    print("--- working tree, the lane (taken from the working tree) ---")
    print(git("status", "--short", "--", LANE), end="")
    committed_dirty = git("status", "--porcelain", "--", *COMMITTED_PATHS).strip()
    if committed_dirty:
        print(
            "note: the committed half differs in the working tree; the bundle "
            "takes the committed bytes, not these.",
            file=sys.stderr,
        )

    members: dict[str, bytes | None] = {}
    archive_bytes = subprocess.run(
        ("git", "archive", commit, *COMMITTED_PATHS),
        cwd=WORKTREE,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as committed:
        for member in committed.getmembers():
            if any(part in EXCLUDED for part in Path(member.name).parts):
                continue
            if member.isdir():
                members[member.name.rstrip("/")] = None
            elif member.isfile():
                extracted = committed.extractfile(member)
                assert extracted is not None
                members[member.name] = extracted.read()

    lane_root = WORKTREE / LANE
    for path in sorted(lane_root.rglob("*")):
        relative = path.relative_to(WORKTREE).as_posix()
        if any(part in EXCLUDED for part in path.relative_to(WORKTREE).parts):
            continue
        members[relative] = None if path.is_dir() else path.read_bytes()
    members[LANE] = None
    for parent in ("chtc", "paper_5", "paper_5/src"):
        members.setdefault(parent, None)

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as bundle:
        for name in sorted(members):
            payload = members[name]
            info = tarfile.TarInfo(name)
            info.mtime = commit_time
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if payload is None:
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                bundle.addfile(info)
            else:
                info.type = tarfile.REGTYPE
                info.mode = 0o755 if payload.startswith(b"#!") else 0o644
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))

    compressed = gzip.compress(raw.getvalue(), compresslevel=9, mtime=0)
    BUNDLE.write_bytes(compressed)
    digest = hashlib.sha256(compressed).hexdigest()
    (OUT_DIR / "bundle.sha256").write_text(f"{digest}  {BUNDLE}\n", encoding="utf-8")
    (OUT_DIR / "bundle.bytes").write_text(f"{len(compressed)}\n", encoding="utf-8")
    (OUT_DIR / "bundle.entries").write_text(
        "\n".join(sorted(members)) + "\n", encoding="utf-8"
    )
    print(f"{digest}  {BUNDLE}")
    print(len(compressed))
    print(f"entries: {len(members)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
