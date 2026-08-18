"""Report-only artifact-retention scan.

Implements the reporting step of `agent_guidance/shared/artifact-retention.md`:
lists compression candidates (uncompressed run-artifact JSON > 100 MB, quiet
for 7+ days) and expiry candidates (raw archive directories quiet for 30+
days). Prints a report and mutates nothing; deletion always requires explicit
user approval per the contract.

Usage:
    python3 pipelines/shell/artifact_retention_report.py [--root PATH]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

COMPRESS_MIN_BYTES = 100 * 1024 * 1024
COMPRESS_QUIET_DAYS = 7
EXPIRY_QUIET_DAYS = 30

SCAN_JSON_UNDER = ("output", "chtc", "raw_outputs", "artifacts")
RAW_ARCHIVE_ROOTS = ("raw_outputs",)
RAW_ARCHIVE_GLOBS = ("chtc/*/retrieved_*", "chtc/*/live_safety_snapshots_*")
PROTECTED_MARKERS = ("_preserved", "storage_archives")


def _protected(path: Path) -> bool:
    text = str(path)
    return any(marker in text for marker in PROTECTED_MARKERS)


def _dir_stats(directory: Path) -> tuple[int, float]:
    total = 0
    newest = 0.0
    for item in directory.rglob("*"):
        try:
            if item.is_file():
                stat = item.stat()
                total += stat.st_size
                newest = max(newest, stat.st_mtime)
        except OSError:
            continue
    return total, newest


def _fmt_gb(size: int) -> str:
    return f"{size / 1e9:7.2f} GB"


def _days_quiet(mtime: float, now: float) -> float:
    return (now - mtime) / 86400.0


def compression_candidates(root: Path, now: float) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for base in SCAN_JSON_UNDER:
        base_path = root / base
        if not base_path.is_dir():
            continue
        for item in base_path.rglob("*.json"):
            try:
                stat = item.stat()
            except OSError:
                continue
            if stat.st_size < COMPRESS_MIN_BYTES:
                continue
            if _days_quiet(stat.st_mtime, now) < COMPRESS_QUIET_DAYS:
                continue
            if _protected(item):
                continue
            found.append((stat.st_size, item))
    return sorted(found, reverse=True)


def expiry_candidates(root: Path, now: float) -> list[tuple[int, float, Path]]:
    directories: set[Path] = set()
    for base in RAW_ARCHIVE_ROOTS:
        base_path = root / base
        if base_path.is_dir():
            directories.update(
                item for item in base_path.iterdir() if item.is_dir()
            )
    for pattern in RAW_ARCHIVE_GLOBS:
        directories.update(
            item for item in root.glob(pattern) if item.is_dir()
        )

    found: list[tuple[int, float, Path]] = []
    for directory in sorted(directories):
        if _protected(directory):
            continue
        size, newest = _dir_stats(directory)
        if size == 0:
            continue
        if _days_quiet(newest, now) < EXPIRY_QUIET_DAYS:
            continue
        found.append((size, newest, directory))
    return sorted(found, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: this checkout)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    now = time.time()

    compress = compression_candidates(root, now)
    expire = expiry_candidates(root, now)

    print(f"artifact retention report — {root}")
    print(
        f"\ncompression candidates (uncompressed .json >= 100 MB, "
        f"quiet >= {COMPRESS_QUIET_DAYS} days): {len(compress)}"
    )
    for size, path in compress:
        print(f"  {_fmt_gb(size)}  {path.relative_to(root)}")

    print(
        f"\nexpiry candidates (raw archive dirs, quiet >= "
        f"{EXPIRY_QUIET_DAYS} days): {len(expire)}"
    )
    for size, newest, path in expire:
        quiet = int(_days_quiet(newest, now))
        print(
            f"  {_fmt_gb(size)}  quiet {quiet:3d}d  "
            f"{path.relative_to(root)}"
        )

    total = sum(size for size, _ in compress) + sum(
        size for size, _, _ in expire
    )
    print(f"\ntotal candidate volume: {_fmt_gb(total)}")
    print(
        "report only — nothing was modified; deletion requires explicit "
        "user approval (see agent_guidance/shared/artifact-retention.md)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
