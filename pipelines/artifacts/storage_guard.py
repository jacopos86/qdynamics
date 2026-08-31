"""Storage guards for verified run-artifact fetches and supersession.

The module deliberately makes no deletion. It answers two questions that must
be settled before an artifact workflow may proceed: whether a fetch fits above
a fixed free-space reserve, and whether every file in an older artifact form is
covered by identical retained content. Directory names and relative filenames
never establish supersession.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

GIB = 1024**3
DEFAULT_FREE_FLOOR_BYTES = 10 * GIB
DEFAULT_UNKNOWN_EXTRACTION_MULTIPLIER = 12.0


@dataclass(frozen=True)
class FetchStorageAssessment:
    """Decision for one prospective verified fetch and extraction."""

    allowed: bool
    free_bytes: int
    free_floor_bytes: int
    archive_bytes: int
    expected_extracted_bytes: int
    estimated_peak_bytes: int
    headroom_after_peak_bytes: int
    reason: str


def assess_fetch_storage(
    *,
    free_bytes: int,
    archive_bytes: int,
    expected_extracted_bytes: int | None = None,
    free_floor_bytes: int = DEFAULT_FREE_FLOOR_BYTES,
    unknown_extraction_multiplier: float = (
        DEFAULT_UNKNOWN_EXTRACTION_MULTIPLIER
    ),
) -> FetchStorageAssessment:
    """Return whether a fetch fits without crossing the free-space floor."""

    values = {
        "free_bytes": free_bytes,
        "archive_bytes": archive_bytes,
        "free_floor_bytes": free_floor_bytes,
    }
    if any(int(value) < 0 for value in values.values()):
        raise ValueError("storage byte counts must be non-negative")
    if unknown_extraction_multiplier < 1.0:
        raise ValueError("unknown extraction multiplier must be at least one")

    extracted = (
        int(expected_extracted_bytes)
        if expected_extracted_bytes is not None
        else int(archive_bytes * unknown_extraction_multiplier)
    )
    if extracted < 0:
        raise ValueError("expected extracted bytes must be non-negative")
    estimated_peak = int(archive_bytes) + extracted
    headroom = int(free_bytes) - estimated_peak - int(free_floor_bytes)
    allowed = headroom >= 0
    reason = (
        "fetch fits above the declared free-space reserve"
        if allowed
        else "fetch would cross the declared free-space reserve"
    )
    return FetchStorageAssessment(
        allowed=allowed,
        free_bytes=int(free_bytes),
        free_floor_bytes=int(free_floor_bytes),
        archive_bytes=int(archive_bytes),
        expected_extracted_bytes=extracted,
        estimated_peak_bytes=estimated_peak,
        headroom_after_peak_bytes=headroom,
        reason=reason,
    )


def assess_fetch_path(
    *,
    path: str | Path,
    archive_bytes: int,
    expected_extracted_bytes: int | None = None,
    free_floor_bytes: int = DEFAULT_FREE_FLOOR_BYTES,
) -> FetchStorageAssessment:
    """Assess a fetch against the volume containing ``path``."""

    free = shutil.disk_usage(Path(path)).free
    return assess_fetch_storage(
        free_bytes=free,
        archive_bytes=archive_bytes,
        expected_extracted_bytes=expected_extracted_bytes,
        free_floor_bytes=free_floor_bytes,
    )


@dataclass(frozen=True)
class ContentFile:
    """One immutable file in a hash inventory."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SupersessionCoverage:
    """Per-file proof that an older form is or is not superseded."""

    covered: tuple[ContentFile, ...]
    uncovered: tuple[ContentFile, ...]

    @property
    def fully_covered(self) -> bool:
        return not self.uncovered

    @property
    def reclaimable_bytes(self) -> int:
        return sum(item.size_bytes for item in self.covered)


def supersession_coverage(
    older: Iterable[ContentFile],
    retained: Iterable[ContentFile],
) -> SupersessionCoverage:
    """Match older files against retained content by SHA-256 and size."""

    retained_content = {
        (item.sha256.lower(), int(item.size_bytes)) for item in retained
    }
    covered: list[ContentFile] = []
    uncovered: list[ContentFile] = []
    for item in older:
        target = (
            covered
            if (item.sha256.lower(), int(item.size_bytes)) in retained_content
            else uncovered
        )
        target.append(item)
    return SupersessionCoverage(tuple(covered), tuple(uncovered))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path.cwd())
    parser.add_argument("--archive-bytes", type=int, required=True)
    parser.add_argument("--expected-extracted-bytes", type=int)
    parser.add_argument(
        "--free-floor-gib",
        type=float,
        default=DEFAULT_FREE_FLOOR_BYTES / GIB,
    )
    args = parser.parse_args()
    assessment = assess_fetch_path(
        path=args.path,
        archive_bytes=args.archive_bytes,
        expected_extracted_bytes=args.expected_extracted_bytes,
        free_floor_bytes=int(args.free_floor_gib * GIB),
    )
    print(json.dumps(asdict(assessment), indent=2, sort_keys=True))
    return 0 if assessment.allowed else 2


if __name__ == "__main__":
    raise SystemExit(_main())

