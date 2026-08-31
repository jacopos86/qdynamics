"""Contracts for shared run-artifact storage guards."""

from pipelines.artifacts.storage_guard import (
    GIB,
    ContentFile,
    assess_fetch_storage,
    supersession_coverage,
)


def test_fetch_is_refused_when_working_space_crosses_reserve() -> None:
    assessment = assess_fetch_storage(
        free_bytes=11 * GIB,
        archive_bytes=100 * 1024**2,
        expected_extracted_bytes=GIB,
        free_floor_bytes=10 * GIB,
    )

    assert assessment.allowed is False
    assert assessment.headroom_after_peak_bytes < 0


def test_fetch_is_allowed_only_with_peak_space_above_reserve() -> None:
    assessment = assess_fetch_storage(
        free_bytes=13 * GIB,
        archive_bytes=100 * 1024**2,
        expected_extracted_bytes=GIB,
        free_floor_bytes=10 * GIB,
    )

    assert assessment.allowed is True
    assert assessment.headroom_after_peak_bytes > 0


def test_unknown_extraction_uses_conservative_multiplier() -> None:
    assessment = assess_fetch_storage(
        free_bytes=20 * GIB,
        archive_bytes=100,
        free_floor_bytes=10 * GIB,
    )

    assert assessment.expected_extracted_bytes == 1200
    assert assessment.estimated_peak_bytes == 1300


def test_same_name_with_different_content_blocks_supersession() -> None:
    older = [ContentFile("run.json", "a" * 64, 10)]
    retained = [ContentFile("run.json", "b" * 64, 10)]

    coverage = supersession_coverage(older, retained)

    assert coverage.fully_covered is False
    assert coverage.uncovered == tuple(older)


def test_identical_content_under_another_name_establishes_coverage() -> None:
    older = [ContentFile("partial/run.json", "a" * 64, 10)]
    retained = [ContentFile("final/result.json", "a" * 64, 10)]

    coverage = supersession_coverage(older, retained)

    assert coverage.fully_covered is True
    assert coverage.covered == tuple(older)
    assert coverage.reclaimable_bytes == 10


def test_one_unique_partial_file_blocks_whole_form_supersession() -> None:
    older = [
        ContentFile("shared.json", "a" * 64, 10),
        ContentFile("partial-only.json", "b" * 64, 20),
    ]
    retained = [ContentFile("shared.json", "a" * 64, 10)]

    coverage = supersession_coverage(older, retained)

    assert coverage.fully_covered is False
    assert coverage.reclaimable_bytes == 10
    assert [item.path for item in coverage.uncovered] == ["partial-only.json"]

