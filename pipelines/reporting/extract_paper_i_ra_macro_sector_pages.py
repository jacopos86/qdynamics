#!/usr/bin/env python3
"""Keep only the two Holstein-sector bar pages in the active diagnostic PDF."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from pypdf import PdfReader, PdfWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_PDF = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_heatmap.pdf"
)
BACKUP_PDF = TARGET_PDF.with_name(
    f"{TARGET_PDF.stem}_pre_pages3_4_only_extraction_four_page_backup.pdf"
)
PROVENANCE_JSON = (
    REPO_ROOT
    / "output/pdf/paper_i_ra_macro_append_only_generator_type_regime_pages3_4_only_extraction_provenance.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    raw = b"" if contents is None else contents.get_data()
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def digested(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out["sha256"] = hashlib.sha256(canonical_json_bytes(out)).hexdigest()
    return out


def main() -> None:
    if not TARGET_PDF.is_file():
        raise FileNotFoundError(TARGET_PDF)
    if BACKUP_PDF.exists():
        raise FileExistsError(f"refusing to overwrite backup {BACKUP_PDF}")
    if PROVENANCE_JSON.exists():
        raise FileExistsError(f"refusing to overwrite provenance {PROVENANCE_JSON}")

    source_reader = PdfReader(str(TARGET_PDF))
    if len(source_reader.pages) != 4:
        raise ValueError(
            f"expected the active four-page diagnostic, found {len(source_reader.pages)} pages"
        )
    retained_pages = [source_reader.pages[2], source_reader.pages[3]]
    retained_content_shas = [page_content_sha256(page) for page in retained_pages]
    source_file_sha = sha256_file(TARGET_PDF)
    shutil.copy2(TARGET_PDF, BACKUP_PDF)
    if sha256_file(BACKUP_PDF) != source_file_sha:
        raise ValueError("four-page extraction backup hash mismatch")

    writer = PdfWriter()
    for page in retained_pages:
        writer.add_page(page)
    temporary = TARGET_PDF.with_name(f".{TARGET_PDF.name}.pages3_4_only.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as handle:
        writer.write(handle)

    extracted_reader = PdfReader(str(temporary))
    if len(extracted_reader.pages) != 2:
        raise ValueError("extracted diagnostic did not contain exactly two pages")
    extracted_content_shas = [
        page_content_sha256(page) for page in extracted_reader.pages
    ]
    if extracted_content_shas != retained_content_shas:
        raise ValueError("retained page content changed during extraction")
    temporary.replace(TARGET_PDF)

    provenance = digested(
        {
            "schema": "paper_i_ra_macro_generator_regime_sector_pages_only_v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed_exact_page_extraction",
            "target_pdf": str(TARGET_PDF.relative_to(REPO_ROOT)),
            "source_page_count": 4,
            "retained_source_page_numbers": [3, 4],
            "final_page_count": 2,
            "final_page_numbers": [1, 2],
            "retained_page_content_sha256_before": retained_content_shas,
            "retained_page_content_sha256_after": extracted_content_shas,
            "source_pdf_sha256": source_file_sha,
            "source_backup": str(BACKUP_PDF.relative_to(REPO_ROOT)),
            "source_backup_sha256": sha256_file(BACKUP_PDF),
            "final_pdf_sha256": sha256_file(TARGET_PDF),
        }
    )
    PROVENANCE_JSON.write_bytes(canonical_json_bytes(provenance) + b"\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
