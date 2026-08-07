#!/usr/bin/env python3
"""Restore the validated singleton plateau-insertion page to the active report."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723"
)
STEM = "paper_i_hh_macro_common_accuracy_20260723"
FINAL_PDF = OUTPUT_DIR / f"{STEM}.pdf"
PAGE_PDF = OUTPUT_DIR / (
    f"{STEM}_singleton_plateau_insertion_batch_page12.pdf"
)
MAIN_PROVENANCE = OUTPUT_DIR / f"{STEM}_provenance.json"
BACKUP_PDF = OUTPUT_DIR / (
    f"{STEM}_pre_singleton_plateau_page12_restore_20260727.pdf"
)
PROVENANCE_PATH = OUTPUT_DIR / (
    f"{STEM}_singleton_plateau_insertion_batch_page12"
    "_restoration_20260727_provenance.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _page_content_sha256(page: Any) -> str:
    contents = page.get_contents()
    payload = b"" if contents is None else contents.get_data()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    base_reader = PdfReader(str(FINAL_PDF))
    page_reader = PdfReader(str(PAGE_PDF))
    if len(base_reader.pages) != 11:
        raise ValueError(
            f"refusing restoration: expected 11 base pages, found {len(base_reader.pages)}"
        )
    if len(page_reader.pages) != 1:
        raise ValueError("singleton plateau page asset must contain exactly one page")

    if BACKUP_PDF.exists():
        if _sha256(BACKUP_PDF) != _sha256(FINAL_PDF):
            raise ValueError("existing restoration backup does not match the active base")
    else:
        shutil.copy2(FINAL_PDF, BACKUP_PDF)

    base_page_hashes = [_page_content_sha256(page) for page in base_reader.pages]
    page_hash = _page_content_sha256(page_reader.pages[0])
    writer = PdfWriter()
    for page in base_reader.pages:
        writer.add_page(page)
    writer.add_page(page_reader.pages[0])
    temporary_pdf = OUTPUT_DIR / f".{STEM}_page12_restore.tmp.pdf"
    with temporary_pdf.open("wb") as handle:
        writer.write(handle)
    temporary_pdf.replace(FINAL_PDF)

    final_reader = PdfReader(str(FINAL_PDF))
    final_page_hashes = [_page_content_sha256(page) for page in final_reader.pages]
    if len(final_reader.pages) != 12:
        raise ValueError("restored report must contain 12 pages")
    if final_page_hashes[:11] != base_page_hashes:
        raise ValueError("restoration changed one or more existing pages")
    if final_page_hashes[11] != page_hash:
        raise ValueError("restored page does not match the validated page asset")

    provenance = {
        "schema": "paper_i_hh_singleton_plateau_page12_restoration_v1",
        "additive_update": True,
        "base_page_count": 11,
        "final_page_count": 12,
        "validation": {
            "preserved_page_content_sha256": base_page_hashes,
            "restored_page_content_sha256": page_hash,
        },
        "sources": {
            "backup_pdf": {
                "path": str(BACKUP_PDF.relative_to(REPO_ROOT)),
                "sha256": _sha256(BACKUP_PDF),
            },
            "page_pdf": {
                "path": str(PAGE_PDF.relative_to(REPO_ROOT)),
                "sha256": _sha256(PAGE_PDF),
            },
        },
        "outputs": {
            "final_pdf": {
                "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
                "sha256": _sha256(FINAL_PDF),
            }
        },
    }
    PROVENANCE_PATH.write_text(
        json.dumps(provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    main_provenance = json.loads(MAIN_PROVENANCE.read_text(encoding="utf-8"))
    main_provenance[
        "singleton_plateau_insertion_batch_page_restoration_20260727"
    ] = provenance
    main_provenance.setdefault("generated", {})["pdf"] = {
        "path": str(FINAL_PDF.relative_to(REPO_ROOT)),
        "sha256": _sha256(FINAL_PDF),
        "pages": 12,
    }
    main_provenance.setdefault("validation", {})["page_count"] = 12
    MAIN_PROVENANCE.write_text(
        json.dumps(main_provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(FINAL_PDF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
