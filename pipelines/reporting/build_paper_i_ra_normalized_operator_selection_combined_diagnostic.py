#!/usr/bin/env python3
"""Combine normalized macro and singleton operator-selection diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

from pypdf import PdfReader


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output/pdf"
DEFAULT_MACRO_PDF = (
    OUTPUT_DIR / "paper_i_ra_macro_append_only_generator_type_regime_heatmap.pdf"
)
DEFAULT_MACRO_PROVENANCE = (
    OUTPUT_DIR
    / "paper_i_ra_macro_append_only_generator_type_regime_normalized_provenance.json"
)
DEFAULT_SINGLETON_PDF = (
    OUTPUT_DIR / "paper_i_ra_singleton_qubit_support_pauli_diagnostic.pdf"
)
DEFAULT_SINGLETON_PROVENANCE = (
    OUTPUT_DIR / "paper_i_ra_singleton_qubit_support_pauli_diagnostic_provenance.json"
)
DEFAULT_OUTPUT_PDF = (
    OUTPUT_DIR / "paper_i_ra_normalized_operator_selection_combined_diagnostic.pdf"
)
DEFAULT_OUTPUT_TEX = DEFAULT_OUTPUT_PDF.with_suffix(".tex")
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_PDF.with_name(
    f"{DEFAULT_OUTPUT_PDF.stem}_provenance.json"
)
DEFAULT_BUILD_DIR = (
    REPO_ROOT
    / "tmp/pdfs/paper_i_ra_normalized_operator_selection_combined_diagnostic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-pdf", type=Path, default=DEFAULT_MACRO_PDF)
    parser.add_argument(
        "--macro-provenance", type=Path, default=DEFAULT_MACRO_PROVENANCE
    )
    parser.add_argument("--singleton-pdf", type=Path, default=DEFAULT_SINGLETON_PDF)
    parser.add_argument(
        "--singleton-provenance",
        type=Path,
        default=DEFAULT_SINGLETON_PROVENANCE,
    )
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    parser.add_argument("--provenance-json", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    return parser.parse_args()


def canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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


def verify_self_digest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(payload.get("sha256", ""))
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if actual != claimed:
        raise ValueError(f"provenance self-digest drifted: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    unsigned["sha256"] = hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()
    encoded = canonical_json_bytes(unsigned) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def build_tex(path: Path, *, macro_pdf: Path, singleton_pdf: Path) -> None:
    macro_relative = repo_relative(macro_pdf)
    singleton_relative = repo_relative(singleton_pdf)
    output_parent_relative = repo_relative(path.parent)
    prefix = "../" * len(Path(output_parent_relative).parts)
    lines = [
        r"\documentclass[letterpaper]{article}",
        r"\usepackage[margin=0in]{geometry}",
        r"\usepackage{pdfpages}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        rf"\includepdf[pages=-,fitpaper=true]{{{prefix}{macro_relative}}}",
        rf"\includepdf[pages=-,fitpaper=true]{{{prefix}{singleton_relative}}}",
        r"\end{document}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(path)


def compile_tex(*, tex_path: Path, output_pdf: Path, build_dir: Path) -> None:
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError("latexmk is required for the combined diagnostic")
    build_dir.mkdir(parents=True, exist_ok=True)
    command = [
        latexmk,
        "-g",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        f"-outdir={build_dir}",
        tex_path.name,
    ]
    completed = subprocess.run(
        command,
        cwd=tex_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "LaTeX build failed:\n"
            + completed.stdout[-8000:]
            + "\n"
            + completed.stderr[-4000:]
        )
    built_pdf = build_dir / output_pdf.name
    if not built_pdf.is_file():
        raise FileNotFoundError(built_pdf)
    temporary = output_pdf.with_name(f".{output_pdf.name}.tmp")
    shutil.copy2(built_pdf, temporary)
    temporary.replace(output_pdf)


def validate_landscape_pdf(path: Path, *, expected_pages: int) -> list[str]:
    reader = PdfReader(str(path))
    if len(reader.pages) != expected_pages:
        raise ValueError(
            f"{path} has {len(reader.pages)} pages, expected {expected_pages}"
        )
    hashes: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if not math.isclose(width, 792.0, abs_tol=1.0) or not math.isclose(
            height, 612.0, abs_tol=1.0
        ):
            raise ValueError(
                f"{path} page {index} is not US-letter landscape: "
                f"{width} x {height}"
            )
        hashes.append(page_content_sha256(page))
    return hashes


def validate_share_sums(values: dict[str, Any], *, context: str) -> None:
    if not values:
        raise ValueError(f"missing normalized share sums: {context}")
    for regime_id, value in values.items():
        if not math.isclose(float(value), 100.0, abs_tol=1.0e-8, rel_tol=0.0):
            raise ValueError(
                f"{context} {regime_id} sums to {value}, not 100 percent"
            )


def main() -> None:
    args = parse_args()
    for path in (
        args.macro_pdf,
        args.macro_provenance,
        args.singleton_pdf,
        args.singleton_provenance,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    macro_provenance = verify_self_digest(args.macro_provenance)
    singleton_provenance = verify_self_digest(args.singleton_provenance)
    if "within_regime_normalized" not in str(macro_provenance.get("schema", "")):
        raise ValueError("macro provenance is not within-regime normalized")
    if singleton_provenance.get("schema") != (
        "paper_i_ra_singleton_qubit_support_pauli_diagnostic_v2"
    ):
        raise ValueError("singleton provenance is not the normalized v2 schema")
    macro_expected_sha = str(
        macro_provenance.get("append", {}).get("final_pdf_sha256", "")
    )
    singleton_expected_sha = str(
        singleton_provenance.get("outputs", {})
        .get("pdf", {})
        .get("sha256", "")
    )
    if sha256_file(args.macro_pdf) != macro_expected_sha:
        raise ValueError("macro PDF does not match normalized provenance")
    if sha256_file(args.singleton_pdf) != singleton_expected_sha:
        raise ValueError("singleton PDF does not match normalized provenance")
    for sector, receipt in macro_provenance.get("pages", {}).items():
        validate_share_sums(
            receipt.get("regime_normalized_share_sums", {}),
            context=f"macro {sector}",
        )
    singleton_bar_pages = [
        receipt
        for receipt in singleton_provenance.get("pages", [])
        if receipt.get("page_kind") == "stacked_3d_exact_support"
    ]
    if len(singleton_bar_pages) != 2:
        raise ValueError("singleton provenance must contain two normalized bar pages")
    for receipt in singleton_bar_pages:
        validate_share_sums(
            receipt.get("regime_normalized_share_sums", {}),
            context=f"singleton {receipt.get('sector')}",
        )

    macro_page_hashes = validate_landscape_pdf(args.macro_pdf, expected_pages=2)
    singleton_page_hashes = validate_landscape_pdf(
        args.singleton_pdf, expected_pages=4
    )
    build_tex(
        args.output_tex,
        macro_pdf=args.macro_pdf,
        singleton_pdf=args.singleton_pdf,
    )
    compile_tex(
        tex_path=args.output_tex,
        output_pdf=args.output_pdf,
        build_dir=args.build_dir,
    )
    combined_page_hashes = validate_landscape_pdf(
        args.output_pdf, expected_pages=6
    )

    provenance = {
        "schema": "paper_i_ra_normalized_operator_selection_combined_diagnostic_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "page_order": [
            "macro_weak_holstein",
            "macro_strong_holstein",
            "singleton_weak_holstein_exact_support",
            "singleton_strong_holstein_exact_support",
            "singleton_weak_holstein_co_support",
            "singleton_strong_holstein_co_support",
        ],
        "normalization": (
            "every displayed energy-drop contribution is divided by the total "
            "realized path drop of its own interaction regime"
        ),
        "sources": {
            "macro": {
                "pdf": repo_relative(args.macro_pdf),
                "pdf_sha256": sha256_file(args.macro_pdf),
                "page_count": 2,
                "page_content_sha256": macro_page_hashes,
                "provenance": repo_relative(args.macro_provenance),
                "provenance_sha256": sha256_file(args.macro_provenance),
            },
            "singleton": {
                "pdf": repo_relative(args.singleton_pdf),
                "pdf_sha256": sha256_file(args.singleton_pdf),
                "page_count": 4,
                "page_content_sha256": singleton_page_hashes,
                "provenance": repo_relative(args.singleton_provenance),
                "provenance_sha256": sha256_file(args.singleton_provenance),
            },
        },
        "outputs": {
            "pdf": {
                "path": repo_relative(args.output_pdf),
                "sha256": sha256_file(args.output_pdf),
                "page_count": 6,
                "page_content_sha256": combined_page_hashes,
            },
            "tex": {
                "path": repo_relative(args.output_tex),
                "sha256": sha256_file(args.output_tex),
            },
        },
    }
    write_json_atomic(args.provenance_json, provenance)
    verify_self_digest(args.provenance_json)
    print(
        json.dumps(
            {
                "pdf": str(args.output_pdf),
                "pdf_sha256": sha256_file(args.output_pdf),
                "provenance": str(args.provenance_json),
                "pages": 6,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
