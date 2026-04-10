#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


FENCED_CODE_BLOCK_RE = re.compile(r"(^```.*?^```[ \t]*$)", re.MULTILINE | re.DOTALL)
INLINE_CODE_RE = re.compile(r"(`[^`\n]*`)")
TOP_LEVEL_HEADING_RE = re.compile(r"(?m)^# .+$")


REPO_MATH_DIR = Path(__file__).resolve().parent
DEFAULT_MD = REPO_MATH_DIR / "Math.md"
DEFAULT_TEX = REPO_MATH_DIR / "Math.tex"
DEFAULT_PDF = REPO_MATH_DIR / "Math.pdf"
DEFAULT_TMP_MD = REPO_MATH_DIR / ".Math.build.normalized.md"
DEFAULT_CONTINUATION_TEX = REPO_MATH_DIR / "adaptive_selection_staged_continuation.tex"
DEFAULT_CONTINUATION_PDF = REPO_MATH_DIR / "adaptive_selection_staged_continuation.pdf"
DEFAULT_CONTINUATION_TMP_MD = REPO_MATH_DIR / ".adaptive_selection_staged_continuation.build.normalized.md"
CONTINUATION_HEADINGS = [
    "# 11. Adaptive Selection and Staged Continuation",
    "# 17A. Projective McLachlan Real-Time Dynamics and Exact-Forecast Checkpoint-Adaptive Snake Control",
    "# 17B. Projective McLachlan Real-Time Dynamics and Checkpoint-Adaptive Snake Control",
    "# 17C. Projective McLachlan Real-Time Geometry and Piecewise Beam-Adaptive Control",
]


# Built-in symbolic description: prose' = Normalize(prose),  code' = code
def _normalize_prose_segment(text: str) -> str:
    parts = INLINE_CODE_RE.split(text)
    out: list[str] = []
    plain_symbol_replacements = {
        "⚡": "[QPU]",
        "✅": "[OK]",
        "❌": "[X]",
        "≈": "$\\approx$",
        "≤": "$\\le$",
        "≥": "$\\ge$",
        "→": "$\\to$",
    }
    for part in parts:
        if part.startswith("`") and part.endswith("`"):
            out.append(part)
            continue
        part = part.replace(r"\(", "$").replace(r"\)", "$")
        part = re.sub(r"(?m)^\\\[\s*$", "$$", part)
        part = re.sub(r"(?m)^\\\]\s*$", "$$", part)
        for old, new in plain_symbol_replacements.items():
            part = part.replace(old, new)
        out.append(part)
    return "".join(out)


# Built-in symbolic description: Math.tex = Pandoc(Normalize(Math.md)),  Math.pdf = XeLaTeX(Math.tex)
def normalize_markdown_for_pandoc(text: str) -> str:
    """Normalize manuscript markdown into a pandoc/XeLaTeX-stable form."""
    parts = FENCED_CODE_BLOCK_RE.split(text)
    out: list[str] = []
    for part in parts:
        if part.startswith("```"):
            out.append(part)
        else:
            out.append(_normalize_prose_segment(part))
    return "".join(out)


def split_top_level_sections(markdown_text: str) -> dict[str, str]:
    """Return top-level # heading blocks keyed by exact heading line."""
    matches = list(TOP_LEVEL_HEADING_RE.finditer(markdown_text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = match.group(0).strip()
        if heading in sections:
            raise ValueError(f"Duplicate top-level heading encountered: {heading}")
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_text)
        sections[heading] = markdown_text[start:end].rstrip() + "\n"
    return sections


def extract_named_top_level_sections(markdown_text: str, headings: list[str]) -> str:
    """Extract exact top-level section blocks in the requested order."""
    sections = split_top_level_sections(markdown_text)
    missing = [heading for heading in headings if heading not in sections]
    if missing:
        raise KeyError(f"Missing top-level headings: {missing}")
    return "\n\n".join(sections[heading].rstrip() for heading in headings) + "\n"


def extract_continuation_markdown(markdown_text: str) -> str:
    """Build the standalone continuation extract from the canonical manuscript."""
    return extract_named_top_level_sections(markdown_text, CONTINUATION_HEADINGS)


# Built-in symbolic description: tmp_md = Normalize(markdown_text)
def write_normalized_markdown_text(markdown_text: str, dst_md: Path) -> None:
    dst_md.write_text(normalize_markdown_for_pandoc(markdown_text), encoding="utf-8")


# Built-in symbolic description: Math.tex = Pandoc(tmp_md)
def generate_tex_from_markdown(src_md: Path, out_tex: Path, *, source_label: str) -> None:
    subprocess.run(
        [
            "pandoc",
            "-f",
            "markdown+raw_tex",
            "-s",
            str(src_md),
            "-o",
            str(out_tex),
        ],
        check=True,
    )
    tex = out_tex.read_text(encoding="utf-8")
    banner = (
        f"% AUTO-GENERATED FROM {source_label} BY build_math_from_md.py.\n"
        "% Edit Math.md or the builder, then regenerate.\n"
    )
    if not tex.startswith(banner):
        tex = banner + tex
    out_tex.write_text(tex, encoding="utf-8")


# Built-in symbolic description: Math.pdf = XeLaTeX^2(Math.tex)
def compile_pdf_from_tex(tex_path: Path, cwd: Path) -> None:
    for _ in range(2):
        subprocess.run(
            [
                "xelatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                tex_path.name,
            ],
            cwd=str(cwd),
            check=True,
        )


def _validate_pdf_path(tex_path: Path, pdf_path: Path) -> None:
    expected_pdf_path = tex_path.with_suffix(".pdf")
    if pdf_path != expected_pdf_path:
        raise ValueError(f"--pdf must equal {expected_pdf_path} because xelatex writes beside the TeX file.")


def _ensure_required_executables(*, tex_only: bool) -> None:
    required_executables = ["pandoc"]
    if not tex_only:
        required_executables.append("xelatex")
    for exe in required_executables:
        if shutil.which(exe) is None:
            raise RuntimeError(f"Required executable not found on PATH: {exe}")


def build_artifact_from_markdown_text(
    markdown_text: str,
    *,
    normalized_md_path: Path,
    tex_path: Path,
    pdf_path: Path,
    source_label: str,
    tex_only: bool,
    keep_normalized_md: bool,
) -> None:
    write_normalized_markdown_text(markdown_text, normalized_md_path)
    generate_tex_from_markdown(normalized_md_path, tex_path, source_label=source_label)

    if not tex_only:
        compile_pdf_from_tex(tex_path, tex_path.parent)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Expected PDF was not produced: {pdf_path}")

    if not keep_normalized_md and normalized_md_path.exists():
        normalized_md_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate MATH/Math.tex and MATH/adaptive_selection_staged_continuation.tex, "
            "and optionally their PDFs, from MATH/Math.md."
        )
    )
    parser.add_argument("--md", type=Path, default=DEFAULT_MD, help="Input markdown manuscript.")
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX, help="Output TeX manuscript.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Expected manuscript PDF path; must match the TeX stem because xelatex writes beside the TeX file.")
    parser.add_argument("--normalized-md", type=Path, default=DEFAULT_TMP_MD, help="Temporary normalized markdown path for the full manuscript.")
    parser.add_argument("--continuation-tex", type=Path, default=DEFAULT_CONTINUATION_TEX, help="Output TeX path for the continuation extract.")
    parser.add_argument("--continuation-pdf", type=Path, default=DEFAULT_CONTINUATION_PDF, help="Expected PDF path for the continuation extract; must match the continuation TeX stem.")
    parser.add_argument("--continuation-normalized-md", type=Path, default=DEFAULT_CONTINUATION_TMP_MD, help="Temporary normalized markdown path for the continuation extract.")
    parser.add_argument("--skip-continuation", action="store_true", help="Only build Math.tex/Math.pdf; skip the continuation extract.")
    parser.add_argument("--tex-only", action="store_true", help="Regenerate TeX outputs only; skip PDF compilation.")
    parser.add_argument("--keep-normalized-md", action="store_true", help="Keep the normalized markdown build helper files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    md_path = args.md.resolve()
    tex_path = args.tex.resolve()
    pdf_path = args.pdf.resolve()
    normalized_md_path = args.normalized_md.resolve()
    continuation_tex_path = args.continuation_tex.resolve()
    continuation_pdf_path = args.continuation_pdf.resolve()
    continuation_normalized_md_path = args.continuation_normalized_md.resolve()

    if not md_path.exists():
        raise FileNotFoundError(f"Missing markdown manuscript: {md_path}")

    _validate_pdf_path(tex_path, pdf_path)
    if not args.skip_continuation:
        _validate_pdf_path(continuation_tex_path, continuation_pdf_path)

    _ensure_required_executables(tex_only=args.tex_only)

    manuscript_markdown = md_path.read_text(encoding="utf-8")
    build_artifact_from_markdown_text(
        manuscript_markdown,
        normalized_md_path=normalized_md_path,
        tex_path=tex_path,
        pdf_path=pdf_path,
        source_label="MATH/Math.md",
        tex_only=args.tex_only,
        keep_normalized_md=args.keep_normalized_md,
    )

    if not args.skip_continuation:
        continuation_markdown = extract_continuation_markdown(manuscript_markdown)
        build_artifact_from_markdown_text(
            continuation_markdown,
            normalized_md_path=continuation_normalized_md_path,
            tex_path=continuation_tex_path,
            pdf_path=continuation_pdf_path,
            source_label="MATH/Math.md (sections 11, 17A, 17B, 17C)",
            tex_only=args.tex_only,
            keep_normalized_md=args.keep_normalized_md,
        )

    print(f"Wrote {tex_path}")
    if not args.tex_only:
        print(f"Wrote {pdf_path}")
    if not args.skip_continuation:
        print(f"Wrote {continuation_tex_path}")
        if not args.tex_only:
            print(f"Wrote {continuation_pdf_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"Build step failed: {exc}", file=sys.stderr)
        raise
