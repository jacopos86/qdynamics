#!/usr/bin/env python3
"""Export the current Paper I manuscript into a GPT-Pro review packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TEX = REPO_ROOT / "MATH/paper_details/Paper_I.tex"
DEFAULT_PDF = REPO_ROOT / "MATH/paper_details/Paper_I.pdf"
DEFAULT_OUT_DIR = REPO_ROOT / "prompt-exports/paper-i-gpt-pro-review"

SUPPORT_DOCS = [
    "MATH/paper_facing/shared/journal_math_skill_supplement.md",
    "MATH/paper_facing/shared/claim_source_types.md",
    "MATH/paper_facing/shared/repo_to_journal_translation.md",
    "MATH/paper_facing/shared/ai_manuscript_style_guardrails.md",
    "MATH/paper_facing/paper_I_static_scaffold/claim_boundaries.md",
]

LOCATOR_PATTERNS = [
    re.compile(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{"),
    re.compile(r"\\(?:caption|captionof)\b"),
    re.compile(r"\\label\{"),
    re.compile(r"\\(?:placeholder|tentative|structlabel)\{"),
    re.compile(r"\b(?:TODO|DO WE|xxx|XX%|L=2 only)\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-path", type=Path, default=DEFAULT_TEX)
    parser.add_argument("--pdf-path", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--timestamp", help="Override iteration timestamp for tests.")
    parser.add_argument(
        "--no-support-docs",
        action="store_true",
        help="Do not embed support-doc excerpts in the GPT-Pro prompt.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\S+\b", text))


def run_pdftotext(pdf_path: Path) -> tuple[str, str]:
    binary = shutil.which("pdftotext")
    if not binary:
        return "", "pdftotext_not_found"

    proc = subprocess.run(
        [binary, "-layout", str(pdf_path), "-"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return "", f"pdftotext_failed: {proc.stderr.strip()}"
    return proc.stdout.strip() + "\n", "pdftotext_layout"


def tex_to_plain_fallback(tex_path: Path) -> str:
    text = tex_path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"(?m)^%.*$", "", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", "\n", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def strip_latex_for_locator(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    if len(line) > 220:
        return line[:217] + "..."
    return line


def build_locator_map(tex_path: Path) -> str:
    lines = tex_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = [
        "# Paper I TeX Locator Map",
        "",
        "Source: `MATH/paper_details/Paper_I.tex`",
        "",
        "Use these line anchors to map GPT-Pro extracted-text findings back to TeX.",
        "",
    ]

    for line_no, line in enumerate(lines, start=1):
        if any(pattern.search(line) for pattern in LOCATOR_PATTERNS):
            out.append(f"- L{line_no}: `{strip_latex_for_locator(line)}`")

    out.append("")
    return "\n".join(out)


def read_support_docs(include_docs: bool) -> str:
    if not include_docs:
        return "Support-doc embedding disabled for this packet.\n"

    chunks: list[str] = []
    for rel_path in SUPPORT_DOCS:
        path = REPO_ROOT / rel_path
        if not path.exists():
            chunks.append(f"## {rel_path}\n\nMissing locally.\n")
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        chunks.append(f"## {rel_path}\n\n{text}\n")
    return "\n".join(chunks).strip() + "\n"


def build_prompt(
    paper_text: str,
    locator_map: str,
    support_docs: str,
    *,
    tex_hash: str,
    pdf_hash: str,
) -> str:
    return f"""# GPT-Pro Critical Review Request: Paper I

## What You Must Assume

- You do not have repo access.
- You do not have attachments.
- You must reason only from this prompt.
- The manuscript text below is extracted from the current rendered Paper I PDF.
- A separate repo agent will edit the TeX only after the user approves candidate changes.

## Objective

Act as a critical journal referee and careful scientific editor for Paper I, "Geometry- and Cost-Aware ADAPT Ansatz Construction for Mixed Fermion--Boson Hamiltonians." Identify the highest-impact fixes needed before the manuscript can be revised by a repo agent.

Do not rewrite the full paper. Do not invent numbers, citations, benchmark results, or missing evidence. Do not delete draft placeholders simply because they look unfinished; mark whether they need locked evidence or explicit user approval.

## Review Priorities

Prioritize:

1. Correctness of mathematical and physical claims.
2. Unsupported numerical or percentage claims.
3. Paper I / Paper II / Paper III scope leakage.
4. ADAPT terminology problems, especially unnecessary controller/scaffold language in Paper I.
5. Overclaims about novelty, hardware readiness, mixed fermion--boson scope, or comparator dominance.
6. Draft artifacts that obstruct a journal referee's reading.
7. Local prose fixes that preserve the author's technical voice.

## Required Output Format

Return exactly these sections:

1. `Referee Summary`
   - 5 to 10 bullets.
   - Put the strongest paper-level concerns first.

2. `Candidate Fix List`
   - A table with columns:
     - `ID`
     - `Priority`
     - `Location`
     - `Problem`
     - `Repo-agent edit instruction`
     - `Evidence or gate needed`
     - `User approval note`
   - Use `P0`, `P1`, or `P2` priority.
   - Keep each repo-agent edit instruction local and actionable.
   - Quote only short phrases when useful; do not provide full replacement sections.

3. `Do Not Auto-Apply`
   - List suggestions requiring run evidence, table/source-map validation, citation verification, user judgment, or broad restructuring.

4. `Possible Reviewer Questions`
   - Concise questions a critical referee might still ask.

## Source Identity

- TeX SHA256: `{tex_hash}`
- PDF SHA256: `{pdf_hash}`

## Paper I Review Guardrails

{support_docs}

## TeX Locator Map

{locator_map}

## Extracted Manuscript Text: paper1.txt

```text
{paper_text.rstrip()}
```
"""


def main() -> int:
    args = parse_args()
    tex_path = resolve_path(args.tex_path)
    pdf_path = resolve_path(args.pdf_path)
    out_root = resolve_path(args.out_dir)

    if not tex_path.exists():
        print(f"ERROR: TeX source not found: {tex_path}", file=sys.stderr)
        return 2
    if not pdf_path.exists():
        print(f"ERROR: PDF source not found: {pdf_path}", file=sys.stderr)
        return 2

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    iteration_dir = out_root / timestamp
    suffix = 1
    while iteration_dir.exists():
        suffix += 1
        iteration_dir = out_root / f"{timestamp}-{suffix:02d}"
    iteration_dir.mkdir(parents=True, exist_ok=False)

    paper_text, extraction_mode = run_pdftotext(pdf_path)
    if not paper_text.strip():
        paper_text = tex_to_plain_fallback(tex_path)
        extraction_mode = f"{extraction_mode}; tex_plain_fallback"

    locator_map = build_locator_map(tex_path)
    support_docs = read_support_docs(not args.no_support_docs)
    tex_hash = sha256_file(tex_path)
    pdf_hash = sha256_file(pdf_path)
    prompt = build_prompt(
        paper_text,
        locator_map,
        support_docs,
        tex_hash=tex_hash,
        pdf_hash=pdf_hash,
    )

    paper_path = iteration_dir / "paper1.txt"
    prompt_path = iteration_dir / "gpt-pro-review-prompt.md"
    locator_path = iteration_dir / "tex-locator-map.md"
    manifest_path = iteration_dir / "manifest.json"

    paper_path.write_text(paper_text, encoding="utf-8")
    locator_path.write_text(locator_map, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")

    manifest = {
        "schema": "paper_i_gpt_pro_review_packet_v1",
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(REPO_ROOT),
        "tex_path": str(tex_path),
        "tex_sha256": tex_hash,
        "pdf_path": str(pdf_path),
        "pdf_sha256": pdf_hash,
        "paper1_txt_path": str(paper_path),
        "paper1_txt_sha256": sha256_file(paper_path),
        "paper1_word_count": word_count(paper_text),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "prompt_word_count": word_count(prompt),
        "locator_map_path": str(locator_path),
        "locator_map_sha256": sha256_file(locator_path),
        "extraction_mode": extraction_mode,
        "support_docs": [] if args.no_support_docs else SUPPORT_DOCS,
        "support_docs_sha256": sha256_text(support_docs),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Iteration directory: {iteration_dir}")
    print(f"paper1.txt: {paper_path}")
    print(f"GPT-Pro prompt: {prompt_path}")
    print(f"Locator map: {locator_path}")
    print(f"Manifest: {manifest_path}")
    print(f"paper1.txt words: {manifest['paper1_word_count']}")
    print(f"prompt words: {manifest['prompt_word_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
