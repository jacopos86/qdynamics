from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "MATH" / "build_math_from_md.py"
MATH_MD_PATH = Path(__file__).resolve().parents[1] / "MATH" / "Math.md"
SPEC = importlib.util.spec_from_file_location("build_math_from_md", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

normalize_markdown_for_pandoc = MODULE.normalize_markdown_for_pandoc
extract_continuation_markdown = MODULE.extract_continuation_markdown
extract_named_top_level_sections = MODULE.extract_named_top_level_sections
CONTINUATION_HEADINGS = MODULE.CONTINUATION_HEADINGS


def test_normalize_preserves_aligned_line_break_commands() -> None:
    text = """$$
\\begin{aligned}
a&:=b,\\[4pt]
c&:=d.
\\end{aligned}
$$
"""
    out = normalize_markdown_for_pandoc(text)
    assert "\\[4pt]" in out
    assert out.count("$$") == 2


def test_normalize_skips_fenced_code_blocks() -> None:
    text = """Outside ≈ text.

```python
example = r"\\(literal math delimiter\\) and ⚡"
```
"""
    out = normalize_markdown_for_pandoc(text)
    assert "Outside $\\approx$ text." in out
    assert 'example = r"\\(literal math delimiter\\) and ⚡"' in out


def test_normalize_skips_inline_code() -> None:
    text = "Use `\\(literal\\)` but normalize \\(a+b\\) and ≤ here."
    out = normalize_markdown_for_pandoc(text)
    assert "`\\(literal\\)`" in out
    assert "$a+b$" in out
    assert "$\\le$" in out


def test_extract_named_top_level_sections_keeps_requested_order() -> None:
    text = """# 1. One
alpha
# 2. Two
beta
# 3. Three
gamma
"""
    out = extract_named_top_level_sections(text, ["# 3. Three", "# 1. One"])
    top_level_lines = [line for line in out.splitlines() if line.startswith("# ")]
    assert top_level_lines == ["# 3. Three", "# 1. One"]
    assert "# 2. Two" not in out


def test_extract_continuation_markdown_matches_canonical_sections() -> None:
    manuscript = MATH_MD_PATH.read_text(encoding="utf-8")
    out = extract_continuation_markdown(manuscript)
    top_level_lines = [line for line in out.splitlines() if line.startswith("# ")]
    assert top_level_lines == CONTINUATION_HEADINGS
    assert out.lstrip().startswith(CONTINUATION_HEADINGS[0])
    assert "# 12. SPSA and Optimizer Semantics" not in out
    assert "Appendix A. Runtime Hardware-Dependent Parameters" not in out



def test_extract_continuation_section_11_uses_updated_canonical_wording() -> None:
    manuscript = MATH_MD_PATH.read_text(encoding="utf-8")
    out = extract_continuation_markdown(manuscript)
    assert "### 11.2.1 Core signal, append position, and one-coordinate local model" in out
    assert "\\Delta E_{1,\\mathrm{TR}}(r)" in out
    assert "### 11.2.1 Core signal, append position, and refit window" not in out
