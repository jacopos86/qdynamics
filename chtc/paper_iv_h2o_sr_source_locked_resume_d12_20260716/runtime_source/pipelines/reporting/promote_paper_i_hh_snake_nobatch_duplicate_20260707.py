#!/usr/bin/env python3
"""Create a duplicate Paper-I manuscript with HH SNAKE no-batch rows promoted.

This is intentionally a one-off, validation-heavy promotion helper.  It leaves
the active Paper-I source untouched and edits only the duplicate TeX file.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_TEX = ROOT / "MATH/paper_details/static_adapt_paper_I.tex"
DUP_TEX = ROOT / "MATH/paper_details/static_adapt_paper_I_snake_nobatch_promoted_20260707.tex"
FIG_DIR = ROOT / "MATH/paper_details/figures/powell_snake_nobatch_promoted_20260707"
SUPPORT_MANIFEST = (
    ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/paper_i_hh_snake_nobatch_duplicate_promotion_20260707.json"
)

NEW_REPORT_DIR = ROOT / "output/pdf/paper_i_hh_powell_visible_batchroute_nobatch_vs_paper1_snake_20260707"
NEW_CSV = NEW_REPORT_DIR / "paper_i_hh_powell_visible_batchroute_nobatch_vs_paper1_snake_20260707_provenance.csv"
NEW_JSON = NEW_REPORT_DIR / "paper_i_hh_powell_visible_batchroute_nobatch_vs_paper1_snake_20260707_provenance.json"
NEW_PDF = NEW_REPORT_DIR / "paper_i_hh_powell_visible_batchroute_nobatch_vs_paper1_snake_20260707.pdf"

OLD_REF_ROOT = Path("/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3")
OLD_SUPPORT_CSV = (
    OLD_REF_ROOT
    / "output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/"
    "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.csv"
)
OLD_SUPPORT_JSON = (
    OLD_REF_ROOT
    / "output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/"
    "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.json"
)
OLD_SUPPORT_PDF = (
    OLD_REF_ROOT
    / "output/pdf/paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630/"
    "paper_i_hh_fullmeta_singleton_symmetry_chtc_current_20260630_powell_pool_exposure_support.pdf"
)
BUILD_ASSET_RESTORES = [
    (
        OLD_REF_ROOT / "MATH/paper_details/figures/static_adapt_selector_flow_attached.png",
        ROOT / "MATH/paper_details/figures/static_adapt_selector_flow_attached.png",
    ),
    (
        OLD_REF_ROOT / "MATH/paper_details/figures/paper_i_hubbard_weak_supportive_diagnostic_20260610.png",
        ROOT / "MATH/paper_details/figures/paper_i_hubbard_weak_supportive_diagnostic_20260610.png",
    ),
    (
        OLD_REF_ROOT / "MATH/paper_details/figures/paper_i_beam_tree_example.png",
        ROOT / "MATH/paper_details/figures/paper_i_beam_tree_example.png",
    ),
    (
        OLD_REF_ROOT / "MATH/paper_details/figures/paper_i_hubbard_weak_u0p25_append_geo_snake_overlay_20260621.png",
        ROOT / "MATH/paper_details/figures/paper_i_hubbard_weak_u0p25_append_geo_snake_overlay_20260621.png",
    ),
    (
        OLD_REF_ROOT / "MATH/paper_details/figures/hubbard_weak_full_noise_fixed_depth8_error_vs_iteration_20260613.pdf",
        ROOT / "MATH/paper_details/figures/hubbard_weak_full_noise_fixed_depth8_error_vs_iteration_20260613.pdf",
    ),
    (
        OLD_REF_ROOT / "MATH/paper_details/figures/hubbard_strong_full_noise_fixed_depth8_error_vs_iteration_20260613.pdf",
        ROOT / "MATH/paper_details/figures/hubbard_strong_full_noise_fixed_depth8_error_vs_iteration_20260613.pdf",
    ),
    (
        OLD_REF_ROOT / "MATH/paper_details/figures/hh_ed_cutoff_log_sensitivity_20260614.pdf",
        ROOT / "MATH/paper_details/figures/hh_ed_cutoff_log_sensitivity_20260614.pdf",
    ),
    (
        OLD_REF_ROOT / "output/pdf/paper_i_true_strong_replacement_20260613_hubbard_u8_strong.pdf",
        ROOT / "MATH/paper_details/output/pdf/paper_i_true_strong_replacement_20260613_hubbard_u8_strong.pdf",
    ),
    (
        OLD_REF_ROOT / "output/pdf/paper_i_data_analysis_spin_boson_weak_repeat_enabled_same_cutoff_error_vs_iteration_20260610.pdf",
        ROOT / "MATH/paper_details/output/pdf/paper_i_data_analysis_spin_boson_weak_repeat_enabled_same_cutoff_error_vs_iteration_20260610.pdf",
    ),
    (
        OLD_REF_ROOT / "output/pdf/paper_i_data_analysis_spin_boson_strong_repeat_enabled_same_cutoff_error_vs_iteration_20260610.pdf",
        ROOT / "MATH/paper_details/output/pdf/paper_i_data_analysis_spin_boson_strong_repeat_enabled_same_cutoff_error_vs_iteration_20260610.pdf",
    ),
]

REGIMES = [
    ("weak_weak", "weak-weak", "weak--weak", "(U/t,\\lambda)=(0.25,0.25)", 2),
    ("intermediate_weak", "intermediate-weak", "intermediate--weak", "(U/t,\\lambda)=(1.25,0.25)", 2),
    ("strong_weak", "strong-weak", "strong--weak", "(U/t,\\lambda)=(8,0.25)", 2),
    ("weak_strong", "weak-strong", "weak--strong", "(U/t,\\lambda)=(0.25,1.25)", 4),
    ("intermediate_strong", "intermediate-strong", "intermediate--strong", "(U/t,\\lambda)=(1.25,1.25)", 4),
    ("strong_strong", "strong-strong", "strong--strong", "(U/t,\\lambda)=(8,1.25)", 4),
]
REGIME_BY_UNDERSCORE = {r[0]: r for r in REGIMES}
REGIME_BY_HYPHEN = {r[1]: r for r in REGIMES}
PLOT_TITLES = {
    "weak_weak": r"weak-weak: $U/t=0.25$, $\lambda=0.25$, $M=2$",
    "intermediate_weak": r"intermediate-weak: $U/t=1.25$, $\lambda=0.25$, $M=2$",
    "strong_weak": r"strong-weak: $U/t=8$, $\lambda=0.25$, $M=2$",
    "weak_strong": r"weak-strong: $U/t=0.25$, $\lambda=1.25$, $M=4$",
    "intermediate_strong": r"intermediate-strong: $U/t=1.25$, $\lambda=1.25$, $M=4$",
    "strong_strong": r"strong-strong: $U/t=8$, $\lambda=1.25$, $M=4$",
}

ROLE_SPECS = {
    "snake_nobatch": {
        "display": "SNAKE",
        "color": "#E45756",
        "marker": "*",
        "linestyle": "-",
    },
    "geo_macro_c": {
        "display": "Geo macro",
        "color": "#54A24B",
        "marker": "^",
        "linestyle": ":",
    },
    "append_macro_c": {
        "display": "Append macro",
        "color": "#4C78A8",
        "marker": "o",
        "linestyle": ":",
    },
    "append_singleton_b1": {
        "display": "Append singleton",
        "color": "#4C78A8",
        "marker": "o",
        "linestyle": "-",
    },
}


@dataclass(frozen=True)
class SnakeRow:
    regime_key: str
    regime_hyphen: str
    k_pl: int
    d_pl: int
    abs_delta_e: float
    one_minus_f: float | None
    fidelity_status: str
    n2q: int
    d2q: int
    dc: int
    s_alg: int
    result_json: str
    result_sha256: str
    qiskit_cost_source: str
    s_alg_source: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"required path missing: {path}")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def as_int(value: Any, field: str) -> int:
    if value in (None, ""):
        raise SystemExit(f"missing integer field {field}")
    return int(round(float(value)))


def as_float(value: Any, field: str) -> float:
    if value in (None, ""):
        raise SystemExit(f"missing float field {field}")
    out = float(value)
    if not math.isfinite(out):
        raise SystemExit(f"non-finite float field {field}: {value}")
    return out


def sci(value: float) -> str:
    return f"{value:.3e}"


def fmt_int(value: int) -> str:
    return f"{value:,d}"


def fmt_one_minus_f(value: float | None) -> str:
    return "--" if value is None else sci(value)


def parse_json_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def snake_history_xy(result_path: Path) -> tuple[list[float], list[float]]:
    payload = json.loads(result_path.read_text())
    history = ((payload.get("adapt_vqe") or {}).get("history") or payload.get("history") or [])
    if not history:
        raise SystemExit(f"missing history in {result_path}")
    x: list[float] = []
    y: list[float] = []
    first = history[0]
    if first.get("delta_abs_prev") is not None:
        x.append(0.0)
        y.append(float(first["delta_abs_prev"]))
    for idx, row in enumerate(history, start=1):
        err = row.get("delta_abs_current")
        if err is None:
            continue
        depth = row.get("depth", row.get("iteration", idx))
        x.append(float(depth))
        y.append(float(err))
    return x, y


def load_snake_rows() -> dict[str, SnakeRow]:
    rows = load_csv(NEW_CSV)
    selected = [
        row
        for row in rows
        if row.get("row_group") == "diagnostic_rerun"
        and row.get("display_label") == "no batch"
        and row.get("optimizer") == "POWELL"
    ]
    by_regime: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_regime.setdefault(row["regime"], []).append(row)
    missing = [key for key, *_rest in REGIMES if key not in by_regime]
    duplicate = {key: rows for key, rows in by_regime.items() if len(rows) != 1}
    if missing or duplicate or len(by_regime) != 6:
        raise SystemExit(f"invalid no-batch SNAKE row set missing={missing} duplicate={duplicate}")

    out: dict[str, SnakeRow] = {}
    for key, hyphen, *_ in REGIMES:
        row = by_regime[key][0]
        if row.get("batching_enabled") != "False":
            raise SystemExit(f"{key}: expected batching_enabled=False, got {row.get('batching_enabled')}")
        if row.get("adapt_beam_lambda") != "0.005":
            raise SystemExit(f"{key}: expected adapt_beam_lambda=0.005, got {row.get('adapt_beam_lambda')}")
        if row.get("qiskit_cost_status") not in {"ok", "done"}:
            raise SystemExit(f"{key}: qiskit cost status not ok: {row.get('qiskit_cost_status')}")
        if row.get("S_alg_status") != "ok":
            raise SystemExit(f"{key}: S_alg status not ok: {row.get('S_alg_status')}")
        fidelity_value = row.get("fidelity")
        one_minus_f: float | None = None
        fidelity_status = "unavailable"
        if fidelity_value not in (None, ""):
            f_val = float(fidelity_value)
            if math.isfinite(f_val):
                one_minus_f = max(0.0, 1.0 - f_val)
                fidelity_status = "computed_from_selected_prefix_fidelity"
        result_json = row["result_json"]
        result_path = parse_json_path(result_json)
        require(result_path)
        out[key] = SnakeRow(
            regime_key=key,
            regime_hyphen=hyphen,
            k_pl=as_int(row.get("plateau_k"), "plateau_k"),
            d_pl=as_int(row.get("plateau_d_ans"), "plateau_d_ans"),
            abs_delta_e=as_float(row.get("plateau_abs_delta_e"), "plateau_abs_delta_e"),
            one_minus_f=one_minus_f,
            fidelity_status=fidelity_status,
            n2q=as_int(row.get("N2q"), "N2q"),
            d2q=as_int(row.get("D2q"), "D2q"),
            dc=as_int(row.get("D_c"), "D_c"),
            s_alg=as_int(row.get("S_alg"), "S_alg"),
            result_json=result_json,
            result_sha256=row.get("result_sha256") or sha256(result_path),
            qiskit_cost_source=row.get("qiskit_cost_source") or "",
            s_alg_source=row.get("S_alg_source") or "",
        )
    return out


def support_rows_for_plots() -> dict[tuple[str, str], dict[str, str]]:
    rows = load_csv(OLD_SUPPORT_CSV)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        role = row.get("role_key")
        regime = row.get("regime")
        if role in {"geo_macro_c", "append_macro_c", "append_singleton_b1"}:
            if regime not in REGIME_BY_HYPHEN:
                raise SystemExit(f"unexpected support regime {regime}")
            out[(REGIME_BY_HYPHEN[regime][0], role)] = row
    expected = [(key, role) for key, *_ in REGIMES for role in ("geo_macro_c", "append_macro_c", "append_singleton_b1")]
    missing = [item for item in expected if item not in out]
    if missing:
        raise SystemExit(f"missing comparator rows: {missing}")
    return out


def parse_points(raw: str) -> tuple[list[float], list[float]]:
    pts = json.loads(raw)
    x = [float(p[0]) for p in pts]
    y = [float(p[1]) for p in pts]
    return x, y


def plot_regime(key: str, snake_row: SnakeRow, support: dict[tuple[str, str], dict[str, str]]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.15, 1.95), dpi=220)

    sx, sy = snake_history_xy(parse_json_path(snake_row.result_json))
    spec = ROLE_SPECS["snake_nobatch"]
    ax.plot(sx, sy, color=spec["color"], linestyle=spec["linestyle"], linewidth=1.3, label=spec["display"])
    ax.scatter(
        [snake_row.k_pl],
        [snake_row.abs_delta_e],
        color=spec["color"],
        marker=spec["marker"],
        s=38,
        zorder=5,
    )

    for role in ("geo_macro_c", "append_macro_c", "append_singleton_b1"):
        row = support[(key, role)]
        x, y = parse_points(row["trajectory_points_json"])
        spec = ROLE_SPECS[role]
        ax.plot(x, y, color=spec["color"], linestyle=spec["linestyle"], linewidth=1.0, label=spec["display"])
        k = int(row["selected_prefix_k"])
        ax.scatter(
            [k],
            [float(row["abs_delta_e"])],
            color=spec["color"],
            marker=spec["marker"],
            s=28,
            zorder=4,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"ADAPT iteration $k$", fontsize=8)
    ax.set_ylabel(r"$|\Delta E|$", fontsize=8)
    ax.set_title(PLOT_TITLES[key], fontsize=7.0, pad=2)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(True, which="both", linewidth=0.25, alpha=0.35)
    ax.legend(fontsize=5.5, loc="best", frameon=True)
    fig.tight_layout(pad=0.12)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"paper_i_hh_snake_nobatch_promoted_20260707__{key}.png"
    pdf = FIG_DIR / f"paper_i_hh_snake_nobatch_promoted_20260707__{key}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": rel(png), "png_sha256": sha256(png), "pdf": rel(pdf), "pdf_sha256": sha256(pdf)}


def snake_tex_row(row: SnakeRow) -> str:
    return (
        "SNAKE & "
        f"{row.k_pl} & {sci(row.abs_delta_e)} & {fmt_one_minus_f(row.one_minus_f)} & "
        f"{fmt_int(row.n2q)} & {fmt_int(row.d2q)} & {fmt_int(row.dc)} & {fmt_int(row.s_alg)} \\\\"
    )


def replace_hh_provenance(text: str, manifest: dict[str, Any]) -> str:
    start = text.index("% BEGIN_TABLE_III_DEPTH_PROVENANCE_HYGIENE_20260609")
    end_marker = "% END_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_SNAKE_RUNTIME_SPLIT_TRACE_20260705"
    end = text.index(end_marker, start) + len(end_marker)
    block_json = json.dumps(manifest, indent=2, sort_keys=True)
    commented = "\n".join("% " + line for line in block_json.splitlines())
    block = (
        "% BEGIN_MACHINE_READABLE_HH_SNAKE_NOBATCH_DUPLICATE_PROMOTION_20260707\n"
        + commented
        + "\n% END_MACHINE_READABLE_HH_SNAKE_NOBATCH_DUPLICATE_PROMOTION_20260707"
    )
    return text[:start] + block + text[end:]


def update_duplicate_tex(snake_rows: dict[str, SnakeRow], plot_files: dict[str, dict[str, str]], manifest: dict[str, Any]) -> None:
    shutil.copy2(SOURCE_TEX, DUP_TEX)
    text = DUP_TEX.read_text()
    text = replace_hh_provenance(text, manifest)
    old_paragraph = (
        "For the Hubbard--Holstein comparison below, all displayed adaptive rows use Powell refits in the pool-exposure diagnostic. "
        "SNAKE uses the native staged hard-guard singleton replacement condition; Geo-ADAPT is shown in its macro-generator form, while append-only ADAPT is shown with macro-generator and common Phase-0 hard-guard singleton-pool exposure. "
        "Each row is evaluated at the plotted selected prefix \\(k_{\\rm pl}\\); circuit resource cells are Qiskit-compiled active-ansatz costs at the same prefix, and the displayed error is the same-cutoff quantity \\(|E_{\\rm ADAPT}-E_{\\rm ED}|\\). "
        "The Geo-ADAPT singleton-pool rows are non-competitive in this diagnostic: the macro-generator Geo-ADAPT row has equal displayed accuracy in weak--weak, lower same-cutoff error in the other five regimes, and approximately one to two orders of magnitude lower \\(S\\) throughout. "
        "We therefore omit Geo-ADAPT singleton rows from the rendered comparison and retain them only in source comments."
    )
    new_paragraph = (
        "For the Hubbard--Holstein comparison below, the SNAKE rows are updated in this duplicate from the local Powell no-batch rerun with the metric-prune route and beam weight \\(0.005\\). "
        "The Geo-ADAPT and append-only ADAPT rows are retained from the existing Powell pool-exposure diagnostic: Geo-ADAPT is shown in its macro-generator form, while append-only ADAPT is shown with macro-generator and common Phase-0 hard-guard singleton-pool exposure. "
        "Each row is evaluated at the plotted selected prefix \\(k_{\\rm pl}\\); circuit resource cells are Qiskit-compiled active-ansatz costs at the same prefix, and the displayed error is the same-cutoff quantity \\(|E_{\\rm ADAPT}-E_{\\rm ED}|\\). "
        "The Geo-ADAPT singleton-pool rows remain omitted from the rendered comparison and retained only in source comments."
    )
    if old_paragraph not in text:
        raise SystemExit("HH paragraph anchor not found")
    text = text.replace(old_paragraph, new_paragraph, 1)

    for key, _hyphen, tex_label, *_ in REGIMES:
        figure_old = re.compile(
            r"\\includegraphics\[width=\\columnwidth\]\{figures/powell_pool_exposure_20260702/[^}]+__"
            + re.escape(key)
            + r"\.png\}"
        )
        figure_new = (
            "\\includegraphics[width=\\columnwidth]{"
            + plot_files[key]["png"].replace("MATH/paper_details/", "")
            + "}"
        )
        text, count = figure_old.subn(lambda _match, replacement=figure_new: replacement, text, count=1)
        if count != 1:
            raise SystemExit(f"{key}: includegraphics replacement count={count}")

        table_regex = re.compile(
            r"(\\textit\{"
            + re.escape(tex_label)
            + r":.*?\\colrule\n)(SNAKE & [^\n]+ \\\\)",
            re.S,
        )
        text, count = table_regex.subn(lambda m, row=snake_rows[key]: m.group(1) + snake_tex_row(row), text, count=1)
        if count != 1:
            raise SystemExit(f"{key}: SNAKE row replacement count={count}")

    layout_anchor = (
        "\\noindent\\begin{minipage}{\\columnwidth}\n"
        "\\begin{center}\n"
        "\\includegraphics[width=\\columnwidth]{"
        + plot_files["intermediate_strong"]["png"].replace("MATH/paper_details/", "")
        + "}"
    )
    if layout_anchor not in text:
        raise SystemExit("intermediate-strong layout anchor not found")
    text = text.replace(
        layout_anchor,
        "% Duplicate-only layout guard: keep the last two HH plot/table pairs in the same visual column.\n"
        "\\newpage\n"
        + layout_anchor,
        1,
    )

    DUP_TEX.write_text(text)


def validate_duplicate(snake_rows: dict[str, SnakeRow], original_sha: str) -> dict[str, Any]:
    if sha256(SOURCE_TEX) != original_sha:
        raise SystemExit("original Paper-I TeX hash changed")
    original_text = SOURCE_TEX.read_text()
    text = DUP_TEX.read_text()
    validation: dict[str, Any] = {
        "source_tex_hash_unchanged": True,
        "rows": {},
        "non_snake_comparator_rows_identical_to_source_tex": {},
    }
    for key, _hyphen, tex_label, *_ in REGIMES:
        expected = snake_tex_row(snake_rows[key])
        table_regex = re.compile(
            r"\\textit\{"
            + re.escape(tex_label)
            + r":.*?\\colrule\n("
            + re.escape(expected)
            + r")",
            re.S,
        )
        count = len(table_regex.findall(text))
        if count != 1:
            raise SystemExit(f"{key}: expected SNAKE row not found exactly once, count={count}: {expected}")
        if f"paper_i_hh_snake_nobatch_promoted_20260707__{key}.png" not in text:
            raise SystemExit(f"{key}: promoted figure path missing")
        validation["rows"][key] = {
            "tex_row": expected,
            "validated": True,
        }
        block_re = re.compile(r"\\textit\{" + re.escape(tex_label) + r":.*?\\end\{tabular\}", re.S)
        original_block = block_re.search(original_text)
        duplicate_block = block_re.search(text)
        if original_block is None or duplicate_block is None:
            raise SystemExit(f"{key}: unable to validate non-SNAKE comparator rows")

        def non_snake_rows(block: re.Match[str]) -> list[str]:
            return [
                line.strip()
                for line in block.group(0).splitlines()
                if "&" in line
                and not line.strip().startswith("SNAKE &")
                and not line.strip().startswith("Method &")
            ]

        original_rows = non_snake_rows(original_block)
        duplicate_rows = non_snake_rows(duplicate_block)
        if original_rows != duplicate_rows:
            raise SystemExit(f"{key}: non-SNAKE comparator rows changed")
        validation["non_snake_comparator_rows_identical_to_source_tex"][key] = True
    if text.count("BEGIN_MACHINE_READABLE_HH_SNAKE_NOBATCH_DUPLICATE_PROMOTION_20260707") != 1:
        raise SystemExit("new machine-readable block missing or duplicated")
    return validation


def main() -> None:
    for path in [SOURCE_TEX, NEW_CSV, NEW_JSON, NEW_PDF, OLD_SUPPORT_CSV, OLD_SUPPORT_JSON, OLD_SUPPORT_PDF]:
        require(path)
    for source, destination in BUILD_ASSET_RESTORES:
        require(source)
        require(destination)
    original_sha = sha256(SOURCE_TEX)
    snake_rows = load_snake_rows()
    support = support_rows_for_plots()
    plot_files = {key: plot_regime(key, snake_rows[key], support) for key, *_ in REGIMES}

    source_hashes = {
        rel(SOURCE_TEX): original_sha,
        rel(NEW_CSV): sha256(NEW_CSV),
        rel(NEW_JSON): sha256(NEW_JSON),
        rel(NEW_PDF): sha256(NEW_PDF),
        str(OLD_SUPPORT_CSV): sha256(OLD_SUPPORT_CSV),
        str(OLD_SUPPORT_JSON): sha256(OLD_SUPPORT_JSON),
        str(OLD_SUPPORT_PDF): sha256(OLD_SUPPORT_PDF),
    }
    changed_snake_cells = {
        REGIME_BY_UNDERSCORE[key][1]: {
            "k_pl": row.k_pl,
            "d_pl": row.d_pl,
            "abs_delta_e": row.abs_delta_e,
            "one_minus_f": row.one_minus_f,
            "fidelity_status": row.fidelity_status,
            "N2q": row.n2q,
            "D2q": row.d2q,
            "Dc": row.dc,
            "S_alg": row.s_alg,
            "result_json": row.result_json,
            "result_sha256": row.result_sha256,
            "qiskit_cost_source": row.qiskit_cost_source,
            "S_alg_source": row.s_alg_source,
        }
        for key, row in snake_rows.items()
    }
    manifest: dict[str, Any] = {
        "schema": "paper_i_hh_snake_nobatch_duplicate_promotion_v1",
        "updated_date": "2026-07-07",
        "scope": "duplicate_only; SNAKE rows and HH plots updated; original manuscript untouched",
        "source_tex": rel(SOURCE_TEX),
        "duplicate_tex": rel(DUP_TEX),
        "source_csv": rel(NEW_CSV),
        "source_json": rel(NEW_JSON),
        "source_pdf": rel(NEW_PDF),
        "retained_comparator_source_csv": str(OLD_SUPPORT_CSV),
        "retained_comparator_source_json": str(OLD_SUPPORT_JSON),
        "source_hashes": source_hashes,
        "optimizer": "POWELL",
        "row_policy": "promote only latest local no-batch SNAKE rows; preserve existing Geo/Append cells",
        "qiskit_compile_convention": "table_i_basis_gate_transpile_v1",
        "qiskit_circuit_scope": "ansatz_circuit_including_reference_state",
        "s_alg_policy": "row-facing winner-lineage/display-prefix S_alg; not S_beam_search_total",
        "fidelity_policy": "display 1-F only if selected-prefix fidelity is source-computable; otherwise --",
        "snake_runtime_contract": {
            "pool": "full_meta_unfiltered_hva_included",
            "runtime_split_mode": "shortlist_pauli_children_v1",
            "runtime_split_selection": "archival_child_set_forward_v1",
            "runtime_split_max_subset_size": 1,
            "batching_enabled": False,
            "metric_prune_route": True,
            "adapt_beam_lambda": 0.005,
            "adapt_beam_live_branches": 3,
            "adapt_beam_children_per_parent": 2,
        },
        "changed_snake_cells": changed_snake_cells,
        "plot_files": plot_files,
        "comparator_policy": "Geo/Append rows and trajectories retained from 20260702 POWELL pool-exposure support artifacts",
        "build_asset_restores": [
            {
                "source": str(source),
                "source_sha256": sha256(source),
                "destination": rel(destination),
                "destination_sha256": sha256(destination),
                "purpose": "build_enablement_only_non_hh_science_input",
            }
            for source, destination in BUILD_ASSET_RESTORES
        ],
        "supersedes_for_snake_rows": [
            "BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_MAIN_UPDATE_20260702",
            "BEGIN_MACHINE_READABLE_HH_POWELL_POOL_EXPOSURE_SNAKE_RUNTIME_SPLIT_TRACE_20260705",
            "older HH native200/Schur/SNAKE provenance blocks in the duplicate",
        ],
    }

    update_duplicate_tex(snake_rows, plot_files, manifest)
    validation = validate_duplicate(snake_rows, original_sha)
    manifest["validation"] = validation
    SUPPORT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # Reinsert final manifest now that validation/source manifest path is available.
    update_duplicate_tex(snake_rows, plot_files, manifest)
    validation = validate_duplicate(snake_rows, original_sha)
    manifest["validation"] = validation
    SUPPORT_MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"duplicate_tex": rel(DUP_TEX), "support_manifest": rel(SUPPORT_MANIFEST), "plot_dir": rel(FIG_DIR)}, indent=2))


if __name__ == "__main__":
    main()
