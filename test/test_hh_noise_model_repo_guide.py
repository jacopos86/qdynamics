from __future__ import annotations

from pathlib import Path

from pipelines.exact_bench import hh_noise_model_repo_guide as guide


def _build_payload(*, include_tests: bool = True, include_docs: bool = True) -> dict:
    args = guide.parse_args(
        [
            "--skip-pdf",
            "--include-tests" if include_tests else "--no-include-tests",
            "--include-docs" if include_docs else "--no-include-docs",
        ]
    )
    return guide._build_index(args=args)


def test_scope_resolver_includes_four_root_modules() -> None:
    payload = _build_payload()
    roots = set(payload["scope"]["root_modules"])
    required = {
        "pipelines.exact_bench.noise_oracle_runtime",
        "pipelines.exact_bench.hh_noise_hardware_validation",
        "pipelines.exact_bench.hh_noise_robustness_seq_report",
        "pipelines.exact_bench.hh_seq_transition_utils",
    }
    assert required.issubset(roots)


def test_symbol_index_is_deterministically_sorted() -> None:
    payload = _build_payload()
    symbols = payload["symbols"]
    expected = sorted(
        symbols,
        key=lambda x: (str(x["path"]), int(x["line"]), str(x["kind"]), str(x["name"])),
    )
    assert symbols == expected


def test_cli_extraction_captures_required_noise_flags() -> None:
    payload = _build_payload()
    flags = {str(x.get("flag", "")) for x in payload["cli_flags"]}
    required = {
        "--noise-mode",
        "--shots",
        "--oracle-repeats",
        "--oracle-aggregate",
        "--backend-name",
        "--allow-aer-fallback",
        "--omp-shm-workaround",
        "--noise-modes",
        "--noisy-methods",
        "--legacy-parity-tol",
    }
    assert required.issubset(flags)


def test_manifest_preview_builder_includes_required_fields() -> None:
    lines = guide._manifest_preview_lines(
        model="Hubbard",
        ansatz="hh_hva",
        drive_enabled=False,
        t=1.0,
        U=4.0,
        dv=0.0,
    )
    text = "\n".join(lines)
    for token in [
        "PARAMETER MANIFEST",
        "Model family/name",
        "Ansatz type(s)",
        "Drive enabled",
        "t:",
        "U:",
        "dv:",
    ]:
        assert token in text


def test_default_paths_match_contract() -> None:
    args = guide.parse_args(["--skip-pdf"])
    assert Path(args.output_pdf).as_posix().endswith("docs/HH_noise_model_repo_guide.pdf")
    assert Path(args.output_json).as_posix().endswith("artifacts/json/hh_noise_model_repo_guide_index.json")


def test_human_front_matter_page_count_is_fixed() -> None:
    pages = guide._human_front_matter_pages()
    assert len(pages) == 20
