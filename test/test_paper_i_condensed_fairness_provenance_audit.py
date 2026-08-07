import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "pipelines"
    / "reporting"
    / "audit_paper_i_condensed_fairness_provenance.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("paper_i_condensed_audit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def test_parse_condensed_hh_plateau_four_regimes():
    text = r'''
\inlinetablecaption{tab:hh_first_plateau_prefix_costs}{caption}
\textit{weak--weak: $(U/t,\lambda)=(0.25,0.25)$}
\begin{tabular}{p{0.26\columnwidth}ccccc}
Method & \metrichead{$k_{\rm pl}$} & \metrichead{$|\Delta E|$} & \metrichead{$N_{\twq}$} & \metrichead{$D_{\twq}$} & \metrichead{$D_c$}\\
append ADAPT & 7 & 1.14e-3 & 180 & 126 & 870 \\
\end{tabular}
\textit{strong--weak: $(U/t,\lambda)=(1.25,0.25)$}
\begin{tabular}{p{0.26\columnwidth}ccccc}
SNAKE & 11 & 1.78e-4 & 314 & 243 & 801 \\
\end{tabular}
\textit{weak--strong: $(U/t,\lambda)=(0.25,1.25)$}
\begin{tabular}{p{0.26\columnwidth}ccccc}
SNAKE & 42 & 1.92e-2 & -- & -- & -- \\
\end{tabular}
\textit{strong--strong: $(U/t,\lambda)=(1.25,1.25)$}
\begin{tabular}{p{0.26\columnwidth}ccccc}
Geo-ADAPT & 21 & 8.58e-3 & 1904 & 1617 & 8899 \\
\end{tabular}
% BEGIN_MACHINE_READABLE_TABLE_IV_ROUTE_A_HH_STRONG_WEAK_ABLATION_20260608
'''
    rows = audit._parse_hh_plateau(text.splitlines())
    assert {(row["method"], row["regime"]) for row in rows} == {
        ("append ADAPT", "weak_weak"),
        ("SNAKE", "strong_weak"),
        ("SNAKE", "weak_strong"),
        ("Geo-ADAPT", "strong_strong"),
    }
    weak_strong = [row for row in rows if row["regime"] == "weak_strong"][0]
    assert weak_strong["values"]["N2q"] == "--"


def test_source_reference_without_expected_hash_is_not_checked(tmp_path: Path):
    source = tmp_path / "output" / "pdf" / "artifact.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    ref = audit.SourceReference(
        path="output/pdf/artifact.json",
        referenced_by="UNIT_TEST",
        expected_sha256=None,
        source_key="source",
    )
    checked = audit.check_source_reference(tmp_path, ref)
    assert checked["exists"] is True
    assert checked["hash_status"] == "not_checked"
    assert checked["actual_sha256"] is not None


def test_metric_policy_flags_hh_appendix_divergence():
    text = (
        "Table~\\ref{tab:fixed_accuracy_hh_cartesian} uses the same-cutoff ED convention used "
        "in the plateau-prefix table.\n"
        "% {\"display_delta_e_policy\": \"raw_external_abs_delta_e=abs(E_alg(n_ph_work)-E_exact(n_ph_ref))\"}\n"
    )
    matrix = audit._build_metric_policy_matrix(text)
    appendix = [row for row in matrix if row["table_label"] == "tab:fixed_accuracy_hh_cartesian"][0]
    assert appendix["status"] == "policy_divergence"


def test_build_audit_is_read_only_and_has_no_transfer_flags(tmp_path: Path):
    (tmp_path / "MATH" / "paper_details").mkdir(parents=True)
    tex = tmp_path / "MATH" / "paper_details" / "static_adapt_paper_I_condensed.tex"
    pdf = tmp_path / "MATH" / "paper_details" / "static_adapt_paper_I_condensed.pdf"
    non = tmp_path / "MATH" / "paper_details" / "static_adapt_paper_I.tex"
    tex.write_text(
        r'''
\inlinetablecaption{tab:hh_first_plateau_prefix_costs}{caption}
\textit{weak--weak: $(U/t,\lambda)=(0.25,0.25)$}
\begin{tabular}{p{0.26\columnwidth}ccccc}
SNAKE & 11 & 4.23e-4 & 124 & 48 & 773 \\
\end{tabular}
% BEGIN_MACHINE_READABLE_TABLE_IV_ROUTE_A_HH_STRONG_WEAK_ABLATION_20260608
\inlinetablecaption{tab:fixed_accuracy_claims}{caption}
\begin{tabular}{p{0.145\textwidth}cccccc|cccccc}
SNAKE & 0 & 0 & 56 & 52 & 219 & 98 & 0 & 0 & 56 & 52 & 219 & 98\\
\end{tabular}
\inlinetablecaption{tab:fixed_accuracy_spin_boson}{caption}
\begin{tabular}{p{0.145\textwidth}cccccc|cccccc}
SNAKE & 0 & 0 & 34 & 34 & 124 & 125 & 0 & 0 & 213 & 198 & 725 & 368\\
\end{tabular}
\inlinetablecaption{tab:fixed_accuracy_hh_cartesian}{caption}
\begin{tabular}{p{0.145\textwidth}cccccc|cccccc}
SNAKE & 0 & -- & 124 & 48 & 773 & 3658 & 0 & -- & 314 & 243 & 801 & 7549\\
\colrule
Method & \multicolumn{6}{c|}{Weak-strong: x} & \multicolumn{6}{c}{Strong-strong: y}\\
SNAKE & 0 & -- & -- & -- & -- & -- & 0 & -- & 956 & 918 & 5350 & --\\
\end{tabular}
''',
        encoding="utf-8",
    )
    pdf.write_bytes(b"%PDF-1.4\n% test only\n")
    non.write_text("", encoding="utf-8")

    report = audit.build_audit(tmp_path, condensed_tex=tex, condensed_pdf=pdf, non_condensed_tex=non)
    assert report["mode"]["manuscript_edited"] is False
    assert report["mode"]["runs_launched"] is False
    assert report["mode"]["pdf_rebuilt"] is False
    assert report["mode"]["promotion_decision"] == "none"
    assert report["mode"]["safe_for_manuscript_transfer"] is False
    assert "safe_for_manuscript_transfer" not in json.dumps(report["findings"])


def test_work_proxy_classifies_missing_and_legacy():
    rows = [
        {"table_label": "tab:fixed_accuracy_hh_cartesian", "method": "SNAKE", "regime": "weak", "values": {"S": "--"}},
        {"table_label": "tab:fixed_accuracy_claims", "method": "HEA VQE", "regime": "weak", "values": {"S": "4.80e4"}},
        {"table_label": "tab:fixed_accuracy_claims", "method": "SNAKE", "regime": "weak", "values": {"S": "98"}},
    ]
    matrix = audit.build_work_proxy_matrix(rows)
    assert matrix[0]["status"] == "blocked"
    assert matrix[1]["source_currency"] == "fixed_structure_energy_eval_proxy"
    assert matrix[2]["blocker"] == "legacy_proxy_only"
