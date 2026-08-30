#!/usr/bin/env python3
"""Build exact-referenced SIAM observable plots from two locked score files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


EXPECTED_SHA256 = {
    "avqds": "8c6125f7eb9cfd41b13fba9f46eb10a097b3d12d0596317048f1c7d661035614",
    "apm": "25414cc6bb79650f4c6abc88db93b53c7b3e7b301432c563eab11e1879a6f2e7",
}
SOURCE_EVIDENCE_REVISION = "d55904498125ca087ee8541772410f66c50a92b7"
SOURCE_FILENAMES = {
    "avqds": "sources/siam_avqds_cluster9714484_score.json",
    "apm": "sources/siam_apm_cluster9715147_score.json",
}
METHOD_LABELS = {"avqds": "AVQDS", "apm": "Full APM"}
COLORS = {"exact": "#1A1A1A", "avqds": "#2878B5", "apm": "#A33D2D"}
LINESTYLES = {"exact": ":", "avqds": "--", "apm": "-"}
PRIMITIVE_LABELS = {
    "C_1": r"$C_1$",
    "C_2": r"$C_2$",
    "C_3": r"$C_3$",
    "C_4": r"$C_4$",
    "D_d": r"$D_d$",
    "m_d": r"$m_d$",
}
DERIVED_LABELS = {
    "Sigma_1": r"$\Sigma_1$",
    "Sigma_2": r"$\Sigma_2$",
    "Sigma_3": r"$\Sigma_3$",
    "Sigma_4": r"$\Sigma_4$",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _array(values: Sequence[Any], *, label: str, expected_length: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) != expected_length:
        raise ValueError(f"{label} must contain {expected_length} scalar values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} contains nonfinite values")
    return result


def _sigma_v2_identity_checks(
    payload: Mapping[str, Any], *, label: str
) -> dict[str, bool]:
    """Verify the stored Sigma-v2 series without importing checkout helpers."""

    checks: dict[str, bool] = {}
    for series_kind in ("exact", "observed"):
        series = payload["series"][series_kind]
        n_d = np.asarray(series["N_d"], dtype=float)
        d_d = np.asarray(series["D_d"], dtype=float)
        s_d2 = np.asarray(series["S_d2"], dtype=float)
        checks[f"{label}_{series_kind}_S_d2_identity"] = bool(
            np.allclose(s_d2, 0.75 * (n_d - 2.0 * d_d), rtol=1.0e-10, atol=1.0e-12)
        )
        sigma = s_d2.copy()
        for radius in range(1, 5):
            sigma = sigma + np.asarray(series[f"C_{radius}"], dtype=float)
            checks[f"{label}_{series_kind}_Sigma_{radius}_v2_identity"] = bool(
                np.allclose(
                    np.asarray(series[f"Sigma_{radius}"], dtype=float),
                    sigma,
                    rtol=1.0e-10,
                    atol=1.0e-12,
                )
            )
    return checks


def _comparability_audit(
    avqds: Mapping[str, Any],
    apm: Mapping[str, Any],
) -> dict[str, bool]:
    checks = {
        "shared_time_grid": avqds["grid"]["times"] == apm["grid"]["times"],
        "shared_time_grid_digest": avqds["grid"]["ordered_values_sha256"]
        == apm["grid"]["ordered_values_sha256"],
        "shared_seed_sha256": avqds["source"]["seed_sha256"]
        == apm["source"]["seed_sha256"],
        "shared_physics_fingerprint": avqds["method"]["physics_fingerprint"]
        == apm["method"]["physics_fingerprint"],
        "shared_comparison_fingerprint": avqds["method"]["comparison_fingerprint"]
        == apm["method"]["comparison_fingerprint"],
        "shared_reference": avqds["reference"] == apm["reference"],
        "shared_observable_contract": avqds["observable_contract"]
        == apm["observable_contract"],
        "shared_accuracy_policy": avqds["accuracy_policy"] == apm["accuracy_policy"],
        "shared_exact_observable_series": avqds["series"]["exact"]
        == apm["series"]["exact"],
        "shared_exact_energy_series": avqds["series"]["exact_energy"]
        == apm["series"]["exact_energy"],
        "exact_reference_reporting_only": bool(avqds["reporting_only"])
        and bool(apm["reporting_only"]),
    }
    checks.update(_sigma_v2_identity_checks(avqds, label="avqds"))
    checks.update(_sigma_v2_identity_checks(apm, label="full_apm"))
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"SIAM comparison identity failed: {', '.join(failed)}")
    return checks


def _load_sources(output_dir: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    source_paths = {
        key: output_dir / relative for key, relative in SOURCE_FILENAMES.items()
    }
    for key, path in source_paths.items():
        actual = _sha256(path)
        if actual != EXPECTED_SHA256[key]:
            raise ValueError(
                f"{key} score SHA-256 mismatch: expected {EXPECTED_SHA256[key]}, got {actual}"
            )

    avqds = _read_json(source_paths["avqds"])
    apm = _read_json(source_paths["apm"])
    checks = _comparability_audit(avqds, apm)

    point_count = int(avqds["grid"]["point_count"])
    time = _array(avqds["series"]["time"], label="time", expected_length=point_count)
    if point_count != 101 or not np.isclose(time[0], 0.0) or not np.isclose(time[-1], 1.0):
        raise ValueError("current SIAM plot contract requires 101 points on t=0..1")

    audit = {
        "schema": "siam_nb4_observable_plot_audit_v1",
        "status": "PASS_DIAGNOSTIC_COMPARISON_IDENTITY",
        "scientific_pair_status": "REJECTED_ACCURACY_CONFOUNDED",
        "exact_reference_role": "reporting_only",
        "qiskit_compilation": "not_performed_not_in_scope",
        "source_evidence_revision": SOURCE_EVIDENCE_REVISION,
        "source_files": {
            key: {
                "path": SOURCE_FILENAMES[key],
                "sha256": EXPECTED_SHA256[key],
                "run_sha256": payload["source"]["run_sha256"],
                "score_sha256": payload["score_sha256"],
                "classification": payload["summary"]["classification"],
            }
            for key, payload in (("avqds", avqds), ("apm", apm))
        },
        "comparability_checks": checks,
        "grid": {
            "point_count": point_count,
            "initial_time": float(time[0]),
            "terminal_time": float(time[-1]),
            "ordered_values_sha256": avqds["grid"]["ordered_values_sha256"],
        },
        "shared_identities": {
            "seed_sha256": avqds["source"]["seed_sha256"],
            "physics_fingerprint": avqds["method"]["physics_fingerprint"],
            "comparison_fingerprint": avqds["method"]["comparison_fingerprint"],
            "exact_reference_data_sha256": avqds["reference"]["data_sha256"],
            "exact_reference_manifest_sha256": avqds["reference"]["manifest_sha256"],
        },
        "plotted_series": {
            "primitive": list(PRIMITIVE_LABELS),
            "derived": list(DERIVED_LABELS),
            "accuracy": ["absolute_energy_error", "infidelity"],
        },
    }
    return avqds, apm, audit


def _style_axis(axis: Any, *, ylabel: str) -> None:
    axis.set_xlabel(r"time $t$")
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#D8DEE5", linewidth=0.65, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.margins(x=0.01)


def _figure_header(figure: Any, title: str, subtitle: str) -> None:
    figure.suptitle(title, x=0.5, y=0.988, fontsize=15, fontweight="bold", color="#17324D")
    figure.text(0.5, 0.949, subtitle, ha="center", va="top", fontsize=9, color="#4C5A67")
    figure.text(
        0.5,
        0.018,
        "Diagnostic exact-reference comparison only | pair verdict: rejected, accuracy-confounded | exact data were not used by either controller",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#6B4E00",
    )


def _plot_primitive_overlays(
    avqds: Mapping[str, Any], apm: Mapping[str, Any], output_path: Path
) -> None:
    time = np.asarray(avqds["series"]["time"], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.8))
    for axis, (key, display) in zip(axes.flat, PRIMITIVE_LABELS.items()):
        exact = np.asarray(avqds["series"]["exact"][key], dtype=float)
        axis.plot(
            time,
            exact,
            color=COLORS["exact"],
            linestyle=LINESTYLES["exact"],
            linewidth=2.2,
            label="Exact",
            zorder=4,
        )
        for method_key, payload in (("avqds", avqds), ("apm", apm)):
            observed = np.asarray(payload["series"]["observed"][key], dtype=float)
            axis.plot(
                time,
                observed,
                color=COLORS[method_key],
                linestyle=LINESTYLES[method_key],
                linewidth=1.65,
                marker="o" if method_key == "avqds" else "s",
                markevery=10,
                markersize=2.7,
                label=METHOD_LABELS[method_key],
                zorder=3 if method_key == "avqds" else 2,
            )
        avqds_rmae = avqds["primitive_response_metrics"][key][
            "response_normalized_mean_absolute_error"
        ]
        apm_rmae = apm["primitive_response_metrics"][key][
            "response_normalized_mean_absolute_error"
        ]
        axis.set_title(
            rf"{display}   rMAE: AVQDS {avqds_rmae:.3g}, APM {apm_rmae:.3g}",
            fontsize=9.3,
        )
        _style_axis(axis, ylabel=display)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    _figure_header(
        figure,
        "SIAM primitive observable trajectories",
        r"$N_b=4$, 101 shared points on $t\in[0,1]$ | AVQDS 9714484 (cut $10^{-3}$) | Full APM 9715147 (cut $10^{-5}$; exchange policy)",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.105, wspace=0.29, hspace=0.36)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def _plot_primitive_errors(
    avqds: Mapping[str, Any], apm: Mapping[str, Any], output_path: Path
) -> None:
    time = np.asarray(avqds["series"]["time"], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.8))
    for axis, (key, display) in zip(axes.flat, PRIMITIVE_LABELS.items()):
        for method_key, payload in (("avqds", avqds), ("apm", apm)):
            errors = np.asarray(payload["series"]["absolute_error"][key], dtype=float)
            axis.plot(
                time,
                errors,
                color=COLORS[method_key],
                linestyle=LINESTYLES[method_key],
                linewidth=1.8,
                label=METHOD_LABELS[method_key],
            )
        avqds_mae = avqds["primitive_response_metrics"][key]["mean_absolute_error"]
        apm_mae = apm["primitive_response_metrics"][key]["mean_absolute_error"]
        axis.set_title(
            rf"{display}   MAE: AVQDS {avqds_mae:.2e}, APM {apm_mae:.2e}",
            fontsize=9.1,
        )
        _style_axis(axis, ylabel=rf"$|\Delta {display.strip('$')}|$")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    _figure_header(
        figure,
        "SIAM primitive observable absolute errors",
        "Physical error against the shared exact diagnostic trajectory; each panel reports the time-mean absolute error",
    )
    figure.subplots_adjust(left=0.078, right=0.985, top=0.84, bottom=0.105, wspace=0.30, hspace=0.38)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def _plot_derived_and_accuracy(
    avqds: Mapping[str, Any], apm: Mapping[str, Any], output_path: Path
) -> None:
    time = np.asarray(avqds["series"]["time"], dtype=float)
    figure, axes = plt.subplots(2, 3, figsize=(11.0, 6.8))
    for axis, (key, display) in zip(axes.flat[:4], DERIVED_LABELS.items()):
        exact = np.asarray(avqds["series"]["exact"][key], dtype=float)
        axis.plot(
            time,
            exact,
            color=COLORS["exact"],
            linestyle=LINESTYLES["exact"],
            linewidth=2.2,
            label="Exact",
            zorder=4,
        )
        for method_key, payload in (("avqds", avqds), ("apm", apm)):
            observed = np.asarray(payload["series"]["observed"][key], dtype=float)
            axis.plot(
                time,
                observed,
                color=COLORS[method_key],
                linestyle=LINESTYLES[method_key],
                linewidth=1.65,
                marker="o" if method_key == "avqds" else "s",
                markevery=10,
                markersize=2.7,
                label=METHOD_LABELS[method_key],
                zorder=3 if method_key == "avqds" else 2,
            )
        avqds_rmae = avqds["derived_screening_metrics"][key][
            "response_normalized_mean_absolute_error"
        ]
        apm_rmae = apm["derived_screening_metrics"][key][
            "response_normalized_mean_absolute_error"
        ]
        axis.set_title(
            rf"{display}   rMAE: AVQDS {avqds_rmae:.3g}, APM {apm_rmae:.3g}",
            fontsize=9.3,
        )
        _style_axis(axis, ylabel=display)

    energy_axis = axes.flat[4]
    exact_energy = np.asarray(avqds["series"]["exact_energy"], dtype=float)
    for method_key, payload in (("avqds", avqds), ("apm", apm)):
        observed_energy = np.asarray(payload["series"]["observed_energy"], dtype=float)
        energy_axis.plot(
            time,
            np.abs(observed_energy - exact_energy),
            color=COLORS[method_key],
            linestyle=LINESTYLES[method_key],
            linewidth=1.8,
            label=METHOD_LABELS[method_key],
        )
    energy_axis.set_title(
        "Energy MAE: AVQDS {:.2e}, APM {:.2e}".format(
            avqds["diagnostics"]["energy"]["mean_absolute_error"],
            apm["diagnostics"]["energy"]["mean_absolute_error"],
        ),
        fontsize=9.3,
    )
    _style_axis(energy_axis, ylabel=r"$|\Delta E|$")
    energy_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)

    fidelity_axis = axes.flat[5]
    for method_key, payload in (("avqds", avqds), ("apm", apm)):
        infidelity = np.asarray(payload["series"]["infidelity"], dtype=float)
        fidelity_axis.plot(
            time,
            infidelity,
            color=COLORS[method_key],
            linestyle=LINESTYLES[method_key],
            linewidth=1.8,
            label=METHOD_LABELS[method_key],
        )
    fidelity_axis.set_title(
        "Mean infidelity: AVQDS {:.2e}, APM {:.2e}".format(
            avqds["summary"]["mean_infidelity"], apm["summary"]["mean_infidelity"]
        ),
        fontsize=9.3,
    )
    _style_axis(fidelity_axis, ylabel=r"$1-F$")
    fidelity_axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, frameon=False)
    _figure_header(
        figure,
        "SIAM derived observables and accuracy diagnostics",
        r"Derived screening observables $\Sigma_1$--$\Sigma_4$, physical energy error, and state infidelity",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.105, wspace=0.29, hspace=0.36)
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def build(output_dir: Path) -> Mapping[str, Any]:
    avqds, apm, audit = _load_sources(output_dir)
    figure_paths = {
        "primitive_observables": output_dir / "siam_nb4_primitive_observables.png",
        "primitive_absolute_errors": output_dir
        / "siam_nb4_primitive_absolute_errors.png",
        "derived_and_accuracy": output_dir / "siam_nb4_derived_and_accuracy.png",
    }
    _plot_primitive_overlays(avqds, apm, figure_paths["primitive_observables"])
    _plot_primitive_errors(avqds, apm, figure_paths["primitive_absolute_errors"])
    _plot_derived_and_accuracy(avqds, apm, figure_paths["derived_and_accuracy"])
    audit["figures"] = {
        key: {
            "path": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for key, path in figure_paths.items()
    }
    audit_path = output_dir / "siam_nb4_observable_plot_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="self-contained output directory containing sources/",
    )
    args = parser.parse_args()
    audit = build(args.output_dir.resolve())
    print(json.dumps({"status": audit["status"], "figures": audit["figures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
