"""Compare two EPH HDF5 files without assuming identical mode ordering/signs."""

import argparse
from pathlib import Path

import h5py
import numpy as np


HARTREE_TO_WAVENUMBER = 219474.6313705


def _read(path):
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["omega"][()]), np.asarray(handle["eph_mat"][()])


def _frequency_groups(frequencies_cm, tolerance_cm):
    """Group adjacent, potentially degenerate modes in frequency order."""
    order = np.argsort(frequencies_cm)
    groups = []
    for index in order:
        if (not groups
                or abs(frequencies_cm[index]
                       - frequencies_cm[groups[-1][-1]]) > tolerance_cm):
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))
    return groups


def compare_eph(reference_path, candidate_path, degeneracy_tolerance_cm=5.0):
    """Return mode-block diagnostics invariant to signs and degenerate rotations.

    A global normal-mode sign and MO phase changes do not alter a coupling
    matrix Frobenius norm.  For a degenerate vibrational subspace, the combined
    norm over the whole block is also invariant to rotations among its modes.
    Consequently a large block-norm mismatch cannot be repaired by merely
    flipping signs or permuting/rotating normal modes.
    """
    ref_omega, ref_eph = _read(reference_path)
    cand_omega, cand_eph = _read(candidate_path)
    ref_cm = np.real(ref_omega) * HARTREE_TO_WAVENUMBER
    cand_cm = np.real(cand_omega) * HARTREE_TO_WAVENUMBER
    ref_groups = _frequency_groups(ref_cm, degeneracy_tolerance_cm)
    cand_groups = _frequency_groups(cand_cm, degeneracy_tolerance_cm)

    rows = []
    unmatched = set(range(len(cand_groups)))
    for ref_group in ref_groups:
        ref_center = float(np.mean(ref_cm[ref_group]))
        if not unmatched:
            rows.append({"reference_modes": ref_group, "candidate_modes": [],
                         "reference_frequency_cm": ref_center})
            continue
        candidate_group_index = min(
            unmatched,
            key=lambda idx: abs(np.mean(cand_cm[cand_groups[idx]]) - ref_center))
        unmatched.remove(candidate_group_index)
        cand_group = cand_groups[candidate_group_index]
        ref_norm = float(np.linalg.norm(ref_eph[ref_group]))
        cand_norm = float(np.linalg.norm(cand_eph[cand_group]))
        rows.append({
            "reference_modes": ref_group,
            "candidate_modes": cand_group,
            "reference_frequency_cm": ref_center,
            "candidate_frequency_cm": float(np.mean(cand_cm[cand_group])),
            "reference_block_norm": ref_norm,
            "candidate_block_norm": cand_norm,
            "norm_ratio": cand_norm / ref_norm if ref_norm else np.nan,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compare EPH results using sign/order-invariant diagnostics")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--degeneracy-tolerance-cm", type=float, default=5.0)
    args = parser.parse_args()
    rows = compare_eph(
        args.reference, args.candidate, args.degeneracy_tolerance_cm)
    print("ref modes -> candidate modes | frequencies/cm^-1 | block norms | ratio")
    for row in rows:
        if not row["candidate_modes"]:
            print(f"{row['reference_modes']} -> unmatched")
            continue
        print(
            f"{row['reference_modes']} -> {row['candidate_modes']} | "
            f"{row['reference_frequency_cm']:.2f} -> "
            f"{row['candidate_frequency_cm']:.2f} | "
            f"{row['reference_block_norm']:.6g} -> "
            f"{row['candidate_block_norm']:.6g} | "
            f"{row['norm_ratio']:.6g}")


if __name__ == "__main__":
    main()
