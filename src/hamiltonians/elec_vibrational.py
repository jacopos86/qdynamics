

# src/hamiltonians/elec_vibrational.py

from .containers import ProblemHamiltonianData


def build_elec_vibrational_hamiltonian(
    electronic_data=None,
    phonon_data=None,
    coupling_data=None,
    **kwargs,
):
    """
    General electron-vibration Hamiltonian.
    """

    # ----------------------------------------
    # TODO
    # ----------------------------------------
    H = None

    return ProblemHamiltonianData(
        problem_type="elec_vibr",
        H=H,
        metadata={},
    )