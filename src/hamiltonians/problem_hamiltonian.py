from src.hamiltonians.hubbard_holstein import build_hubbard_holstein_hamiltonian
from src.hamiltonians.molecular_electronic import build_molecular_electronic_hamiltonian
from src.hamiltonians.spin_system import build_spin_hamiltonian
from src.hamiltonians.elec_vibrational import build_elec_vibrational_hamiltonian

#
#   build hamiltonian dispatcher
#

def build_problem_hamiltonian(problem_type: str, **kwargs):
    """
    Global dispatcher.

    This is the ONLY entrypoint used by workflows.
    """
    if problem_type is None:
        log.error("problem_type must be provided")
    key = problem_type.strip().lower()
    if key == "hh":
        return build_hubbard_holstein_hamiltonian(**kwargs)
    if key in ("molecular", "electronic"):
        return build_molecular_electronic_hamiltonian(**kwargs)
    if key == "spin":
        return build_spin_hamiltonian(**kwargs)
    if key in ("elec_vibr", "electron_phonon"):
        return build_elec_vibrational_hamiltonian(**kwargs)
    log.error(f"Unsupported problem_type: {problem_type}")