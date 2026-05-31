from dataclasses import dataclass, field

@dataclass
class ProblemHamiltonianData:
    """
    Unified container passed to solvers.

    This is the ONLY object solvers should depend on.
    
    Attributes:
        problem_type (str): Type of problem ('hh', 'molecular', 'spin', 'elec_vibr').
        
        H (object): The main Hamiltonian operator in various representations:
            - matrix: Dense or sparse numpy/scipy matrix
            - MPO: Matrix Product Operator for tensor networks
            - Pauli: Pauli polynomial representation for quantum circuits
            - Other backend-specific formats
        
        basis (object): Description of the Hilbert space/basis used:
            - Fock space configuration
            - Quantum numbers / symmetry labels
            - Dimension information for each subspace
        
        subspaces (dict): Optional decomposition of the system into sectors:
            - Keys: subspace labels (e.g., 'electronic', 'phonon', 'spin')
            - Values: Hamiltonian components for each sector or sector data
            - Useful for problems with clear physical decomposition
        
        backend_data (object): Backend-specific data and configurations:
            - Qiskit/Cirq circuit objects (if quantum backend)
            - Classical simulation parameters
            - Device-specific mappings
        
        metadata (dict): Reproducibility and tracking information:
            - timestamps, parameters used, software versions
            - Links between problem definition and solver results
            - Problem configuration snapshot
    """
    problem_type: str
    # main operator (matrix, MPO, Pauli, etc.)
    H: object = None
    # basis / Hilbert space description
    basis: object = None
    # optional decomposition (electronic / phonon / spin)
    subspaces: dict = field(default_factory=dict)
    # backend data (optional)
    backend_data: object = None
    # metadata for reproducibility
    metadata: dict = field(default_factory=dict)