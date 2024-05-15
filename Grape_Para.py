import numpy as np 
from itertools import combinations
from quantum_optimal_control.core import hamiltonian

class Grape_Para:
    N: int # Number of Qubit
    dt: float # Unit of Pulse
    connected_qubit_pairs: list # All combinations of connected qubits
    d: int # Dimension of Hilbert space
    max_iterations: int # Maximum number of iterations
    H0: np.ndarray # Initial Hamiltonian
    Hops: list # List of Hamiltonian operators
    states_concerned_list: list # List of concerned states
    maxA: float # Maximum amplitude of the pulse
    decay: float # Decay rate
    convergence: dict # Convergence parameters
    reg_coeffs: dict # Regularization coefficients

    def __init__(self, N, dt=0.5, connected_qubit_pairs=None) -> None:
        self.N = N
        self.dt = dt
        if connected_qubit_pairs is None:
            self.connected_qubit_pairs = list(combinations(list(range(N)), 2))
        else:
            self.connected_qubit_pairs = connected_qubit_pairs
        self.d = 2
        self.max_iterations = 1000
        self.H0 = hamiltonian.get_H0(N, self.d)
        self.Hops, self.Hnames = hamiltonian.get_Hops_and_Hnames(N, self.d, self.connected_qubit_pairs)
        self.states_concerned_list = hamiltonian.get_full_states_concerned_list(N, self.d)
        self.maxA = hamiltonian.get_maxA(N, self.d, self.connected_qubit_pairs)
        self.decay =  self.max_iterations / 2
        self.convergence = {'rate': 0.01, 'max_iterations': self.max_iterations,
                    'conv_target':1e-3, 'learning_rate_decay':self.decay, 'min_grad': 1e-12}
        self.reg_coeffs = {'envelope': 5, 'dwdt': 0.001, 'd2wdt2': 0.00001}
    