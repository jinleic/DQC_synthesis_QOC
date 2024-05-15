import numpy as np
import numbers


def is_sequence(obj):
    if type(obj) is str:
        return False
    try:
        len(obj)
    except Exception:
        return False
    return True


def is_real_number(obj):
    return isinstance(obj, numbers.Real)


def check_unitary(matrix: np.ndarray):
    if matrix.shape[0] != matrix.shape[1]:
        return False
    if np.allclose(np.eye(matrix.shape[0]), matrix @ matrix.conj().T):
        return True
    return False


class unitary_matrix:
    unitary: np.ndarray
    num_qubits: np.ndarray
    num_params: int

    def __init__(self,
                 matrix: np.ndarray,
                 num_params: int,
                 num_qubits: np.ndarray = None,
                 check: bool = True):
        self.unitary = matrix
        self.num_params = num_params
        if num_qubits is None:
            self.num_qubits = np.arange(0, matrix.shape[0], 1)
        if check and not check_unitary(matrix):
            raise ValueError("The matrix is not unitary.")
        if check and len(num_qubits) != matrix.shape[0]:
            raise ValueError("The number of qubits does not match the matrix.")

    def check_params(self, params):
        if not is_sequence(params):
            raise TypeError("The parameters must be a sequence insteasd of %s."
                            % type(params))
        if not all(is_real_number(param) for param in params):
            raise TypeError("The parameters must be real numbers.")

        if len(params) != self.num_params:
            raise ValueError(
                "The number of parameters does not match the matrix.")


class U3_gate(unitary_matrix):
    num_qubits = 1
    num_params = 3

    def get_unitary(self, params):
        self.check_params(params)
        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        cp = np.cos(params[1])
        sp = np.sin(params[1])
        cl = np.cos(params[2])
        sl = np.sin(params[2])
        el = cl + 1j * sl
        ep = cp + 1j * sp

        return unitary_matrix(
            np.array([[ct, -el * st],
                      [ep * st, ep * el * ct]]),
            self.num_params,
            self.num_qubits
        )

    def get_grad(self, params):
        self.check_params(params)
        ct = np.cos(params[0] / 2)
        st = np.sin(params[0] / 2)
        cp = np.cos(params[1])
        sp = np.sin(params[1])
        cl = np.cos(params[2])
        sl = np.sin(params[2])
        el = cl + 1j * sl
        ep = cp + 1j * sp
        del_ = -sl + 1j * cl
        dep_ = -sp + 1j * cp

        return np.array(
            [
                [
                    [-0.5 * st, -0.5 * ct * el],
                    [0.5 * ct * ep, -0.5 * st * ep * el]
                ],
                [
                    [0, 0],
                    [st * dep_, ct * el * dep_]
                ],
                [
                    [0, -st * del_],
                    [0, -ct * ep * del_]
                ]
            ], dtype=np.complex128,
        )

    def cal_params(self, uty: unitary_matrix):
        if uty.unitary.shape != (2, 2):
            raise ValueError("The unitary matrix for U3 gate must be 2x2.")
        mag = np.linalg.det(uty.unitary) ** (-1/2)
        s_uty = unitary_matrix(uty.unitary * mag, uty.num_params, uty.num_qubits)
        a = np.angle(s_uty.unitary[1, 1])
        b = np.angle(s_uty.unitary[1, 0])
        c = np.abs(s_uty.unitary[1, 0])
        d = np.abs(s_uty.unitary[0, 0])
        theta = 2 * float(np.arctan2(c, d))
        phi = a + b
        lamb = a - b
        return np.array([theta, phi, lamb])
    
    def get_inverse_params(self, params):
        self.check_params(params)
        return np.array([-params[0], -params[2], -params[1]])
    
    def get_inverse(self):
        return U3_gate()


class CNOT(unitary_matrix):
    num_qubits = 2
    num_params = 0

    def get_unitary(self):
        return unitary_matrix(
            np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]]),
            self.num_params,
            self.num_qubits
        )

    