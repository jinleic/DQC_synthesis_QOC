from bqskitrs import HilbertSchmidtResidualsFunction, LeastSquaresMinimizerNative
from bqskitrs import HilbertSchmidtCostFunction
from qiskit import QuantumCircuit
import numpy as np
from bqskit.ext import qiskit_to_bqskit
from bqskit.qis.unitary import UnitaryMatrix
from typing import TypeVar
Self = TypeVar('Self', bound='Minimizer')

class HilbertSchmidtResiduals(
        HilbertSchmidtResidualsFunction,
):
    '''
    A cost function that uses the Hilbert-Schmidt norm of the difference
    between two matrices as the cost where the cost is zero if the target and circuit unitary 
    differ only by a global phase.
    '''

class HilbertSchmidtResidualsGenerator:
    def gen_cost(
            self,
            circuit: QuantumCircuit,
            target:  np.ndarray
    ):
        # circuit = qiskit_to_bqskit(circuit)
        target = UnitaryMatrix(target)
        return HilbertSchmidtResiduals(circuit, target)

class HilbertSchmidtCost(
    HilbertSchmidtCostFunction,
):
    '''
    A cost function that uses the Hilbert-Schmidt norm of the difference
    between two matrices as the cost where the cost is zero if the target and circuit unitary 
    differ only by a global phase.
    '''

class HilbertSchmidtCostGenerator:
    def gen_cost(
            self,
            circuit,
            target:  np.ndarray
    ):
        # circuit = qiskit_to_bqskit(circuit)
        target = UnitaryMatrix(target)
        return HilbertSchmidtCost(circuit, target)

class Minimizer(
        LeastSquaresMinimizerNative
):
    '''
    A minimizer that uses the least squares method to minimize the cost function.
    '''
    def __new__(
            cls, num_threads: int = 4, ftol: float = 1e-6,
            gtol: float = 1e-10, report: bool = False,
    ) -> Self:
        return super().__new__(cls, num_threads, ftol, gtol, report)


class minimization:
    def __init__(
            self,
            cost_fn_gen=None,
            minimizer=None) -> None:
        
        if cost_fn_gen is None:
            cost_fn_gen = HilbertSchmidtResidualsGenerator()
        
        if minimizer is None:
            minimizer = Minimizer()
        
        self.cost_fn_gen = cost_fn_gen
        self.minimizer = minimizer

    def instaniate(
            self,
            circuit: QuantumCircuit,
            target: np.ndarray,
            x0: np.ndarray
    ):
        circuit = qiskit_to_bqskit(circuit)
        target = UnitaryMatrix(target)
        cost = self.cost_fn_gen.gen_cost(circuit, target)
        return self.minimizer(cost, x0)


    def multi_start_instaniate_inplace(
            self,
            circuit: QuantumCircuit,
            target: np.ndarray,
            num_starts: int
    ):
        circuit = qiskit_to_bqskit(circuit)
        target = UnitaryMatrix(target)
        start_gen = [
            2 * np.pi * np.random.random(circuit.num_params)
            for i in range(num_starts)
        ]
        cost = self.cost_fn_gen.gen_cost(circuit, target)
        params_list = [self.instaniate(circuit, target, x0) for x0 in start_gen]
        params = sorted(params_list, key=lambda x: cost(x))[0]
        circuit.set_params(params)








if __name__ == '__main__':
    import qiskit.quantum_info as qi
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc_a = qi.Operator(qc).data
    h = HilbertSchmidtResidualsGenerator()
    # h = HilbertSchmidResidualsGenerator()
    bqc = qiskit_to_bqskit(qc)
    params = np.random.random(bqc.num_params)
    res = h.gen_cost(qc, qc_a).get_cost(params)
    print(res)
