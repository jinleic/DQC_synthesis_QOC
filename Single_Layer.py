import numpy as np
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.qis.unitary import UnitaryMatrix 
from bqskit import Circuit

class single_layer:
    def __init__(self, two_qubit_gate: VariableUnitaryGate, 
                 single_qubit_gate: VariableUnitaryGate, 
                 single_qubit_gate2: VariableUnitaryGate,
                 initial_gate: VariableUnitaryGate):
        self.two_qubit_gate = two_qubit_gate
        self.single_qubit_gate2 = single_qubit_gate2
        self.single_qubit_gate = single_qubit_gate
        self.initial_gate = initial_gate

    def gen_initial_layer(
            self,
            target: np.ndarray,
            data
    ):
        target = UnitaryMatrix(target)
        init_circuit = Circuit(target.num_qudits, target.radixes) 
        for i in range(init_circuit.num_qudits):
            init_circuit.append_gate(self.initial_gate, [i])
        return init_circuit
    
    def gen_successor(
            self,
            circuit: Circuit,
            data
    ):
        coupling_graph = data.connectivity
        successors = []
        for edge in coupling_graph:
            successor = circuit.copy()
            successor.append_gate(self.two_qubit_gate, [edge[0], edge[1]])
            successor.append_gate(self.single_qubit_gate, edge[0])
            successor.append_gate(self.single_qubit_gate2, edge[1])
            successors.append(successor)
        
        return successors
    

if __name__ == "__main__":
    pass


