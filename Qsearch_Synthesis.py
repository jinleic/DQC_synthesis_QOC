from Minimization import HilbertSchmidtResidualsGenerator
from Minimization import HilbertSchmidtCostGenerator
from typing import Any, Union
from qiskit import QuantumCircuit
import numpy as np
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit
from Single_Layer import single_layer
from bqskit.qis.unitary import UnitaryMatrix
from typing import NamedTuple
import itertools
import heapq
from bqskit import Circuit
from bqskit.runtime import get_runtime
from bqskit.ir.gates import VariableUnitaryGate
from pprint import pprint
import asyncio


class AStarHeuristic:
    def __init__(
            self,
            heuristic_factor: float = 10.0,
            cost_factor: float = 1.0,
            cost_gen = HilbertSchmidtCostGenerator()
    ):
        self.heuristic_factor = heuristic_factor
        self.cost_factor = cost_factor
        self.cost_gen = cost_gen
    
    def get_value(
            self,
            circuit,
            target: np.ndarray
    ):
        cost = 0.0
        # circuit = qiskit_to_bqskit(circuit)
        for gate in circuit.gate_set:
            if gate.num_qudits == 1:
                continue
            cost += float(circuit.count(gate))
        
        heuristic = self.cost_gen.gen_cost(circuit, target).get_cost(circuit.params)
        return self.heuristic_factor * heuristic + self.cost_factor * cost


class GreedyHeuristic:
    def __init__(
            self,
            cost_gen = HilbertSchmidtCostGenerator()
    ):
        self.cost_gen = cost_gen
    
    def get_value(
            self,
            circuit,
            target: np.ndarray
    ):
        return self.cost_gen.gen_cost(circuit, target).get_cost(circuit.params)


class FrontierElement(NamedTuple):
    """The Frontier contains FrontierElements."""
    cost: float
    element_id: int
    circuit: QuantumCircuit
    extra_data: Any



class Frontier:
    def __init__(
            self,
            target: np.ndarray,
            heuristic_function
    ):
        self.target = target
        self.heuristic_function = heuristic_function
        self._frontier = []
        self._counter = itertools.count()
    
    def add(
            self,
            circuit: QuantumCircuit,
            extra_data: Any
    ):
        heuristic_value = self.heuristic_function.get_value(circuit, self.target)
        count = next(self._counter)
        elem = FrontierElement(heuristic_value, count, circuit, extra_data)
        heapq.heappush(self._frontier, elem)
    
    def pop(self):
        elem = heapq.heappop(self._frontier)
        return elem.circuit, elem.extra_data
    
    def empty(self):
        return len(self._frontier) == 0
    
    def clear(self):
        self._frontier.clear()

class QSearch_Synthesis:
    def __init__(
            self,
            layer_generator,
            # cost,
            heuristic_function = AStarHeuristic(),
            success_threshold: float = 1e-6,
            cost = HilbertSchmidtResidualsGenerator(),
            max_layers : Union[int, None] = 10,
            store_partial_results: bool = False, 
            partials_per_depth: int = 25,
            instantiate_options = {}
    ):
        self.heuristic_function = heuristic_function
        self.layer_generator = layer_generator
        self.success_threshold = success_threshold
        self.cost = cost
        self.max_layers = max_layers
        self.store_partial_results = store_partial_results
        self.partials_per_depth = partials_per_depth
        self.instantiate_options = {
            'cost_fn_gen': self.cost,
        }

    
    async def synthesize(
            self,
            utry,
            data
    ):
        instantiate_options = self.instantiate_options.copy()

        instantiate_options['seed'] = data.seed
        # layer_gen = self.layer_generator(data)
        layer_gen = self.layer_generator

        frontier = Frontier(utry, self.heuristic_function)
        initial_layer = layer_gen.gen_initial_layer(utry, data)
        initial_layer.instantiate(utry, **instantiate_options)
        frontier.add(initial_layer, 0)

        best_dist = self.cost.gen_cost(initial_layer, utry).get_cost(initial_layer.params)
        best_circuit = initial_layer
        best_layer = 0

        psols: dict[int, list[tuple[Circuit, float]]] = {}

        if best_dist < self.success_threshold:
            return initial_layer
        
        while not frontier.empty():
            top_circuit, layer = frontier.pop()
            successors = layer_gen.gen_successor(top_circuit, data)
            if len(successors) == 0:
                continue
            
            # circuits = await get_runtime().map(
            #     Circuit.instantiate,
            #     successors,
            #     target=utry,
            #     **instantiate_options
            # )
            circuits = []
            tasks = []
            loop = asyncio.get_event_loop()
            async def instantiate_thread(successor):
                def instantiate_sync():
                    return successor.instantiate(utry, **instantiate_options)
                circuit = await loop.run_in_executor(None, instantiate_sync)
                circuits.append(circuit)

            for successor in successors:
                # circuit = successor.instantiate(utry, **instantiate_options)
                task = asyncio.create_task(instantiate_thread(successor))
                tasks.append(task)
            await asyncio.gather(*tasks)


            for circuit in circuits:
                dist = self.cost.gen_cost(circuit, utry).get_cost(circuit.params)
                
                if dist < self.success_threshold:
                    if self.store_partial_results:
                        data['psols'] = psols
                    return circuit
                
                if dist < best_dist:
                    best_dist = dist
                    best_circuit = circuit
                    best_layer = layer 

                if self.store_partial_results:
                    if layer not in psols:
                        psols[layer] = []
                    psols[layer].append((circuit.copy(), dist))

                    if len(psols[layer]) > self.partials_per_depth:
                        psols[layer].sort(key=lambda x: x[1])
                        del psols[layer][-1]

                if self.max_layers is None or (layer + 1) < self.max_layers:
                    frontier.add(circuit, layer + 1)
            
        if self.store_partial_results:
            data['psols'] = psols

        return best_circuit
    
if __name__ == '__main__':
    import asyncio
    from PassData import PassData
    from bqskit.runtime import worker, default_worker_port
    VUG1 = VariableUnitaryGate(1, [2])
    VUG2 = VariableUnitaryGate(1, [2])
    VUG3 = VariableUnitaryGate(2, [2, 2])
    VUG4 = VariableUnitaryGate(1, [2])
    layer_gen = single_layer(VUG3, VUG1, VUG2, VUG4)
    syn = QSearch_Synthesis(layer_generator=layer_gen,
                      heuristic_function=AStarHeuristic(),
                      success_threshold=1e-6,
                      cost=HilbertSchmidtResidualsGenerator(),
                      max_layers=None,
                      store_partial_results=True,
                        partials_per_depth=25,
                        instantiate_options={})
    toffoli = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    ]
    qc = Circuit(3, [2, 2, 2])
    passdata = PassData(qc)
   # passdata.seed = 0
    # print(type(passdata.seed))
    # print(type(qc.num_qudits))
    best = asyncio.run(syn.synthesize(np.array(toffoli), passdata))
    pprint(best.coupling_graph)