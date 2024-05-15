from qiskit import QuantumCircuit
from Unitary_Matrix import unitary_matrix
from typing import Union, Any
import numpy as np
from sys import platform
import ctypes
from ctypes.util import find_library
import random
from Minimization import minimization
from bqskit.passes import QuickPartitioner
from bqskit import Circuit
from bqskit.ir.gates import barrier, BarrierPlaceholder, VariableUnitaryGate   
from bqskit.compiler.compiler import Compiler
# import asyncio
from PassData import PassData
from Single_Layer import single_layer
from Grape_Para import Grape_Para
from PulseLib import PulseLib
from bqskit.passes import QSearchSynthesisPass, GreedyHeuristic
from bqskit.passes import SimpleLayerGenerator


instantiate_method = []

def random_seed_gen(seed:int):
    if platform != 'win32':
        libc = ctypes.CDLL(find_library('c'))
        libc.srand(seed)
    # set numpy seed
    np.random.seed(seed)
    random.seed(seed)

def remove_inactive(
        circuit: Circuit
):
    while len(list(set(range(circuit.num_qudits)) - set(circuit.active_qudits))) > 0:
        circuit.pop_qudit(list(set(range(circuit.num_qudits)) - set(circuit.active_qudits))[0])
    return circuit


def circuit_partition(
        circuit: Circuit,
        maxsize: int
):
    if circuit.num_qudits <= maxsize:
        maxsize = circuit.num_qudits - 2
        if maxsize < 4:
            maxsize = 4
    partitioner = QuickPartitioner(maxsize)
    circuit_par = Compiler().compile(circuit, [partitioner])
    # print(circuit_par.num_qudits)
    circuits = []
    for i in range(circuit_par.depth):
        is_barrier = False
        if len(circuit_par[i]) == 1:
            for _ in range(1, circuit_par.num_qudits+1):
                if circuit_par[i][0].gate == BarrierPlaceholder(_):
                    is_barrier = True
                    break
        if not is_barrier:
            if len(circuit_par[i]) == 1:
                qc = Circuit(circuit_par.num_qudits)
                qc.append(circuit_par[i][0])
                qc = remove_inactive(qc)
                circuits.append(qc)
            elif len(circuit_par[i]) < 2*maxsize - 1:
                parallel_circuits = []
                qc1 = Circuit(circuit_par.num_qudits)
                qc2 = Circuit(circuit_par.num_qudits)
                for j in range(int(len(circuit_par[i])/2)):
                    # print(circuit_par[i][j].location)
                    qc1.append(circuit_par[i][j])
                qc1 = remove_inactive(qc1)
                parallel_circuits.append(qc1)
                for k in range(int(len(circuit_par[i])/2), len(circuit_par[i])):
                    qc2.append(circuit_par[i][k])
                qc2 = remove_inactive(qc2)
                parallel_circuits.append(qc2)
                circuits.append(parallel_circuits)
            else:
                parallel_circuits = []
                num = 0
                length = len(circuit_par[i])
                for op in circuit_par[i]:
                    if num == 0:
                        qc = Circuit(circuit_par.num_qudits)
                    if num < maxsize - 1:
                        qc.append(op)
                        num += 1
                    if num == maxsize - 1:
                        qc = remove_inactive(qc)
                        parallel_circuits.append(qc)
                        num = 0
                if num != 0:
                    qc = remove_inactive(qc)
                    parallel_circuits.append(qc)
                circuits.append(parallel_circuits)
    return circuits

# async def cal_synthesis(
#     partitioned_circuit
# ):
#     VUG1 = VariableUnitaryGate(1, [2])
#     VUG2 = VariableUnitaryGate(1, [2])
#     VUG3 = VariableUnitaryGate(2, [2, 2])
#     VUG4 = VariableUnitaryGate(1, [2])
#     layer_gen = single_layer(VUG3, VUG1, VUG2, VUG4)
#     syn = QSearch_Synthesis(layer_generator=layer_gen,
#                       heuristic_function=AStarHeuristic(),
#                       success_threshold=1e-6,
#                       cost=HilbertSchmidtResidualsGenerator(),
#                       max_layers=None,
#                       store_partial_results=True,
#                         partials_per_depth=25,
#                         instantiate_options={})
#     loop = asyncio.get_event_loop()
#     synthesized_circuits = []
#     async def cal_synthesis_thread(circuit):
#         def cal_synthesis_sync():    
#             if (type(circuit) == list):
#                 parallel_circuits = []
#                 for qc in circuit:
#                     passdata = PassData(qc)
#                     best = asyncio.run(
#                         syn.synthesize(
#                             np.array(qc.get_unitary()),
#                             passdata
#                             ))
#                     parallel_circuits.append(best)
#                 return parallel_circuits
#             else:
#                 circuit_unitary = circuit.get_unitary()               
#                 passdata = PassData(circuit)
#                 best = asyncio.run(
#                     syn.synthesize(
#                         np.array(circuit_unitary),
#                         passdata
#                         ))
#                 return best
#         synthesized_circuit = await loop.run_in_executor(None, cal_synthesis_sync)
#         synthesized_circuits.append(synthesized_circuit)
#     tasks = []
#     for _ in partitioned_circuit:
#         task = asyncio.create_task(cal_synthesis_thread(_))
#         tasks.append(task)
#     await asyncio.gather(*tasks)
#     return synthesized_circuits

def cal_synthesis(
        partitioned_circuit,
        numworker: int
):
    VUG1 = VariableUnitaryGate(1, [2])
    VUG2 = VariableUnitaryGate(1, [2])
    VUG3 = VariableUnitaryGate(2, [2, 2])
    VUG4 = VariableUnitaryGate(1, [2])
    
    synthesized_circuits = []
    layer_gen = SimpleLayerGenerator(VUG3, VUG1, VUG2, VUG4)
    leap = QSearchSynthesisPass(
        heuristic_function=GreedyHeuristic(),
        layer_generator=layer_gen
    )
    num = 0
    for circuit in partitioned_circuit:
        num += 1
        print(f"Processing layer {num}")
        if type(circuit) == list:
            parallel_circuits = []
            for qc in circuit:
                const_circuit = Circuit.from_unitary(qc.get_unitary())
                best = Compiler(
                    num_workers=numworker
                ).compile(const_circuit, [leap])
                parallel_circuits.append(best)
            synthesized_circuits.append(parallel_circuits)
        else:
            circuit_unitary = circuit.get_unitary()
            const_circuit = Circuit.from_unitary(circuit_unitary)
            best = Compiler(
                num_workers=numworker
            ).compile(const_circuit, [leap])
            synthesized_circuits.append(best)
    return synthesized_circuits


def cal_latency(
    circuit: Circuit,
    libpath: str
):
    pulse_lib = PulseLib(libpath)
    latency_total = 0
    fidelity_total = 1
    depth = circuit.depth
    for _ in range(depth):
        if len(circuit[_]) == 1:
            operation = circuit[_][0]
            unitary = np.array(operation.get_unitary())
            grape_para = Grape_Para(int(np.log2(unitary.shape[0])))
            latency, fidelity, compilation_time = pulse_lib.get_pulse2(unitary, grape_para)
            latency_total += latency
            fidelity_total = fidelity_total * ( 1- fidelity)
        else:
            latency_set = []
            for operation in circuit[_]:
                unitary = np.array(operation.get_unitary())
                grape_para = Grape_Para(int(np.log2(unitary.shape[0])))
                latency, fidelity, compilation_time = pulse_lib.get_pulse2(unitary, grape_para)
                latency_set.append(latency)
                fidelity_total = fidelity_total * (1 - fidelity)
            latency_total += max(latency_set)
    return latency_total, fidelity_total


def cal_latency_all(
        synthesized_circuits,
        libpath: str
):
    latency_total = 0
    fidelity_total = 1
    tasks = []

    for _ in synthesized_circuits:
        if type(_) == list:
            latency_set = []
            for qc in _:
                latency, fidelity = cal_latency(qc, libpath)
                latency_set.append(latency)
                fidelity_total = fidelity_total * (fidelity)
            latency_total += max(latency_set)
        else:
            latency, fidelity =  cal_latency(_, libpath)
            latency_total += latency
            fidelity_total = fidelity_total * (fidelity)
    return latency_total, fidelity_total


def regroup_and_cal_latency(
        synthesized_circuits,
        libpath: str
):
    total_latency = 0
    tasks = []
    fidelity_total = 1
    for _ in synthesized_circuits:
        if type(_) == list:
            parallel_latency = []
            for qc in _:
                regroup_qc = circuit_partition(qc, 4)
                latency, fidelity = (cal_latency_all(regroup_qc, libpath))
                parallel_latency.append(latency)
                fidelity_total = fidelity_total * ( fidelity )
            total_latency += max(parallel_latency)
        else:
            regroup_qc = circuit_partition(_, 4)
            latency, fidelity =  (cal_latency_all(regroup_qc, libpath))
            total_latency += latency
            fidelity_total = fidelity_total * ( fidelity)
    return total_latency, fidelity_total



                

            




