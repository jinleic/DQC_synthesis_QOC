import pyzx as zx
import numpy as np
import os 
from tqdm import tqdm
import polars as pl

def depth_optimize(path: str):
    '''
    Simplify the circuit to a minimal depth via ZX-calculus
    '''
    try:
        circuit = zx.Circuit.load(path)
    except FileNotFoundError:
        raise FileNotFoundError("The file does not exist.")
    except TypeError:
        return None 
    except KeyError:
        return None
    depth = np.array([])
    qasm_str = []
    depth = np.append(depth, circuit.depth())
    qasm_str.append(circuit.to_qasm())
    if depth[0] > 10000:
        return None
    if circuit.qubits > 300:
        return None 
    circuit_o = circuit.copy()
    circuit_o2 = circuit.copy()
    try:
        zx.optimize.full_optimize(circuit_o)
        depth = np.append(depth, circuit_o.depth())
        qasm_str.append(circuit_o.to_qasm())
    except TypeError:
        pass
    try:
        zx.optimize.basic_optimization(circuit_o2)   
        depth = np.append(depth, circuit_o2.depth())
        qasm_str.append(circuit_o2.to_qasm())
    except TypeError:
        pass
    circuit_g = circuit.to_graph()
    zx.simplify.full_reduce(circuit_g)
    # zx.simplify.lcomp_simp() 
    circuit_r = zx.extract.extract_circuit(circuit_g)
    depth = np.append(depth, circuit_r.depth())
    qasm_str.append(circuit_r.to_qasm())
    circuit_r1 = circuit_r.copy()
    try:
        zx.optimize.full_optimize(circuit_r)
        depth = np.append(depth, circuit_r.depth()) 
        qasm_str.append(circuit_r.to_qasm())
    except TypeError:
        pass
    zx.optimize.basic_optimization(circuit_r1.to_basic_gates())
    depth = np.append(depth, circuit_r1.depth())
    qasm_str.append(circuit_r1.to_qasm())
    min_index = np.argmin(depth) 
    return depth[min_index], qasm_str[min_index], depth[0], qasm_str[0]

if __name__ == '__main__':
    directory = './QASMBench'
    files = []

    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.qasm'):
                files.append(os.path.join(root, filename))

    print(len(files))
    count = 0
    dataset = pl.DataFrame({'file_path':[], 'depth':[], 'depth_o':[], 'qasm_str':[], 'qasm_str_o':[]})
    for i in tqdm(range(len(files))):
        if i == 160:
            continue # skip too complicated circuit
        result = depth_optimize(files[-i])
        if result: 
            dataset = pl.concat([pl.DataFrame({
                                    'file_path': files[-i], 
                                    'depth': result[0], 
                                    'depth_o': result[2], 
                                    'qasm_str': result[1], 
                                    'qasm_str_o': result[3]}), 
                                 dataset], how="vertical") 
            if result[0] < result[2]:
                count += 1

    print(count)
    dataset.write_csv('depth_optimize.csv')
    # try:
    #     depth, qasm_str, depth_o, qasm_str_o = depth_optimize([files[-58]])
    # except TypeError:
    #     pass
