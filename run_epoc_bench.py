import polars as pl
from Circuit_Process import circuit_partition, cal_synthesis, regroup_and_cal_latency, cal_latency_all
from tqdm import tqdm
from bqskit import Circuit
from qiskit import QuantumCircuit
from bqskit.ext import qiskit_to_bqskit
import argparse
import time
import asyncio
import pyzx
import os 
from Depth_Optimize import depth_optimize

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run the EPOC algorithm with parameters.")
    parser.add_argument(
        "--maxsize",
        metavar="m",
        type=int,
        nargs=1,
        help="The maximum size of the partition.")
    parser.add_argument(
        "--dir",
        dest="dirpath",
        default=False,
        type=str,
        nargs=1,
        help="Specify the qasm directory."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        nargs=1,
        help="The path to the file containing the pulses."
    )
    parser.add_argument(
        "--dest",
        dest="dest_file",
        type=str,
        nargs=1,
        help="The path to the file containing the performance data."
    )
    args = parser.parse_args()
    maxsize = args.maxsize[0]
    dirpath = args.dirpath[0]
    filepath = args.filepath[0]
    dest_file = args.dest_file[0]


    print(f"Maxsize: {maxsize}")
 
    print(f"Filepath: {filepath}")

    print(f"Directory: {dirpath}")
    
    # performance = pl.read_csv("performance_bench.csv")

    files = []

    for root, dirs, filenames in os.walk(dirpath):
        for filename in filenames:
            if filename.endswith('.qasm'):
                files.append(os.path.join(root, filename))
    # files = ["./QASMBench/small/adder_n4/adder_n4.qasm"]
    total_len = (len(files))

    for _ in tqdm(range(0, total_len)):
        print()
        print(files[_])
        dataset = pl.read_csv(dest_file)
        print("running ZX")

        result = depth_optimize(files[_])
        if result == None:
            continue
        qasm_str = result[1]
        qc = QuantumCircuit.from_qasm_str(qasm_str)  
        qc = qiskit_to_bqskit(qc)
        start_time = time.time()
        qc.remove_all_measurements()
        partitioned = circuit_partition(qc, maxsize)
        synthesized = cal_synthesis(
            partitioned,
            numworker=8
            )
        time1 = time.time() - start_time
        start_time = time.time()
        latency_regroup, fidelity_group = (regroup_and_cal_latency(synthesized, filepath))
        time2 = time.time() - start_time
        start_time = time.time()
        latency, fidelity = (cal_latency_all(synthesized, filepath))
        time3 = time.time() - start_time
        data = pl.DataFrame({
            "path":[files[_]],
            "latency": [latency],
            "latency_group": [latency_regroup],
            "compilation_time": [time1 + time3],
            "compilation_time_group": [time1 + time2],
            "fidelity": [fidelity],
            "fidelity_group": [fidelity_group]
        }) 

    

        dataset = pl.concat(
            [dataset, data], 
            how="vertical")
        dataset.write_csv(dest_file)

        

