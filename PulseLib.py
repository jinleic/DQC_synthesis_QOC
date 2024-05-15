from quantum_optimal_control.main_grape.grape import Grape
import numpy as np
import time
import polars as pl
import base64
# import msgpack
import pickle
import pandas as pd
class PulseLib:
    csv_file: str = None
    pulse_lib: pl.DataFrame = None
    pulse_runtime: dict = None
    pre_compile_time: float = None

    def __init__(self, csv_file: str) -> None:
        self.csv_file = csv_file
        self.pulse_lib = pl.read_csv(self.csv_file)
        self.pulse_runtime = {}
        self.pre_compile_time = 0
    
    def cal_pulse(self, U: np.ndarray, grape_para, initial_guess=[]):
        # Utilization of GRAPE algorithm
        steps = [0, 0, 20, 40, 60, 120, 200]
        time_para = grape_para.N
        checked = set()
        all_records = []
        valid_records = []
        latency = 0
        iter = 0
        res = None
        while time_para not in checked:
            checked.add(time_para)
            if ( time_para + 1 > len(steps) or time_para < 0 ):
                break
            min_step = steps[time_para]
            max_step = steps[time_para + 1]
            delta = max(int((max_step - min_step) / 20), 1)
            while min_step + delta < max_step:
                iter += 1
                mid_step = int((min_step + max_step) / 2)
                total_time = mid_step * grape_para.dt
                # print(f'iter: {iter}, steps: {mid_step} / [{min_step} - {max_step}] total_time: {total_time}')
                res = Grape(grape_para.H0, grape_para.Hops, grape_para.Hnames, 
                            U, total_time, mid_step, grape_para.states_concerned_list,
                            grape_para.convergence, reg_coeffs=grape_para.reg_coeffs,
                            initial_guess=initial_guess, use_gpu=True, sparse_H=False,
                            method='ADAM', maxA=grape_para.maxA, show_plots=False,
                            save=False, return_converged=True)
                # print(f'total_time: {total_time}, res: {res.l}')
                if( res.l <= grape_para.convergence['conv_target'] ):
                    max_step = mid_step
                    valid_records.append((total_time, res))
                else:
                    min_step = mid_step
            if len(valid_records) == 0:
                time_para += 1
            elif min_step == steps[time_para] and time_para != 0:
                time_para -= 1
            else:
                break
        if len(valid_records) == 0:
            raise ValueError('No valid records found')
        else:
            return total_time, (res.__dict__)['l']
        
    
    def get_pulse(self, U: np.ndarray, grape_para, initial_guess=[]):
        self.dataset = pd.read_csv(self.csv_file)
        dataset = self.dataset
        def decode_base64(series):
            return series.apply(lambda x: pickle.loads(base64.b64decode(x)))
        dataset_pulse = decode_base64(dataset['unitary'])
        def check_similarity(x, U):
            if x.shape != U.shape:
                return False
            elif np.allclose(x, U, atol=1e-1, rtol=1e-1):
                return True
            else:
                inner_product = np.abs(np.dot(x.conj().T, U))[0, 0]
                return np.isclose(inner_product, 1, atol=1e-1, rtol=1e-1)

        matched = dataset_pulse.apply(lambda x: check_similarity(x, U))
        # matched = matched[matched == True].index
        filtered_dataset = dataset[matched == True]
        if len(filtered_dataset) == 0:
            print('No pulse found, calculating...')
            start_time = time.time()
            res = self.cal_pulse(U, grape_para, initial_guess)
            compilation_time = time.time() - start_time
            unitary_base64 = base64.b64encode(
                pickle.dumps(U)).decode('utf-8'),
            data_new = pd.DataFrame({
                    'unitary': [unitary_base64[0]],
                    'total_time': [res[0]],
                    'fidelity': [res[1]],
                    'compilation_time': [compilation_time]
            })
            dataset = pd.concat(
                    [dataset, data_new],
                    ignore_index=True)
            dataset.to_csv(self.csv_file, index=False)
            print('New pulse added')
            return res[0], res[1], compilation_time
        else:
            print('Pulse found')
            # print(filtered_dataset)
            # print(filtered_dataset['total_time'][0])
            total_time = filtered_dataset['total_time'].values[0]
            # print(total_time)
            fidelity = filtered_dataset['fidelity'].values[0]
            compilation_time = filtered_dataset['compilation_time'].values[0]
            return total_time, fidelity, compilation_time





if __name__ == "__main__":
    from Grape_Para import Grape_Para
    pulse_lib = PulseLib('pulse_lib.csv')
    unitary = np.array([[0, 1], [1, 0]])
    grape_para = Grape_Para(int(np.log2(unitary.shape[0])))
    pulse_lib.get_pulse(unitary, grape_para)
    # pulse_lib.get_pulse(unitary, None, None)
    # dataset0 = pl.read_csv('pulse_lib.csv')
    # unitary_1 = base64.b64encode(
    #         pickle.dumps(np.array([[1, 0, 0, 0], 
    #                                [0, 1, 0, 0], 
    #                                [0, 0, 1, 0], 
    #                                [0, 0, 0, 1]]))
    #     ).decode('utf-8'),
    # dataset = pl.DataFrame({
    #     'unitary': unitary_1[0],
    #     'total_time': 0,
    #     'fidelity': 1,
    #     'compilation_time': 0
    # }, schema={
    #     'unitary': pl.String,
    #     'total_time': pl.Float64,
    #     'fidelity': pl.Float64,
    #     'compilation_time': pl.Float64
    # })
    # dataset0 = pl.concat(
    #     [dataset0, dataset],
    #     how='vertical')
    # dataset0.write_csv('pulse_lib.csv')
     