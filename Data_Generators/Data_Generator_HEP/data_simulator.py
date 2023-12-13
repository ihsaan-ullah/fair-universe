import os
import pandas as pd
import numpy as np
from systematics import Systematics, DER_data
from bootstrap import bootstrap_data



def dataSimulator(size = 1000, n=10, verbose=0, tes=1.0, mu=1.0,Bootstrap=False):
    '''
    Simulates data for the HEP competition using reference data and systematics.

    Args:
        size (int): Number of samples to simulate. Default is 1000.
        n (int): Number of samples to simulate. Default is 10.
        verbose (int): Verbosity level. Default is 0.
        tes (float): TES value. Default is 1.0.
        mu (float): Mu value. Default is 1.0.
        Bootstrap (bool): Bootstrap flag. Default is False.
    Returns:
        data_set_bs (list): List of simulated data sets.

    '''

    # Get the directory of the current script (datagen_temp.py)
    module_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the absolute path to reference_data.csv
    csv_file_path = os.path.join(module_dir, 'public_data.csv')
    df = pd.read_csv(csv_file_path)

    # Sample n rows from the reference data
    # Apply systematics to the sampled data
    data_syst = Systematics(
        data=df,
        verbose=verbose,
        tes=tes
    ).data


    data_syst['Weight'] *= mu
    data_set_bs = []
    # Apply bootstrap to the data
    if Bootstrap:

        for i in range(n):
            data = data_syst.sample(n=size, replace=True)
            data_bs = bootstrap_data(data) 
            data_set_bs.append(data_bs)

    else :
        data_bs = data_syst.sample(n=size, replace=False)
        data_set_bs.append(data_bs)

    return data_set_bs

if __name__ == '__main__':
    data = dataSimulator()

    print(data.head())

    print(data.describe())

    print("Data Simulation Completed")
    

