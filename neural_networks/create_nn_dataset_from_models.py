# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2203.xxxxx
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           os for filesystem operations
#           pandas for dataset operations
#           time for time tracking
#           torch for ML
#           tqdm for progress bar
# Last modified: Feb, 2022

################################################################################
# This file takes all trained neural networks and generates a dataset containing
# all the parameters of each model.
################################################################################
import os
import numpy as np
import pandas as pd

from time import time
from torch import load
from tqdm import tqdm
from utils_nn import SimpleNNModel

parent_dir = '..'
data_dir   = f'{parent_dir}/datasets'
model_dir  = f'{parent_dir}/models/nn'

# Generate column names
model       = SimpleNNModel()
weights_idx = 0
biases_idx  = 0
param_names = []
for param in model.parameters():
    if param.ndim == 1:
        names = [f'b{biases_idx}_{ii}' for ii in range(param.shape[0])]
        biases_idx += 1
    elif param.ndim == 2:
        names = np.array([[f'w{weights_idx}_{ii}_{jj}'
                           for ii in range(param.shape[0])]
                           for jj in range(param.shape[1])]).flatten().tolist()
        weights_idx += 1
    else:
        print('Something is wrong with')
        print(param)
    param_names += names

all_columns = ['dataset', 'id', 'imbalance', 'type', 'accuracy'] + param_names

# Read and process models
time0 = time()
row_list = []
for instance in tqdm(os.listdir(model_dir)):
    if len(instance.split('.')) == 1:
        for params in os.listdir(f'{model_dir}/{instance}'):
            if params[-3:] == '.pt':
                try:
                    imbalance = float(params[:-3].split('_')[1][:4])
                except ValueError:
                    imbalance = float(params[:-3].split('_')[1][:3])
                model.load_state_dict(load(f'{model_dir}/{instance}/{params}'))
                info = params[:-3].split('_')
                id = info[0]
                accuracy = float(info[-1][3:])
                typ = info[1][-3:]
                param_values = []
                if typ == 'ven':
                    typ = 'even'
                # Read parameters
                for param in model.parameters():
                    if param.ndim <= 2:
                        param_values += param.detach().cpu() \
                                             .numpy().flatten().tolist()
                row_data = [instance, id, imbalance, typ, accuracy]+param_values
                row_list.append(dict(zip(all_columns, row_data)))

dataframe = pd.DataFrame(row_list, columns=all_columns)
dataframe[['dataset', 'id']] = dataframe[['dataset', 'id']].astype(int)
dataframe[['imbalance']]     = dataframe[['imbalance']].astype(float)

print('Dataframe creation completed\n')
print(f'Time elapsed: {time()-time0} seconds')
print('\nStatistics of the dataset:')
print(dataframe.describe())

dataframe = dataframe.sort_values(by=['dataset', 'type', 'imbalance', 'id'])
dataframe = dataframe.reset_index(drop=True)
dataframe.to_csv(f'{data_dir}/dataset_of_nn_models.csv')
