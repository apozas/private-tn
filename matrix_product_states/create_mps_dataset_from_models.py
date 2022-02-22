# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2203.xxxxx
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           os, sys for filesystem operations
#           pandas for dataset operations
#           pickle for data loading
#           time for time tracking
#           tqdm for progress bar
# Last modified: Feb, 2022

################################################################################
# This file takes all trained MPS architectures and generates a dataset
# containing all the parameters of each model, either in canonical form or not.
################################################################################
import numpy as np
import os, sys
import pandas as pd
import pickle

from classifier import MatrixProductState
from time import time
from tqdm import tqdm
from utils_mps import flatten_mps_tensors

################################################################################
# Argument parsing
# Arguments to be input: whether canonical form is generated (c) or not (n)
################################################################################
args = sys.argv
if len(args) != 2:
    raise Exception('Please, specify whether you want to compute the canonical'
                    + 'form of the MPS by adding the argument `c` (for'
                    + 'canonical) or `n` (for non-canonical) after calling the '
                    + 'file.')
else:
    if args[1] == 'c':
        canonical = True
    elif args[1] == 'n':
        canonical = False
    else:
        raise Exception('Please, only use the arguments `c` for generating the '
                        + 'database of canonical-form MPS, or `n` for the '
                        + 'database of non-canonical MPS.')

parent_dir  = '..'
data_dir    = f'{parent_dir}/datasets'
model_dir   = f'{parent_dir}/models/mps'
MPS         = MatrixProductState(5, 2, 2, 2)
param_names = []

for ii, param in enumerate(MPS.tensors_in_finiteMPS_notation()):
    names = np.array([[[f'w{ii}_{jj}_{kk}_{ll}'
                        for jj in range(param.shape[0])]
                        for kk in range(param.shape[1])]
                        for ll in range(param.shape[2])]).flatten().tolist()
    param_names += names

all_columns = ['dataset', 'id', 'imbalance', 'type', 'accuracy'] + param_names

# Read and process models
time0 = time()
row_list = []
for instance in tqdm(os.listdir(model_dir)):
    if len(instance.split('.')) == 1:
        for params in os.listdir(f'{model_dir}/{instance}'):
            if params[-7:] == '.pickle':
                try:
                    imbalance = float(params[:-3].split('_')[1][:4])
                except ValueError:
                    imbalance = float(params[:-3].split('_')[1][:3])
                with open(f'{model_dir}/{instance}/{params}', 'rb') as file:
                    MPS.load_numpy(pickle.load(file))
                info = params[:-7].split('_')
                id = info[0]
                accuracy = float(info[-1][3:])
                typ = info[1][-3:]
                param_values = []
                if typ == 'ven':
                    typ = 'even'
                # Read parameters
                if canonical:
                    tensors = MPS.canonical_form(
                                               ).tensors_in_finiteMPS_notation()
                else:
                    tensors = MPS.tensors_in_finiteMPS_notation()

                params   = flatten_mps_tensors(tensors)
                row_data = [instance, id, imbalance, typ, accuracy] + params
                row_list.append(dict(zip(all_columns, row_data)))

dataframe = pd.DataFrame(row_list, columns=all_columns)
dataframe[['dataset', 'id']] = dataframe[['dataset', 'id']].astype(int)
dataframe[['imbalance']]     = dataframe[['imbalance']].astype(float)

print('Dataframe creation completed\n')
print('Time elapsed: {} seconds'.format(time() - time0))
print('\nStatistics of the dataset:')
print(dataframe.describe())

dataframe = dataframe.sort_values(by=['dataset', 'type', 'imbalance', 'id'])
dataframe = dataframe.reset_index(drop=True)
can_str = '' if canonical else 'n'
dataframe.to_csv(f'{data_dir}/dataset_of_mps_models_{can_str}can.csv')
