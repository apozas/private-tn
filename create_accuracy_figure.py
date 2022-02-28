# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: glob, os for filesystem operations
#           matplotlib for plotting
#           numpy for array operations
#           pandas for dataset operations
#           pickle for data loading
#           seaborn for plot visuals
#           tensorflow, torch for ML
#           tqdm for progress bar
# Last modified: Feb, 2022

################################################################################
# This file generates Figure 2c in the paper, which compares the accuracy of
# neural networks and MPS trained to predict the outcome of a COVID-19 infection
# given demographics and symptoms.
################################################################################
import glob, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import tensorflow as tf
import torch

from matrix_product_states.utils_mps import preprocess_mps
from matrix_product_states.classifier import MatrixProductState
from neural_networks.utils_nn import SimpleNNModel, preprocess_nn
from tqdm import tqdm

################################################################################
# Style data
################################################################################
sns.set_theme()
sns.set(style='whitegrid', font_scale=1.8)
mpl.rcParams['lines.linewidth']  = 0.75
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams.update({'font.size': 18})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')

################################################################################
# Common information
################################################################################
imbalance_levels = np.array([0.5, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
data_dir         = 'datasets'
models_dir       = 'models'
results_dir      = 'results'
all_data         = f'{data_dir}/covid_argentina_colombia_until20210322.csv'
df               = pd.read_csv(all_data, index_col=0)
device           = torch.device('cpu')

################################################################################
# Neural network data
################################################################################
nn_data_exists = (len([file for file in glob.glob(f'{results_dir}/*_nn_*')])
                  == 4)
if not nn_data_exists:
    print('Calculating data for neural networks')
    df_nn = df.copy()
    df_nn = preprocess_nn(df_nn)

    labels          = df_nn['recovered'].to_numpy()
    feature_dataset = df_nn.drop(['recovered'], axis=1)
    data            = feature_dataset.to_numpy()

    data_tensor = torch.from_numpy(data).type(torch.FloatTensor).to(device)
    data_labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)

    NN = SimpleNNModel().to(device)

    all_accuracies = []
    for dataset in tqdm(os.listdir(f'{models_dir}/nn')):
        if len(dataset.split('.')) == 1:    # Discard non-folders
            accuracies_odd  = [[], [], [], [], [], [], [], [], []]
            accuracies_even = [[], [], [], [], [], [], [], [], []]
            accuracies      = [accuracies_odd, accuracies_even]
            for model in os.listdir(f'{models_dir}/nn/{dataset}'):
                if len(model.split('.')[-1]) == 2:    # Take .pt models
                    try:
                        imbalance = float(model.split('_')[1][:4])
                    except ValueError:
                        imbalance = float(model.split('_')[1][:3])
                    imbalance_pos = np.where(
                                             imbalance_levels == imbalance
                                             )[0].item()
                    typ = 0 if model.split('_')[1][-3:] == 'odd' else 1
                    NN.load_state_dict(torch.load(
                                          f'{models_dir}/nn/{dataset}/{model}'))
                    acc = (NN(data_tensor).argmax(dim=1)
                           == data_labels).sum().item()
                    acc /= len(data_labels)
                    accuracies[typ][imbalance_pos].append(acc)
        all_accuracies.append(accuracies)
    all_accuracies = np.array(all_accuracies)
    avg_accuracies_per_dataset = all_accuracies.mean(-1)
    acc_nns_odd    = avg_accuracies_per_dataset[:,0,:].mean(0)
    std_nns_odd    = avg_accuracies_per_dataset[:,0,:].std(0)
    acc_nns_even   = avg_accuracies_per_dataset[:,1,:].mean(0)
    std_nns_even   = avg_accuracies_per_dataset[:,1,:].std(0)
    np.savetxt(f'{results_dir}/acc_nn_odd.txt',  acc_nns_odd)
    np.savetxt(f'{results_dir}/acc_nn_even.txt', acc_nns_even)
    np.savetxt(f'{results_dir}/std_nn_odd.txt',  std_nns_odd)
    np.savetxt(f'{results_dir}/std_nn_even.txt', std_nns_even)
else:
    print('Loading data for neural networks')
    acc_nns_odd  = np.loadtxt(f'{results_dir}/acc_nn_odd.txt')
    acc_nns_even = np.loadtxt(f'{results_dir}/acc_nn_even.txt')
    std_nns_odd  = np.loadtxt(f'{results_dir}/std_nn_odd.txt')
    std_nns_even = np.loadtxt(f'{results_dir}/std_nn_even.txt')

################################################################################
# MPS data
################################################################################
mps_data_exists = (len([file
                           for file in glob.glob(f'{results_dir}/*_mps_*.txt')])
                   == 4)
if not mps_data_exists:
    print('Calculating data for MPS')
    MPS = MatrixProductState(5, 2, 2, 2)
    data_tensor, data_labels = preprocess_mps(df)
    data_tensor    = tf.cast(data_tensor, tf.float32)
    data_labels    = tf.cast(data_labels, tf.float32)
    all_accuracies = []
    for dataset in tqdm(os.listdir(f'{models_dir}/mps')):
        if len(dataset.split('.')) == 1:    # Discard non-folders
            accuracies_odd  = [[], [], [], [], [], [], [], [], []]
            accuracies_even = [[], [], [], [], [], [], [], [], []]
            accuracies      = [accuracies_odd, accuracies_even]
            for model in os.listdir(f'{models_dir}/mps/{dataset}'):
                if len(model.split('.')[-1]) == 6:    # Take .pickle models
                    try:
                        imbalance = float(model.split('_')[1][:4])
                    except ValueError:
                        imbalance = float(model.split('_')[1][:3])
                    imbalance_pos = np.where(
                                             imbalance_levels == imbalance
                                             )[0].item()
                    typ = 0 if model.split('_')[1][-3:] == 'odd' else 1
                    with open(f'{models_dir}/mps/{dataset}/{model}', 'rb') as f:
                        params = pickle.load(f)
                    MPS.load_numpy(params)
                    acc = MPS.accuracy(data_tensor, data_labels)
                    accuracies[typ][imbalance_pos].append(acc)
        all_accuracies.append(accuracies)
    all_accuracies = np.array(all_accuracies)
    avg_accuracies_per_dataset = all_accuracies.mean(-1)
    acc_mps_odd  = avg_accuracies_per_dataset[:,0,:].mean(0)
    std_mps_odd  = avg_accuracies_per_dataset[:,0,:].std(0)
    acc_mps_even = avg_accuracies_per_dataset[:,1,:].mean(0)
    std_mps_even = avg_accuracies_per_dataset[:,1,:].std(0)
    np.savetxt(f'{results_dir}/acc_mps_odd.txt',  acc_mps_odd)
    np.savetxt(f'{results_dir}/acc_mps_even.txt', acc_mps_even)
    np.savetxt(f'{results_dir}/std_mps_odd.txt',  std_mps_odd)
    np.savetxt(f'{results_dir}/std_mps_even.txt', std_mps_even)
else:
    print('Loading data for MPS')
    acc_mps_odd  = np.loadtxt(f'{results_dir}/acc_mps_odd.txt')
    acc_mps_even = np.loadtxt(f'{results_dir}/acc_mps_even.txt')
    std_mps_odd  = np.loadtxt(f'{results_dir}/std_mps_odd.txt')
    std_mps_even = np.loadtxt(f'{results_dir}/std_mps_even.txt')

################################################################################
# Plot
################################################################################
print(acc_mps_odd)
print(acc_mps_even)
plt.errorbar(imbalance_levels, acc_mps_odd,  std_mps_odd,
             marker='o', label='MPS, odd')
plt.errorbar(imbalance_levels, acc_mps_even, std_mps_even,
             marker='o', label='MPS, even')
plt.errorbar(imbalance_levels, acc_nns_odd,  std_nns_odd,
             marker='*', markersize=10, label='NNs, odd')
plt.errorbar(imbalance_levels, acc_nns_even, std_nns_even,
             marker='*', markersize=10, label='NNs, even')
plt.legend(loc=3, prop={'size': 14})
plt.xlabel('Percentage of majority irrelevant class')
plt.ylabel('Model accuracy')
plt.savefig('models.pdf', bbox_inches='tight')
