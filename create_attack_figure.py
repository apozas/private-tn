# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2203.xxxxx
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: matplotlib for plotting
#           pandas for dataset operations
#           seaborn for plot visuals
# Last modified: Feb, 2022

################################################################################
# This file generates Figure 2d in the paper, which compares the robustness of
# white-box neural networks and MPS against guessing the bias of an irrelevant
# input feature in the training dataset.
################################################################################
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Style data
sns.set_theme()
sns.set(style='whitegrid', font_scale=1.8)
mpl.rcParams['lines.linewidth']  = 0.75
mpl.rcParams['errorbar.capsize'] = 3
mpl.rcParams.update({'font.size': 18})
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('font', family='serif')

results_dir = 'results'

# X axis
imbalances = [0.5, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Neural networks
nn_data = pd.read_csv(f'{results_dir}/attacks_nn.csv', index_col=0)
acc_nn  = []
std_nn  = []
for imbalance, attacks in nn_data.groupby('imbalance'):
    acc_nn.append(attacks['test_acc'].mean())
    std_nn.append(attacks['test_acc'].std())

# MPS
mps_ncan_data = pd.read_csv(f'{results_dir}/attacks_mps_ncan.csv', index_col=0)
acc_mps_ncan  = []
std_mps_ncan  = []
for imbalance, attacks in mps_ncan_data.groupby('imbalance'):
    acc_mps_ncan.append(attacks['test_acc'].mean())
    std_mps_ncan.append(attacks['test_acc'].std())

# MPS in canonical form
mps_can_data = pd.read_csv(f'{results_dir}/attacks_mps_can.csv', index_col=0)
acc_mps_can  = []
std_mps_can  = []
for imbalance, attacks in mps_can_data.groupby('imbalance'):
    acc_mps_can.append(attacks['test_acc'].mean())
    std_mps_can.append(attacks['test_acc'].std())

# Plot
plt.errorbar(imbalances, acc_nn, std_nn,
             marker='*', markersize=10, label='Neural networks')
plt.errorbar(imbalances, acc_mps_ncan, std_mps_ncan,
             marker='o', label='MPS')
plt.errorbar(imbalances, acc_mps_can, std_mps_can,
             marker='d', label=('MPS, canonical'))
plt.legend()
plt.xlabel('Percentage of majority irrelevant class')
plt.ylabel('Average attack accuracy')
plt.savefig('attacks.pdf', bbox_inches='tight')
