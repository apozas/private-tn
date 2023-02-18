# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           os for filesystem operations
#           pandas for dataset operations
#           pickle for data loading
#           scikit-learn for ML utils
#           seaborn for plot visuals
#           tensorflow for ML
#           tqdm for progress bar
# Last modified: Feb, 2023

###############################################################################
# This file trains the MPS models, in the same way that the neural network
# models are trained.
###############################################################################
import numpy as np
import os, sys
import pandas as pd
import pickle
import tensorflow as tf
import training

from classifier import MatrixProductState as MPS
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from utils_mps import preprocess_mps

###############################################################################
# Argument parsing
# Arguments to be (optionally) input: Start dataset, end dataset
###############################################################################
args = sys.argv
if len(args) == 1:
    beginning = 0
    end = 100
elif len(args) == 2:
    beginning = int(args[1])
    end = 100
elif len(args) == 3:
    beginning = int(args[1])
    end = int(args[2])

###############################################################################
# Parameters
###############################################################################
parent_dir        = '..'
data_dir          = f'{parent_dir}/datasets'
model_dir         = f'{parent_dir}/models/mps'
d_bond            = 2       # Dimension of the connections between MPS tensors
d_phys            = 2       # Dimension of each component of the input
batch_size        = 100
learning_rate     = 0.1
n_epochs          = 20
mps_acc_threshold = 0.81    # Minimum training accuracy expected
black_box_diff    = 0.05    # Maximum difference between black-box accuracies

###############################################################################
# Check directories and models already trained
###############################################################################
data = pd.read_csv(f'{data_dir}/covid_argentina_colombia_until20210322.csv',
                   header=0, index_col=0, low_memory=False)
print('Dataframe loaded')
# Data directories
if not all([os.path.exists(f'{data_dir}/{instance}')
            for instance in range(1, 101)]):
    print('Data directories not found. Creating...')
    for instance in range(1, 101):
        os.makedirs(f'{data_dir}/{instance}')
        df[df.odd == True].sample(n=1000
                                  ).to_csv(f'{data_dir}/{instance}/odd.csv')
        df[df.odd == False].sample(n=1000
                                   ).to_csv(f'{data_dir}/{instance}/even.csv')

# Model directories
if not all([os.path.exists(f'{model_dir}/{instance}')
            for instance in range(1, 101)]):
    print('Model directories not found. Creating...')
    for instance in range(1, 101):
        os.makedirs(f'{model_dir}/{instance}')

print('Checking how many models we have already trained')
all_models = []
imbalances = [0.50, 0.51, 0.55, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
for instance in range(1, 101):
    fifty      = [0, 0, 0.50]
    fiftyone   = [0, 0, 0.51]
    fiftyfive  = [0, 0, 0.55]
    sixty      = [0, 0, 0.60]
    seventy    = [0, 0, 0.70]
    eighty     = [0, 0, 0.80]
    ninety     = [0, 0, 0.90]
    ninetyfive = [0, 0, 0.95]
    ninetynine = [0, 0, 0.99]
    for filename in [name for name in os.listdir(f'{model_dir}/{instance}')
                          if name.split('.')[-1] == 'pickle']:
        relevant_info = filename.split('_')[1]
        try:
            imbalance_level = float(relevant_info[:4])
        except ValueError:
            imbalance_level = float(relevant_info[:3])
        sign = 0 if relevant_info[-3:] == 'odd' else 1
        if imbalance_level == 0.5:
            fifty[sign] += 1
        elif imbalance_level == 0.51:
            fiftyone[sign] += 1
        elif imbalance_level == 0.55:
            fiftyfive[sign] += 1
        elif imbalance_level == 0.6:
            sixty[sign] += 1
        elif imbalance_level == 0.7:
            seventy[sign] += 1
        elif imbalance_level == 0.8:
            eighty[sign] += 1
        elif imbalance_level == 0.9:
            ninety[sign] += 1
        elif imbalance_level == 0.95:
            ninetyfive[sign] += 1
        elif imbalance_level == 0.99:
            ninetynine[sign] += 1
    models = np.array([fifty, fiftyone, fiftyfive, sixty, seventy, eighty,
                       ninety, ninetyfive, ninetynine],
                      dtype=object)
    all_models.append(models)

###############################################################################
# Prepare datasets and train
###############################################################################
print('Preparing test dataset')
# Test dataset, from the full dataset
X_test, y_test = preprocess_mps(data.sample(n=5000))
# Complete dataset, for black-box comparison
X_total, y_total = preprocess_mps(data)
odd_indices  = np.where(X_total[:, -1, 0] > 0.5)[0]
even_indices = list(set(range(len(data))) - set(odd_indices))
X_even_total = tf.cast(X_total[even_indices], tf.float32)
y_even_total = tf.cast(y_total[even_indices], tf.float32)
X_odd_total  = tf.cast(X_total[odd_indices], tf.float32)
y_odd_total  = tf.cast(y_total[odd_indices], tf.float32)

n_features = X_total.shape[1]
n_labels   = y_total.shape[1]

for jj in range(len(imbalances)):
    for instance, models_done in list(enumerate(all_models))[beginning:end]:
        instance += 1
        even = pd.read_csv(f'{data_dir}/{instance}/even.csv',
                           index_col=0)
        odd = pd.read_csv(f'{data_dir}/{instance}/odd.csv',
                          index_col=0)
        n_odd, n_even, percent_good = models_done[jj]
        # Generate or use existing imbalanced datasets
        if f'{percent_good}even.csv' in os.listdir(f'{data_dir}/{instance}'):
            even_dataset = pd.read_csv(
                               f'{data_dir}/{instance}/{percent_good}even.csv',
                               index_col=0)
        else:
            even_dataset = even.sample(frac=percent_good
                                       ).append(odd.sample(frac=1-percent_good),
                                                ignore_index=True)
            even_dataset.to_csv(f'{data_dir}/{instance}/{percent_good}even.csv')

        X_even, y_even = preprocess_mps(even_dataset)
        X_even, y_even = shuffle(X_even, y_even)

        if f'{percent_good}odd.csv' in os.listdir(f'{data_dir}/{instance}'):
            odd_dataset = pd.read_csv(
                                f'{data_dir}/{instance}/{percent_good}odd.csv',
                                index_col=0)
        else:
            odd_dataset = odd.sample(frac=percent_good
                                     ).append(even.sample(frac=1-percent_good),
                                              ignore_index=True)
            odd_dataset.to_csv(f'{data_dir}/{instance}/{percent_good}odd.csv')

        X_odd, y_odd = preprocess_mps(odd_dataset)
        X_odd, y_odd = shuffle(X_odd, y_odd)

        t = tqdm(total=100, initial=n_even,
                 desc=f'Dataset {instance}, {percent_good} imbalance')
        ii = 0
        while ii < 100:
            if n_even < 100:
                # Train model with even dataset
                test_acc = 0
                while test_acc < mps_acc_threshold:    # Avoid faulty trainings
                    mps            = MPS(n_features, n_labels, d_phys, d_bond)
                    optimizer      = Adam(learning_rate=learning_rate)
                    mps_trained, _ = training.fit(mps, optimizer,
                                                  X_even, y_even,
                                                  n_epochs,
                                                  batch_size=batch_size)
                    _, test_acc    = training.evaluate(mps_trained, X_test,
                                                       y_test, batch_size)
                # Only save if the black boxes are similar enough
                even_acc = mps_trained.accuracy(X_even_total, y_even_total)
                odd_acc  = mps_trained.accuracy(X_odd_total,  y_odd_total)
                if np.abs(even_acc - odd_acc) < black_box_diff:
                    filedir = (f'{model_dir}/{instance}'
                      + f'/{n_even}_{percent_good}even_acc{round(test_acc, 3)}'
                      + '.pickle')
                    with open(filedir, 'wb') as file:
                        mps_numpy = [tensor.numpy()
                                     for tensor in mps_trained.tensors]
                        pickle.dump(mps_numpy, file)
                    n_even += 1
            if n_odd < 100:
                # Train model with odd dataset
                test_acc = 0
                while test_acc < mps_acc_threshold:    # Avoid faulty trainings
                    mps            = MPS(n_features, n_labels, d_phys, d_bond)
                    optimizer      = Adam(learning_rate=learning_rate)
                    mps_trained, _ = training.fit(mps, optimizer,
                                                  X_odd, y_odd,
                                                  n_epochs,
                                                  batch_size=batch_size)
                    _, test_acc    = training.evaluate(mps_trained, X_test,
                                                       y_test, batch_size)
                # Only save if the black boxes are similar enough
                even_acc = mps_trained.accuracy(X_even_total, y_even_total)
                odd_acc  = mps_trained.accuracy(X_odd_total,  y_odd_total)
                if np.abs(even_acc - odd_acc) < black_box_diff:
                    filedir = (f'{model_dir}/{instance}'
                        + f'/{n_odd}_{percent_good}odd_acc{round(test_acc, 3)}'
                        + '.pickle')
                    with open(filedir, 'wb') as file:
                        mps_numpy = [tensor.numpy()
                                     for tensor in mps_trained.tensors]
                        pickle.dump(mps_numpy, file)
                    n_odd += 1
                    t.update(1)
            ii += 1
        t.close()
