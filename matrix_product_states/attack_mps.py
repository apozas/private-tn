# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           pandas for dataset operations
#           scikit-learn for ML utils
#           os, sys for filesystem operations
#           tensorflow for ML
#           tqdm for progress bar
# Last modified: Feb, 2023

###############################################################################
# This file performs the attacks for inferring the irrelevant variable in MPS.
# architectures. The attacks are, for each level of imbalance of the irrelevant
# variable, neural networks trained on all the models trained on 80 of the
# datasets (a total of 80x100x2). The accuracy is the evaluation of the
# resulting neural network in all the models trained on the remaining datasets.
###############################################################################
import numpy as np
import os, sys
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm

###############################################################################
# Argument parsing
# Arguments to be input: whether canonical form is generated (c) or not (n)
###############################################################################
args = sys.argv
sample = ''
if len(args) not in [2, 3]:
    raise Exception('Please, specify whether you want to attack the canonical '
                    + 'form of the MPS by adding the argument `c` (for '
                    + 'canonical), `n` (for non-canonical) or `u` (for '
                    + 'univocal) after calling the file.')
else:
    if args[1] not in ['c', 'n', 'u']:
        raise Exception('Please, only use the arguments `c` for attacking the '
                        + 'database of canonical-form MPS, or `n` for the '
                        + 'database of non-canonical MPS.')
    else:
        can = args[1]
        if can == 'c':
            can_str    = 'can'
            canoni_str = 'canoni'
        elif can == 'n':
            can_str    = 'ncan'
            canoni_str = 'non-canoni'
        else:
            can_str    = 'uni'
            canoni_str = 'univo'
    if len(args) == 3:
        if args[2] != 's':
            raise Exception('The last argument must only be `s` in case you '
                            + 'want to sample resudial gauges. Otherwise, '
                            + 'leave it blank')
        else:
            sample = '_sample'


###############################################################################
# Attack: a neural network that is trained on many collections of MPS
# parameters and whose task is to infer the nature of one of the variables used
# in the training dataset of such models
###############################################################################
def define_attack_and_train(train_data, train_labels, val_data, val_labels,
                            test_data, test_labels, hparams,
                            verbose=False):
    regularizer = L2(l2=hparams['REGULARIZER'])
    layers = []
    layers.append(Input(shape=(n_features,)))
    layers.append(Dropout(rate=hparams['DROPOUT'], input_shape=(n_features,)))
    for ii in range(hparams['N_LAYERS']):
        layers.append(Dense(hparams['N_UNITS']//min(2**ii, hparams['N_UNITS']),
                            activation="relu",
                            kernel_regularizer=regularizer))
        layers.append(Dropout(rate=hparams['DROPOUT']))
        layers.append(Dense(hparams['N_UNITS']//min(2**ii, hparams['N_UNITS']),
                            activation="relu",
                            kernel_regularizer=regularizer))
        layers.append(Dropout(rate=hparams['DROPOUT']))
    layers.append(Dense(n_labels,
                        activation="relu",
                        kernel_regularizer=regularizer))
    layers.append(Dropout(rate=hparams['DROPOUT']))
    layers.append(Dense(n_labels,
                        activation="softmax",
                        kernel_regularizer=regularizer))
    model = Sequential(layers)

    model.compile(optimizer=Adam(learning_rate=hparams['LEARNING_RATE']),
                  loss='categorical_crossentropy',
                  metrics=["mae", "accuracy"])

    # Train
    if hparams['PATIENCE'] > 0:
        callback = EarlyStopping(monitor='val_loss',
                                 patience=hparams['PATIENCE'],
                                 restore_best_weights=True)
        training_history = model.fit(x=train_data, y=train_labels,
                                     validation_data=(val_data, val_labels),
                                     epochs=hparams['N_EPOCHS'],
                                     batch_size=hparams['BATCH_SIZE'],
                                     callbacks=[callback],
                                     verbose=verbose)
    else:
        training_history = model.fit(x=train_data, y=train_labels,
                                     validation_data=(val_data, val_labels),
                                     epochs=hparams['N_EPOCHS'],
                                     batch_size=hparams['BATCH_SIZE'],
                                     verbose=verbose)
    # Evaluate on untrained data. This gives the attack's accuracy
    _, _, accuracy_test = model.evaluate(test_data, test_labels,
                                         verbose=verbose)

    return model, accuracy_test

###############################################################################
# Parameters
###############################################################################
parent_dir   = '..'
data_dir     = f'{parent_dir}/datasets'
attack_dir   = f'{parent_dir}/results'
all_models   = pd.read_csv(data_dir + '/dataset_of_mps_models_'
                           + f'{can_str}{sample}.csv',
                           index_col=0)
test_size        = 0.2
n_train_datasets = 80  # No. of datasets whose models will be used for training
data_columns     = all_models.columns[5:]
n_features       = len(data_columns)
n_labels         = 2
hparams          = {
                    'N_LAYERS': 2,
                    'N_UNITS': n_features//2,
                    'LEARNING_RATE': 0.001,
                    'REGULARIZER': 0.0001,
                    'DROPOUT': 0.,
                    'BATCH_SIZE': 1000,
                    'N_EPOCHS': 1000,
                    'PATIENCE': 0
                    }

###############################################################################
# Attacks
###############################################################################
test_accs   = []
test_stds   = []
attack_rows = []
attack_cols = ['imbalance', 'test_acc', 'train_datasets']
for imbalance, dataset in tqdm(all_models.groupby('imbalance'),
                               desc='Level of imbalance'):
    scaler = MinMaxScaler()
    dataset[data_columns] = scaler.fit_transform(dataset[data_columns])

    one_hot = pd.get_dummies(dataset['type'])
    dataset = dataset.join(one_hot)
    dataset = dataset.drop(columns=['type', 'imbalance'])

    test_accs_attacks = []
    for _ in tqdm(range(1000), leave=False):

        choice = np.random.choice(range(1, 101),
                                  n_train_datasets,
                                  replace=False)

        train  = dataset[dataset.dataset.isin(choice)]
        test   = dataset[~dataset.dataset.isin(choice)]
        train_labels, test_labels = train[['even', 'odd']],test[['even', 'odd']]
        train = train.drop(columns=['dataset', 'id', 'accuracy', 'even', 'odd'])
        test  = test.drop( columns=['dataset', 'id', 'accuracy', 'even', 'odd'])

        real_train        = train.sample(frac=1-test_size)
        real_train_labels = train_labels.loc[real_train.index]
        val_indices       = set(train.index) - set(real_train.index)

        X_val   = train.loc[val_indices].to_numpy()
        y_val   = train_labels.loc[val_indices].to_numpy()
        X_train = real_train.to_numpy()
        y_train = real_train_labels.to_numpy()
        X_test  = test.to_numpy()
        y_test  = test_labels.to_numpy()

        # Shuffle
        X_train, y_train = shuffle(X_train, y_train)
        X_val,   y_val   = shuffle(X_val,   y_val)

        X_train = tf.cast(X_train, dtype=tf.float32)
        y_train = tf.cast(y_train, dtype=tf.float32)
        X_val   = tf.cast(X_val, dtype=tf.float32)
        y_val   = tf.cast(y_val, dtype=tf.float32)

        # Attack: Shadow training
        test_acc = 0.5
        while abs(test_acc - 0.5) < 1e-6:    # Discard obviously wrong trainings
            _, test_acc = define_attack_and_train(X_train, y_train,
                                                  X_val,   y_val,
                                                  X_test,  y_test,
                                                  hparams)
        attack_rows.append([imbalance, test_acc, sorted(choice)])
        test_accs_attacks.append(test_acc)
    test_accs.append(np.mean(test_accs_attacks))
    test_stds.append(np.std(test_accs_attacks))

print('Test accuracies, shadow training attacks on '
      + f'{canoni_str}cal MPS models:')
print('Mean: ', test_accs)
print('Std: ',  test_stds)

# Export
if not os.path.exists(attack_dir):
    print('Attack directory not found. Creating...')
    os.makedirs(attack_dir)

attack_rows = np.array(attack_rows, dtype=object).T
pd.DataFrame(dict(zip(attack_cols, attack_rows)), dtype=object) \
            .sort_values(by=['imbalance']).reset_index(drop=True) \
            .to_csv(f'{attack_dir}/attacks_mps_{can_str}{sample}.csv')
