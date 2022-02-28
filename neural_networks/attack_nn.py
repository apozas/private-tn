# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           os for filesystem operations
#           pandas for dataset operations
#           random for random numbers
#           scikit-learn for logistic regression
#           torch for ML
#           tqdm for progress bar
# Last modified: Feb, 2022

################################################################################
# This file performs the attacks for inferring the irrelevant variable in neural
# networks. The attacks are, for each level of imbalance of the irrelevant
# variable, logistic regressors trained on all the models trained on 80 of the
# datasets (a total of 80x100x2). The accuracy is the evaluation of the
# regressor in all the models trained on the remaining 20 datasets.
################################################################################
import numpy as np
import os, random
import pandas as pd

from sklearn.linear_model import LogisticRegression
from torch import device, from_numpy, FloatTensor, LongTensor
from tqdm import tqdm

parent_dir       = '..'
data_dir         = f'{parent_dir}/datasets'
attack_dir       = f'{parent_dir}/results'
n_train_datasets = 80 # No. of datasets whose models will be used for training
train_test_split = 0.8
device           = device('cpu')

all_models  = pd.read_csv(f'{data_dir}/dataset_of_nn_models.csv', index_col=0)

train_accs  = []
train_stds  = []
test_accs   = []
test_stds   = []
val_accs    = []
val_stds    = []
attack_rows = []
attack_cols = ['imbalance', 'test_acc', 'train_datasets']

for imbalance, dataset in tqdm(all_models.groupby('imbalance'),
                               desc='Level of imbalance'):
    # Dataset preprocessing
    for col in dataset.columns:
        if col not in ['dataset', 'id', 'imbalance', 'accuracy', 'type']:
            dataset[col] -= dataset[col].mean()
            dataset[col] /= dataset[col].std()
    one_hot = pd.get_dummies(dataset['type'])
    dataset = dataset.join(one_hot)
    dataset = dataset.drop(columns=['type', 'imbalance'])

    train_accs_attacks = []
    test_accs_attacks  = []
    val_accs_attacks   = []

    for _ in tqdm(range(1000), leave=False):
        choice = random.sample(range(1, 101), n_train_datasets)
        train  = dataset[dataset.dataset.isin(choice)]
        test   = dataset[~dataset.dataset.isin(choice)]
        train_labels = train[['even', 'odd']]
        test_labels  = test[['even',  'odd']]
        train = train.drop(columns=['dataset', 'id', 'accuracy', 'even', 'odd'])
        test  = test.drop( columns=['dataset', 'id', 'accuracy', 'even', 'odd'])

        # Split training set in train-validation
        real_train        = train.sample(frac=train_test_split)
        real_train_labels = train_labels.loc[real_train.index]
        val_indices       = set(train.index) - set(real_train.index)

        val_set      = train.loc[val_indices].to_numpy()
        val_labels   = train_labels.loc[val_indices].to_numpy()
        train_set    = real_train.to_numpy()
        train_labels = real_train_labels.to_numpy()
        test_set     = test.to_numpy()
        test_labels  = test_labels.to_numpy()

        train_tensor = from_numpy(train_set).type(FloatTensor).to(device)
        train_labels = from_numpy(train_labels).type(LongTensor).to(device)
        test_tensor  = from_numpy(test_set).type(FloatTensor).to(device)
        test_labels  = from_numpy(test_labels).type(LongTensor).to(device)
        val_tensor   = from_numpy(val_set).type(FloatTensor).to(device)
        val_labels   = from_numpy(val_labels).type(LongTensor).to(device)

        # Attack: logistic regression
        reg = LogisticRegression().fit(train_set, train_labels[:,0].cpu())
        train_acc = reg.score(train_set, train_labels[:,0].cpu())
        test_acc  = reg.score(test_set, test_labels[:,0].cpu())
        val_acc   = reg.score(val_set, val_labels[:,0].cpu())
        train_accs_attacks.append(train_acc)
        test_accs_attacks.append(test_acc)
        val_accs_attacks.append(val_acc)

        attack_rows.append([imbalance, test_acc, sorted(choice)])

    train_accs.append(np.mean(train_accs_attacks))
    train_stds.append(np.std(train_accs_attacks))
    test_accs.append(np.mean(test_accs_attacks))
    test_stds.append(np.std(test_accs_attacks))
    val_accs.append(np.mean(val_accs_attacks))
    val_stds.append(np.std(val_accs_attacks))

print('Train accuracies:')
print('Mean: ', train_accs)
print('Std: ',  train_stds)
print('Validation accuracies:')
print('Mean: ', val_accs)
print('Std: ',  val_stds)
print('Test accuracies:')
print('Mean: ', test_accs)
print('Std: ',  test_stds)

# Export
if not os.path.exists(attack_dir):
    print('Attack directory not found. Creating...')
    os.makedirs(attack_dir)

attack_rows = np.array(attack_rows, dtype=object).T
pd.DataFrame(dict(zip(attack_cols, attack_rows)), dtype=object) \
            .sort_values(by=['imbalance']).reset_index(drop=True) \
            .to_csv(f'{attack_dir}/attacks_nn.csv')
