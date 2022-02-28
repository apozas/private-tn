# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: copy for object copying
#           numpy for array operations
#           os, sys for filesystem operations
#           pandas for dataset operations
#           torch for ML
#           tqdm for progress bar
# Last modified: Feb, 2022

################################################################################
# This file trains the neural network models.
# We produce 100 datasets from the global.health database, each with 1000 points
# with even registration day and 1000 points with odd registration day. From
# these, we sample datasets that have some percentages of odd and even points,
# and we train 100 models on each of the resulting datasets (all this, x2 for
# majority of odd vs. majority of even).
################################################################################
import os, sys
import torch
import numpy as np
import pandas as pd

from copy import deepcopy
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils_nn import *

################################################################################
# Argument parsing
# Arguments to be (optionally) input: Start dataset, end dataset, device
################################################################################
args   = sys.argv
dev_no = 'cuda:0'
if len(args) == 1:
    beginning = 0
    end       = 100
elif len(args) == 2:
    beginning = 0
    end       = 100
    dev_no    = args[1]
elif len(args) == 3:
    beginning = int(args[1])
    end       = int(args[2])
else:
    beginning = int(args[1])
    end       = int(args[2])
    dev_no    = args[3]

################################################################################
# Parameters
################################################################################
parent_dir       = '..'
data_dir         = f'{parent_dir}/datasets'
model_dir        = f'{parent_dir}/models/nn'
criterion        = nn.CrossEntropyLoss()
num_epochs       = 1250
learning_rate    = 3e-4
weight_decay     = 6e-3
early_stop_crit  = 100
target           = 'recovered'
batch_size       = 8
nn_acc_threshold = 0.84    # Minimum training accuracy expected
device           = torch.device(dev_no if torch.cuda.is_available() else 'cpu')
np.random.seed(1)

################################################################################
# Check directories and models already trained
################################################################################
df = pd.read_csv(f'{data_dir}/covid_argentina_colombia_until20210322.csv',
                 index_col=0)
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
                          if name.split('.')[-1] == 'pt']:
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

# print('Count of models in the working range')
# print(all_models[beginning:end])

################################################################################
# Prepare datasets and train
################################################################################
print('Preparing test dataset')
# Test dataset, from the full dataset
test_dataset    = preprocess_nn(df.sample(n=5000))
labels          = test_dataset[target].to_numpy()
feature_dataset = test_dataset.drop([target], axis=1)
data            = feature_dataset.to_numpy()

test_tensor = torch.from_numpy(data).type(torch.FloatTensor).to(device)
test_labels = torch.from_numpy(labels).type(torch.LongTensor).to(device)

print('Begin training')
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
        even_dataset = preprocess_nn(even_dataset)

        input_tensor_even, label_tensor_even, _, _ = \
            dataset_to_torch(even_dataset, 1)

        input_tensor_even = input_tensor_even.to(device)
        label_tensor_even = label_tensor_even.to(device)

        if f'{percent_good}odd.csv' in os.listdir(f'{data_dir}/{instance}'):
            odd_dataset = pd.read_csv(
                                 f'{data_dir}/{instance}/{percent_good}odd.csv',
                                      index_col=0)
        else:
            odd_dataset = odd.sample(frac=percent_good
                                     ).append(even.sample(frac=1-percent_good),
                                              ignore_index=True)
            odd_dataset.to_csv(f'{data_dir}/{instance}/{percent_good}odd.csv')
        odd_dataset = preprocess_nn(odd_dataset)

        input_tensor_odd, label_tensor_odd, _, _ = \
            dataset_to_torch(odd_dataset, 1)

        input_tensor_odd = input_tensor_odd.to(device)
        label_tensor_odd = label_tensor_odd.to(device)

        t = tqdm(total=100, initial=n_even,
                 desc=f'Dataset {instance}, {percent_good} imbalance')
        ii = 0
        while ii < 100:
            if n_even < 100:
                # Train model with even dataset
                test_acc = 0
                while test_acc < nn_acc_threshold:    # Avoid faulty trainings
                    net = SimpleNNModel().to(device)
                    optimizer = torch.optim.Adam(net.parameters(),
                                                 lr=learning_rate,
                                                 weight_decay=weight_decay)
                    loader = DataLoader([[point, label]
                                    for point, label in zip(input_tensor_even,
                                                            label_tensor_even)],
                                        shuffle=True,
                                        batch_size=batch_size)
                    best_test_acc = 0
                    for epoch in range(num_epochs):
                        for batch, label in loader:
                            output = net(batch)
                            loss   = criterion(output, label)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        test_acc = (net(test_tensor).argmax(dim=1)
                                    == test_labels).sum().item()
                        test_acc /= len(test_labels)
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                            early_stop    = 0
                            best_model    = deepcopy(net.state_dict())
                        elif abs(test_acc - best_test_acc) < 1e-4:
                            best_model = deepcopy(net.state_dict())
                        else:
                            early_stop += 1
                        if early_stop > early_stop_crit:
                            break
                    net.load_state_dict(best_model)
                    test_acc = (net(test_tensor).argmax(dim=1)
                                == test_labels).sum().item()
                    test_acc /= len(test_labels)
                    if test_acc >= nn_acc_threshold:
                        torch.save(net.cpu().state_dict(),
                                   f'{model_dir}/{instance}'
                                   + f'/{n_even}_{percent_good}'
                                   + f'even_acc{round(test_acc, 3)}.pt')
                        n_even += 1
            if n_odd < 100:
                # Train model with odd dataset
                test_acc = 0
                while test_acc < nn_acc_threshold:    # Avoid faulty trainings
                    net = SimpleNNModel().to(device)
                    optimizer = torch.optim.Adam(net.parameters(),
                                                 lr=learning_rate,
                                                 weight_decay=weight_decay)
                    loader = DataLoader([[point, label]
                                     for point, label in zip(input_tensor_odd,
                                                             label_tensor_odd)],
                                        shuffle=True,
                                        batch_size=batch_size)
                    best_test_acc = 0
                    for epoch in range(num_epochs):
                        for batch, label in loader:
                            output = net(batch)
                            loss   = criterion(output, label)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        test_acc = (net(test_tensor).argmax(dim=1)
                                    == test_labels).sum().item()
                        test_acc /= len(test_labels)
                        if test_acc > best_test_acc:
                            best_test_acc = test_acc
                            early_stop    = 0
                            best_model    = deepcopy(net.state_dict())
                        elif abs(test_acc - best_test_acc) < 1e-4:
                            best_model = deepcopy(net.state_dict())
                        else:
                            early_stop += 1
                        if early_stop > early_stop_crit:
                            break
                    net.load_state_dict(best_model)
                    test_acc = (net(test_tensor).argmax(dim=1)
                                == test_labels).sum().item()
                    test_acc /= len(test_labels)
                    if test_acc >= nn_acc_threshold:
                        torch.save(net.cpu().state_dict(),
                                   f'{model_dir}/{instance}'
                                   + f'/{n_odd}_{percent_good}'
                                   + f'odd_acc{round(test_acc, 3)}.pt')
                        n_odd += 1
                        t.update(1)
            ii += 1
        t.close()
