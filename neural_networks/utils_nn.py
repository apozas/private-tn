# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2203.xxxxx
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           pandas for dataset operations
#           torch for ML
# Last modified: Feb, 2022

################################################################################
# This file contains helper functions for the training of neural network models.
################################################################################
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class SimpleNNModel(nn.Module):
    '''Neural network model used in the experiments
    '''
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(9, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 8)
        self.linear4 = nn.Linear(8, 4)
        self.lastlayer = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        out1 = self.activation(self.linear1(x))
        out2 = self.activation(self.linear2(out1))
        out3 = self.activation(self.linear3(out2))
        out4 = self.activation(self.linear4(out3))
        out5 = self.lastlayer(out4)
        return out5

def preprocess_nn(dataset):
    '''Data processing in order to feed it to the neural network models.
    Categorical variables are one-hot encoded, and continuous variables are
    brought to a normal distribution.

    :param dataset: Pandas DataFrame with the data

    :returns: a pandas.DataFrame with the processed data
    '''
    categorical = ['argentina', 'female', 'symptomatic', 'odd']
    target      = 'recovered'
    continuous  = list(set(dataset.columns) - set(categorical) - set([target]))
    for col in categorical:
        one_hot = pd.get_dummies(dataset[col])
        one_hot.columns = [col + '_' + str(col_name)[0]
                           for col_name in one_hot.columns]
        dataset = dataset.drop(col, axis=1)
        dataset = dataset.join(one_hot)

    for col in continuous:
        mean = dataset[col].mean()
        std = dataset[col].std()
        dataset[col] = (dataset[col] - mean) / std

    return dataset

def dataset_to_torch(dataset, train_ratio):
    '''Transform a Pandas DataFrame into train-test pytorch Tensors.

    :param dataset: Pandas DataFrame with the data
    :param train_ratio: percentage of the dataset used for training
    :type train_ratio: float 0 <= x <= 1

    :returns input_tensor: torch.Tensor with the training data
    :returns label_tensor: torch.Tensor with the training labels
    :returns test_input_tensor: torch.Tensor with the test data
    :returns test_label_tensor: torch.Tensor with the test labels
    '''
    # Convert features and labels to numpy arrays.'
    target = 'recovered'
    labels = dataset[target].to_numpy()
    feature_dataset = dataset.drop([target], axis=1)
    data = feature_dataset.to_numpy()

    # Separate training and test sets
    train_indices  = np.random.choice(len(labels),
                                      int(train_ratio*len(labels)),
                                      replace=False)
    test_indices   = list(set(range(len(labels))) - set(train_indices))
    train_features = data[train_indices]
    train_labels   = labels[train_indices]
    test_features  = data[test_indices]
    test_labels    = labels[test_indices]

    input_tensor      = torch.from_numpy(train_features).type(torch.FloatTensor)
    label_tensor      = torch.from_numpy(train_labels).type(torch.LongTensor)
    test_input_tensor = torch.from_numpy(test_features).type(torch.FloatTensor)
    test_label_tensor = torch.from_numpy(test_labels).type(torch.LongTensor)

    return input_tensor, label_tensor, test_input_tensor, test_label_tensor
