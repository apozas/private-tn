# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           pandas for dataset operations
#           scikit-learn for ML utils
#           tensorflow for ML
# Last modified: Feb, 2023

###############################################################################
# This file contains helper functions for the training of MPS models.
###############################################################################
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from tensorflow import Variable
from typing import List, Tuple

# Values needed for normalization, extracted from the complete database
database_max_age = 120.
database_min_age = 0.


def convert_binary_to_continuous(array: np.array,
                                 gap: float = 0.1) -> np.array:
    '''Converts a list of binary variables to continuous. This is done by
    assigning a random value in [0, 1/2-gap] for one category, and a random
    value in [1/2+gap, 1] for the other one.

    Args:
      array: array of categorical binary variables
      gap: width of the separation between categories

    Returns:
      array_cont: numpy.array with continuous variables
    '''
    delta = gap / 2
    array_below = np.random.uniform(low=0, high=0.5-delta, size=[len(array)])
    array_above = np.random.uniform(low=0.5+delta, high=1, size=[len(array)])
    array_cont = np.array([array_above[idx] if array[idx] else array_below[idx]
                           for idx in range(len(array))])
    return array_cont


def convert_data(X: np.array) -> np.array:
    '''Wrapper to convert a series of categorical columns to continuous.

    Args:
      X: array of categorical variables

    Returns:
      X_cont: array with continuous variables
    '''
    n_features = X.shape[1]
    X_cont = X.copy()
    for idx in range(n_features):
        X_cont[:, idx] = convert_binary_to_continuous(X[:, idx])
    return X_cont


def dataset_to_mps_input(data: np.array) -> np.array:
    '''Encodes the data for contraction with an MPS. The encoding consists in
    performing the map psi(p) = (1-p, p) to every element of each datapoint.

    Args:
      data: data array

    Returns:
      encoded dataset
    '''
    return np.array([1 - data, data]).transpose([1, 2, 0]).astype('float32')


def flatten_mps_tensors(mps_tensors: List[Variable]) -> np.array:
    '''Flatten out all the tensors in an MPS, so the result is a
    one-dimensional list of numbers.

    Args:
      mps_tensors: List of tensors, stored as Tensorflow Variables.

    Returns:
      all_params: one-dimensional numpy.array with the flattened tensors.
    '''
    mps_tensors_flat = []
    for tensor in mps_tensors:
        mps_tensors_flat.append(np.reshape(tensor.numpy(), [-1]))
    all_params = np.concatenate(mps_tensors_flat, axis=0).tolist()
    return all_params


def preprocess_mps(dataset: pd.DataFrame) -> Tuple[np.array, np.array]:
    '''Data processing in order to feed it to the matrix product state models.
    Categorical variables are encoded in continuous, non-overlapping ranges of
    the interval [0, 1], and continuous variables are
    brought to a normal distribution.

    Args:
      dataset: Pandas DataFrame with the data

    Returns:
      X: numpy.array with the processed dataset
      y: numoy.array with the corresponding labels
    '''
    # Scale age
    dataset['age'] = ((dataset['age'] - database_min_age)
                      / (database_max_age - database_min_age))

    features = ['age', 'female', 'symptomatic', 'argentina', 'odd']
    label = ['recovered']
    X = dataset[features].values
    y = dataset[label].values

    # Encode categorical variables in continuous ranges
    X[:, 1:] = convert_data(X[:, 1:])

    # One-hot encode labels
    y = OneHotEncoder().fit_transform(y).toarray()

    # Prepare data for inputting to an MPS (i.e., apply the function Psi)
    X = dataset_to_mps_input(X)

    return X, y
