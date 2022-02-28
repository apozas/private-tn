# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           tensorflow for ML
# Last modified: Feb, 2022

################################################################################
# This file contains the functions needed for training the MPS models created in
# classifier.py.
################################################################################
import classifier
import numpy as np
import tensorflow as tf

from typing import Tuple, Optional, Dict


def evaluate(mps, x, y, batch_size: int = 0):
    '''Evaluation of an MPS classifier on a dataset.

    Args:
      mps: MatrixProductState classifier object.
      x: Input data of shape (n_data, n_features, d_phys)
      y: Corresponding labels in one-hot format of shape (n_data, n_labels)
      batch_size: Batch size for evaluation.

    Returns:
      loss: Evaluation of the loss function between the labels and MPS(data).
      accuracy: Fraction of labels correctly guessed.
    '''
    if batch_size == 0:
        batch_size = len(x)
        n_batch = 1
    else:
        n_batch = len(x) // batch_size
    data, labels = tf.cast(x, dtype=mps.dtype), tf.cast(y, dtype=mps.dtype)
    generator = ((data[i*batch_size:(i+1)*batch_size],
                  labels[i*batch_size:(i+1)*batch_size])
                 for i in range(n_batch))
    loss, logits = run_epoch(mps, generator)
    accuracy = (logits.numpy().argmax(axis=1) == y.argmax(axis=1)).mean()
    return loss.numpy() / len(x), accuracy


def fit(mps: classifier.MatrixProductState,
        optimizer,
        x: tf.Tensor,
        y: tf.Tensor,
        n_epochs: int = 20,
        batch_size: int = 10,
        x_val: Optional[tf.Tensor] = None,
        y_val: Optional[tf.Tensor] = None
        ) -> Tuple[classifier.MatrixProductState, Dict[str, float]]:
    '''Supervised training of an MPS classifier on a dataset.

    Args:
      mps: MatrixProductState classifier object.
      optimizer: TensorFlow optimizer object to use in training.
      x: Training data (encoded images) of shape (n_data, n_features, d_phys)
      y: Training labels in one-hot format of shape (n_data, n_labels)
      x_val: Validation data to calculate loss and accuracy during training.
      y_val: Validation labels to calculate loss and accuracy during training.
      n_epochs: Total number of epochs to train.
      batch_size: Batch size for training.

    Returns:
      mps: The trained MatrixProductState classifier object.
      history: History of training and validation loss and accuracy.
    '''
    data       = tf.cast(x, dtype=mps.dtype)
    labels     = tf.cast(y, dtype=mps.dtype)
    n_batch    = len(x) // batch_size
    n_features = x.shape[1]

    if x_val is not None:
        data_val   = tf.cast(x_val, dtype=mps.dtype)
        labels_val = tf.cast(y_val, dtype=mps.dtype)
        if len(x_val) > batch_size:
            batch_size_val = batch_size
            n_batch_val    = len(x_val) // batch_size
        else:
            batch_size_val = len(x_val)
            n_batch_val    = 1

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}

    epoch = 0
    while epoch < n_epochs:
        # Shuffle data and labels
        x     = data.numpy()
        y     = labels.numpy()
        order = np.arange(len(x))
        np.random.shuffle(order)
        x      = x[order]
        y      = y[order]
        data   = tf.cast(x, dtype=mps.dtype)
        labels = tf.cast(y, dtype=mps.dtype)

        generator = ((data[i*batch_size:(i+1)*batch_size],
                      labels[i*batch_size:(i+1)*batch_size])
                     for i in range(n_batch))
        # Train
        loss, logits = run_epoch(mps, generator, optimizer)
        history["loss"].append(loss / len(x))

        history["acc"].append(
                (logits.numpy().argmax(axis=1) == y.argmax(axis=1)).mean())

        # Evaluate for validation
        if x_val is not None:
            val_generator = ((data_val[i*batch_size_val:(i+1)*batch_size_val],
                              labels_val[i*batch_size_val:(i+1)*batch_size_val])
                             for i in range(n_batch_val))
            val_loss, val_logits = run_epoch(mps, val_generator)
            history["val_loss"].append(val_loss / len(x_val))
            history["val_acc"].append((val_logits.numpy().argmax(axis=1)
                                       == y_val.argmax(axis=1)).mean())
        # Increment epoch
        epoch += 1

    return mps, history


def run_epoch(mps: classifier.MatrixProductState, data_generator,
              optimizer=None) -> Tuple[float, tf.Tensor]:
    '''Performs a whole training epoch.
    One epoch corresponds to one full iteration over the training set.

    Args:
      mps: The MatrixProductState to be trained.
      data_generator: Iterator with the training dataset
      optimizer: The optimizer to be used. If not provided, no training (only
                 evaluation) is performed.

    Returns:
      loss: Evaluation of the loss function between the labels and MPS(data).
      logits_output: Activations of the MPS in the data.
    '''
    loss, logits = 0.0, []
    for data, labels in data_generator:
        if optimizer is None:
            batch_results = mps.loss(data, labels)
        else:
            batch_results = run_step(mps, optimizer, data, labels)
        loss += batch_results[0]
        logits.append(batch_results[1])

    if len(logits) == 1:
        logits_output = logits[0]
    else:
        logits_output = tf.concat(logits, axis=0)

    return loss, logits_output


def run_step(mps: classifier.MatrixProductState, optimizer,
             data: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    '''Runs a single training step for one batch

    Args:
      mps: The MatrixProductState to be trained.
      optimizer: The optimizer to be used.
      data: Input datapoints to be trained on.
      labels: Expected labels corresponding to the input datapoints.

    Returns:
      loss: Evaluation of the loss function between the labels and MPS(data).
      logits: Activations of the MPS in the data.
    '''
    with tf.GradientTape() as tape:
        tape.watch(mps.tensors)
        loss, logits = mps.loss(data, labels)

    grads = tape.gradient(loss, mps.tensors)
    optimizer.apply_gradients(zip(grads, mps.tensors))

    return loss, logits
