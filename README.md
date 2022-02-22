## Code to accompany *[Physics solutions to machine learning privacy leaks](https://www.arxiv.org/abs/2203.xxxxx)*
#### Alejandro Pozas-Kerstjens, Senaida Hernández-Santana, José Ramón Pareja Monturiol, Marcos Castrillón López, Giannicola Scarpa, Carlos E. González-Guillén, and David Pérez-García

This repository contains the codes used for the article "*Physics solutions to machine learning privacy leaks*. Alejandro Pozas-Kerstjens, Senaida Hernández-Santana, José Ramón Pareja Monturiol, Carlos E. González-Guillén, Giannicola Scarpa, and David Pérez-García. [arXiv:2202.xxxxx](https://www.arxiv.org/abs/2202.xxxxx)." It provides the codes for cleaning the [global.health database](https://global.health/), training neural network and matrix product state models on the dataset generated, and attacking the models via shadow training.

All code is written in Python.

Libraries required:
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for plots
- [numpy](https://numpy.org) for array operations
- [pandas](https://pandas.pydata.org/) for database operations
- [scikit-learn](https://scikit-learn.org) for machine learning utils
- [tensorflow](https://www.tensorflow.org) and [pytorch](https://pytorch.org) for training the matrix product states and the neural networks, respectively
- [tensornetwork](https://tensornetwork.readthedocs.io/en/latest/) for defining the matrix product states
- [tqdm](https://tqdm.github.io/) for progress bars
- [copy](https://docs.python.org/3/library/copy.html), [glob](https://docs.python.org/3/library/glob.html), [os](https://docs.python.org/3/library/os.html), [pickle](https://docs.python.org/3/library/pickle.html), [random](https://docs.python.org/3/library/random.html), [sys](https://docs.python.org/3/library/sys.html), [time](https://docs.python.org/3/library/time.html) and [typing](https://docs.python.org/3/library/typing.html)

Files:

- General files
  - [create_accuracy_figure](https://github.com/apozas/private-tn/blob/main/create_accuracy_figure.py): Create Figure 2c in the paper, showing the accuracies of the models in predicting the outcome of COVID-19 patients given demographics and symptoms.
  - [create_attack_figure](https://github.com/apozas/private-tn/blob/main/create_attack_figure.py): Create Figure 2d in the paper, showing the accuracies of attacks inferring the parity of the registration day of the models' training data.
  - [create_vulnerability_figure](https://github.com/apozas/private-tn/blob/main/create_vulnerability_figure.py): Create Figure 1 in the paper, showing how neural networks store data from the training set that is irrelevant for the target task.
  - [database_processing](https://github.com/apozas/private-tn/blob/main/database_processing.py): Clean the [global.health](https://global.health/) database to generate the dataset used in the experiments.


- Neural networks
  - [attack_nn](https://github.com/apozas/private-tn/blob/main/neural_networks/attack_nn.py): Attacks inferring the parity of the registration day of the neural networks' training data.
  - [create_nn_dataset_from_models](https://github.com/apozas/private-tn/blob/main/neural_networks/create_nn_dataset_from_models.py): Generate the dataset with all the neural networks' model parameters.
  - [generate_nn_models](https://github.com/apozas/private-tn/blob/main/neural_networks/generate_nn_models.py): Train neural network models on predicting COVID-19 outcome given demographics and symptoms.
  - [utils_nn](https://github.com/apozas/private-tn/blob/main/neural_networks/utils_nn.py): Helper function for data processing and model training.


- Matrix product states
  - [attack_mps](https://github.com/apozas/private-tn/blob/main/matrix_product_states/attack_mps.py): Attacks, based on shadow training, inferring the parity of the registration day of the matrix product states' training data.
  - [batchtensornetwork](https://github.com/apozas/private-tn/blob/main/matrix_product_states/batchtensornetwork.py): Functions for evaluating matrix product states on input data.
  - [classifier](https://github.com/apozas/private-tn/blob/main/matrix_product_states/classifier.py): Definition of the _classifier_ matrix product state model.
  - [create_mps_dataset_from_models](https://github.com/apozas/private-tn/blob/main/matrix_product_states/create_mps_dataset_from_models.py): Generate the dataset with all the matrix product states' model parameters, either in standard or in canonical form.
  - [generate_mps_models](https://github.com/apozas/private-tn/blob/main/matrix_product_states/generate_mps_models.py): Train matrix product state models on predicting COVID-19 outcome given demographics and symptoms.
  - [training](https://github.com/apozas/private-tn/blob/main/matrix_product_states/training.py): Functions for training matrix produc state models.
  - [utils_mps](https://github.com/apozas/private-tn/blob/main/matrix_product_states/utils_mps.py): Helper function for data processing and model training.

If you would like to cite this work, please use the following format:

A. Pozas-Kerstjens, S. Hernández-Santana, J. R. Pareja Monturiol, M. Castrillón López, G. Scarpa, C. E. González-Guillén, and D. Pérez-García, _Physics solutions to machine learning privacy leaks_, arXiv:2203.xxxxx

```
@misc{pozaskerstjens2022privatetn,
author = {Pozas-Kerstjens, Alejandro and Hern\'andez-Santana, Senaida and Pareja Monturiol, Jos\'e Ram\'on and Castrill\'on L\'opez, Marco and Scarpa, Giannicola and Gonz\'alez-Guill\'en, Carlos E. and P\'erez-Garc\'ia, David},
title = {Physics solutions to machine learning privacy leaks},
eprint = {2203.xxxxx},
archivePrefix={arXiv}
}
```
