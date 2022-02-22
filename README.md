## Code to accompany *[Physics solutions to machine learning privacy leaks](https://www.arxiv.org/abs/2203.xxxxx)*
#### Alejandro Pozas-Kerstjens, Senaida Hernández-Santana, José Ramón Pareja Monturiol, Marcos Castrillón López, Giannicola Scarpa, Carlos González-Guillén, and David Pérez-García

This repository contains the codes used for the article "*Physics solutions to machine learning privacy leaks*. Alejandro Pozas-Kerstjens, Senaida Hernández-Santana, José Ramón Pareja Monturiol, Carlos González-Guillén, Giannicola Scarpa, and David Pérez-García. [arXiv:2202.xxxxx](https://www.arxiv.org/abs/2202.xxxxx)." It provides the codes for cleaning the [global.health database](https://global.health/), training neural network and matrix product state models on the dataset generated, and attacking the models via shadow training.

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


If you would like to cite this work, please use the following format:

A. Pozas-Kerstjens, S. Hernández-Santana, J. R. Pareja Monturiol, M. Castrillón López, G. Scarpa, C. González-Guillén, and D. Pérez-García, _Physics solutions to machine learning privacy leaks_, arXiv:2203.xxxxx

```
@misc{pozaskerstjens2022privatetn,
author = {Pozas-Kerstjens, Alejandro and Hern\'andez-Santana, Senaida and Pareja Monturiol, Jos\'e Ram\'on and Castrill\'on L\'opez, Marco and Scarpa, Giannicola and Gonz\'alez-Guill\'en, Carlos and P\'erez-Garc\'ia, David},
title = {Physics solutions to machine learning privacy leaks},
eprint = {2203.xxxxx},
archivePrefix={arXiv}
}
```
