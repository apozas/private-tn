# Code for
# Physics solutions for machine learning privacy leaks
# arXiv:2202.12319
#
# Authors: Alejandro Pozas-Kerstjens and Senaida Hernandez-Santana
#
# Requires: numpy for array operations
#           tensorflow for ML
#           tensornetwork for tensor network operations
# Last modified: Feb, 2023

###############################################################################
# This file defines the MPS classifier object.
###############################################################################
import numpy as np
import tensorflow as tf
import tensornetwork as tn
from tensornetwork import FiniteMPS
from typing import Tuple, List, Optional
try:
    from matrix_product_states.batchtensornetwork import \
                                   batched_contract_between, pairwise_reduction
except ModuleNotFoundError:
    from batchtensornetwork import batched_contract_between, pairwise_reduction


def random_initializer(d_phys_aux: int, d_bond: int, d_phys: int = None,
                       boundary: bool = False, std: float = 0.,
                       right: bool = False) -> np.ndarray:
    '''Initializes MPS tensors randomly and close to identity matrices.

    Args:
        d_phys: Physical dimension of the MPS tensor.
        d_bond: Bond dimension of the MPS tensor.
        std: standard deviation of normal distribution for random initialization.
        boundary: If True returns a tensor of shape (d_phys, d_bond).
                  Otherwise returns a tensor of shape (d_phys, d_bond, d_bond).
        Note that d_phys given in this function does not have to be the actual
        MPS physical dimension (eg. it can also be n_labels to initialize there
        label MPS tensor).

    Returns:
        tensor: Random numpy array with shape described above.
    '''
    if boundary:
        d_phys = d_phys_aux
        x      = np.zeros((d_phys, d_bond))
        x[:, 0] = 1
    else:
        x = np.array(d_phys_aux * [np.eye(d_bond)])
    x += np.random.normal(0.0, std, size=x.shape)

    return x


class Environment:
    '''MatrixProductState environments.

    Perform the core calculation required for the inner product by building the
    relevant TensorNetwork. An environment consists of a boundary vector of
    shape (d_phys, d_bond) and MPS matrices of shape (d_phys, d_bond, d_bond).
    Note that the MPS matrices are stored as a tensor of shape
    (right, n_features, d_phys, d_bond, d_bond), namely with an additional
    "space" index for efficiency.
    The right argument is used to indicate whether the environment is right or
    left. This parameter has been added to be able to provide all the final
    tensors in the correct order during the training of the full MPS. This is
    convenient to obtain the canonical form. If this is not done one needs to
    invert the order of the right matrices in the mps classifier, transpose
    them along the bond dimension edges before a canonical transform, obtain
    the canonical transform, and then do the reverse operation to insert the
    resulting canonical form into the class MatrixProductState.
    '''
    def __init__(self, right: bool, n_features: int, d_phys: int,
                 d_bond: int, std: float = 0., dtype=tf.float32):
        self.n_features, self.dtype  = n_features, dtype
        self.d_phys,     self.d_bond = d_phys,     d_bond
        self.right = right

        tensor_v = random_initializer(d_phys_aux=d_phys,
                                      d_bond=d_bond,
                                      boundary=True,
                                      std=std,
                                      right=right)
        self.vector = tf.Variable(tensor_v, dtype=dtype)

        if n_features > 1:
            tensor_m = random_initializer(d_phys_aux=d_phys*(n_features-1),
                                          d_bond=d_bond,
                                          d_phys=d_phys,
                                          std=std,
                                          right=right)
            self.matrices = tf.Variable(
                      tensor_m.reshape((n_features-1, d_phys, d_bond, d_bond)),
                      dtype=dtype)
        else:
            self.matrices = None

    def create_network(self, data_vector: tf.Tensor,
                       data_matrices: tf.Tensor) -> [List[tn.Node],
                                                     Tuple[tn.Node]]:
        '''Creates TensorNetwork with MPS and data.

        Args:
          data_vector: Tensor of input data at the boundary of shape
                       (n_batch, d_phys).
          data_matrices: Tensor of input data of shape
                         (n_batch, n_features, d_phys).

        Returns:
          var_nodes: List of the MPS nodes.
          data_nodes: Tuple of the data nodes.
        '''
        # Set tensorflow as default backend
        tn.set_default_backend('tensorflow')

        # Connect the bond edges of the MPS tensors
        if self.matrices is None:
            var_nodes = [tn.Node(self.vector)]
        else:
            var_nodes = [tn.Node(self.vector), tn.Node(self.matrices)]
            var_nodes[0][1] ^ var_nodes[1][2]   # Connect d_bond edges

        # Connect the data nodes with the physical edges of the MPS tensors
        data_nodes = (tn.Node(data_vector), tn.Node(data_matrices))
        data_nodes[0][1] ^ var_nodes[0][0]
        if not (self.matrices is None):
            data_nodes[1][2] ^ var_nodes[1][1]

        return var_nodes, data_nodes

    def contract_network(self, var_nodes: List[tn.Node],
                         data_nodes: Tuple[tn.Node]) -> tf.Tensor:
        '''Contracts the TensorNetwork created in `create_network`.

        Args:
          var_nodes: List of the MPS nodes.
          data_nodes: Tuple of the data nodes.

        Returns:
          tensorflow.Tensor with the contraction.
        '''
        # Contract data with the MPS tensors
        var_nodes[0] = data_nodes[0] @ var_nodes[0]
        if len(var_nodes) >= 2:
            var_nodes[1] = batched_contract_between(data_nodes[1],
                                                    var_nodes[1],
                                                    data_nodes[1][1],
                                                    var_nodes[1][0])
        # the previous line gets rid of all the connected edges -> below, we
        # have to reconnect var_nodes[0] and var_nodes[1] post data-mps product

        # Contract the artificial "space" index. This step is equivalent to
        # contracting the MPS over the bond dimensions.
            space_edge = var_nodes[1][1]
            var_nodes[1] = pairwise_reduction(var_nodes[1], space_edge)

        # Contract the final bond edge with the boundary
            var_nodes[0] = var_nodes[0].copy()  # copy wo connected edges
            var_nodes[0][1] ^ var_nodes[1][1+self.right]  # connect dbond edges
            # this is necessary since var_nodes[1] is not connected
            batch_edge1, batch_edge2 = var_nodes[0][0], var_nodes[1][0]
            var_nodes = batched_contract_between(var_nodes[0],
                                                 var_nodes[1],
                                                 batch_edge1,
                                                 batch_edge2)
            # notice this is contracting over the second batch_edge
        else:
            var_nodes = var_nodes[0]
        return var_nodes.tensor

    def predict(self, data_vector: tf.Tensor,
                data_matrices: tf.Tensor) -> tf.Tensor:
        '''Compute the predictions by contracting the network and the data

        Args:
          data_vector: tensorflow.Tensor with the data to be predicted on.
          data_matrices: tensorflow.Tensor with the parameters of the MPS.

        Returns:
          tensorflow.Tensor with the contraction.
        '''
        var_nodes, data_nodes = self.create_network(data_vector, data_matrices)
        return self.contract_network(var_nodes, data_nodes)


class MatrixProductState:
    '''MPS classifier prediction graph.

    Contains the MPS tensors which are our variational parameters and the
    methods that define the forward pass. These methods are used by
    `training.py` to fit data using automatic differentation and can also be
    used for predictions. Each MatrixProductState consists of a left and right
    environment which are connected by the label tensor to get the final
    prediction. These environments are defined in the `Environment` class.

    Args:
      n_features: Number of input features.
      n_labels: Number of output labels.
      d_phys: Dimension of the encoding of each input feature.
      d_bond: Dimension of the MPS matrices for each dimension of the input.
      l_position: Position of the tensor that encodes the output.
      std: Standard deviation of Gaussian noise for the initial parameters.
      dtype: Variable type
    '''
    def __init__(self,
                 n_features: int,
                 n_labels: int,
                 d_phys: int,
                 d_bond: int,
                 l_position: Optional[int] = None,
                 std: float = 1e-14,
                 dtype=tf.float32):
        self.dtype = dtype
        if l_position is None:
            l_position = n_features // 2
        self.position = l_position

        tensor_l = random_initializer(n_labels, d_bond, std=std)
        self.labeled = tf.Variable(tensor_l, dtype=dtype)
        right_left, right_right = False, True
        self.left_env = Environment(right_left, l_position, d_phys,
                                    d_bond, std=std, dtype=dtype)
        self.right_env = Environment(right_right, n_features - l_position,
                                     d_phys, d_bond, std=std, dtype=dtype)

        self.left_env.vector = tf.Variable(self.left_env.vector)
        if self.left_env.matrices is not None:
            self.left_env.matrices = tf.Variable(self.left_env.matrices)
        else:
            self.left_env.matrices = None
        self.right_env.vector = tf.Variable(self.right_env.vector)
        if self.right_env.matrices is not None:
            self.right_env.matrices = tf.Variable(self.right_env.matrices)
        else:
            self.right_env.matrices = None

        self.tensors = [self.left_env.vector]
        if self.left_env.matrices is not None:
            self.tensors.append(self.left_env.matrices)
        self.tensors.append(self.labeled)
        if self.right_env.matrices is not None:
            self.tensors.append(self.right_env.matrices)
        self.tensors.append(self.right_env.vector)

    def accuracy(self, data: tf.Tensor, labels: tf.Tensor):
        '''Computes the accuracy of prediction on a given dataset.

        Args:
          data: Tensor with input data, of shape (n_batch, n_features, d_phys)
          labels: Tensor with corresponding labels, of shape (n_batch,n_labels)

        Returns:
          accuracy: (number of correct predictions) / (number of predictions)
        '''
        logits = self.flx(data)
        return (logits.numpy().argmax(axis=1)
                == labels.numpy().argmax(axis=1)).mean()

    def canonical_form(self, normalize: bool = False):
        '''Computes the parameters describing the canonical form of the MPS.

        Args:
          normalize: Normalize the MPS, so its trace equals to 1.
        '''
        finite_mps = self.to_finite()
        finite_mps.canonicalize(normalize=normalize)
        self.from_finite(finite_mps)
        self.update_environment_attributes()

        return self

    def flx(self, data: tf.Tensor) -> tf.Tensor:
        '''Calculates prediction given by contracting input data with MPS.
        This is equivalent to the "forward pass" of a neural network.

        Args:
          data: Tensor with input data of shape (n_batch, n_features, d_phys).

        Returns:
          flx: Prediction (value of the function f^l(x)) with
               shape (n_batch, n_labels).
        '''
        i = 0
        self.left_env.vector = self.tensors[i]
        i = i + 1
        if self.left_env.matrices is not None:
            self.left_env.matrices = self.tensors[i]
            i = i + 1
        self.labeled = self.tensors[i]
        i = i + 1
        if self.right_env.matrices is not None:
            self.right_env.matrices = self.tensors[i]
            i = i + 1
        self.right_env.vector = self.tensors[i]

        left  = self.left_env.predict(data[:, 0], data[:, 1:self.position])
        right = self.right_env.predict(data[:, -1],
                                       data[:, -2:self.position-1:-1])
        return tf.einsum('bl,olr,br->bo', left, self.labeled, right)

    def from_finite(self, finite_mps, right_biased=True):
        '''Convert a FiniteMPS from TensorNetwork to a MatrixProductState.

        Args:
          finite_mps: The FiniteMPS to transform.
          right_biased: For MPS with three sites, determines where the output
                        tensor is located.
        '''
        # Move the physical index to the first
        tensors_transposed = list(map(
                              lambda x: tf.transpose(x,
                                                     perm=[1, 0, 2]).numpy(),
                                  finite_mps.tensors))

        # Make the extreme tensors become matrices
        tensors_transposed[0]  = np.squeeze(tensors_transposed[0])
        tensors_transposed[-1] = np.squeeze(tensors_transposed[-1])

        # Obtaining MPS characterizing parameters
        n_features   = len(tensors_transposed) - 1
        d_phys_list  = [tensor.shape[0] for tensor in tensors_transposed]
        d_phys       = d_phys_list[0]
        n_labels     = d_phys_list[n_features // 2]
        d_bonds_list = [tensor.shape[1:] for tensor in tensors_transposed]
        d_bond_list  = [array[-1] for array in d_bonds_list[:-1]]
        d_bond       = max(d_bond_list)

        # Pad to obtain the same dimensions everywhere
        tensors_transposed_and_padded_left = np.pad(tensors_transposed[0],
                                 pad_width=((0, 0),
                                            (0, d_bond-d_bonds_list[0][0])))
        tensors_transposed_and_padded_right = np.pad(tensors_transposed[-1],
                                pad_width=((0, 0),
                                           (0, d_bond-d_bonds_list[-1][0])))
        tensors_transposed_and_padded_bulk = [
            np.pad(tensors_transposed[i], pad_width=(
                                                (0, 0),
                                                (0, d_bond-d_bonds_list[i][0]),
                                                (0, d_bond-d_bonds_list[i][1]))
                   ) for i in range(1, len(tensors_transposed) - 1)]

        # Create tensors
        mps_left_edge = tf.Variable(tensors_transposed_and_padded_left,
                                    dtype=finite_mps.dtype)
        mps_tensors = [mps_left_edge]

        if ((n_features == 3) & (not right_biased)) | (n_features >= 4):
            mps_left_bulk = tf.Variable(np.array(
                      tensors_transposed_and_padded_bulk[0:(n_features-2)//2]),
                      dtype=finite_mps.dtype)
            mps_tensors.append(mps_left_bulk)

        mps_center = tf.Variable(
                         tensors_transposed_and_padded_bulk[(n_features-2)//2],
                         dtype=finite_mps.dtype)
        mps_tensors.append(mps_center)

        if ((n_features == 3) & right_biased) | (n_features >= 4):
            mps_right_bulk = tf.Variable(np.array(
                     tensors_transposed_and_padded_bulk[(n_features-2)//2+1:]),
                                         dtype=finite_mps.dtype)
            mps_tensors.append(mps_right_bulk)

        mps_right_edge = tf.Variable(tensors_transposed_and_padded_right,
                                     dtype=finite_mps.dtype)
        mps_tensors.append(mps_right_edge)

        self.tensors = mps_tensors
        self.update_environment_attributes()

    def load_numpy(self, params):
        '''Loads MPS parameters from numpy arrays.

        Args:
          params: List[numpy.array] with the parameters of the MPS.
        '''
        self.tensors = [tf.Variable(tensor) for tensor in params]
        self.update_environment_attributes()

    def loss(self, data: tf.Tensor, labels: tf.Tensor) -> [tf.Tensor,tf.Tensor]:
        '''Calculates loss in a batch of (data, labels).

        Args:
          data: Tensor with input data of shape (n_batch, n_features, d_phys).
          labels: Tensor with the corresponding labels of shape
                  (n_batch, n_labels).

        Returns:
          loss: Loss of the given batch.
          logits: flx prediction as returned from self.flx method.
        '''
        logits = self.flx(data)
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels,
                                                                     logits))
        return loss, logits

    def tensors_in_finiteMPS_notation(self) -> List[tf.Tensor]:
        '''Extracts the tensors of an MPS in the form compatible with the
        FiniteMPS structure of TensorNetwork.

        Returns:
          tensors: tensors of the MPS in the format of FiniteMPS.
        '''
        # Compute number of features depending on the amount of tensors.
        if len(self.tensors) == 5:
            n_features = 2 + self.tensors[1].shape[0] + self.tensors[3].shape[0]
        else:
            n_features = len(self.tensors) - 1

        # Extract tensors
        # First and last tensors must be transformed from matrices to tensors
        tensor_idx = 0
        tensors = [tf.expand_dims(self.tensors[tensor_idx], axis=0)]

        # Bulk tensors' indices must be reordered
        if n_features >= 4:
            tensor_idx += 1
            for tensor in self.tensors[tensor_idx]:
                tensors.append(tf.transpose(tensor, perm=[1, 0, 2]))
        tensor_idx += 1
        tensors.append(tf.transpose(self.tensors[tensor_idx], perm=[1, 0, 2]))
        if n_features >= 3:
            tensor_idx += 1
            for tensor in self.tensors[tensor_idx]:
                tensors.append(tf.transpose(tensor, perm=[1, 0, 2]))

        # First and last tensors must be transformed from matrices to tensors
        tensor_idx += 1
        tensors.append(tf.expand_dims(tf.transpose(self.tensors[tensor_idx],
                                                   perm=[1, 0]), axis=2))
        return tensors

    def to_finite(self, canonicalize: bool = False) -> FiniteMPS:
        '''Convert a MatrixProductState to a FiniteMPS from TensorNetwork.

        Args:
          canonicalize: Give the MPS parameters in canonical form.

        Returns:
          finite_mps: TensorNetwork.FiniteMPS
        '''
        mps_tensors = self.tensors_in_finiteMPS_notation()
        finite_mps  = FiniteMPS(mps_tensors,
                                canonicalize=canonicalize,
                                backend='tensorflow')
        return finite_mps

    def update_environment_attributes(self):
        '''Propagates changes in tensors to the rest of the structures in the
        MatrixProductState.
        '''
        i = 0
        self.left_env.vector = self.tensors[i]
        i = i + 1
        if self.left_env.matrices is not None:
            self.left_env.matrices = self.tensors[i]
            i = i + 1
        self.labeled = self.tensors[i]
        i = i + 1
        if self.right_env.matrices is not None:
            self.right_env.matrices = self.tensors[i]
            i = i + 1
        self.right_env.vector = self.tensors[i]
