import os
from typing import Tuple

# Temporary import. It will be removed in the final vserion
import sys

import numpy as np


# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)

from cutagitest import Utils
from cutagi import HrSoftmax


class HierarchicalSoftmax(HrSoftmax):
    """Hierarchical softmax wrapper. Further details can be found here
    https://building-babylon.net/2017/08/01/hierarchical-softmax
    """

    def __init__(self) -> None:
        super().__init__()


class Utils:
    """Frontend for utility functions from C++/CUDA backend

    Attributes:
        backend_utils: Utility functionalities from the backend
    """

    backend_utils = Utils()

    def __init__(self) -> None:
        pass

    def label_to_obs(
        self, labels: np.ndarray, num_classes: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Get observations and observation indices of the binary tree for
            classification

        Args:
            labels: Labels of dataset
            num_classes: Total number of classes
        Returns:
            obs: Encoded observations of the labels
            obs_idx: Indices of the encoded observations in the output vector
            num_obs: Number of encoded observations
        """

        obs, obs_idx, num_obs = self.backend_utils.label_to_obs_wrapper(
            labels, num_classes
        )

        return np.array(obs), np.array(obs_idx), np.array(num_obs)

    def label_to_one_hot(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Get the one hot encoder for each class

        Args:
            labels: Labels of dataset
            num_classes: Total number of classes
        Returns:
            one_hot: One hot encoder
        """

        return self.backend_utils.label_to_one_hot_wrapper(labels, num_classes)

    def load_mnist_images(
        self, image_file: str, label_file: str, num_images: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load mnist dataset

        Args:
            image_file: Location of the Mnist image file
            label_file: Location of the Mnist label file
            num_images: Number of images to be loaded
        Returns:
            images: Image dataset
            labels: Label dataset
            num_images: Total number of images
        """
        images, labels = self.backend_utils.load_mnist_dataset_wrapper(
            image_file, label_file, num_images
        )

        return images, labels

    def load_cifar_images(
        self, image_file: str, num: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load cifar dataset

        Args:
            image_file: Location of image file
            num: Number of images to be loaded
        Returns:
            images: Image dataset
            labels: Label dataset
        """

        images, labels = self.backend_utils.load_cifar_dataset_wrapper(image_file, num)

        return images, labels

    def get_labels(
        self,
        ma: np.ndarray,
        Sa: np.ndarray,
        hr_softmax: HierarchicalSoftmax,
        num_classes: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert last layer's hidden state to labels

        Args:
            ma: Mean of activation units for the output layer
            Sa: Variance of activation units for the output layer
            hr_softmax: Hierarchical softmax
            num_classes: Total number of classes
            batch_size: Number of data in a batch
        Returns:
            pred: Label prediciton
            prob: Probability for each label
        """

        pred, prob = self.backend_utils.get_labels_wrapper(
            ma, Sa, hr_softmax, num_classes, batch_size
        )

        return pred, prob

    def get_errors(
        self,
        ma: np.ndarray,
        Sa: np.ndarray,
        labels: np.ndarray,
        hr_softmax: HierarchicalSoftmax,
        num_classes: int,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert last layer's hidden state to labels

        Args:
            ma: Mean of activation units for the output layer
            Sa: Variance of activation units for the output layer
            labels: Label dataset
            hr_softmax: Hierarchical softmax
            num_classes: Total number of classes
            batch_size: Number of data in a batch
        Returns:
            pred: Label prediction
            prob: Probability for each label
        """

        pred, prob = self.backend_utils.get_error_wrapper(
            ma, Sa, labels, hr_softmax, num_classes, batch_size
        )

        return pred, prob

    def get_hierarchical_softmax(self, num_classes: int) -> HierarchicalSoftmax:
        """Convert labels to binary tree

        Args:
            num_classes: Total number of classes
        Returns:
            hr_softmax: Hierarchical softmax
        """
        hr_softmax = self.backend_utils.hierarchical_softmax_wrapper(num_classes)

        return hr_softmax

    def obs_to_label_prob(
        self,
        ma: np.ndarray,
        Sa: np.ndarray,
        hr_softmax: HierarchicalSoftmax,
        num_classes: int,
    ) -> np.ndarray:
        """Convert observation to label probabilities

        Args:
            ma: Mean of activation units for the output layer
            Sa: Variance of activation units for the output layer
            hr_softmax: Hierarchical softmax
            num_classes: Total number of classes
        Returns:
            prob: Probability for each label
        """

        prob = self.backend_utils.obs_to_label_prob_wrapper(
            ma, Sa, hr_softmax, num_classes
        )

        return np.array(prob)

    def create_rolling_window(
        self,
        data: np.ndarray,
        output_col: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
        num_features: int,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create rolling window for time series data

        Args:
            data: dataset
            output_col: Indices of the output columns
            input_seq_len: Length of the input sequence
            output_seq_len: Length of the output sequence
            num_features: Number of features
            stride: Controls number of steps for the window movements
        Returns:
            input_data: Input data for neural networks in sequence
            output_data: Output data for neural networks in sequence
        """
        num_data = int(
            (len(data) / num_features - input_seq_len - output_seq_len) / stride + 1
        )

        input_data, output_data = self.backend_utils.create_rolling_window_wrapper(
            data.flatten(),
            output_col,
            input_seq_len,
            output_seq_len,
            num_features,
            stride,
        )
        input_data = input_data.reshape((num_data, input_seq_len))
        output_data = output_data.reshape((num_data, output_seq_len))

        return input_data, output_data

    def get_upper_triu_cov(
        self, batch_size: int, num_data: int, sigma: float
    ) -> np.ndarray:
        """Create an upper triangle covriance matrix for inputs"""

        vx_f = self.backend_utils.get_upper_triu_cov_wrapper(
            batch_size, num_data, sigma
        )

        return np.array(vx_f)