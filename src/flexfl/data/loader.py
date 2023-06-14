"""
====================
Federated Data Submodule
====================
"""
from typing import List

import torch
from torch.utils.data import Dataset

import pytorch_lightning as pl

import random
import numpy as np


import random
import numpy as np

def create_shards(dataset, num_shards, target_iidness, sample_size_iidness):
    """
    Split a dataset into shards, allowing control over the target and sample size iidness.

    Args:
        dataset (list or iterable): The dataset to be split into shards.
        num_shards (int): The number of desired shards.
        target_iidness (float): The target iidness level, ranging from 0 (completely non-iid) to 1 (completely iid).
        sample_size_iidness (float): The sample size iidness level, ranging from 0 (completely non-iid) to 1 (completely iid).

    Returns:
        list: A list of shards, where each shard is a subset of the original dataset.
    """
    # Shuffle the dataset
    dataset = list(dataset)
    random.shuffle(dataset)

    # Compute the number of samples per shard
    num_samples = len(dataset)
    shard_sizes = [int(num_samples / num_shards)] * num_shards

    # Adjust shard sizes based on sample size iidness
    if 0 < sample_size_iidness < 1:
        total_samples = sum(shard_sizes)
        remainder = total_samples % num_shards
        shard_sizes = [total_samples // num_shards] * num_shards
        for i in range(remainder):
            shard_sizes[i] += 1

    # Adjust shard sizes based on target iidness
    if 0 < target_iidness < 1:
        class_counts = np.zeros(10)  # Assuming 10 classes for MNIST (modify according to your dataset)
        shard_sizes_cumulative = np.cumsum(shard_sizes)

        for i in range(1, num_shards):
            num_samples_per_class = int(target_iidness * shard_sizes_cumulative[i] / num_shards)
            class_counts[i % 10] += num_samples_per_class

        for i in range(num_shards):
            class_counts[i % 10] += shard_sizes_cumulative[i] - shard_sizes_cumulative[i - 1]

        shard_sizes = [int(class_count) for class_count in class_counts]

    # Split the dataset into shards
    shards = []
    start_index = 0
    for shard_size in shard_sizes:
        end_index = start_index + shard_size
        shard = dataset[start_index:end_index]
        shards.append(shard)
        start_index = end_index

    return shards



class FederatedDataset:
    """Class for splitting a dataset into shards."""
    ALLOWED_NON_IIDNESS_LEVELS = ['low', 'medium', 'high']

    def __init__(
            self,
            dataset: Dataset,
            n_clients: int,
            weights: List[float],
            validation_split: float = 0.2,
            test_split: float = 0.2,
            non_iidness_level: str = 'low',
    ):
        # Save hyperparameters
        self.dataset = dataset
        self.n_clients = n_clients
        self.weights = weights
        self.validation_split = validation_split
        self.test_split = test_split
        self.non_iidness_level = non_iidness_level

        # Assert weights are valid
        assert len(self.weights) == self.n_clients, "Number of weights must equal number of clients."
        assert sum(self.weights) == 1, "Sum of weights must equal 1."

        # Assert non-IIDness level is valid
        assert self.non_iidness_level in self.ALLOWED_NON_IIDNESS_LEVELS, \
            f"Non-IIDness level must be one of {self.ALLOWED_NON_IIDNESS_LEVELS}."

        # Sample class concentration per site following a dirichlet distribution
        self.concentrations = self._sample_concentrations()

        # Split the dataset into shards/clients. Receives a list of pl.LightningDataModule objects
        self.shards = create_shards(self.dataset, self.n_clients, self.non_iidness_level)

    def _sample_concentrations(self):
        # Get number of classes (targets) in dataset: n_classes
        self.n_classes = len(set(self.dataset.targets))

        # Define concentration parameters for dirichlet distribution
        if self.non_iidness_level == 'low':
            alpha = torch.ones(self.n_classes)
        elif self.non_iidness_level == 'medium':
            alpha = torch.ones(self.n_classes) * 0.5
        elif self.non_iidness_level == 'high':
            alpha = torch.ones(self.n_classes) * 0.1
        else:
            raise ValueError(f"Non-IIDness level must be one of {self.ALLOWED_NON_IIDNESS_LEVELS}.")

        # Sample class concentration per site following a dirichlet distribution
        self.concentrations = torch.distributions.dirichlet.Dirichlet(alpha).sample(torch.Size(self.n_clients, ))

