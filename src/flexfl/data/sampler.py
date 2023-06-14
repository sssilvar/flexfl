"""
=============================
A submodule for splitting datasets into clients.
=============================

This submodule contains the classes and functions for splitting datasets into clients.
Functionality includes:
    - Splitting a dataset into clients.
    - Accounting for non-IIDness in the dataset based in either one or multiple variables of choice.
        Example:
            - Splitting MNIST into clients based on the digit class.
            - Splitting MNIST into clients based on the digit class and the writer of the digit.
            - Splitting a medical dataset into clients based on the disease and the age of the patient.

"""
import numpy as np
import torch
from torch.utils.data import Subset


class FederatedSampler:
    """Base class for sampling a dataset into clients."""
    ALLOWED_SAMPLE_SIZE_IIDNESS = ['low', 'medium', 'high']
    ALLOWED_TARGET_IIDNESS = ['low', 'medium', 'high']

    def __init__(
            self,
            dataset,
            num_clients,
            sample_size_non_iidness='low',
            target_non_iidness='low',
            random_state=42,
    ):
        """
        Args:
            dataset (Dataset): The dataset to be split into clients.
            num_clients (int): The number of desired clients.
            sample_size_non_iidness (str): The sample size non-iidness level, ranging from 'low' to 'high'.
            target_non_iidness (str): The target non-iidness level, ranging from 'low' to 'high'.

        """
        self.dataset = dataset
        self.num_clients = num_clients
        self.sample_size_non_iidness = sample_size_non_iidness
        self.target_non_iidness = target_non_iidness
        self.random_state = random_state

        # Compute proportions and size of samples per client
        self.sample_size_proportions = self._compute_proportions()

        # Compute proportion of classes per client if target iidness is specified
        self.class_proportions = self._compute_proportions_per_class()

        # Split the dataset into clients
        self.clients = self._split_dataset()

    def _compute_concentration_parameter(self, level):
        """Compute the concentration parameter of the Dirichlet distribution."""
        if level == 'low':
            concentration_parameter = 0.001
        elif level == 'medium':
            concentration_parameter = 0.3
        elif level == 'high':
            concentration_parameter = 1
        else:
            raise ValueError(
                f'Invalid non-iidness level: {level}. '
                f'Please choose from: {self.ALLOWED_SAMPLE_SIZE_IIDNESS}.'
            )
        return concentration_parameter

    def _compute_proportions(self):
        """Compute the proportions of samples per client."""
        # Use a Dirichlet distribution to sample proportions
        alpha_sample_concentration = self._compute_concentration_parameter(self.sample_size_non_iidness)
        proportions = np.random.dirichlet(np.ones(self.num_clients) * 1 / alpha_sample_concentration, size=1)[0]
        return proportions

    def _compute_proportions_per_class(self):
        """Compute the proportion of classes per client."""
        # Use a Dirichlet distribution to sample proportions of classes per client
        alpha_target_concentration = self._compute_concentration_parameter(self.target_non_iidness)
        class_proportions = np.random.dirichlet(np.ones(self.n_classes) * 1 / alpha_target_concentration,
                                                size=self.num_clients)
        return class_proportions

    @property
    def sample_sizes(self):
        """Compute the sample sizes per client."""
        # Compute the sample sizes per client
        sample_sizes = [int(len(self.dataset) * p) for p in self.sample_size_proportions]
        return sample_sizes

    @property
    def num_shards(self):
        """Compute the number of shards."""
        # Compute the number of shards
        num_shards = self.num_clients
        return num_shards

    @property
    def n_classes(self):
        """Compute the number of classes in the dataset."""
        # Compute the number of classes in the dataset
        n_classes = len(self.classes)
        return n_classes

    @property
    def classes(self):
        """Compute the classes in the dataset."""
        # Compute the classes in the dataset
        classes = np.unique(self.dataset.targets)
        return classes

    def _split_dataset(self):
        """Split the dataset into clients."""
        # Get the indices per class in the dataset
        self.class_idxs = [np.where(self.dataset.targets == i)[0] for i in self.classes]

        # Sample using choice without replacement from the indices per class
        # respecting the proportions per class per client
        self.client_idxs = []
        for i in range(self.num_clients):
            self.client_idxs.append([])
            for j in range(self.n_classes):
                n_desired_samples = int(self.sample_sizes[i] * self.class_proportions[i, j])
                n_available_samples = len(self.class_idxs[j])

                # Account for the case where the number of desired samples is larger
                # than the number of available samples
                if n_desired_samples > n_available_samples:
                    n_desired_samples = n_available_samples

                self.client_idxs[i] += list(np.random.choice(self.class_idxs[j], size=n_desired_samples, replace=False))

                # Eliminate the sampled indices from the class indices
                self.class_idxs[j] = np.setdiff1d(self.class_idxs[j], self.client_idxs[i])

        # Assert there's no intersection between clients indices: compare all pairs of clients
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                assert len(np.intersect1d(self.client_idxs[i], self.client_idxs[j])) == 0, \
                    f'There are intersecting indices between clients {i} and {j}.'

        # Update total samples attribute
        self.total_samples = sum([len(idxs) for idxs in self.client_idxs])

        return [Subset(self.dataset, idxs) for idxs in self.client_idxs]
