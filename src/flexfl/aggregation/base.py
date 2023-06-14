"""
====================
FL Aggregation Base API
====================
"""
from typing import OrderedDict, List


def _normalize_weights(weights):
    """Normalizes the weights so they sum to one."""
    return [weight / sum(weights) for weight in weights]


class AggregatorBase:
    """Base class for aggregators in FL."""

    def __init__(self):
        pass

    def __call__(
            self,
            model_dicts: List[OrderedDict],
            weights: List[float],
            normalize_weights=False,
            assert_weights=False
    ) -> OrderedDict:
        """Aggregates the clients' weights.

        Args:
            model_dicts (List[OrderedDict]): The clients' weights in the form of a list of dictionaries.
            weights (List[float]): The clients' weights in the form of a list of floats.
            normalize_weights (bool): Whether to normalize the weights before aggregation, so they sum to one
                (useful when sampling clients).
            assert_weights (bool): Whether to assert that the weights sum to one.

        Returns (OrderedDict):
            The aggregated weights in the form of a dictionary.
        """
        if normalize_weights:
            weights = _normalize_weights(weights)

        if assert_weights:
            assert sum(weights) == 1
        return self.aggregate(model_dicts, weights)

    def aggregate(
            self,
            model_dicts: List[OrderedDict],
            weights: List[float]
    ) -> OrderedDict:
        """Aggregates the clients' weights.

        Args:
            model_dicts (List[OrderedDict]): The clients' weights in the form of a list of dictionaries.
            weights (List[float]): The clients' weights in the form of a list of floats.

        Returns (OrderedDict):
            The aggregated weights in the form of a dictionary.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"
