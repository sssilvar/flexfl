"""
=====================
Federated Averaging Submodule
=====================
"""
from .base import AggregatorBase


class FedAveragingAggregator(AggregatorBase):
    """Federated averaging aggregator class."""

    def __init__(self):
        super().__init__()

    def aggregate(self, model_dicts, weights):
        """Aggregates the clients' weights.

        Args:
            model_dicts (List[OrderedDict]): The clients' weights in the form of a list of dictionaries.
            weights (List[float]): The clients' weights in the form of a list of floats.

        Returns (OrderedDict):
            The aggregated weights in the form of a dictionary.
        """
        aggregated_weights = {}
        for model_dict, weight in zip(model_dicts, weights):
            for key, value in model_dict.items():
                if key not in aggregated_weights:
                    aggregated_weights[key] = value * weight
                else:
                    aggregated_weights[key] += value * weight
        return aggregated_weights
