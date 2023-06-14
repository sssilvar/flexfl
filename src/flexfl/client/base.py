"""
====================
FlexFL Client Base Class
====================
"""
from typing import OrderedDict

import pytorch_lightning as pl


class ClientBase:
    """Base class for clients in FL."""

    def __init__(
            self,
            client_id: str,
            data_module: pl.LightningDataModule,
            model: pl.LightningModule,
            weight: float,
            logger: None
    ):
        """
        Args:
            client_id: The client's ID.
            data_module: The client's data module.
            model: The client's model.
            weight: The client's weight.
        """
        self.id = client_id
        self.data_module = data_module
        self.model = model
        self.weight = weight

        # Assert weight is less than 1
        assert self.weight <= 1, "Client weight must be less than or equal to 1."

        # Stateful attributes
        # Number of iterations the client has been trained on
        self.n_iterations = 0
        # The client's current state dictionary
        self.state_dict = None

    def partial_fit(self, n_steps: int, initial_state_dict: OrderedDict):
        """Performs a partial fit of the model on the client's data.

        Args:
            n_steps: The number of gradient steps to train the model.
            initial_state_dict: Initialization point for the model's weights.
        """
        # Set the model's weights to the initial state dict
        self.model.load_state_dict(initial_state_dict)

        # Create a trainer with no loger and no checkpoint callback specifying the number of steps to train
        trainer = pl.Trainer(
            logger=False,
            checkpoint_callback=False,
            max_steps=n_steps
        )

        # Train the model
        trainer.fit(self.model, datamodule=self.data_module)

        # Update the number of iterations
        self.n_iterations += n_steps

        # Update the state dict
        self.state_dict = self.model.state_dict()
