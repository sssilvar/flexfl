"""
=====================
Client submodule for FlexFL
=====================

In federated learning, the client is the entity that holds the data and trains the model. The only information that
the client sends to the server is the model's weights. The server then aggregates the weights from all the clients
and sends the updated model back to the clients.

The client is responsible for the following tasks:
- Loading the data
- Training the model
- Sending the model's weights to the server
- Receiving the updated model from the server

The client is implemented as a Python class that inherits from the :class:`Client` class.
"""