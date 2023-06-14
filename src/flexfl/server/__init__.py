"""
=====================
FlexFL Server Submodule
=====================

The server is responsible for the following tasks:
- Receiving the model's weights from the clients
- Aggregating the weights
- Sending the updated model to the clients

The server is implemented as a Python class that inherits from the :class:`Server` class. To add a client to the
server, it is necessary to use the `add_client` method.

"""