"""
====================
Base Class Submodule for Servers
====================
"""
from typing import List

from ..client.base import ClientBase


class ServerBase:
    """Base class for servers in FL."""
    clients: List[ClientBase]

    def __init__(self):
        pass

    def _assert_client_uniqueness(self, client: ClientBase):
        """Asserts that the client is unique.

        Args:
            client: The client to check.
        """
        if client in self.clients:
            raise ValueError(f"Client {client.id} already exists in the server.")

    def add_client(self, client: ClientBase):
        """Adds a client to the server.

        Args:
            client: The client to add to the server.
        """
        self._assert_client_uniqueness(client)
        self.clients.append(client)

    def aggregate(self):
        """Aggregates the clients' weights."""
        raise NotImplementedError
