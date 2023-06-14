import logging

logger = logging.getLogger("flexfl")


class App:
    def __init__(self, settings) -> None:
        self.settings = settings

    def start(self) -> None:
        logger.info("flexfl Starting")

    def stop(self) -> None:
        logger.info("flexfl Stopping")
