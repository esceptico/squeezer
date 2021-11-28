from squeezer.logging.abstract import AbstractLogger


class StubLogger(AbstractLogger):
    """Stub logger. Does nothing."""
    def log(self, *args, **kwargs) -> None:
        pass

    def log_dict(self, *args, **kwargs) -> None:
        pass

    def dispose(self) -> None:
        pass
