class AbstractLogger:
    """Abstract logger"""
    def log(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def log_dict(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def dispose(self) -> None:
        raise NotImplementedError
