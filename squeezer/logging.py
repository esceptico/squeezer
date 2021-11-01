import os
from datetime import datetime
from logging import getLogger
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter


logger = getLogger(__name__)


class TensorboardLogger:
    """TensorBoard logger."""
    def __init__(self, log_dir: str, name: str):
        self.log_dir = log_dir
        self._name = name
        self.writer = SummaryWriter(self.experiment_dir)

    @property
    def name(self) -> str:
        if self._name is None:
            now = datetime.now()
            return now.strftime('%Y_%m_%d_%H:%M:%S')
        return self._name

    @property
    def experiment_dir(self) -> str:
        directory = os.path.join(self.log_dir, self.name)
        if os.path.exists(directory):
            directory = f'{directory}_copy'
            logger.warning(f'Experiment directory is exists. This run will be saved to {directory}')
        os.makedirs(directory, exist_ok=True)
        return directory

    def log(self, name: str, value: float, step: Optional[int] = None) -> None:
        self.writer.add_scalar(name, value, step)

    def log_dict(self, values: Dict[str, float], step: Optional[int] = None) -> None:
        for name, value in values.items():
            self.log(name, value, step)

    def dispose(self) -> None:
        self.writer.flush()
        self.writer.close()
