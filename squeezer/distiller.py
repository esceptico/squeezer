import sys
from typing import List, Optional, TypeVar, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from squeezer.logging.abstract import AbstractLogger
from squeezer.logging.stub import StubLogger
from squeezer.policy import AbstractDistillationPolicy
from squeezer.reduce import Average, DictAverage
from squeezer.utils import move_to_device


BatchT = TypeVar('BatchT')


# TODO
# 1. implement metrics
# 2. checkpoints
class Distiller:
    """Base class for distiller.

    Notes:
        You must implement teacher_forward and student_forward methods
        in the child class. By default those methods define as:
        >>> def teacher_forward(self, batch):
        >>>     return self.teacher(**batch)

        >>> def student_forward(self, batch):
        >>>     return self.student(**batch)

    """
    def __init__(
        self,
        teacher: Union[nn.Module, List[nn.Module]],
        student: nn.Module,
        loss_policy: AbstractDistillationPolicy,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional = None,
        device: Union[str, torch.device] = 'cpu',
        logger: Optional[AbstractLogger] = None
    ):
        """Constructor.

        Args:
            teacher: Teacher model or list of models.
            student: Student model.
            optimizer: Optimizer.
            scheduler: Learning Rate scheduler. Defaults to None.
            loss_policy: Distillation loss policy.
            device: Device to which you want to move data and models.
                Defaults to cpu.
            logger: Instance of AbstractLogger class. If None, will be replaced to stub logger.
                Defaults to None.
        """
        self.device = device
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.loss_policy = loss_policy
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.logger = StubLogger() if logger is None else logger

    def __call__(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, n_epochs: int = 10):
        self.teacher.eval()

        for epoch in range(n_epochs):
            try:
                self._train(train_loader, epoch)
                if val_loader is not None:
                    self._validate(val_loader, epoch)
            except KeyboardInterrupt:
                print('Keyboard interrupt...')
                return
            finally:
                self.logger.dispose()

    def move_batch_to_device(self, batch: BatchT) -> BatchT:
        return move_to_device(batch, self.device)

    def teacher_forward(self, batch):
        """Implements teacher forward on given batch.

        Args:
            batch: Batch of data.

        Returns:
            Required teacher output.

        Examples:
            >>> def teacher_forward(self, batch):
            >>>     return self.teacher(**batch)
        """
        raise NotImplementedError

    def student_forward(self, batch):
        """Implements student forward on given batch.

        Args:
            batch: Batch of data.

        Returns:
            Required student output.

        Examples:
            >>> def student_forward(self, batch):
            >>>     return self.student(**batch)
        """
        raise NotImplementedError

    def _train(self, loader: DataLoader, epoch: int, accumulation_steps: int = 1):
        self.student.train()

        loss_average = Average()
        average = DictAverage()

        last_batch_index = len(loader) - 1
        bar_desc = f'[{epoch}th epoch]'
        loader_bar = tqdm(loader, desc=bar_desc, leave=True, file=sys.stdout)
        for i, batch in enumerate(loader_bar):
            batch = self.move_batch_to_device(batch)
            with torch.no_grad():
                teacher_output = self.teacher_forward(batch)
            student_output = self.student_forward(batch)
            batch_loss, batch_loss_dict = self.loss_policy(
                teacher_output=teacher_output,
                student_output=student_output,
                batch=batch,
                epoch=epoch
            )
            batch_loss = batch_loss / accumulation_steps
            batch_loss.backward()
            if (i + 1) % accumulation_steps == 0 or i == last_batch_index:
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            self.optimizer.step()

            loss_average.update(batch_loss.item())
            average.update(batch_loss_dict)
            loader_bar.set_postfix({'loss': loss_average.compute()})
        loss_dict = {f'Train/{k}': v for k, v in average.compute().items()}
        self.logger.log_dict(loss_dict, step=epoch)

    @torch.inference_mode()
    def _validate(self, loader: DataLoader, epoch: int):
        self.student.train()

        loss_average = Average()
        average = DictAverage()

        loader_bar = tqdm(loader, desc=f'Validation', leave=True, file=sys.stdout)
        for batch in loader_bar:
            batch = self.move_batch_to_device(batch)
            teacher_output = self.teacher_forward(batch)
            student_output = self.student_forward(batch)
            batch_loss, batch_loss_dict = self.loss_policy(
                teacher_output=teacher_output,
                student_output=student_output,
                batch=batch,
                epoch=epoch
            )
            loss_average.update(batch_loss.item())
            average.update(batch_loss_dict)
            loader_bar.set_postfix({'loss': loss_average.compute()})
        loss_dict = {f'Val/{k}': v for k, v in average.compute().items()}
        self.logger.log_dict(loss_dict, step=epoch)
