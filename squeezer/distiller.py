import os
import sys
from collections import defaultdict
from typing import Optional, TypeVar, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from squeezer.policy import AbstractDistillationPolicy
from squeezer.utils import move_to_device


BatchT = TypeVar('BatchT')


# TODO
# 1. implement metrics
# 2. gradient accumulating
# 3. save / load
# 4. checkpoints
# 5. logging
# 6. scheduler
class Distiller:
    """Base class for distiller."""
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        loss_policy: AbstractDistillationPolicy,
        device: Union[str, torch.device] = 'cpu',
        optimizer=None,
    ):
        self.device = device
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.loss_policy = loss_policy
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

    def __call__(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, n_epochs: int = 10):
        self.teacher.eval()

        for epoch in range(n_epochs):
            try:
                self._train(train_loader, epoch)
                if val_loader is not None:
                    self._validate(val_loader, epoch)
            except InterruptedError:
                print('Keyboard interrupt...')

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
            >>> def teacher_forward(self, batch):
            >>>     return self.student(**batch)
        """
        raise NotImplementedError

    def _train(self, loader: DataLoader, epoch: int):
        self.student.train()
        loss_dict = defaultdict(float)

        loader_bar = tqdm(loader, desc=f'Train [{epoch}th epoch]', leave=False, file=sys.stdout)
        for i, batch in enumerate(loader_bar):
            batch = self.move_batch_to_device(batch)
            self.optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = self.teacher_forward(batch)
            student_output = self.student_forward(batch)
            batch_loss, batch_loss_dict = self.loss_policy(
                teacher_output=teacher_output,
                student_output=student_output,
                batch=batch,
                epoch=epoch
            )
            for loss_name, loss_value in batch_loss_dict.items():
                loss_dict[loss_name] += loss_value
            batch_loss.backward()
            self.optimizer.step()

            loss_dict = {k: v / (i + 1) for k, v in loss_dict.items()}
        report = '   '.join(f'{k}={v:.5f}' for k, v in loss_dict.items())
        print(f'Epoch {epoch}: {report}')

    @torch.no_grad()
    def _validate(self, loader: DataLoader, epoch: int):
        self.student.train()
        loss_dict = defaultdict(float)

        loader_bar = tqdm(loader)
        for batch in loader_bar:
            batch = self.move_batch_to_device(batch)
            teacher_output = self.teacher_forward(batch)
            student_output = self.student_forward(batch)
            _, batch_loss_dict = self.loss_policy(
                teacher_output=teacher_output,
                student_output=student_output,
                batch=batch,
                epoch=epoch
            )
            for loss_name, loss_value in batch_loss_dict.items():
                loss_dict[loss_name] += loss_value
        loss_dict = {k: v / len(loader) for k, v in loss_dict.items()}
        report = '   '.join(f'{k}={v:.5f}' for k, v in loss_dict.items())
        print(f'Epoch {epoch}: {report}')

    def save(self, save_path: str):
        """Saves weights (student, optimizer and loss policy) to the given directory."""
        torch.save(self.student.state_dict(), os.path.join(save_path, 'student.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(save_path, 'optimizer.pth'))
        torch.save(self.loss_policy.state_dict(), os.path.join(save_path, 'loss_policy.pth'))

    def load(self, load_path: str, device: Union[str, torch.device] = 'cpu'):
        """Loads weights (student, optimizer and loss policy) from the given directory."""
        student_state_dict = torch.load(os.path.join(load_path, 'student.pth', device))
        self.student.load_state_dict(student_state_dict)
        optimizer_state_dict = torch.load(os.path.join(load_path, 'optimizer.pth', device))
        self.optimizer.load_state_dict(optimizer_state_dict)
        loss_policy_state_dict = torch.load(os.path.join(load_path, 'loss_policy.pth', device))
        self.loss_policy.load_state_dict(loss_policy_state_dict)
