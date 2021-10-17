from collections import defaultdict
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchmetrics import Metric, MetricCollection

MetricT = Callable[[torch.Tensor, torch.Tensor], float]


# TODO
# 1. implement metrics
# 2. gradient accumulating
# 3. save / load
# 4. checkpoints
# 5. logging
class Distiller:
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        loss_policy: Callable,
        metrics: Optional[Dict[str, MetricT]] = None,
        device: Union[str, torch.device] = 'cpu',
    ):
        self.device = device
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        self.loss_policy = loss_policy
        self.metrics = metrics

    def __call__(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, n_epochs: int = 10):
        self.teacher.eval()

        for epoch in range(n_epochs):
            self._train(train_loader, epoch)
            if val_loader is not None:
                self._validate(val_loader, epoch)

    def _train(self, loader, epoch):
        self.student.train()
        loss_dict = defaultdict(float)

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            with torch.no_grad():
                teacher_output = self.teacher(batch.data)
            student_output = self.student(batch.data)
            batch_loss, batch_loss_dict = self.loss_policy(
                teacher_output=teacher_output,
                student_output=student_output,
                target=batch.target,
                epoch=epoch
            )
            for loss_name, loss_value in batch_loss_dict.items():
                loss_dict[loss_name] += loss_value
            batch_loss.backward()
            self.optimizer.step()
        loss_dict = {k: v / len(loader) for k, v in loss_dict.items()}

        report = '\t'.join(f'{k}={v:.5f}' for k, v in loss_dict.items())
        print(f'Epoch {epoch}: {report}')

    @torch.inference_mode()
    def _validate(self, loader, epoch):
        self.student.train()
        loss_dict = defaultdict(float)
        metric_dict = defaultdict(float)

        for batch in loader:
            batch = batch.to(self.device)
            teacher_output = self.teacher(batch.data)
            student_output = self.student(batch.data)
            batch_loss, batch_loss_dict = self.loss_policy(
                teacher_output=teacher_output,
                student_output=student_output,
                target=batch.target,
                epoch=epoch
            )
