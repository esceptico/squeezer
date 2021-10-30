from abc import ABC
from typing import Dict, Tuple

import torch
from torch import nn


LossDictT = Dict[str, float]


class AbstractDistillationPolicy(ABC, nn.Module):
    """Abstract class for all distillation policies.
    """
    def forward(self, teacher_output, student_output, batch, epoch: int) -> Tuple[torch.Tensor, LossDictT]:
        """Forward method.

        Args:
            teacher_output: Output of teacher model.
            student_output: Output of student model.
            batch: Batch from loader.
            epoch: Number of epoch.

        Returns:
            Tuple of overall loss tensor and dictionary of sub-loss values.
            Overall loss must support autograd, i.e. requires_grad=True.
        """
        raise NotImplementedError()
