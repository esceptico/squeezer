from abc import ABC
from typing import Dict, Optional, Tuple

import torch
from torch import nn


ValuesDictT = Dict[str, float]


class AbstractDistillationPolicy(ABC, nn.Module):
    """Abstract class for all distillation policies.
    """
    def forward(
        self,
        teacher_output,
        student_output,
        batch,
        epoch: Optional[int] = None
    ) -> Tuple[torch.Tensor, ValuesDictT]:
        """Forward method.

        Args:
            teacher_output: Output of teacher model.
            student_output: Output of student model.
            batch: Batch from loader.
            epoch: Number of epoch.

        Returns:
            Tuple that contains loss tensor and values to log.
        """
        raise NotImplementedError
