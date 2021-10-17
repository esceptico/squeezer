from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as func

from squeezer.containers import ModelOutput
from squeezer.criterion import kld_loss

LossDictT = Dict[str, float]


class AbstractDistillationPolicy(ABC, nn.Module):
    def forward(self, teacher_output, student_output, target, epoch) -> Tuple[torch.Tensor, LossDictT]:
        raise NotImplementedError()


class DistillationPolicy(AbstractDistillationPolicy):
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        teacher_output: ModelOutput,
        student_output: ModelOutput,
        target: torch.Tensor,
        epoch: int
    ) -> Tuple[torch.Tensor, LossDictT]:
        loss_kld = kld_loss(
            teacher_output.logits,
            student_output.logits,
            self.temperature
        )
        loss_ce = func.cross_entropy(student_output.logits, target)
        overall = loss_kld * self.alpha + loss_ce * (1 - self.alpha)
        loss_dict = {
            'kld': loss_kld.item(),
            'cross_entropy': loss_ce.item(),
            'overall': overall.item(),
        }
        return overall, loss_dict




