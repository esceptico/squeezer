from typing import Tuple

import torch

from squeezer.criterion.functional import kld_loss
from squeezer.policy.abstract import AbstractDistillationPolicy, ValuesDictT


class BertForSequenceClassificationPolicy(AbstractDistillationPolicy):
    """Loss policy for BertForSequenceClassification model."""
    def __init__(self, alpha: float = 0.5, temperature: float = 2):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, teacher_output, student_output, batch, *args, **kwargs) -> Tuple[torch.Tensor, ValuesDictT]:
        kld = kld_loss(
            teacher_logits=teacher_output.logits,
            student_logits=student_output.logits,
            temperature=self.temperature
        )
        cross_entropy = student_output.loss
        overall = self.alpha * cross_entropy + (1 - self.alpha) * kld
        value_dict = {
            'cross_entropy': cross_entropy.item(),
            'kld': kld.item(),
            'overall': overall.item()
        }
        return overall, value_dict
