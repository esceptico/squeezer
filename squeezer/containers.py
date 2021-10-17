from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor


@dataclass
class Batch:
    data: torch.Tensor
    target: torch.Tensor

    def to(self, device):
        self.data.to(device)
        self.target.to(device)
        return self
