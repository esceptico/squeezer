from typing import Tuple

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from squeezer import AbstractDistillationPolicy
from squeezer.policy.abstract import ValuesDictT

NUM_FEATURES = 32
NUM_TARGETS = 1


@pytest.fixture
def loader():
    data = torch.randn(10, NUM_FEATURES)
    targets = torch.randn(10, NUM_TARGETS)
    dataset = TensorDataset(data, targets)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture
def model():
    return torch.nn.Linear(NUM_FEATURES, NUM_TARGETS)


@pytest.fixture
def policy():
    class Policy(AbstractDistillationPolicy):
        def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, ValuesDictT]:
            return torch.tensor(42.0, requires_grad=True), {'value': 42}
    return Policy()
