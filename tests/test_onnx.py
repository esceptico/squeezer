import tempfile

import torch
from torch import nn

from squeezer.onnx.export import export_to_onnx


def test_onnx_export():
    with tempfile.TemporaryFile() as fp:
        model = nn.Linear(16, 2)
        dummy_input = torch.randn(2, 16)
        export_to_onnx(model, dummy_input, fp)
