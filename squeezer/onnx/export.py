from logging import getLogger
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.onnx import export

logger = getLogger(__name__)


def export_to_onnx(
    model: nn.Module,
    dummy_input: [Union[Tuple], torch.Tensor],
    file,
    opset_version: int = 12,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None
) -> None:
    """Exports PyTorch model to ONNX format.

    Args:
        model: PyTorch module.
        dummy_input: Dummy input.
        file: Path to save converted model or file-like object.
        opset_version: Version of ONNX operator set. Defaults to 12.
        input_names: Names of model inputs. Defaults to None.
        output_names: Names of model outputs. Defaults to None.
        dynamic_axes: Axes (input or/and outputs) with dynamic shapes.
            Defaults to None.

    Examples:
        >>> from transformers import AutoModel, AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        >>> model = AutoModel.from_pretrained('bert-base-uncased')
        >>> encoded = tokenizer('aboba', return_tensors='np')
        >>>
        >>> export_to_onnx(
        >>>     model,
        >>>     dummy_input=tuple(encoded.values()),
        >>>     path_to_save='model.onnx',
        >>>     input_names=list(encoded.keys()),
        >>>     output_names=['last_hidden_state', 'pooler_output'],
        >>>     dynamic_axes={
        >>>         'input_ids' : {0 : 'batch_size', 1: 'seq'},
        >>>         'token_type_ids' : {0 : 'batch_size', 1: 'seq'},
        >>>         'attention_mask' : {0 : 'batch_size', 1: 'seq'},
        >>>         'last_hidden_state' : {0 : 'batch_size', 1: 'seq'},
        >>>         'pooler_output' : {0 : 'batch_size', 1: 'seq'}
        >>>     }
        >>> )
    """
    model.eval()
    export(
        model,
        dummy_input,
        file,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    logger.warning(f'Model was exported to ONNX.')
