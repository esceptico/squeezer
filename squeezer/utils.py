import os
from typing import Any, Union

import torch


def move_to_device(obj: Any, device: Union[str, torch.device]):
    """Moves object to the given device.

    Args:
        obj: Object (tensor, list or dict of tensors or nested object).
        device: Device to move object to.

    Returns:
        Moved to given device object

    Raises:
        ValueError: If the object cannot be moved to the device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_to_device(value, device)
        return obj
    if isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = move_to_device(value, device)
        return obj
    if isinstance(obj, tuple):
        obj = list(obj)
        for i, value in enumerate(obj):
            obj[i] = move_to_device(value, device)
        return tuple(obj)
    else:
        raise ValueError(f'Cannot move given object of type ({type(obj)}) to {device}.')


def save_weights(module, path: str, skip_if_empty: bool = True) -> None:
    """Saves parameters of model, optimizer or scheduler to the given path.

    Args:
        module: Object whose parameters need to be saved.
        path: File path to save parameters.
        skip_if_empty: Do not save empty object if object have no parameters to save?
    """
    weights = module.state_dict()
    if not any(weights) and skip_if_empty:
        return
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    torch.save(weights, path)
