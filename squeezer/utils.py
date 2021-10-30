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
