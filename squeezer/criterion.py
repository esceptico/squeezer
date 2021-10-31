from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as func


def distill_loss(
    teacher_logits: Tensor,
    student_logits: Tensor,
    labels: Tensor,
    temperature: float = 1.0,
    alpha: float = 0.5
) -> Tuple[Tensor, Tensor, Tensor]:
    """Default distillation loss (KLD + CE).

    Args:
        teacher_logits: Logits from teacher model.
        student_logits: Logits from student model.
        labels: Targets.
        temperature: Temperature for softening distributions.
            Larger temperature -> softer distribution.
        alpha: Weight to KLD loss and 1 - alpha to CrossEntropy loss

    Returns:
        Tuple of KLD loss, CE loss and combined weighted loss tensors.
    """
    loss_kld = kld_loss(teacher_logits, student_logits, temperature)
    loss_ce = func.cross_entropy(student_logits, labels)
    overall = loss_kld * alpha + loss_ce * (1. - alpha)
    return loss_kld, loss_ce, overall


def kld_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """Kullback–Leibler divergence loss.

    Args:
        teacher_logits: Logits from teacher model.
        student_logits: Logits from student model.
        temperature: Temperature for softening distributions.
            Larger temperature -> softer distribution.

    Returns:
        Tensor of loss value.
    """
    soft_log_probs = func.log_softmax(student_logits / temperature, dim=-1)
    soft_targets = func.softmax(teacher_logits / temperature, dim=-1)
    distillation_loss = func.kl_div(
        input=soft_log_probs,
        target=soft_targets,
        reduction='batchmean'
    )
    distillation_loss_scaled = distillation_loss * temperature ** 2
    return distillation_loss_scaled


def cosine_loss(x1: Tensor, x2: Tensor) -> Tensor:
    """Cosine distance loss calculated on last dimension.

    Args:
        x1: First input.
        x2: Second input (of size matching x1).

    Returns:
        Averaged cosine distance between x1 and x2.
    """
    distance = 1 - func.cosine_similarity(x1, x2, dim=-1)
    mean_distance = distance.mean()
    return mean_distance

