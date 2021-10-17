import torch
from torch.nn import functional as func


def distill_loss(teacher_logits, student_logits, labels, temperature, alpha):
    loss_kld = kld_loss(teacher_logits, student_logits, temperature)
    loss_ce = func.cross_entropy(student_logits, labels)
    overall = loss_kld * alpha + loss_ce * (1. - alpha)
    return loss_kld, loss_ce, overall


def kld_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    soft_log_probs = func.log_softmax(student_logits / temperature, dim=-1)
    soft_targets = func.softmax(teacher_logits / temperature, dim=-1)
    distillation_loss = func.kl_div(
        input=soft_log_probs,
        target=soft_targets,
        reduction='batchmean'
    )
    distillation_loss_scaled = distillation_loss * temperature ** 2
    return distillation_loss_scaled


def cosine_similarity_loss(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor
) -> torch.Tensor:
    # TODO: refactor
    bs, seq, dim = student_hidden.size()
    target = student_hidden.new_ones(bs*seq)
    teacher_hidden = teacher_hidden.view(bs*seq, -1).clone()
    return func.cosine_embedding_loss(
        teacher_hidden,
        student_hidden.contiguous().view(-1, dim),
        target,
        reduction='mean'
    )
