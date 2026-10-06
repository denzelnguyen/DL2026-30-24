import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """alpha * CE(hard labels) + (1 - alpha) * T^2 * KL(student_soft || teacher_soft)."""

    def __init__(self, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        loss_ce = self.ce_loss(student_logits, labels)

        soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss_kd = self.kl_loss(soft_student, soft_teacher) * (self.temperature ** 2)

        total = self.alpha * loss_ce + (1.0 - self.alpha) * loss_kd
        return total, loss_ce, loss_kd
