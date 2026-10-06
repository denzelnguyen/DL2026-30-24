"""One training loop shared by every model (student, TinyBERT) so comparisons are fair."""
import time

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .losses import DistillationLoss
from .models import evaluate_accuracy


def train_model(
    model,
    train_loader,
    eval_loader,
    device,
    *,
    teacher=None,
    epochs: int = 3,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    temperature: float = 3.0,
    alpha: float = 0.5,
    bf16: bool = False,
):
    """Train `model`. If `teacher` is given -> knowledge distillation, otherwise plain CE.

    Returns a list with one record per epoch (train loss, validation accuracy, seconds).
    """
    use_kd = teacher is not None
    criterion = DistillationLoss(temperature, alpha if use_kd else 1.0)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    history = []
    for epoch in range(epochs):
        model.train()
        running, t0 = 0.0, time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop("labels")

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                teacher_logits = None
                if use_kd:
                    with torch.no_grad():
                        teacher_logits = teacher(**inputs).logits
                logits = model(**inputs).logits

            if use_kd:
                loss, _, _ = criterion(logits.float(), teacher_logits.float(), labels)
            else:
                loss = criterion.ce_loss(logits.float(), labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()

        record = {
            "epoch": epoch + 1,
            "train_loss": running / len(train_loader),
            "val_accuracy": evaluate_accuracy(model, eval_loader, device),
            "seconds": time.time() - t0,
        }
        history.append(record)
        print(record)
    return history
