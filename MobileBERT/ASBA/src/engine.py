"""Model creation, training loop (plain CE or knowledge distillation), prediction, metrics."""
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, get_linear_schedule_with_warmup)

from src import config
from src.data import MODEL_KEYS

device = config.device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def new_model(name, label2id):
    id2label = {i: l for l, i in label2id.items()}
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(label2id), id2label=id2label, label2id=label2id)


def model_inputs(batch, dev):
    return {k: batch[k].to(dev) for k in MODEL_KEYS if k in batch}


@torch.no_grad()
def predict(model, loader, amp=False):
    model.eval()
    outs = []
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(**model_inputs(batch, device)).logits
        outs.append(logits.float().cpu())
    return torch.cat(outs)


def metrics(logits, y):
    y = np.asarray(y)
    pred = logits.argmax(-1).numpy()
    present = sorted(set(y.tolist()))      # macro-F1 only over classes present in this split
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, labels=present, average="macro", zero_division=0)
    return {"acc": float(acc), "macro_f1": float(f1), "score": float((acc + f1) / 2)}


def train_model(model, train_loader, val_loader, epochs, lr, amp, kd=None, tag=""):
    """kd=None -> cross-entropy on hard labels.
    kd=(T, alpha) -> alpha*CE + (1-alpha)*T^2*KL(softmax(teacher/T) || softmax(student/T)).
    The best epoch is picked on the validation set and its weights are restored at the end."""
    model.to(device)
    no_decay = lambda n: n.endswith("bias") or "LayerNorm" in n
    groups = [
        {"params": [p for n, p in model.named_parameters() if not no_decay(n)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters() if no_decay(n)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    total = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best = {"score": -1.0, "epoch": 0, "state": None}
    history = []
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for batch in train_loader:
            labels = batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=amp and device.type == "cuda"):
                logits = model(**model_inputs(batch, device)).logits
            logits = logits.float()
            loss = F.cross_entropy(logits, labels)
            if kd is not None:
                T, alpha = kd
                t_logits = batch["teacher_logits"].to(device)
                kl = F.kl_div(F.log_softmax(logits / T, dim=-1),
                              F.softmax(t_logits / T, dim=-1),
                              reduction="batchmean") * (T * T)
                loss = alpha * loss + (1 - alpha) * kl
            if not torch.isfinite(loss):
                raise RuntimeError(f"[{tag}] loss is {loss.item()} - if this is MobileBERT make sure "
                                   f"STUDENT_AMP=False, otherwise lower the learning rate")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            n_batches += 1

        val_m = metrics(predict(model, val_loader, amp), val_loader.dataset.y)
        history.append({"epoch": epoch, "train_loss": running / n_batches,
                        **{f"val_{k}": v for k, v in val_m.items()}})
        marker = ""
        if val_m["score"] > best["score"]:
            best = {"score": val_m["score"], "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            marker = "  <- best"
        print(f"[{tag}] epoch {epoch}/{epochs} | loss {running / n_batches:.4f} | "
              f"val acc {val_m['acc']:.4f} f1 {val_m['macro_f1']:.4f}{marker}")

    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "val_score": best["score"], "history": history,
            "train_time_s": time.time() - t_start}
