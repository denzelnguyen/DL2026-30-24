import copy

import evaluate
import torch
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification

from .config import TEACHER_ID


def load_teacher(device, teacher_id: str = TEACHER_ID):
    """Load the teacher and freeze it."""
    model = AutoModelForSequenceClassification.from_pretrained(teacher_id)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def default_layer_map(num_student_layers: int, num_teacher_layers: int):
    """Evenly spaced teacher layers, e.g. 24 -> 6 gives [0, 4, 8, 12, 16, 20]."""
    step = num_teacher_layers // num_student_layers
    return [i * step for i in range(num_student_layers)]


def build_student(teacher, num_layers: int = 6, layer_map=None, device=None):
    """Create a shallower BERT and initialise it from selected teacher layers."""
    t_cfg = teacher.config
    if layer_map is None:
        layer_map = default_layer_map(num_layers, t_cfg.num_hidden_layers)
    assert len(layer_map) == num_layers, "layer_map length must equal num_layers"

    s_cfg = copy.deepcopy(t_cfg)
    s_cfg.num_hidden_layers = num_layers
    student = BertForSequenceClassification(s_cfg)

    student.bert.embeddings.load_state_dict(teacher.bert.embeddings.state_dict())
    for s_idx, t_idx in enumerate(layer_map):
        student.bert.encoder.layer[s_idx].load_state_dict(
            teacher.bert.encoder.layer[t_idx].state_dict()
        )
    student.bert.pooler.load_state_dict(teacher.bert.pooler.state_dict())
    student.classifier.load_state_dict(teacher.classifier.state_dict())

    if device is not None:
        student.to(device)
    return student, layer_map


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device) -> float:
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        metric.add_batch(predictions=torch.argmax(logits, dim=-1), references=batch["labels"])
    return metric.compute()["accuracy"] * 100
