# MobileBERT — Bottleneck Layer-wise Knowledge Distillation on SST-2

Distill a pre-fine-tuned DistilBERT teacher into a **MobileBERT** student
initialised **from scratch** (random weights) using a binary soft-target
regression loss. The student uses a custom single-output classification head
with `BCEWithLogitsLoss`, where the teacher's class-1 probability serves as a
continuous soft label.

---

## Core Architecture & Innovation

Unlike standard KD that uses KL divergence over full probability distributions
with a temperature parameter, this implementation frames distillation as a
**binary regression** problem:

* The **teacher** (`distilbert-base-uncased-finetuned-sst-2-english`) produces a
  2-class softmax distribution. Only the **class-1 probability** is extracted as
  the soft target.
* The **student** (`google/mobilebert-uncased`) outputs a **single scalar logit**
  via `nn.Linear(hidden_size, 1)` with Xavier initialisation. The same
  `BCEWithLogitsLoss` is applied twice — once against hard labels and once
  against the teacher's soft probability.

**Loss Function:**

```
L = (1 − α) · BCE(student_logit, hard_label) + α · BCE(student_logit, teacher_prob)
```

where `α = 0.5`, balancing hard and soft supervision equally.

---

## Setup

```bash
pip install torch transformers datasets scikit-learn
```

## Run

```bash
python main.py
```

* Training runs for **3 epochs** with `AdamW` (lr = 5e-5, weight_decay = 0.01)
  and a **cosine warmup** schedule (50 warmup steps).
* Gradient clipping at `max_norm = 1.0`.
* Best model (by weighted F1) is saved to `./student_sst2_distilled/best_model.pt`.
* The tokenizer is also saved to `./student_sst2_distilled/` for inference.

---

## Files

| File | What it does |
| --- | --- |
| `config.py` | Teacher/student IDs, hyperparameters (α, lr, epochs, batch size), device |
| `dataset.py` | `get_dataloaders()` → SST-2 train/val DataLoaders + tokenizer |
| `model.py` | `MobileBertForSequenceClassificationBCE` — custom BCE classification head |
| `engine.py` | `train_one_epoch` (KD loop) and `evaluate` (accuracy + weighted F1) |
| `main.py` | Entry point — loads teacher, initialises student from scratch, trains, saves |

---

## Key Design Choices

| Choice | Reason |
| --- | --- |
| BCE instead of KL-div | Binary sentiment naturally maps to a single sigmoid output; teacher confidence is a direct regression target |
| Student from scratch (`AutoConfig`) | Tests whether KD alone is sufficient to train MobileBERT without pretrained LM weights |
| Xavier init on classifier | Balanced initial gradients for the single-output head |
| `teacher.eval()` + `torch.no_grad()` | Teacher is frozen — no gradient memory overhead |
| Cosine schedule with warmup | Smooth learning rate decay; warmup prevents early divergence |
| Best model by F1 (not accuracy) | Weighted F1 is more robust for imbalanced evaluation splits |

---

## Hyperparameter Tuning Guide

Change one setting at a time and compare validation accuracy and F1:

| Knob | Values to try | Location |
| --- | --- | --- |
| `ALPHA` | 0.3, 0.5, 0.7 | `config.py` |
| `LEARNING_RATE` | 2e-5, 5e-5, 1e-4 | `config.py` |
| `EPOCHS` | 3, 5, 8 | `config.py` |
| `BATCH_SIZE` | 16, 32, 64 | `config.py` |
| `num_warmup_steps` | 0, 50, 100 | `main.py` |

---

## Reproducibility

* **Seed**: Dataset shuffle uses `seed=42` in `dataset.py`.
* **Hardware**: Verified on Kaggle GPU (NVIDIA T4). Results may vary slightly
  on different GPUs due to non-deterministic CUDA operations.
* **Dependencies**: `torch`, `transformers`, `datasets`, `scikit-learn`.
