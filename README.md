# Knowledge Distillation: How Small Can a Neural Network Become?

> A systematic study of five knowledge distillation strategies for compressing
> BERT-family models on **SST-2** binary sentiment classification.
> Each student architecture implements a distinct distillation method and resides
> in its own self-contained directory with Kaggle-verified training
> configurations and results.

---

## Table of Contents

| #   | Student Model   | Distillation Strategy                             | Directory         | Status             |
| --- | --------------- | ------------------------------------------------- | ----------------- | ------------------ |
| 1   | DistilBERT      | Response-based & Cosine Hidden KD                 | `1_distilbert/`   | ✅ Completed       |
| 2   | TinyBERT        | Two-stage Transformer Distillation                | `2_tinybert/`     | ✅ Completed       |
| 3   | MobileBERT      | Bottleneck Layer-wise KD                          | `3_mobilebert/`   | ✅ Completed       |
| 4   | PKD-BERT        | Patient Knowledge Distillation                    | `4_pkd_bert/`     | ✅ Completed       |
| 5   | MiniLM          | Self-Attention & Value-Relation Matrix KD         | `5_minilm/`       | ✅ Completed       |

---

## Quick Start

```bash
# Install common dependencies
pip install torch transformers datasets evaluate accelerate

# Run any single model (example — MiniLM)
cd 5_minilm
python -m Bert.train_distill --layers 4 --epochs 5
```

> **Kaggle users:** Enable **GPU** + **Internet** in notebook settings, then
> follow the per-model run instructions below.

---

## 1. DistilBERT

**Response-based & Cosine Hidden Knowledge Distillation**

* **Directory**: `1_distilbert/`
* **Dataset Support**: SST-2
* **Teacher Model**: `bert-base-uncased` (12 layers, ~110 M parameters)
* **Student Model Details**:
  * **Architecture**: BERT-base encoder truncated to **2 / 4 / 6 layers**
    (hidden size 768 — same as the teacher). Layers are selected via
    `alternate` (evenly spaced, DistilBERT recipe) or `first` (bottom-k)
    strategy from the pretrained teacher weights.
  * **Core Files**:
    | File | Role |
    | ---- | ---- |
    | `1_distilbert/Bert/config.py` | Hyperparameters & output paths (T = 2.0, α = 0.5, β = 0.0) |
    | `1_distilbert/Bert/dataset.py` | `load_sst2()` → tokenized HF dataset + tokenizer |
    | `1_distilbert/Bert/model_teacher.py` | `build_teacher(path)` — load / fine-tune teacher |
    | `1_distilbert/Bert/model_student.py` | `build_student()` — layer-sliced student |
    | `1_distilbert/Bert/train_teacher.py` | Step 1 — fine-tune the teacher on SST-2 |
    | `1_distilbert/Bert/train_student.py` | Step 2 — baseline student (hard labels only) |
    | `1_distilbert/Bert/train_distill.py` | Step 3 — `DistillationTrainer` (KD + optional cosine) |
    | `1_distilbert/Bert/benchmark.py` | Accuracy, agreement, latency, compression table & plot |
    | `1_distilbert/Bert/utils.py` | `compute_metrics`, `make_training_args`, `count_parameters` |
    | `1_distilbert/Bert/run_all.py` | End-to-end pipeline (trains teacher → students → benchmark) |
* **Distillation Mechanics**:
  * **Loss Function**:

    ```
    L = α · L_CE  +  (1 − α) · T² · L_KL  +  β · L_cosine
    ```

  * **Transferred Knowledge**:
    * **Logits** — KL divergence on temperature-softened output distributions
    * **Hidden States** (optional, β > 0) — Cosine embedding loss on the
      last hidden layer between student and teacher
* **How to Run**:
  ```bash
  cd 1_distilbert

  # Full pipeline (teacher → 3 student sizes × 2 modes → benchmark)
  python -m Bert.run_all --layers 2 4 6

  # Or step-by-step
  python -m Bert.train_teacher
  python -m Bert.train_student  --layers 6 --init alternate
  python -m Bert.train_distill  --layers 6 --temperature 2 --alpha 0.5 --beta 0
  python -m Bert.benchmark
  ```
* **Tuning Guide** (change one at a time, compare `accuracy` and
  `agreement_with_teacher` in the benchmark table):

  | Knob | Values to try |
  | ---- | ------------- |
  | `--temperature` | 1, 2, 4 |
  | `--alpha` | 0.3, 0.5, 0.7 |
  | `--beta` | 0, 1 |
  | `--init` | alternate, first |
  | `--epochs` | 3, 5 |

* **Implementation Status**: ✅ Completed

---

## 2. TinyBERT

**Two-stage Transformer Distillation**

* **Directory**: `2_tinybert/`
* **Dataset Support**: SST-2
* **Teacher Model**: `yoshitomo-matsubara/bert-large-uncased-sst2` (24 layers, BERT-large, pre-fine-tuned on SST-2)
* **Student Model Details**:
  * **Architecture**:
    * **TinyBERT** — `huawei-noah/TinyBERT_General_4L_312D` (4 layers, 312 hidden, ~14.5 M params). Pre-distilled at the general stage; this repo performs the **task-specific** distillation stage.
    * **Custom 6 L Student** — 6 layers copied from the BERT-large teacher via evenly-spaced layer mapping (e.g. layers 0, 4, 8, 12, 16, 20).
  * **Core Files**:
    | File | Role |
    | ---- | ---- |
    | `2_tinybert/src/config.py` | Constants (teacher ID, dataset, seed, paths) |
    | `2_tinybert/src/data.py` | `build_dataloaders(tokenizer)` → train / eval DataLoaders |
    | `2_tinybert/src/models.py` | `load_teacher`, `build_student`, `evaluate_accuracy` |
    | `2_tinybert/src/losses.py` | `DistillationLoss` — α · CE + (1 − α) · T² · KL |
    | `2_tinybert/src/trainer.py` | Unified training loop (AdamW, linear warmup, grad clipping) |
    | `2_tinybert/src/benchmark.py` | Accuracy, F1, latency (batch = 1), throughput, peak VRAM |
    | `2_tinybert/src/utils.py` | `set_seed`, `get_device`, `count_parameters`, `model_size_mb` |
    | `2_tinybert/train_student.py` | Train the 6 L student (`--mode kd` or `--mode ce`) |
    | `2_tinybert/train_tinybert.py` | Train TinyBERT (`--mode kd` or `--mode ce`) |
    | `2_tinybert/run_benchmark.py` | Benchmark a single model (one process for clean VRAM) |
    | `2_tinybert/run_all.sh` | Full grid: multiple seeds × models × GPU/CPU benchmark |
* **Distillation Mechanics**:
  * **Loss Function**:

    ```
    L = α · L_CE  +  (1 − α) · T² · L_KL        (T = 3.0, α = 0.5)
    ```

  * **Transferred Knowledge**:
    * **Logits** — KL divergence on temperature-softened distributions
    * Tokenizer vocabulary compatibility is checked at runtime to ensure
      valid logit-level distillation
* **How to Run**:
  ```bash
  cd 2_tinybert

  # TinyBERT — with / without distillation
  python train_tinybert.py --mode kd --name tinybert_kd_s42 --bf16
  python train_tinybert.py --mode ce --name tinybert_ce_s42 --bf16

  # 6 L Student — with / without distillation
  python train_student.py --mode kd --num-layers 6 --name student_kd_6L_s42 --bf16
  python train_student.py --mode ce --num-layers 6 --name student_ce_6L_s42 --bf16

  # Benchmark (run each model in a separate process)
  python run_benchmark.py --name teacher \
      --model yoshitomo-matsubara/bert-large-uncased-sst2
  python run_benchmark.py --name tinybert_kd_s42 \
      --model checkpoints/tinybert_kd_s42
  python run_benchmark.py --name tinybert_kd_s42 \
      --model checkpoints/tinybert_kd_s42 --device cpu

  # Or the full grid in one go
  SEEDS_STUDENT="42" SEEDS_TINY="42 43 44" bash run_all.sh
  ```
* **Results**: Written to `results/<name>.json` (`*_cpu.json` for CPU runs);
  training curves saved to `results/<name>_train_log.json`.
* **Implementation Status**: ✅ Completed

---

## 3. MobileBERT

**Bottleneck Layer-wise Knowledge Distillation**

* **Directory**: `3_mobilebert/`
* **Dataset Support**: SST-2
* **Teacher Model**: `distilbert-base-uncased-finetuned-sst-2-english`
* **Student Model Details**:
  * **Architecture**: `google/mobilebert-uncased` initialized **from scratch**
    (random weights via `AutoConfig`) with a custom single-output classification
    head using `BCEWithLogitsLoss`. Xavier-initialized classifier with
    `nn.Linear(hidden_size, 1)`.
  * **Core Files**:
    | File | Role |
    | ---- | ---- |
    | `3_mobilebert/config.py` | Hyperparameters (α = 0.5, lr = 5e-5, epochs = 3, device) |
    | `3_mobilebert/dataset.py` | `get_dataloaders()` → train / eval DataLoaders + tokenizer |
    | `3_mobilebert/model.py` | `MobileBertForSequenceClassificationBCE` (custom BCE head) |
    | `3_mobilebert/engine.py` | `train_one_epoch` and `evaluate` (accuracy + weighted F1) |
    | `3_mobilebert/main.py` | Entry point — orchestrates full training pipeline |
* **Distillation Mechanics**:
  * **Loss Function**:

    ```
    L = (1 − α) · L_BCE_hard  +  α · L_BCE_soft       (α = 0.5)
    ```

  * **Transferred Knowledge**:
    * **Soft Targets** — Teacher's class-1 probability (via softmax) used as a
      continuous soft label for the student's BCE output.
    * This is a **binary regression** framing: the teacher's confidence score
      guides the student, rather than the standard KL divergence on full
      probability distributions.
* **How to Run**:
  ```bash
  cd 3_mobilebert
  python main.py
  ```
  Best model (by F1) is saved to `./student_sst2_distilled/best_model.pt`.
* **Implementation Status**: ✅ Completed

---

## 4. PKD-BERT

**Patient Knowledge Distillation**

* **Directory**: `4_pkd_bert/`
* **Dataset Support**: —
* **Teacher Model**: —
* **Student Model Details**:
  * **Architecture**: —
  * **Core Files**: —
* **Distillation Mechanics**:
  * **Loss Function**: —
  * **Transferred Knowledge**: —
* **How to Run**: —
* **Implementation Status**: ⏳ Pending Upload

---

## 5. MiniLM

**Self-Attention & Value-Relation Matrix Knowledge Distillation**

* **Directory**: `5_minilm/`
* **Dataset Support**: SST-2
* **Teacher Model**: `textattack/bert-base-uncased-SST-2` (12 layers, 768 hidden, ~109.5 M params)
* **Student Model Details**:
  * **Architecture**: Custom BERT — **4 layers, 384 hidden, 12 attention heads,
    1536 intermediate** (~19.2 M parameters, **−82.5 %** vs. teacher). Built
    from scratch via a hand-written BERT implementation (not HuggingFace
    `AutoModel`), enabling QKV tensor exposure for relation-based distillation.
  * **Core Files**:
    | File | Role |
    | ---- | ---- |
    | `5_minilm/Bert/config.py` | Hyperparameters (α\_ce = 1.0, α\_attn = 1.0, α\_val = 1.0, 12 relation heads) |
    | `5_minilm/Bert/dataset.py` | `load_sst2()` → tokenized HF dataset + tokenizer |
    | `5_minilm/Bert/modeling_bert.py` | Full custom BERT: `BertConfig`, `BertModel`, `BertClassifier` (with `save/from_pretrained`) |
    | `5_minilm/Bert/minilm.py` | MiniLM core — `relation_log_probs`, `masked_relation_kl` |
    | `5_minilm/Bert/model_student.py` | `build_student(num_layers)` — 4 L-384 D config |
    | `5_minilm/Bert/model_teacher.py` | `build_teacher()` — loads weights from HuggingFace Hub |
    | `5_minilm/Bert/train_distill.py` | `MiniLMTrainer` — relation-based distillation via HF `Trainer` |
    | `5_minilm/Bert/benchmark.py` | Inference throughput & latency measurement |
    | `5_minilm/Bert/utils.py` | `compute_metrics`, `make_training_args`, `count_parameters` |
* **Distillation Mechanics**:
  * **Loss Function**:

    ```
    L = α_ce · L_CE  +  α_attn · L_KL(A_teacher ∥ A_student)
                      +  α_val  · L_KL(VR_teacher ∥ VR_student)
    ```

  * **Transferred Knowledge** (from the teacher's **last transformer layer**):
    * **Self-Attention Distributions (A)** — Softmax of scaled query-key dot
      products: `A = Softmax(Q·Kᵀ / √d_k)`. KL divergence aligns the
      student's attention patterns to the teacher's.
    * **Value-Relation Matrices (VR)** — Softmax of scaled value-value dot
      products: `VR = Softmax(V·Vᵀ / √d_k)`. Captures semantic token
      relationships through value vectors.
    * **Key innovation**: Both A and VR have shape `[seq_len × seq_len]`,
      so the hidden dimension **cancels out mathematically**. This enables
      distillation from a 768-D teacher to a 384-D student **without any
      projection layers**.
* **How to Run**:
  ```bash
  cd 5_minilm
  python -m Bert.train_distill --layers 4 --epochs 5
  python -m Bert.benchmark --model_path ./outputs/student_distill_4L
  ```
* **Key Results**:

  | Metric | Teacher (12 L, 768 D) | MiniLM Student (4 L, 384 D) |
  | ------ | --------------------- | --------------------------- |
  | Parameters | 109.5 M | **19.2 M (−82.5 %)** |
  | SST-2 Accuracy | 92.4 % | 81.1 % |
  | Throughput (T4) | ~3,000 samples/s | **12,618 samples/s (~4× ↑)** |

* **Implementation Status**: ✅ Completed

---

## Project-Wide Setup

### Dependencies

```bash
# Core (all models)
pip install torch transformers datasets evaluate accelerate

# Model-specific extras
pip install scikit-learn matplotlib tabulate pandas   # 2_tinybert benchmarks
pip install safetensors huggingface_hub               # 5_minilm weight loading
pip install scikit-learn                              # 3_mobilebert F1 metrics
```

### Hardware Requirements

| Environment | Notes |
| ----------- | ----- |
| **Kaggle GPU** (recommended) | Enable **GPU** + **Internet** in notebook settings |
| **Local GPU** | NVIDIA T4 or better; CUDA 11.8+ |
| **CPU-only** | Supported but training is significantly slower |

### Reproducibility

All experiments use `SEED = 42` by default. Results reported in this README
were verified on Kaggle GPU runtimes. To reproduce:

1. Use the same seed (`--seed 42` where applicable)
2. Use the same batch size, learning rate, and epoch count as defined in each
   model's `config.py`
3. Run on a comparable GPU (results may vary slightly across hardware)

---

## Directory Mapping

> The numbered directory names correspond to the following original working
> directories. All internal code, imports, and module paths are **preserved
> unchanged** to maintain Kaggle reproducibility.

| Canonical Name | Original Directory | Python Module Prefix |
| -------------- | ------------------- | -------------------- |
| `1_distilbert/` | `Bert/` | `Bert.*` |
| `2_tinybert/` | `Knowledge-Distillation-Project/` | `src.*` |
| `3_mobilebert/` | `deep_learning_distillation_mobileBERT/` | (flat imports) |
| `4_pkd_bert/` | — (pending) | — |
| `5_minilm/` | `minilm-distillation/` | `Bert.*` |

---

## References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531
2. Sanh, V., et al. (2019). *DistilBERT, a distilled version of BERT.* arXiv:1910.01108
3. Jiao, X., et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding.* EMNLP 2020
4. Sun, S., et al. (2020). *MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices.* ACL 2020
5. Sun, S., et al. (2019). *Patient Knowledge Distillation for BERT Model Compression.* EMNLP 2019
6. Wang, W., et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS 2020
