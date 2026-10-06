# Knowledge Distillation: How Small Can a Neural Network Become?

Distill a BERT-large teacher (SST-2) into smaller students and study the trade-off
between model size, computational efficiency and prediction performance.

## Setup
```bash
pip install -r requirements.txt
```

## Run
Every run name ends with its seed (`_s42`) so results over seeds can be aggregated.
```bash
# Student (6 layers copied from the teacher): with / without distillation
python train_student.py --mode kd --num-layers 6 --name student_kd_6L_s42 --bf16
python train_student.py --mode ce --num-layers 6 --name student_ce_6L_s42 --bf16

# TinyBERT-4L-312D baseline: direct fine-tuning vs. + distillation from the teacher
python train_tinybert.py --mode ce --name tinybert_ce_s42 --bf16
python train_tinybert.py --mode kd --name tinybert_kd_s42 --bf16

# Benchmark (one process per model; run all models on the SAME machine)
python run_benchmark.py --name teacher --model yoshitomo-matsubara/bert-large-uncased-sst2
python run_benchmark.py --name tinybert_ce_s42 --model checkpoints/tinybert_ce_s42
python run_benchmark.py --name tinybert_ce_s42 --model checkpoints/tinybert_ce_s42 --device cpu

# Or the whole grid (multiple seeds) in one go
SEEDS_STUDENT="42" SEEDS_TINY="42 43 44" bash run_all.sh
```
Results are written to `results/<name>.json` (`*_cpu.json` for CPU) with identical keys
for every model; training curves go to `results/<name>_train_log.json`.

## Layout
- `src/` shared code (data, models, loss, benchmark, utils)
- `train_student.py`, `train_tinybert.py`, `run_benchmark.py`, `run_all.sh` entry points
- `src/trainer.py` the single training loop used by all models
- `results/` JSON results (committed) | `checkpoints/` weights (git-ignored)

## Model weights
Weights are not stored in git. Download link: _TODO (Hugging Face Hub / Google Drive)_.

## Team
_TODO: member — task_
