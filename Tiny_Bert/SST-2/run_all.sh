#!/usr/bin/env bash
# Run the full experiment grid, then benchmark every model on the SAME machine.
# TinyBERT is cheap (a few minutes) -> use several seeds. The 6L student is slow
# (~1h/run without bf16) -> default to 1 seed; override via env vars, e.g.
#   SEEDS_STUDENT="42 43" SEEDS_TINY="42 43 44" bash run_all.sh
set -euo pipefail

SEEDS_STUDENT=${SEEDS_STUDENT:-"42"}
SEEDS_TINY=${SEEDS_TINY:-"42 43 44"}
FLAGS=${FLAGS:-"--bf16"}

for s in $SEEDS_TINY; do
  python train_tinybert.py --mode ce --name tinybert_ce_s$s --seed $s $FLAGS
  python train_tinybert.py --mode kd --name tinybert_kd_s$s --seed $s $FLAGS
done

for s in $SEEDS_STUDENT; do
  python train_student.py --mode kd --num-layers 6 --name student_kd_6L_s$s --seed $s $FLAGS
  python train_student.py --mode ce --num-layers 6 --name student_ce_6L_s$s --seed $s $FLAGS
done

# Benchmarks: one process per model (meaningful VRAM). GPU and CPU latency.
python run_benchmark.py --name teacher --model yoshitomo-matsubara/bert-large-uncased-sst2
for dir in checkpoints/*/; do
  name=$(basename "$dir")
  python run_benchmark.py --name "$name" --model "$dir"
  python run_benchmark.py --name "$name" --model "$dir" --device cpu
done
