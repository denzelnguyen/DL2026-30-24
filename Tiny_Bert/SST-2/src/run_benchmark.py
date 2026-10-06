"""Benchmark ONE model per process so VRAM numbers are meaningful.

Examples:
  python run_benchmark.py --name teacher --model yoshitomo-matsubara/bert-large-uncased-sst2
  python run_benchmark.py --name student_kd_6L --model checkpoints/student_kd_6L
  python run_benchmark.py --name tinybert --model checkpoints/tinybert   # same JSON format
  python run_benchmark.py --name student_kd_6L --model checkpoints/student_kd_6L --device cpu
"""
import argparse
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.benchmark import benchmark_model
from src.config import RESULTS_DIR, SEED
from src.data import build_dataloaders
from src.utils import get_device, save_json, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True, help="Local checkpoint dir or Hugging Face model id")
    p.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = p.parse_args()

    set_seed(SEED)
    device = torch.device(args.device) if args.device else get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    _, eval_loader = build_dataloaders(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    stats = benchmark_model(model, eval_loader, device)
    stats["name"] = args.name
    suffix = "" if device.type == "cuda" else "_cpu"
    path = os.path.join(RESULTS_DIR, f"{args.name}{suffix}.json")
    save_json(stats, path)
    print(stats)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
