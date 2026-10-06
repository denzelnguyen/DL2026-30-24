"""Train a shallow BERT student (initialised from the teacher's layers) on SST-2.

Examples:
  python train_student.py --mode kd --num-layers 6 --name student_kd_6L_s42 --bf16
  python train_student.py --mode ce --num-layers 6 --name student_ce_6L_s42 --bf16   # ablation
"""
import argparse
import os
import time

from transformers import AutoTokenizer

from src.config import CHECKPOINT_DIR, RESULTS_DIR, SEED, TEACHER_ID
from src.data import build_dataloaders
from src.models import build_student, load_teacher
from src.trainer import train_model
from src.utils import count_parameters, get_device, save_json, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Run name (checkpoint/result file names)")
    p.add_argument("--mode", choices=["kd", "ce"], default="kd")
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--layer-map", type=int, nargs="*", default=None,
                   help="Teacher layers to copy, e.g. --layer-map 3 7 11 15 19 23")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.5, help="Weight of the hard-label CE loss")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device} | args: {vars(args)}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID)
    train_loader, eval_loader = build_dataloaders(tokenizer)

    # The teacher is always needed to initialise the student; it is only used
    # for the loss in KD mode.
    teacher = load_teacher(device)
    student, layer_map = build_student(teacher, args.num_layers, args.layer_map, device)
    print(f"Layer map: {layer_map} | student params: {count_parameters(student):,}")

    start = time.time()
    history = train_model(
        student, train_loader, eval_loader, device,
        teacher=teacher if args.mode == "kd" else None,
        epochs=args.epochs, lr=args.lr, temperature=args.temperature,
        alpha=args.alpha, bf16=args.bf16,
    )

    save_dir = os.path.join(CHECKPOINT_DIR, args.name)
    student.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_json(
        {"name": args.name, "args": vars(args), "layer_map": layer_map,
         "history": history, "train_minutes": (time.time() - start) / 60},
        os.path.join(RESULTS_DIR, f"{args.name}_train_log.json"),
    )
    print(f"Saved model to {save_dir}")


if __name__ == "__main__":
    main()
