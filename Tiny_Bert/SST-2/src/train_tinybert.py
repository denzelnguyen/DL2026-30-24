"""Fine-tune a small pretrained model (TinyBERT-4L-312D by default) on SST-2.

--mode ce : direct fine-tuning, no teacher        (the baseline from tinybert.ipynb)
--mode kd : same model, but also learns from the BERT-large teacher's logits

Examples:
  python train_tinybert.py --mode ce --name tinybert_ce_s42
  python train_tinybert.py --mode kd --name tinybert_kd_s42 --bf16
"""
import argparse
import os
import time

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import CHECKPOINT_DIR, RESULTS_DIR, SEED, TEACHER_ID
from src.data import build_dataloaders
from src.models import load_teacher
from src.trainer import train_model
from src.utils import count_parameters, get_device, save_json, set_seed

TINYBERT_ID = "huawei-noah/TinyBERT_General_4L_312D"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["kd", "ce"], default="ce")
    p.add_argument("--model-id", default=TINYBERT_ID)
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

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    train_loader, eval_loader = build_dataloaders(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=2)
    model.to(device)
    print(f"{args.model_id}: {count_parameters(model):,} parameters")

    teacher = None
    if args.mode == "kd":
        # Teacher logits are computed on the SAME token ids, so both vocabularies must match.
        teacher_tok = AutoTokenizer.from_pretrained(TEACHER_ID)
        if teacher_tok.get_vocab() != tokenizer.get_vocab():
            raise ValueError("Teacher and student tokenizers differ; logit KD would be invalid.")
        teacher = load_teacher(device)

    start = time.time()
    history = train_model(
        model, train_loader, eval_loader, device,
        teacher=teacher, epochs=args.epochs, lr=args.lr,
        temperature=args.temperature, alpha=args.alpha, bf16=args.bf16,
    )

    save_dir = os.path.join(CHECKPOINT_DIR, args.name)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    save_json(
        {"name": args.name, "args": vars(args), "history": history,
         "train_minutes": (time.time() - start) / 60},
        os.path.join(RESULTS_DIR, f"{args.name}_train_log.json"),
    )
    print(f"Saved model to {save_dir}")


if __name__ == "__main__":
    main()
