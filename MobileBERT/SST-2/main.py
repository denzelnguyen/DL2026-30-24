import os
import warnings
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

import config
from model import MobileBertForSequenceClassificationBCE
from dataset import get_dataloaders
from engine import train_one_epoch, evaluate

warnings.filterwarnings("ignore")

def main():
    print(f"[INFO] Using device: {config.DEVICE}")
    train_loader, eval_loader, tokenizer, num_train, num_eval = get_dataloaders()

    print(f"[INFO] Loading Teacher ({config.TEACHER_ID})...")
    teacher_model = AutoModelForSequenceClassification.from_pretrained(config.TEACHER_ID).to(config.DEVICE)
    teacher_model.eval()

    print(f"[INFO] Initializing Student ({config.STUDENT_ID}) FROM SCRATCH...")
    student_model = MobileBertForSequenceClassificationBCE(AutoConfig.from_pretrained(config.STUDENT_ID)).to(config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=len(train_loader) * config.EPOCHS)

    print("\n" + "="*85)
    print(f"{'Epoch':<8}{'Train Loss':<22}{'Val Loss':<15}{'Accuracy':<14}{'F1':<10}")
    print("="*85)

    best_f1 = 0.0
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(student_model, teacher_model, train_loader, optimizer, scheduler, criterion)
        eval_loss, acc, f1 = evaluate(student_model, eval_loader, criterion)

        print(f"{epoch:<8}{train_loss/num_train:<22.6f}{eval_loss/num_eval:<15.6f}{acc:<14.6f}{f1:<10.6f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(student_model.state_dict(), f"{config.OUTPUT_DIR}/best_model.pt")

    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print("="*85)
    print(f"[INFO] Done! Best model saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()