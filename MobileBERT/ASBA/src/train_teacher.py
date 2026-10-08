"""Step 1: fine-tune the BERT-base teacher and save its soft labels for the training set.

Run:  python -m src.train_teacher            (add --smoke for a quick check)
Saves: teacher/, teacher_train_logits.pt, teacher_results.json, teacher_report.txt
"""
import json
import os

import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.data import AbsaDataset, fingerprint, load_splits, make_loader
from src.engine import device, metrics, new_model, predict, set_seed, train_model


def main():
    sp = load_splits()
    seed = config.SEEDS[0]
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(config.TEACHER_NAME)
    train_ds = AbsaDataset(sp.train_df, tok)
    train_loader = make_loader(train_ds, tok, True, config.BATCH_SIZE, seed=seed)
    train_eval_loader = make_loader(train_ds, tok, False, 64)          # fixed order, for logits
    val_loader = make_loader(AbsaDataset(sp.val_df, tok), tok, False, 64)
    test_loader = make_loader(AbsaDataset(sp.test_df, tok), tok, False, 64)

    amp = config.TEACHER_AMP
    if os.path.isdir(config.TEACHER_DIR) and not config.FORCE_RETRAIN_TEACHER:
        print("loading saved teacher from", config.TEACHER_DIR)
        teacher = AutoModelForSequenceClassification.from_pretrained(config.TEACHER_DIR).to(device)
        info = {"best_epoch": None, "train_time_s": None}
    else:
        teacher = new_model(config.TEACHER_NAME, sp.label2id)
        info = train_model(teacher, train_loader, val_loader, config.T_EPOCHS, config.T_LR,
                           amp, kd=None, tag="teacher")
        teacher.save_pretrained(config.TEACHER_DIR)
        tok.save_pretrained(config.TEACHER_DIR)

    val_m = metrics(predict(teacher, val_loader, amp), sp.val_df.y)
    test_logits = predict(teacher, test_loader, amp)
    test_m = metrics(test_logits, sp.test_df.y)
    print("\nTEACHER  val:", {k: round(v, 4) for k, v in val_m.items()})
    print("TEACHER test:", {k: round(v, 4) for k, v in test_m.items()})
    report = classification_report(sp.test_df.y, test_logits.argmax(-1).numpy(),
                                   labels=list(range(sp.num_labels)), target_names=sp.labels,
                                   zero_division=0)
    print(report)
    with open(os.path.join(config.OUT_DIR, "teacher_report.txt"), "w") as f:
        f.write(report)

    # soft labels for the (frozen) teacher over the training set, in training-set order
    train_logits = predict(teacher, train_eval_loader, amp)
    assert train_logits.shape == (len(sp.train_df), sp.num_labels)
    train_acc = metrics(train_logits, sp.train_df.y)["acc"]
    print(f"teacher accuracy on its own training set: {train_acc:.4f} "
          f"(close to 1.0 = very confident soft labels = less 'dark knowledge')")
    torch.save({"logits": train_logits, "fingerprint": fingerprint(sp.train_df)},
               config.TEACHER_LOGITS_PATH)

    with open(config.TEACHER_RESULTS_PATH, "w") as f:
        json.dump({"val": val_m, "test": test_m, "train_acc": train_acc,
                   "params": sum(p.numel() for p in teacher.parameters()),
                   "best_epoch": info["best_epoch"], "seed": seed}, f, indent=1)
    print("saved teacher + soft labels to", config.OUT_DIR)


if __name__ == "__main__":
    main()
