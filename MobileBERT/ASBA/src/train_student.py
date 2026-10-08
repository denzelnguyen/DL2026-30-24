"""Step 2: train MobileBERT with and without distillation (several seeds).

Run:  python -m src.train_student            (add --smoke, --sweep, --drop-conflict as needed)
Needs the output of train_teacher. Saves: runs.json, student_kd_best/
"""
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

from src import config
from src.data import AbsaDataset, fingerprint, load_splits, make_loader
from src.engine import metrics, new_model, predict, set_seed, train_model


def main():
    sp = load_splits()
    payload = torch.load(config.TEACHER_LOGITS_PATH)
    assert payload["fingerprint"] == fingerprint(sp.train_df), (
        "teacher logits do not match this training set - rerun train_teacher with the same flags "
        "(--smoke / --drop-conflict)")
    teacher_logits = payload["logits"]

    s_tok = AutoTokenizer.from_pretrained(config.STUDENT_NAME)
    t_tok = AutoTokenizer.from_pretrained(config.TEACHER_NAME)
    train_ds = AbsaDataset(sp.train_df, s_tok, teacher_logits=teacher_logits)
    val_loader = make_loader(AbsaDataset(sp.val_df, s_tok), s_tok, False, 64)
    test_loader = make_loader(AbsaDataset(sp.test_df, s_tok), s_tok, False, 64)

    same = np.mean([t_tok(a, b, truncation=True)["input_ids"] == s_tok(a, b, truncation=True)["input_ids"]
                    for a, b in zip(sp.train_df.text[:200], sp.train_df.span[:200])])
    print(f"teacher/student tokenizers agree on {same:.0%} of examples (informative only)")

    def run_student(seed, kd, tag, eval_test=True):
        set_seed(seed)
        model = new_model(config.STUDENT_NAME, sp.label2id)
        loader = make_loader(train_ds, s_tok, True, config.BATCH_SIZE, seed=seed)
        info = train_model(model, loader, val_loader, config.S_EPOCHS, config.S_LR,
                           config.STUDENT_AMP, kd=kd, tag=tag)
        rec = {"tag": tag, "seed": seed, "kd": kd, **info}
        if eval_test:
            test_logits = predict(model, test_loader, config.STUDENT_AMP)
            rec.update({f"test_{k}": v for k, v in metrics(test_logits, sp.test_df.y).items()})
            rec["test_pred"] = test_logits.argmax(-1).numpy().tolist()
        return rec, model

    temp, alpha = config.TEMP, config.ALPHA

    # optional: tune T / alpha on the VALIDATION set (test is never used for this)
    if config.RUN_SWEEP:
        rows = []
        for T in config.SWEEP_TEMPS:
            for a in config.SWEEP_ALPHAS:
                rec, m = run_student(config.SEEDS[0], (T, a), f"sweep T={T} a={a}", eval_test=False)
                rows.append({"T": T, "alpha": a, "val_score": rec["val_score"],
                             "best_epoch": rec["best_epoch"]})
                del m
                torch.cuda.empty_cache()
        sweep_df = pd.DataFrame(rows).sort_values("val_score", ascending=False)
        print(sweep_df.to_string(index=False))
        sweep_df.to_csv(f"{config.OUT_DIR}/sweep_val.csv", index=False)
        temp, alpha = float(sweep_df.iloc[0]["T"]), float(sweep_df.iloc[0]["alpha"])
    print(f"using TEMP={temp}, ALPHA={alpha}")

    runs, best_kd_val, best_kd_seed = [], -1.0, None
    for seed in config.SEEDS:
        for variant in ["baseline", "kd"]:
            kd_cfg = (temp, alpha) if variant == "kd" else None
            rec, model = run_student(seed, kd_cfg, f"{variant} s{seed}")
            rec["variant"] = variant
            runs.append(rec)
            print(f"==> {variant} seed {seed}: test acc {rec['test_acc']:.4f} | "
                  f"test macro-F1 {rec['test_macro_f1']:.4f} | best epoch {rec['best_epoch']}\n")
            # keep the KD model with the best VALIDATION score (not test)
            if variant == "kd" and rec["val_score"] > best_kd_val:
                best_kd_val, best_kd_seed = rec["val_score"], seed
                model.save_pretrained(config.STUDENT_DIR)
                s_tok.save_pretrained(config.STUDENT_DIR)
            del model
            torch.cuda.empty_cache()
        # save after every seed so a crash does not lose finished runs
        with open(config.RUNS_PATH, "w") as f:
            json.dump({"temp": temp, "alpha": alpha, "best_kd_seed": best_kd_seed, "runs": runs},
                      f, indent=1, default=str)
    print("saved", config.RUNS_PATH)


if __name__ == "__main__":
    main()
