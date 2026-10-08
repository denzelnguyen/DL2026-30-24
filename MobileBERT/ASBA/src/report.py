"""Step 4: aggregate results over seeds, print tables, save plots and summary.md.

Run:  python -m src.report
"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402
from src.data import load_splits  # noqa: E402


def main():
    sp = load_splits()
    with open(config.TEACHER_RESULTS_PATH) as f:
        teacher = json.load(f)
    with open(config.RUNS_PATH) as f:
        blob = json.load(f)
    runs, best_kd_seed = blob["runs"], blob["best_kd_seed"]
    bench_df = pd.read_csv(config.BENCH_PATH) if os.path.exists(config.BENCH_PATH) else None

    rows = [{"model": "BERT-base teacher", "variant": "teacher", "seed": teacher["seed"],
             "test_acc": teacher["test"]["acc"], "test_macro_f1": teacher["test"]["macro_f1"]}]
    rows += [{"model": "MobileBERT " + r["variant"], "variant": r["variant"], "seed": r["seed"],
              "test_acc": r["test_acc"], "test_macro_f1": r["test_macro_f1"]} for r in runs]
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(config.OUT_DIR, "per_run_results.csv"), index=False)

    agg = (per_run.groupby("model")[["test_acc", "test_macro_f1"]].agg(["mean", "std"]) * 100).round(2)
    print(agg.to_string())

    base = per_run[per_run.variant == "baseline"]
    kd = per_run[per_run.variant == "kd"]
    d_acc = (kd.test_acc.mean() - base.test_acc.mean()) * 100
    d_f1 = (kd.test_macro_f1.mean() - base.test_macro_f1.mean()) * 100
    print(f"\nKD minus baseline: accuracy {d_acc:+.2f} pts | macro-F1 {d_f1:+.2f} pts "
          f"(mean over {len(config.SEEDS)} seeds; compare with the std above - a small gap may be noise)")

    best_kd = [r for r in runs if r["variant"] == "kd" and r["seed"] == best_kd_seed][0]
    print(f"\nPer-class test report, KD student (seed {best_kd_seed}):")
    print(classification_report(sp.test_df.y, best_kd["test_pred"], labels=list(range(sp.num_labels)),
                                target_names=sp.labels, zero_division=0))

    # plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for variant, color in [("baseline", "tab:orange"), ("kd", "tab:blue")]:
        curves = np.array([[h["val_score"] for h in r["history"]] for r in runs if r["variant"] == variant])
        ep = np.arange(1, curves.shape[1] + 1)
        ax[0].plot(ep, curves.mean(0), marker="o", label=variant, color=color)
        ax[0].fill_between(ep, curves.mean(0) - curves.std(0), curves.mean(0) + curves.std(0),
                           alpha=0.15, color=color)
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("validation score (mean of acc and macro-F1)")
    ax[0].set_title("Student validation curves")
    ax[0].legend()

    order = ["BERT-base teacher", "MobileBERT baseline", "MobileBERT kd"]
    w = 0.35
    for j, (col, lab) in enumerate([("test_acc", "accuracy"), ("test_macro_f1", "macro-F1")]):
        means = [per_run[per_run.model == m][col].mean() * 100 for m in order]
        stds = [np.nan_to_num(per_run[per_run.model == m][col].std()) * 100 for m in order]
        ax[1].bar(np.arange(3) + j * w, means, w, yerr=stds, capsize=3, label=lab)
    ax[1].set_xticks(np.arange(3) + w / 2)
    ax[1].set_xticklabels(order, rotation=10)
    ax[1].set_ylim(max(0, min(per_run.test_macro_f1.min(), per_run.test_acc.min()) * 100 - 10), 100)
    ax[1].set_title("Test set")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUT_DIR, "results.png"), dpi=150)

    with open(os.path.join(config.OUT_DIR, "summary.md"), "w") as f:
        f.write(f"# KD for ABSA - summary\n\nteacher={config.TEACHER_NAME}, student={config.STUDENT_NAME}, "
                f"data={config.DATASET}\nT={blob['temp']}, alpha={blob['alpha']}, seeds={config.SEEDS}, "
                f"student lr={config.S_LR}, epochs={config.S_EPOCHS}\n\n"
                f"## Test results (mean/std %)\n\n```\n{agg.to_string()}\n```\n\n")
        if bench_df is not None:
            f.write(f"## Efficiency\n\n```\n{bench_df.to_string(index=False)}\n```\n")
    print("saved to", config.OUT_DIR, "->", sorted(os.listdir(config.OUT_DIR)))


if __name__ == "__main__":
    main()
