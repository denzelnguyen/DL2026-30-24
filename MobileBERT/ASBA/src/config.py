"""Central config for the KD-for-ABSA project.

Command-line flags understood by every step (they are just looked up in sys.argv):
    --smoke          tiny fast run to check the pipeline works (also: env KD_SMOKE=1)
    --sweep          grid-search TEMP/ALPHA on the validation set before the main runs
    --drop-conflict  classic 3-class setup (removes the 'conflict' rows)
"""
import os
import sys

import torch

SMOKE_TEST = "--smoke" in sys.argv or os.environ.get("KD_SMOKE") == "1"
RUN_SWEEP = "--sweep" in sys.argv
DROP_CONFLICT = "--drop-conflict" in sys.argv

DATASET = "tomaarsen/setfit-absa-semeval-restaurants"
TEACHER_NAME = "bert-base-uncased"
STUDENT_NAME = "google/mobilebert-uncased"

OUT_DIR = os.environ.get("KD_OUT") or ("/kaggle/working" if os.path.exists("/kaggle") else "./outputs")

MAX_LEN = 128
SPLIT_SEED = 42
BATCH_SIZE = 16

# teacher
T_EPOCHS, T_LR = 5, 3e-5
TEACHER_AMP = True            # fp16 autocast, fine for BERT-base
FORCE_RETRAIN_TEACHER = False  # if a saved teacher exists in OUT_DIR it is reused

# student (MobileBERT gives NaN losses in fp16, so it trains in fp32)
S_EPOCHS, S_LR = 8, 5e-5
STUDENT_AMP = False

# distillation: loss = ALPHA * CE(hard labels) + (1 - ALPHA) * T^2 * KL(teacher || student)
TEMP, ALPHA = 2.0, 0.5

SEEDS = [42, 43, 44]          # every seed is run for the baseline AND for KD

SWEEP_TEMPS = [1.0, 2.0, 4.0]
SWEEP_ALPHAS = [0.3, 0.5, 0.7]

if SMOKE_TEST:
    T_EPOCHS, S_EPOCHS, SEEDS = 1, 1, [42]
    RUN_SWEEP = False
    FORCE_RETRAIN_TEACHER = True
    OUT_DIR = os.path.join(OUT_DIR, "smoke")

os.makedirs(OUT_DIR, exist_ok=True)

TEACHER_DIR = os.path.join(OUT_DIR, "teacher")
STUDENT_DIR = os.path.join(OUT_DIR, "student_kd_best")
TEACHER_LOGITS_PATH = os.path.join(OUT_DIR, "teacher_train_logits.pt")
TEACHER_RESULTS_PATH = os.path.join(OUT_DIR, "teacher_results.json")
RUNS_PATH = os.path.join(OUT_DIR, "runs.json")
BENCH_PATH = os.path.join(OUT_DIR, "benchmark.csv")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
