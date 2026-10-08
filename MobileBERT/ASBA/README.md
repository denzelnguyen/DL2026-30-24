# Knowledge Distillation for ABSA (BERT-base -> MobileBERT)

Teacher: `bert-base-uncased` | Student: `google/mobilebert-uncased`
Data: `tomaarsen/setfit-absa-semeval-restaurants` (SemEval-2014 Task 4, restaurants)

Task: given a review sentence and one aspect term, predict the sentiment towards that aspect.
Input is the pair `[CLS] sentence [SEP] aspect [SEP]`.

The Hub "test" split has no labels, so `src/data.py` builds its own train/val/test split
(~71/14/14) from the 3693 labelled rows, grouped by sentence so no sentence is in two splits.

## Layout

```
kd_absa/
  run_all.py            runs the 4 steps below in order
  requirements.txt
  src/
    config.py           all settings (epochs, lr, T, alpha, seeds, paths) + CLI flags
    data.py             load data, grouped split, tokenization, DataLoaders
    engine.py           train loop (CE or KD), predict, metrics
    train_teacher.py    step 1: fine-tune BERT-base, save teacher + soft labels
    train_student.py    step 2: MobileBERT baseline vs KD, several seeds
    benchmark.py        step 3: params, size, latency (GPU + CPU)
    report.py           step 4: tables, plots, summary.md
```

## Run on Kaggle

Settings: Accelerator GPU T4 x2, Internet On. Then, in notebook cells:

```
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>/kd_absa        # or just %cd <your-repo> if kd_absa is the repo root
!python run_all.py --smoke     # quick check, 2-3 minutes
!python run_all.py             # full run
```

Or run the steps one by one (same result):

```
!python -m src.train_teacher
!python -m src.train_student
!python -m src.benchmark
!python -m src.report
```

Flags (work on every step; use the same flags for all steps):
`--smoke` quick check | `--sweep` tune T/alpha on validation | `--drop-conflict` 3-class setup

Everything is written to `/kaggle/working` (or `./outputs` elsewhere): `teacher/`, `student_kd_best/`,
`runs.json`, `per_run_results.csv`, `benchmark.csv`, `results.png`, `summary.md`.

Notes
- MobileBERT is trained in fp32 on purpose (it gives NaN losses in fp16).
- The model is picked on the validation set; test is only used for the final numbers.
- MobileBERT targets CPU/phones. On a GPU at batch size 1 it can be slower than BERT-base.
