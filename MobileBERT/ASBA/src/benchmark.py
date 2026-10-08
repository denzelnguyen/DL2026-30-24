"""Step 3: parameters, model size and batch-size-1 latency (GPU and CPU) of teacher vs student.

Run:  python -m src.benchmark
MobileBERT is built for CPU/phones; on a big GPU it can be slower than BERT-base, so look at CPU.
"""
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config
from src.data import load_splits
from src.engine import device


@torch.no_grad()
def latency_ms(model, tok, dev, test_df, n=100, warmup=10):
    model.to(dev).eval()
    ts = []
    for i in range(n + warmup):
        r = test_df.iloc[i % len(test_df)]
        enc = tok(r.text, r.span, truncation=True, max_length=config.MAX_LEN, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(**enc)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def main():
    sp = load_splits()
    cpu = torch.device("cpu")
    rows = []
    for name, path in [("BERT-base teacher", config.TEACHER_DIR), ("MobileBERT student", config.STUDENT_DIR)]:
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tok = AutoTokenizer.from_pretrained(path)
        params = sum(p.numel() for p in model.parameters())
        row = {"model": name, "params_M": round(params / 1e6, 2), "size_MB_fp32": round(params * 4 / 1e6, 1)}
        if device.type == "cuda":
            row["gpu_latency_ms"] = round(latency_ms(model, tok, device, sp.test_df), 2)
        row["cpu_latency_ms"] = round(latency_ms(model, tok, cpu, sp.test_df), 2)
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    p_t, p_s = rows[0]["params_M"], rows[1]["params_M"]
    print(f"\nparameter reduction: {100 * (1 - p_s / p_t):.1f}% ({p_t / p_s:.1f}x smaller)")
    print(f"CPU speed-up: {rows[0]['cpu_latency_ms'] / rows[1]['cpu_latency_ms']:.2f}x")
    df.to_csv(config.BENCH_PATH, index=False)


if __name__ == "__main__":
    main()
