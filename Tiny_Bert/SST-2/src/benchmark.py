import time

import evaluate
import numpy as np
import torch

from .utils import count_parameters, model_size_mb


def _single_samples(dataloader, n):
    """Collect n real validation samples as batch-size-1 inputs, trimmed to true length."""
    samples = []
    for batch in dataloader:
        for i in range(batch["input_ids"].size(0)):
            length = int(batch["attention_mask"][i].sum())
            samples.append({k: v[i : i + 1, :length] for k, v in batch.items() if k != "labels"})
            if len(samples) >= n:
                return samples
    return samples


@torch.no_grad()
def benchmark_model(model, dataloader, device, num_warmup: int = 10, num_runs: int = 200):
    """Accuracy, F1, latency (batch=1, CPU/GPU), throughput and peak VRAM.

    IMPORTANT: run this with ONLY ONE model loaded in the process (use run_benchmark.py),
    otherwise peak VRAM includes other models/optimizer states and is meaningless.
    """
    model.eval()
    use_cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if use_cuda else (lambda: None)

    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # 1) Quality
    acc, f1 = evaluate.load("accuracy"), evaluate.load("f1")
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        preds = torch.argmax(logits, dim=-1)
        acc.add_batch(predictions=preds, references=batch["labels"])
        f1.add_batch(predictions=preds, references=batch["labels"])
    accuracy = acc.compute()["accuracy"] * 100
    f1_score = f1.compute()["f1"] * 100

    # 2) Throughput (full validation set, batched)
    total = 0
    sync()
    start = time.perf_counter()
    for batch in dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        model(**inputs)
        total += inputs["input_ids"].size(0)
    sync()
    throughput = total / (time.perf_counter() - start)

    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if use_cuda else 0.0

    # 3) Latency over many different real samples (batch = 1)
    samples = _single_samples(dataloader, num_runs)
    for s in samples[:num_warmup]:
        model(**{k: v.to(device) for k, v in s.items()})
    times = []
    for s in samples:
        s = {k: v.to(device) for k, v in s.items()}
        sync()
        t0 = time.perf_counter()
        model(**s)
        sync()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "params_million": count_parameters(model) / 1e6,
        "weights_mb": model_size_mb(model),
        "accuracy": accuracy,
        "f1": f1_score,
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_median": float(np.median(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
        "throughput_fps": throughput,
        "peak_vram_mb": peak_vram,
        "device": str(device),
    }
