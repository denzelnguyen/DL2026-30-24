"""Runs all four steps one after another, each in its own process (frees GPU memory in between).

    python run_all.py            full run
    python run_all.py --smoke    quick check (2-3 min)
Extra flags (--sweep, --drop-conflict) are passed on to every step.
"""
import subprocess
import sys

STEPS = ["train_teacher", "train_student", "benchmark", "report"]

if __name__ == "__main__":
    flags = sys.argv[1:]
    for step in STEPS:
        print(f"\n{'=' * 20} {step} {'=' * 20}", flush=True)
        subprocess.run([sys.executable, "-m", f"src.{step}"] + flags, check=True)
