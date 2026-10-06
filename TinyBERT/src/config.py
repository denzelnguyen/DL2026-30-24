"""Shared constants. Keep every experiment on the same setup for fair comparison."""

TEACHER_ID = "yoshitomo-matsubara/bert-large-uncased-sst2"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"

MAX_LENGTH = 128
BATCH_SIZE = 32
SEED = 42

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
