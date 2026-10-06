import torch

# --- Models ---
TEACHER_ID = "distilbert-base-uncased-finetuned-sst-2-english"
STUDENT_ID = "google/mobilebert-uncased"

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
ALPHA = 0.5  # Trọng số Knowledge Distillation

# --- Paths & Env ---
OUTPUT_DIR = "./student_sst2_distilled"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")