from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .config import BATCH_SIZE, DATASET_CONFIG, DATASET_NAME, MAX_LENGTH


def build_dataloaders(tokenizer, batch_size: int = BATCH_SIZE, max_length: int = MAX_LENGTH):
    """Load SST-2 and return (train_loader, eval_loader)."""
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG)

    def tokenize(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=max_length)

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(["sentence", "idx"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        tokenized["train"], shuffle=True, batch_size=batch_size, collate_fn=collator
    )
    eval_loader = DataLoader(
        tokenized["validation"], batch_size=batch_size, collate_fn=collator
    )
    return train_loader, eval_loader
