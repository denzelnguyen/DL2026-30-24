from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
import config

def get_dataloaders():
    print("[INFO] Loading and Tokenizing SST-2 Dataset...")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_ID)

    def preprocess(examples):
        tokenized = tokenizer(examples["sentence"], truncation=True, max_length=128)
        tokenized["labels"] = [float(l) for l in examples["label"]]
        return tokenized

    train_data = dataset["train"].shuffle(seed=42).map(preprocess, batched=True, remove_columns=["sentence", "label", "idx"])
    eval_data = dataset["validation"].map(preprocess, batched=True, remove_columns=["sentence", "label", "idx"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_data, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator)

    return train_loader, eval_loader, tokenizer, len(train_data), len(eval_data)