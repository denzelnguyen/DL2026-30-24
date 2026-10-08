"""Data loading, grouped train/val/test split, tokenization and DataLoaders."""
import hashlib
from types import SimpleNamespace

import torch
from datasets import load_dataset
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from src import config

MODEL_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def _one_fold(df, n_splits):
    """Hold out one label-stratified fold, keeping all rows of a sentence together."""
    groups = df.text.factorize()[0]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.SPLIT_SEED)
    rest_idx, held_idx = next(iter(sgkf.split(df, df.y, groups)))
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[held_idx].reset_index(drop=True)


def load_splits():
    """Returns a namespace with train_df, val_df, test_df, labels, label2id, id2label, num_labels.

    The Hub 'test' split has NO labels (blank strings), so we split the 3693 labelled rows
    ourselves: ~71% train / 14% val / 14% test. Several rows share one sentence (one per aspect),
    so the split is grouped by sentence to avoid leakage.
    """
    raw = load_dataset(config.DATASET)
    full = raw["train"].to_pandas()
    label_col = "label" if "label" in full.columns else "polarity"
    test_labels = raw["test"].to_pandas()[label_col].astype(str).unique()[:5]
    print("columns:", list(full.columns), "| label column:", label_col)
    print("original Hub test labels:", test_labels, "(blank = unlabeled, not usable for scoring)")

    full = full[full[label_col].astype(str).str.strip() != ""]
    if config.DROP_CONFLICT:
        full = full[full[label_col] != "conflict"]
    print("label counts:\n", full[label_col].value_counts().to_string())
    print("sentences:", full.text.nunique(), "| rows:", len(full))

    labels = sorted(str(l) for l in full[label_col].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    full = full[["text", "span", label_col]].copy()
    full["text"] = full["text"].astype(str)
    full["span"] = full["span"].astype(str)
    full["label_name"] = full[label_col].astype(str)
    full["y"] = full["label_name"].map(label2id).astype(int)
    full = full[["text", "span", "label_name", "y"]].reset_index(drop=True)

    rest_df, test_df = _one_fold(full, 7)
    train_df, val_df = _one_fold(rest_df, 6)
    assert not (set(train_df.text) & set(test_df.text)) and not (set(train_df.text) & set(val_df.text))

    if config.SMOKE_TEST:
        train_df, val_df, test_df = train_df.iloc[:200], val_df.iloc[:60], test_df.iloc[:60]
        train_df, val_df, test_df = [d.reset_index(drop=True) for d in (train_df, val_df, test_df)]

    print(f"split sizes -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")
    for n, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(" ", n, d.label_name.value_counts().to_dict())

    return SimpleNamespace(train_df=train_df, val_df=val_df, test_df=test_df, labels=labels,
                           label2id=label2id, id2label=id2label, num_labels=len(labels))


def fingerprint(df):
    """Short hash of the rows (in order). Used to check teacher logits match the training set."""
    return hashlib.md5("|".join(df.text + "#" + df.span).encode()).hexdigest()


class AbsaDataset(Dataset):
    """Tokenizes (sentence, aspect) pairs as `[CLS] sentence [SEP] aspect [SEP]`.
    Optionally carries the teacher's logits for each example."""

    def __init__(self, df, tokenizer, teacher_logits=None):
        enc = tokenizer(list(df.text), list(df.span), truncation=True, max_length=config.MAX_LEN)
        self.keys = [k for k in MODEL_KEYS if k in enc]
        self.enc = enc
        self.y = df.y.tolist()
        self.teacher_logits = teacher_logits

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.keys}
        item["labels"] = self.y[i]
        if self.teacher_logits is not None:
            item["teacher_logits"] = self.teacher_logits[i]
        return item


def make_collate(tokenizer):
    def collate(batch):
        feats = [{k: b[k] for k in MODEL_KEYS if k in b} for b in batch]
        out = tokenizer.pad(feats, return_tensors="pt")      # pads to longest in the batch
        out["labels"] = torch.tensor([b["labels"] for b in batch])
        if "teacher_logits" in batch[0]:
            out["teacher_logits"] = torch.stack([b["teacher_logits"] for b in batch])
        return out
    return collate


def make_loader(ds, tokenizer, shuffle, bs, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=make_collate(tokenizer),
                      generator=g if shuffle else None)
