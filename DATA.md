# 📊 Dataset Specifications & Data Preparation Guide (`DATA.md`)

This document provides complete documentation for all datasets utilized across the 5 knowledge distillation experiments (**DistilBERT, TinyBERT, MobileBERT, PKD-BERT, and MiniLM**). It includes official download links, dataset versions, exact split sizes, preprocessing pipelines, and reproducibility scripts.

---

## 📌 Summary of Benchmark Datasets

| Dataset | Task Type | Primary Domain | Official URL | Downloadable HF ID |
|---|---|---|---|---|
| **SST-2 (GLUE Benchmark)** | Binary Sentiment Classification | Movie Reviews | [GLUE Benchmark](https://gluebenchmark.com/) | `nyu-mll/glue` (`sst2`) / `stanfordnlp/sst2` |
| **SemEval-2014 Task 4 (ABSA)** | Aspect-Based Sentiment Analysis | Restaurant / Laptop Reviews | [SemEval-2014 Task 4](https://alt.qcri.org/semeval2014/task4/) | `tomaarsen/setfit-absa-semeval-restaurants` |

---

## 1. Stanford Sentiment Treebank (SST-2)

### 1.1 Overview & Metadata
* **Official Dataset URL**: [https://huggingface.co/datasets/nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) (GLUE Benchmark) / [https://nlp.stanford.edu/sentiment/](https://nlp.stanford.edu/sentiment/)
* **Hugging Face Library ID**: `nyu-mll/glue` (Config: `sst2`) or `stanfordnlp/sst2`
* **Version**: GLUE Benchmark Release (v1.0.0)
* **License**: CC BY 4.0 / Stanford NLP License

### 1.2 Data Split Statistics
The dataset contains single-sentence movie reviews extracted from Rotten Tomatoes, annotated for binary sentiment (`0` = Negative, `1` = Positive):

| Data Split | Sample Count | Column Schema | Purpose in Experiments |
|---|---|---|---|
| **Train** | 67,349 | `sentence` (string), `label` (int64), `idx` (int32) | Model training / Knowledge Distillation |
| **Validation (Dev)** | 872 | `sentence` (string), `label` (int64), `idx` (int32) | Evaluation & hyperparameter tuning |
| **Test** | 1,821 | `sentence` (string), `label` (int64), `idx` (int32) | Benchmark evaluation (unlabeled in GLUE) |

### 1.3 Preprocessing Procedure
All models process SST-2 inputs using the following standardized pipeline:
1. **Tokenization**: Standard WordPiece tokenization via `bert-base-uncased` (Vocabulary Size: 30,522).
2. **Special Tokens**: Prepend `[CLS]` token at sentence start, append `[SEP]` token at sentence end.
3. **Sequence Length**: Fixed truncation / padding to `max_length = 128` tokens.
4. **Label Mapping**: Target column `label` renamed to `labels` tensor (`torch.long`).
5. **Batching**: Dynamic collate via `DataCollatorWithPadding` or static tensor formatting (`set_format("torch")`).

---

## 2. SemEval-2014 Task 4 (Aspect-Based Sentiment Analysis - ABSA)

### 2.1 Overview & Metadata
* **Official Dataset URL**: [https://alt.qcri.org/semeval2014/task4/](https://alt.qcri.org/semeval2014/task4/)
* **Hugging Face Library ID**: `tomaarsen/setfit-absa-semeval-restaurants`
* **Version**: SemEval-2014 Task 4 (Restaurant Domain SetFit release)
* **License**: Open Research Dataset

### 2.2 Data Split Statistics
The ABSA task evaluates fine-grained sentiment expressed toward a specific aspect term (`span`) within a context sentence (`text`):

| Data Split | Sample Count | Column Schema | Target Sentiment Labels |
|---|---|---|---|
| **Train** | 2,252 | `text` (string), `span` (string), `label` (string/int) | Positive, Negative, Neutral |
| **Test** | 800 | `text` (string), `span` (string), `label` (string/int) | Benchmark evaluation |

### 2.3 Preprocessing Procedure
ABSA models require sentence-pair input formatting to associate the sentence with its target aspect:
1. **Sentence-Pair Tokenization**:
   - `text_a`: Context sentence (`examples["text"]`)
   - `text_b`: Aspect term (`examples["span"]`)
   - Formatted sequence: `[CLS] text_a [SEP] text_b [SEP]`
2. **Sequence Truncation & Padding**: Truncate pairs to `max_length = 128`, pad with `[PAD]` tokens.
3. **Label Encoding**:
   - **Classification Setup**: Categorical label index (`0`: Negative, `1`: Neutral, `2`: Positive).
   - **Regression Setup** (MiniLM ABSA): Continuous float mapping (`1.0` = Positive, `0.0` = Negative, `0.5` = Neutral) evaluated via `MSELoss`.

---

## 3. Data Reproduction Scripts

Each student model directory includes standalone data loading and preprocessing scripts to reproduce the exact datasets used in all experiments.

### 3.1 Script Inventory & File Paths

| Model Directory | Dataset Script Path | Loading Function | Target Dataset |
|---|---|---|---|
| `1_distilbert/` | [`1_distilbert/Bert/dataset.py`](file:///mnt/data/AI/FINAL_DEEPLEARNING/1_distilbert/Bert/dataset.py) | `load_sst2(tokenizer, max_length)` | SST-2 |
| `2_tinybert/` | [`2_tinybert/src/data.py`](file:///mnt/data/AI/FINAL_DEEPLEARNING/2_tinybert/src/data.py) | `build_dataloaders(tokenizer, batch_size, max_length)` | SST-2 |
| `3_mobilebert/` | [`3_mobilebert/dataset.py`](file:///mnt/data/AI/FINAL_DEEPLEARNING/3_mobilebert/dataset.py) | `get_dataloaders(tokenizer_name, batch_size)` | SST-2 |
| `4_pkd_bert/` | [`4_pkd_bert/SST-2/dataset.py`](file:///mnt/data/AI/FINAL_DEEPLEARNING/4_pkd_bert/SST-2/dataset.py) | `load_sst2_dataset()` | SST-2 |
| `4_pkd_bert/` | [`4_pkd_bert/ABSA/dataset.py`](file:///mnt/data/AI/FINAL_DEEPLEARNING/4_pkd_bert/ABSA/dataset.py) | `prepare_absa_dataset()` | ABSA |
| `5_minilm/` | [`5_minilm/SST-2/dataset.py`](file:///mnt/data/AI/FINAL_DEEPLEARNING/5_minilm/SST-2/dataset.py) | `load_sst2(tokenizer, max_length)` | SST-2 |
| `5_minilm/` | [`5_minilm/Aspect/minilm-absa/dataset.py`](file:///mnt/data/AI/FINAL_DEEPLEARNING/5_minilm/Aspect/minilm-absa/dataset.py) | `prepare_absa_dataset(config)` | ABSA |

### 3.2 Executing Data Reproduction

To test or reproduce dataset downloading and tokenization independently, execute any of the following commands:

```bash
# 1. Reproduce DistilBERT SST-2 Preprocessing
python -m 1_distilbert.Bert.dataset

# 2. Reproduce TinyBERT SST-2 Data Loaders
python -c "from 2_tinybert.src.data import build_dataloaders; from transformers import AutoTokenizer; build_dataloaders(AutoTokenizer.from_pretrained('bert-base-uncased'))"

# 3. Reproduce MobileBERT Data Preparation
python -m 3_mobilebert.dataset

# 4. Reproduce MiniLM SST-2 Data Tokenization
python -m 5_minilm.SST-2.dataset

# 5. Reproduce MiniLM ABSA Data Tokenization
python 5_minilm/Aspect/minilm-absa/dataset.py
```

---

## 4. Download Links for Data & Pretrained Weights

| Resource | Source / Provider | Download Method / Link |
|---|---|---|
| **GLUE SST-2 Dataset** | NYU MLL / Hugging Face | [HuggingFace Datasets: nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue) |
| **Stanford SST-2 Original** | Stanford NLP | [Stanford Sentiment Treebank Page](https://nlp.stanford.edu/sentiment/) |
| **SemEval 2014 Task 4 ABSA** | Qatar Computing Research Institute | [SemEval-2014 Task 4 Official Webpage](https://alt.qcri.org/semeval2014/task4/) |
| **SetFit ABSA Restaurant Dataset** | Hugging Face Hub | [HuggingFace Datasets: tomaarsen/setfit-absa-semeval-restaurants](https://huggingface.co/datasets/tomaarsen/setfit-absa-semeval-restaurants) |
| **Teacher Model Checkpoint (BERT-large SST-2)** | Hugging Face Hub | [yoshitomo-matsubara/bert-large-uncased-sst2](https://huggingface.co/yoshitomo-matsubara/bert-large-uncased-sst2) |
| **Teacher Model Checkpoint (BERT-base SST-2)** | Hugging Face Hub | [textattack/bert-base-uncased-SST-2](https://huggingface.co/textattack/bert-base-uncased-SST-2) |
