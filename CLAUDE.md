# CLAUDE.md — slm-from-scratch

This file is read by Claude Code at the start of every session. It contains everything needed to understand the project, navigate the codebase, and assist effectively.

---

## Project Overview

An end-to-end Small Language Model (SLM) training project built in PyTorch. Covers the full LLM pipeline from scratch: custom BPE tokenizer, GPT-style transformer, pretraining, SFT, DPO, PEFT, quantization, inference optimization, and evaluation.

**Two pipelines exist in this repo:**

1. **TinyStories SLM** (notebooks 01–08) — complete, used as learning scaffold. Trains a ~25M param GPT on the TinyStories dataset. All supporting infrastructure (tokenizer, model, trainer, DPO, evaluation) was written here and is reused by the medical pipeline.

2. **Medical Q&A SLM** (notebooks 09–15) — the primary deliverable. A domain-specific model trained on medical text and fine-tuned on USMLE-style Q&A. Goal: uploadable to HuggingFace, benchmarkable on MCQ accuracy.

**Target environment:** Google Colab with a single T4 GPU (16GB VRAM). Also runs locally (MPS available on Mac). Local Python environment is a `.venv` at the repo root.

---

## Repo Structure

```
slm-from-scratch/
├── config.py                  # All paths, hyperparams, env detection — source of truth
├── requirements.txt
├── .venv/                     # Local virtual environment (gitignored)
├── tokenizer/                 # Custom BPE tokenizer, trained from scratch
│   └── preprocess.py          # train_tokenizer(), tokenize_and_save(), load_tokenizer()
├── data/                      # Raw + tokenized data (gitignored, lives on Drive)
├── loaders/                   # PyTorch Dataset classes for pretrain/SFT/DPO
│   └── medical.py             # MedQADataset, PubMedQADataset, DPO pair builder
│                              # NOTE: named 'loaders/' (not 'datasets/') to avoid
│                              # shadowing the HuggingFace `datasets` library
├── model/                     # GPT architecture: attention, MLP, blocks, KV-cache
├── training/                  # Training loops, optimizer, scheduler, utils
├── generation/                # Inference, sampling strategies, KV-cache inference
├── evaluation/                # Perplexity, repetition, prefix completion, benchmarks
│   └── medical_metrics.py     # MCQ accuracy, USMLE benchmark, medical perplexity
├── dpo/                       # DPO loss, trainer, preference dataset generation
├── checkpoints/               # Gitignored, saved to Drive
└── experiments/
    ├── logs/                  # Gitignored, saved to Drive
    └── notebooks/             # Colab notebooks versioned in git
```

---

## Notebooks

### TinyStories pipeline (complete)
| Notebook | Topic |
|----------|-------|
| 01 | BPE tokenizer training |
| 02 | Data preprocessing |
| 03 | GPT pretraining |
| 04 | Evaluation (perplexity, sampling) |
| 05 | SFT (instruction following) |
| 06 | DPO (preference optimization) |
| 07 | PEFT / LoRA |
| 08 | Quantization, KV-cache, benchmarking |

### Medical Q&A pipeline (in progress)
| Notebook | Topic |
|----------|-------|
| 09 | Medical data exploration (guidelines, MedQA, PubMedQA) |
| 10 | Medical tokenizer training + data preprocessing |
| 11 | Medical pretraining (random init, 20k steps) |
| 12 | Medical SFT (MedQA + PubMedQA labeled) |
| 13 | Medical evaluation (MCQ accuracy, USMLE benchmark) |
| 14 | Medical DPO (labeled wrong options as rejected) |
| 15 | HuggingFace upload + model card |

---

## Medical SLM — Datasets

| Role | Dataset | HF ID | Key field(s) |
|------|---------|--------|--------------|
| Pretraining corpus | Medical guidelines | `epfl-llm/guidelines` | `clean_text` |
| Pretraining corpus | PubMed unlabeled Q&A | `pubmed_qa` (pqa_unlabeled) | `context["contexts"]` |
| SFT Q&A | USMLE-style questions | `medalpaca/medical_meadow_medqa` | `input`, `output` |
| SFT Q&A | PubMed labeled Q&A | `pubmed_qa` (pqa_labeled) | `question`, `context`, `long_answer`, `final_decision` |
| Eval benchmark | Held-out MedQA | same as above | `input`, `output` |

**MedQA format details:**
- `input`: `"Q: <question>? {'A': 'opt1', 'B': 'opt2', 'C': 'opt3', 'D': 'opt4', 'E': 'opt5'}"` — options are embedded as a Python dict literal
- `output`: `"C: Cortical laminar necrosis"` — correct answer as `"LETTER: text"`
- 5 options (A–E), random baseline = 20%
- Parse with `loaders.medical._parse_medqa_input()` using `ast.literal_eval`

**PubMedQA format details:**
- `context` is a nested dict: `context["contexts"]` is a list of strings
- `final_decision`: "yes" / "no" / "maybe"
- `long_answer`: paragraph-length answer

**Pretraining data volume:**
- ~33k guidelines docs + ~60k pubmedqa_unlabeled docs ≈ 410–480M tokens
- Chinchilla optimal for 25M param model: ~500M tokens — this is within range
- pqa_labeled (1k docs) also included in tokenizer training for vocabulary coverage

**Val/test splits:** Each split draws 1,000 examples from guidelines and 1,000 from pubmedqa, so they reflect the training distribution.

---

## Medical SLM — config.py Variables

All medical variables live under the `# ── Medical SLM ──` section in config.py:

```python
# Dataset names
MED_TEXTBOOK_DATASET          = "epfl-llm/guidelines"
MED_MEDQA_DATASET             = "medalpaca/medical_meadow_medqa"
MED_PUBMEDQA_DATASET          = "pubmed_qa"
MED_PUBMEDQA_PRETRAIN_SUBSET  = "pqa_unlabeled"
MED_PUBMEDQA_PRETRAIN_SIZE    = 60_000
MED_TEXTBOOK_DATA_SIZE        = 33_000
MED_VAL_SIZE                  = 2_000   # 1000 per source
MED_TEST_SIZE                 = 2_000   # 1000 per source
MED_MEDQA_TRAIN_SIZE          = 10_000
MED_PUBMEDQA_TRAIN_SIZE       = 2_000   # pqa_labeled for SFT

# Paths (all derived from BASE)
MED_DATA_DIR, MED_TOKENIZER_DIR, MED_CHECKPOINT_DIR
MED_TOKENIZER_VOCAB, MED_TOKENIZER_MERGES
MED_TRAIN_TXT, MED_VAL_TXT, MED_TEST_TXT
MED_TRAIN_TOKENS, MED_VAL_TOKENS, MED_TEST_TOKENS
MED_DAPT_CKPT, MED_DAPT_FINAL_CKPT  # kept for historical name; these are pretrain ckpts
MED_SFT_CKPT, MED_SFT_FINAL_CKPT, MED_DPO_FINAL_CKPT
MED_HF_MODEL_REPO, MED_HF_TOKENIZER_REPO

# Hyperparams
MED_DAPT_LR             = 1e-4
MED_DAPT_MAX_STEPS      = 20_000
MED_DAPT_WARMUP_STEPS   = 300
MED_DAPT_BATCH_SIZE     = 16
MED_SFT_LR              = 3e-5
MED_SFT_MAX_STEPS       = 5_000
MED_SFT_BATCH_SIZE      = 16
MED_DPO_LR              = 1e-5
MED_DPO_MAX_STEPS       = 1_000
MED_DPO_BETA            = 0.1
```

---

## Medical SLM — Key Modules

### `loaders/medical.py`
- `_parse_medqa_input(raw_input)` — splits MedQA `input` field into question + options dict
- `format_medqa_instruction(example)` — returns `(prompt, answer)` pair for SFT
- `format_pubmedqa_instruction(example)` — same, for PubMedQA labeled examples
- `MedQADataset` — PyTorch Dataset; tracks `prompt_len` for loss masking
- `PubMedQADataset` — same interface as MedQADataset
- `collate_sft_batch(batch, pad_id)` — pads, sets prompt tokens to -100
- `build_dpo_pairs_from_medqa(examples)` — uses labeled wrong options as `rejected`

### `evaluation/medical_metrics.py`
- `_score_text(model, token_ids, device)` — mean per-token log-prob (used for MCQ scoring)
- `evaluate_mcq_accuracy(model, tokenizer, device, examples)` — likelihood scoring over all 5 options
- `evaluate_medical_perplexity(model, device, split)` — perplexity on medical token splits
- `run_usmle_benchmark(checkpoints, model_factory, tokenizer, device, examples)` — comparison table

### `tokenizer/preprocess.py`
- `train_tokenizer(train_file, save_dir, vocab_size)` — `train_file` accepts `str | list[str]`; pass multiple files to train on both guidelines text and pqa_labeled text

---

## Key Design Decisions (always respect these)

### 1. Environment-agnostic paths

All paths are controlled by `config.py` at the repo root. Never hardcode paths anywhere else.

```python
# config.py detects environment like this:
ON_COLAB = os.path.exists('/content/drive')
BASE = DRIVE_BASE if ON_COLAB else LOCAL_BASE
```

- Always import paths from `config.py`.
- Never hardcode `/content/drive/...` or `/home/user/...` in any `.py` or notebook.
- When adding a new path (data dir, checkpoint dir, log dir), add it to `config.py` first, then import it.

### 2. Dataset-agnostic

No dataset name is hardcoded anywhere. Pretraining and SFT datasets are configured via `config.py`. Datasets may be pushed to or pulled from HuggingFace Hub. When writing data-loading code, always accept a dataset name or path as a parameter sourced from config.

### 3. Single T4 GPU — 16GB VRAM constraint

Every code change must fit within 16GB VRAM. Apply these strategies by default:

| Situation | Preferred approach |
|---|---|
| Large batch needed | Use gradient accumulation (`grad_accum_steps` in config) |
| Full fine-tuning requested | Suggest LoRA/PEFT instead; warn if proceeding with full FT |
| Multiple models needed | Load and unload sequentially, never simultaneously |
| Long sequences | Enable gradient checkpointing |
| Training precision | Use fp16 or bf16 mixed precision |

Flag any suggestion that risks exceeding VRAM with a `# WARNING: may exceed T4 VRAM` comment.

Medical pipeline uses `block_size=512` (vs 256 for TinyStories) — note the 4× memory increase for attention.

### 4. Notebook-first, script-backed

- Reusable logic lives in `.py` modules under `model/`, `training/`, `evaluation/`, `loaders/`, etc.
- Notebooks under `experiments/notebooks/` import from those modules and run experiments.
- When adding a feature: write the logic in the appropriate `.py` module, then demonstrate it in a notebook cell.
- Never put non-trivial logic directly in notebook cells — it won't be reusable.

### 5. Pedagogical code style

This is a learning project. Prioritize clarity over cleverness:

- Add docstrings to **every** new function and class (what it does, args, returns).
- Comment the *why*, not just the *what* — explain design choices inline.
- Break logic into small, named functions (avoid long monolithic blocks).
- Name all constants in `config.py`; no magic numbers in code.
- Explicit over implicit: spell out shapes, dtypes, and tensor operations.
- No premature optimization — a clear slow implementation beats an opaque fast one.

### 6. `loaders/` not `datasets/`

The local package that holds PyTorch Dataset classes is named `loaders/`, not `datasets/`. This is intentional: Python resolves local packages before installed packages, so a local `datasets/` folder would shadow the HuggingFace `datasets` library and break `from datasets import load_dataset`. Always import from `loaders.medical`, never from `datasets.medical`.

---

## config.py Conventions

- `ON_COLAB` — bool, detected via `os.path.exists('/content/drive')`
- `DRIVE_BASE` / `LOCAL_BASE` — root paths for the two environments
- All derived paths (data, checkpoints, logs, tokenizer) flow from the base path
- TinyStories model hyperparams: `D_MODEL`, `N_HEADS`, `N_LAYERS`, `FFN_DIM`, `MAX_SEQ_LEN`, `VOCAB_SIZE`
- TinyStories training hyperparams: `LR`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `MAX_STEPS`, `WARMUP_STEPS`, `WEIGHT_DECAY`
- Medical hyperparams: all prefixed with `MED_` — see Medical SLM section above
- Never modify `config.py` for one-off experiments — override values at the top of the notebook instead:
  ```python
  # notebook override example
  import config
  config.MED_DAPT_LR = 3e-4  # override for this run only
  ```

Before suggesting any path or hyperparameter value, check `config.py` to see if it is already defined there.

---

## Local Environment Notes

- **Python venv:** `.venv/` at repo root. Activate with `source .venv/bin/activate`. In Jupyter, select the `.venv` kernel.
- **MPS available:** Mac with Apple Silicon — use `device = "mps"` locally; `device = "cuda"` on Colab.
- **bitsandbytes:** CUDA-only; not installed locally. Do not import it in code paths that run locally without guarding with `try/except ImportError`.
- **To verify correct kernel in Jupyter:** `import sys; print(sys.executable)` — should show `.venv` path.

---

## Dependencies

```
torch                  # primary framework
transformers           # model utilities, tokenizer helpers
datasets               # HuggingFace dataset loading (import as: from datasets import load_dataset)
tokenizers             # fast BPE tokenizer (Rust-backed)
peft                   # LoRA, prefix tuning
bitsandbytes           # INT8/INT4 quantization (CUDA only — not available locally)
huggingface_hub        # push/pull models, datasets, tokenizer
numpy
pandas
matplotlib
tqdm
wandb                  # optional — for experiment tracking
```

---

## What Claude must always do

- **Read `config.py` first** before suggesting any path, hyperparameter, or dataset name.
- **Add docstrings** to every new function and class.
- **Include checkpoint saving** in every training loop (save to the Drive path from config).
- **Add markdown cells** above notebook code cells explaining what the cell does and why.
- **Suggest HuggingFace Hub uploads** for tokenizer artifacts, model weights, and processed datasets.
- **Prefer memory-efficient patterns**: gradient checkpointing, mixed precision, LoRA over full FT, gradient accumulation over large batches.
- **Flag VRAM risk** with an inline comment when a suggestion may push close to or over 16GB.
- **Import from `loaders.medical`**, never `datasets.medical`.
- **Use `clean_text` field** when loading `epfl-llm/guidelines`.
- **Use `context["contexts"]`** (nested dict) when loading PubMedQA context.

## What Claude must never do

- Hardcode file paths anywhere outside `config.py`.
- Hardcode dataset names — always pull from config or accept as a parameter.
- Write a training loop without checkpoint saving logic.
- Suggest full fine-tuning without first proposing LoRA/PEFT as the default and warning about memory cost.
- Load multiple large models into GPU memory at the same time.
- Put reusable logic only in a notebook cell — it must live in a `.py` module.
- Use magic numbers — all constants belong in `config.py`.
- Import `bitsandbytes` without a try/except guard (CUDA-only library).
- Use `datasets/` as a package name — it shadows the HuggingFace library.
