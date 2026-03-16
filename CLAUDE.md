# CLAUDE.md — slm-from-scratch

This file is read by Claude Code at the start of every session. It contains everything needed to understand the project, navigate the codebase, and assist effectively.

---

## Project Overview

An end-to-end Small Language Model (SLM) training project built in PyTorch. Covers the full LLM pipeline from scratch: custom BPE tokenizer, GPT-style transformer, pretraining, SFT, DPO, PEFT, quantization, inference optimization, and evaluation.

**Target environment:** Google Colab with a single T4 GPU (16GB VRAM). All code must also run on local machines.

---

## Repo Structure

```
slm-from-scratch/
├── config.py                  # All paths, hyperparams, env detection — source of truth
├── requirements.txt
├── tokenizer/                 # Custom BPE tokenizer, trained from scratch
├── data/                      # Raw + tokenized data (gitignored, lives on Drive)
├── datasets/                  # PyTorch Dataset classes for pretrain/SFT/DPO
├── model/                     # GPT architecture: attention, MLP, blocks, KV-cache
├── training/                  # Training loops, optimizer, scheduler, utils
├── generation/                # Inference, sampling strategies, KV-cache inference
├── evaluation/                # Perplexity, repetition, prefix completion, benchmarks
├── dpo/                       # DPO loss, trainer, preference dataset generation
├── checkpoints/               # Gitignored, saved to Drive
└── experiments/
    ├── logs/                  # Gitignored, saved to Drive
    └── notebooks/             # Colab notebooks versioned in git
```

---

## Pipeline Stages (in order)

1. Tokenizer training — BPE from scratch
2. Data preprocessing and tokenization
3. Pretraining — causal LM objective
4. Supervised Fine-Tuning (SFT) — instruction following
5. DPO — preference optimization
6. PEFT — LoRA, prefix tuning
7. Quantization — INT8 / INT4 via bitsandbytes
8. Inference optimization — KV-cache, sampling (greedy, top-k, top-p, temperature)
9. Evaluation — perplexity, repetition score, prefix completion
10. Benchmarking — throughput (tokens/sec), latency, memory profiling

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

### 4. Notebook-first, script-backed

- Reusable logic lives in `.py` modules under `model/`, `training/`, `evaluation/`, etc.
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

---

## config.py Conventions

- `ON_COLAB` — bool, detected via `os.path.exists('/content/drive')`
- `DRIVE_BASE` / `LOCAL_BASE` — root paths for the two environments
- All derived paths (data, checkpoints, logs, tokenizer) flow from the base path
- Model hyperparams: `D_MODEL`, `N_HEADS`, `N_LAYERS`, `FFN_DIM`, `MAX_SEQ_LEN`, `VOCAB_SIZE`
- Training hyperparams: `LR`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `MAX_STEPS`, `WARMUP_STEPS`, `WEIGHT_DECAY`
- Never modify `config.py` for one-off experiments — override values at the top of the notebook instead:
  ```python
  # notebook override example
  import config
  config.LR = 3e-4  # override for this run only
  ```

Before suggesting any path or hyperparameter value, check `config.py` to see if it is already defined there.

---

## Dependencies

```
torch                  # primary framework
transformers           # model utilities, tokenizer helpers
datasets               # HuggingFace dataset loading
tokenizers             # fast BPE tokenizer (Rust-backed)
peft                   # LoRA, prefix tuning
bitsandbytes           # INT8/INT4 quantization
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

## What Claude must never do

- Hardcode file paths anywhere outside `config.py`.
- Hardcode dataset names — always pull from config or accept as a parameter.
- Write a training loop without checkpoint saving logic.
- Suggest full fine-tuning without first proposing LoRA/PEFT as the default and warning about memory cost.
- Load multiple large models into GPU memory at the same time.
- Put reusable logic only in a notebook cell — it must live in a `.py` module.
- Use magic numbers — all constants belong in `config.py`.
