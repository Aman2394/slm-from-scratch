"""
config.py — Single source of truth for all paths and hyperparameters.

How to use:
    from config import cfg

    # Access paths
    model = torch.load(cfg.PRETRAIN_CKPT)

    # Access hyperparams
    optimizer = AdamW(model.parameters(), lr=cfg.PRETRAIN_LR)

Notebook overrides (never edit this file for one-off runs):
    import config
    config.cfg.PRETRAIN_LR = 1e-4   # override for this run only
"""

import os

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

# True when running on Google Colab with Drive mounted
ON_COLAB: bool = os.path.exists("/content/drive")

# ---------------------------------------------------------------------------
# Base paths — everything flows from here, never hardcode below this point
# ---------------------------------------------------------------------------

if ON_COLAB:
    # Google Drive root for all persistent storage
    DRIVE_BASE: str = "/content/drive/MyDrive/slm_project"
    # Repo root inside Colab (cloned from GitHub)
    REPO_ROOT: str = "/content/slm-from-scratch"
else:
    # Local machine root — adjust to match your local clone
    LOCAL_BASE: str = os.path.join(os.path.expanduser("~"), "Desktop", "slm-learning")
    DRIVE_BASE: str = LOCAL_BASE   # same variable name so code below is env-agnostic
    REPO_ROOT: str = LOCAL_BASE

BASE: str = DRIVE_BASE   # single alias used everywhere below

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------

DATA_DIR:        str = os.path.join(BASE, "data")
TOKENIZER_DIR:   str = os.path.join(BASE, "tokenizer")
CHECKPOINT_DIR:  str = os.path.join(BASE, "checkpoints")
LOG_DIR:         str = os.path.join(BASE, "experiments", "logs")

# ---------------------------------------------------------------------------
# Tokenizer file paths
# ---------------------------------------------------------------------------

TOKENIZER_VOCAB:  str = os.path.join(TOKENIZER_DIR, "vocab.json")
TOKENIZER_MERGES: str = os.path.join(TOKENIZER_DIR, "merges.txt")

# ---------------------------------------------------------------------------
# Dataset file paths
# ---------------------------------------------------------------------------

# Raw text splits (written during preprocessing)
TRAIN_TXT: str = os.path.join(DATA_DIR, "train.txt")
VAL_TXT:   str = os.path.join(DATA_DIR, "val.txt")
TEST_TXT:  str = os.path.join(DATA_DIR, "test.txt")

# Tokenized binary splits (uint16 numpy arrays)
TRAIN_TOKENS: str = os.path.join(DATA_DIR, "train_tokens.npy")
VAL_TOKENS:   str = os.path.join(DATA_DIR, "val_tokens.npy")
TEST_TOKENS:  str = os.path.join(DATA_DIR, "test_tokens.npy")

# SFT tokenized dataset (pickle)
SFT_TOKENS: str = os.path.join(DATA_DIR, "sft_tokens.pkl")

# DPO preference pairs (JSON Lines)
DPO_DATASET: str = os.path.join(DATA_DIR, "tinystories_dpo.jsonl")

# ---------------------------------------------------------------------------
# Checkpoint file paths
# ---------------------------------------------------------------------------

# Pretraining: rolling checkpoint (overwritten every PRETRAIN_SAVE_INTERVAL steps)
PRETRAIN_CKPT:       str = os.path.join(CHECKPOINT_DIR, "gpt_checkpoint.pt")
# Pretraining: final model saved at the end of the run
PRETRAIN_FINAL_CKPT: str = os.path.join(CHECKPOINT_DIR, "gpt_final_v2.pt")

# SFT: rolling checkpoint
SFT_CKPT:       str = os.path.join(CHECKPOINT_DIR, "model_sft.pt")
# SFT: final model
SFT_FINAL_CKPT: str = os.path.join(CHECKPOINT_DIR, "model_sft_final.pt")

# DPO: final model (add when DPO trainer is built)
DPO_FINAL_CKPT: str = os.path.join(CHECKPOINT_DIR, "model_dpo_final.pt")

# ---------------------------------------------------------------------------
# HuggingFace dataset names
# ---------------------------------------------------------------------------

# Pretraining source dataset (loaded via HF datasets library)
PRETRAIN_DATASET_NAME: str = "roneneldan/TinyStories"

# SFT source dataset — same corpus, instruction-formatted subset
SFT_DATASET_NAME: str = "roneneldan/TinyStories"

# Number of examples to use for SFT (subset of the full dataset)
SFT_DATA_SIZE: int = 100_000

# HuggingFace Hub repo names for pushing artifacts (set before pushing)
HF_MODEL_REPO:     str = ""   # e.g. "your-username/slm-pretrained"
HF_TOKENIZER_REPO: str = ""   # e.g. "your-username/slm-tokenizer"
HF_DATASET_REPO:   str = ""   # e.g. "your-username/slm-dpo-pairs"

# ---------------------------------------------------------------------------
# Model architecture hyperparameters (GPT-style transformer)
# ---------------------------------------------------------------------------

VOCAB_SIZE:  int   = 8192   # BPE vocabulary size (matches trained tokenizer)
BLOCK_SIZE:  int   = 256    # Maximum context length (tokens per sequence)
N_LAYER:     int   = 8      # Number of transformer blocks
N_HEAD:      int   = 8      # Number of attention heads per block
N_EMBD:      int   = 512    # Embedding / hidden dimension
DROPOUT:     float = 0.1    # Dropout probability (applied in attn + MLP)

# Derived: FFN inner dimension is 4x the embedding dimension (standard GPT ratio)
FFN_DIM: int = 4 * N_EMBD   # = 2048

# Derived: dimension per attention head
HEAD_DIM: int = N_EMBD // N_HEAD   # = 64

# ---------------------------------------------------------------------------
# Pretraining hyperparameters
# ---------------------------------------------------------------------------

PRETRAIN_LR:            float = 3e-4    # Peak learning rate for AdamW
PRETRAIN_WEIGHT_DECAY:  float = 0.1    # L2 regularisation coefficient
PRETRAIN_BATCH_SIZE:    int   = 16     # Sequences per gradient step
PRETRAIN_MAX_STEPS:     int   = 20_000 # Total gradient update steps
PRETRAIN_EVAL_INTERVAL: int   = 500    # Evaluate val loss every N steps
PRETRAIN_SAVE_INTERVAL: int   = 1_000  # Save rolling checkpoint every N steps
PRETRAIN_EVAL_BATCHES:  int   = 50     # Number of batches used for loss estimation

# Gradient accumulation: simulate a larger effective batch size
# Effective batch = PRETRAIN_BATCH_SIZE × PRETRAIN_GRAD_ACCUM_STEPS
PRETRAIN_GRAD_ACCUM_STEPS: int = 1     # Increase if VRAM is tight

# Mixed precision: use "fp16" or "bf16" to halve VRAM; "fp32" to disable
# WARNING: T4 supports fp16 natively; bf16 requires Ampere or newer
PRETRAIN_PRECISION: str = "fp16"

# Gradient checkpointing: trades compute for memory (enable for long sequences)
PRETRAIN_GRAD_CHECKPOINT: bool = False

# ---------------------------------------------------------------------------
# Supervised Fine-Tuning (SFT) hyperparameters
# ---------------------------------------------------------------------------

SFT_LR:            float = 3e-5   # Lower LR than pretraining — fine-tuning regime
SFT_WEIGHT_DECAY:  float = 0.01
SFT_BATCH_SIZE:    int   = 16
SFT_MAX_STEPS:     int   = 3_000
SFT_EVAL_INTERVAL: int   = 200
SFT_SAVE_INTERVAL: int   = 500
SFT_EVAL_BATCHES:  int   = 50

SFT_GRAD_ACCUM_STEPS:  int  = 1
SFT_PRECISION:         str  = "fp16"
SFT_GRAD_CHECKPOINT:   bool = False

# ---------------------------------------------------------------------------
# DPO hyperparameters
# ---------------------------------------------------------------------------

DPO_LR:           float = 1e-5   # Even lower than SFT
DPO_WEIGHT_DECAY: float = 0.01
DPO_BATCH_SIZE:   int   = 8      # Pairs per step (chosen + rejected)
DPO_MAX_STEPS:    int   = 1_000
DPO_BETA:         float = 0.1    # KL penalty coefficient in DPO loss

DPO_GRAD_ACCUM_STEPS: int  = 2
DPO_PRECISION:        str  = "fp16"

# DPO dataset generation settings
DPO_NUM_CANDIDATES:       int   = 4     # Candidate responses sampled per prompt
DPO_GEN_MAX_TOKENS:       int   = 120
DPO_GEN_TEMPERATURE:      float = 1.1
DPO_GEN_TOP_P:            float = 0.95
DPO_GEN_TOP_K:            int   = 50
DPO_REPETITION_NGRAM_LEN: int   = 3     # N-gram length for repetition detection
DPO_LENGTH_PENALTY_MIN:   int   = 40    # Minimum word count for a valid response

# ---------------------------------------------------------------------------
# Generation / inference hyperparameters (defaults, override per notebook)
# ---------------------------------------------------------------------------

GEN_MAX_NEW_TOKENS:   int   = 120
GEN_TEMPERATURE:      float = 0.9
GEN_TOP_K:            int   = 50
GEN_TOP_P:            float = 0.9
GEN_REPETITION_PENALTY: float = 1.15   # > 1.0 penalises repeated tokens

# ---------------------------------------------------------------------------
# Evaluation hyperparameters
# ---------------------------------------------------------------------------

EVAL_BATCH_SIZE:    int = 16
EVAL_NUM_BATCHES:   int = 200   # For perplexity estimation over val set
EVAL_PREFIX_LEN:    int = 20    # Tokens used as prefix for completion tasks
EVAL_CONTINUATION_LEN: int = 120  # Tokens generated for prefix completion

# ---------------------------------------------------------------------------
# Medical SLM — fully isolated paths so TinyStories artifacts are safe
#
# Why separate paths?
#   - Different tokenizer (trained on medical vocabulary)
#   - Different checkpoint files (fresh random init, completely independent model)
#   - Keeps the medical experiment cleanly separated on Drive
#
# Pipeline: medical textbooks → pretrain from scratch → SFT on MedQA/PubMedQA → DPO → HF upload
# ---------------------------------------------------------------------------

# HuggingFace dataset IDs
MED_TEXTBOOK_DATASET: str = "epfl-llm/guidelines"    # Clinical guidelines (WHO, CDC, NICE) — used to train Meditron
MED_MEDQA_DATASET:    str = "medalpaca/medical_meadow_medqa"
MED_PUBMEDQA_DATASET: str = "pubmed_qa"              # pqa_labeled split — used for SFT

# PubMedQA unlabeled split — used as additional PRETRAINING text (not SFT)
# pqa_unlabeled has 61k examples with rich biomedical abstract context paragraphs.
# We extract: question + context paragraphs + long_answer as one document per example.
MED_PUBMEDQA_PRETRAIN_SUBSET: str = "pqa_unlabeled"
MED_PUBMEDQA_PRETRAIN_SIZE:   int = 60_000           # nearly all 61k available

# Number of textbook examples to use for DAPT pretraining corpus
# ~80k examples ≈ 40–60M tokens — enough for meaningful domain adaptation
MED_TEXTBOOK_DATA_SIZE: int = 33_000   # epfl-llm/guidelines has ~37,970 total; leaving ~5k as buffer
MED_VAL_SIZE:           int = 2_000    # Held out from textbook stream
MED_TEST_SIZE:          int = 2_000    # Total used: 37,000 of 37,970

# Isolated data, tokenizer, and checkpoint directories for this run
MED_DATA_DIR:        str = os.path.join(BASE, "data",        "medical")
MED_TOKENIZER_DIR:   str = os.path.join(BASE, "tokenizer",   "medical")
MED_CHECKPOINT_DIR:  str = os.path.join(BASE, "checkpoints", "medical")

# Tokenizer files
MED_TOKENIZER_VOCAB:  str = os.path.join(MED_TOKENIZER_DIR, "vocab.json")
MED_TOKENIZER_MERGES: str = os.path.join(MED_TOKENIZER_DIR, "merges.txt")

# Raw text splits (one document per line)
MED_TRAIN_TXT: str = os.path.join(MED_DATA_DIR, "train.txt")
MED_VAL_TXT:   str = os.path.join(MED_DATA_DIR, "val.txt")
MED_TEST_TXT:  str = os.path.join(MED_DATA_DIR, "test.txt")

# Tokenized binary splits (uint16 numpy arrays, memory-mapped during training)
MED_TRAIN_TOKENS: str = os.path.join(MED_DATA_DIR, "train_tokens.npy")
MED_VAL_TOKENS:   str = os.path.join(MED_DATA_DIR, "val_tokens.npy")
MED_TEST_TOKENS:  str = os.path.join(MED_DATA_DIR, "test_tokens.npy")

# SFT tokenized dataset (pickle of list of token arrays)
MED_SFT_TOKENS: str = os.path.join(MED_DATA_DIR, "sft_tokens.pkl")

# DPO preference pairs derived from MedQA wrong-option labels (JSONL)
MED_DPO_DATASET: str = os.path.join(MED_DATA_DIR, "medical_dpo.jsonl")

# Checkpoints
MED_DAPT_CKPT:       str = os.path.join(MED_CHECKPOINT_DIR, "med_dapt_checkpoint.pt")
MED_DAPT_FINAL_CKPT: str = os.path.join(MED_CHECKPOINT_DIR, "med_dapt_final.pt")
MED_SFT_CKPT:        str = os.path.join(MED_CHECKPOINT_DIR, "med_sft_checkpoint.pt")
MED_SFT_FINAL_CKPT:  str = os.path.join(MED_CHECKPOINT_DIR, "med_sft_final.pt")
MED_DPO_FINAL_CKPT:  str = os.path.join(MED_CHECKPOINT_DIR, "med_dpo_final.pt")

# HuggingFace Hub repo names (fill in with your username before uploading)
MED_HF_MODEL_REPO:     str = ""   # e.g. "your-username/medical-slm"
MED_HF_TOKENIZER_REPO: str = ""   # e.g. "your-username/medical-slm-tokenizer"

# ---------------------------------------------------------------------------
# Medical Pretraining hyperparameters
#
# Fresh random initialisation — no weights transferred from TinyStories or any other model.
# The medical tokenizer has a different vocabulary, so this is a purpose-built model.
#   - LR 1e-4 with warmup → cosine decay (standard for transformer pretraining)
#   - block_size overridden to 512 in notebook (medical text is longer than stories)
# ---------------------------------------------------------------------------

MED_DAPT_LR:            float = 1e-4   # Peak LR for medical pretraining
MED_DAPT_WEIGHT_DECAY:  float = 0.1
MED_DAPT_BATCH_SIZE:    int   = 16
MED_DAPT_MAX_STEPS:     int   = 20_000
MED_DAPT_WARMUP_STEPS:  int   = 300    # Shorter warmup (weights already warm from TinyStories pretraining)
MED_DAPT_GRAD_CLIP:     float = 1.0
MED_DAPT_EVAL_INTERVAL: int   = 500
MED_DAPT_SAVE_INTERVAL: int   = 1_000
MED_DAPT_EVAL_BATCHES:  int   = 50
MED_DAPT_GRAD_ACCUM_STEPS: int = 1     # Increase to 2 if block_size=512 causes OOM

# ---------------------------------------------------------------------------
# Medical SFT hyperparameters
#
# Fine-tunes on MedQA (USMLE-style) + PubMedQA (research Q&A).
# Loss is masked to response tokens only (question/options are not trained on).
# ---------------------------------------------------------------------------

MED_SFT_LR:            float = 3e-5   # Standard SFT: 10× lower than DAPT LR
MED_SFT_WEIGHT_DECAY:  float = 0.01
MED_SFT_BATCH_SIZE:    int   = 16
MED_SFT_MAX_STEPS:     int   = 5_000  # More steps than TinyStories SFT (harder domain)
MED_SFT_EVAL_INTERVAL: int   = 200
MED_SFT_SAVE_INTERVAL: int   = 500
MED_SFT_EVAL_BATCHES:  int   = 50
MED_SFT_GRAD_ACCUM_STEPS: int = 1

# MedQA / PubMedQA dataset sizes for SFT
MED_MEDQA_TRAIN_SIZE:    int   = 10_000  # Examples from medalpaca medqa split
MED_PUBMEDQA_TRAIN_SIZE: int   = 2_000   # Examples from pubmed_qa labeled split
MED_SFT_MIX_RATIO:       float = 0.8    # Fraction of batches drawn from MedQA vs PubMedQA

# ---------------------------------------------------------------------------
# Medical DPO hyperparameters
#
# Unlike TinyStories DPO (which generates candidates and heuristically ranks them),
# medical DPO uses *labeled wrong options* from USMLE data as rejected responses.
# This gives a cleaner supervision signal: chosen = correct answer, rejected = wrong option.
# ---------------------------------------------------------------------------

MED_DPO_LR:           float = 1e-5
MED_DPO_WEIGHT_DECAY: float = 0.01
MED_DPO_BATCH_SIZE:   int   = 8     # Pairs per step (chosen + rejected)
MED_DPO_MAX_STEPS:    int   = 1_000
MED_DPO_BETA:         float = 0.1   # KL penalty (same as TinyStories DPO)
MED_DPO_GRAD_ACCUM_STEPS: int = 2

# ---------------------------------------------------------------------------
# Utility: create all directories that must exist before any run
# ---------------------------------------------------------------------------

def make_dirs() -> None:
    """Create all required project directories (safe to call repeatedly)."""
    dirs = [
        DATA_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, LOG_DIR,
        MED_DATA_DIR, MED_TOKENIZER_DIR, MED_CHECKPOINT_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Pretty-print summary for quick sanity-checking in notebooks
# ---------------------------------------------------------------------------

def print_config() -> None:
    """Print a human-readable summary of the active configuration."""
    print(f"{'='*50}")
    print(f"  slm-from-scratch config")
    print(f"{'='*50}")
    print(f"  Environment : {'Google Colab' if ON_COLAB else 'Local'}")
    print(f"  Base path   : {BASE}")
    print(f"  --- Model ---")
    print(f"  vocab_size  : {VOCAB_SIZE}")
    print(f"  block_size  : {BLOCK_SIZE}")
    print(f"  n_layer     : {N_LAYER}  |  n_head: {N_HEAD}  |  n_embd: {N_EMBD}")
    print(f"  --- Pretraining ---")
    print(f"  lr          : {PRETRAIN_LR}  |  steps: {PRETRAIN_MAX_STEPS}")
    print(f"  batch_size  : {PRETRAIN_BATCH_SIZE}  |  grad_accum: {PRETRAIN_GRAD_ACCUM_STEPS}")
    print(f"  precision   : {PRETRAIN_PRECISION}")
    print(f"  --- SFT ---")
    print(f"  lr          : {SFT_LR}  |  steps: {SFT_MAX_STEPS}")
    print(f"{'='*50}")
