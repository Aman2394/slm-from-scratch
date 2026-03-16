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
CONFIG_DIR:      str = os.path.join(BASE, "config")

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
# Utility: create all directories that must exist before any run
# ---------------------------------------------------------------------------

def make_dirs() -> None:
    """Create all required project directories (safe to call repeatedly)."""
    for d in [DATA_DIR, TOKENIZER_DIR, CHECKPOINT_DIR, LOG_DIR, CONFIG_DIR]:
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
