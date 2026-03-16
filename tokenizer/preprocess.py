"""
tokenizer/preprocess.py — Tokenizer training and data tokenization utilities.

Two responsibilities:
  1. Train a BPE tokenizer from scratch on raw text files.
  2. Tokenize text splits into flat uint16 numpy arrays for fast pretraining.

Why uint16?
    Our vocab is 8192 tokens (fits in uint16: max 65535). Using uint16 halves
    the storage and memory cost vs int32, which matters when the full training
    file is hundreds of MB.

Why tokenize upfront (not on-the-fly)?
    Tokenizing 2M+ stories on every run is slow. We do it once, save the
    binary array, and use np.memmap to read it lazily during training.

Usage:
    from tokenizer.preprocess import train_tokenizer, tokenize_and_save
    import config as cfg

    # Step 1: train the tokenizer on the training text file
    tokenizer = train_tokenizer(cfg.TRAIN_TXT, cfg.TOKENIZER_DIR)

    # Step 2: tokenize all splits
    for split in ["train", "val", "test"]:
        tokenize_and_save(split, tokenizer)
"""

import os
import numpy as np
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer

import config as cfg


def train_tokenizer(
    train_file: str = None,
    save_dir:   str = None,
    vocab_size: int = None,
) -> ByteLevelBPETokenizer:
    """
    Train a Byte-Level BPE tokenizer from scratch on a raw text file.

    BPE (Byte-Pair Encoding) starts with individual bytes as the vocabulary
    and iteratively merges the most frequent adjacent pair until reaching
    vocab_size. 'Byte-Level' means we operate on UTF-8 bytes, so any Unicode
    text is representable without an <unk> token.

    Special tokens added:
        <s>     — beginning of sequence
        <pad>   — padding (id=1, used in SFT batching)
        </s>    — end of sequence (used as generation stop signal)
        <unk>   — fallback (should never be used with byte-level BPE)
        <mask>  — future use (e.g. masked language modelling)

    Args:
        train_file: path to the raw training text file (defaults to cfg.TRAIN_TXT)
        save_dir:   directory to save vocab.json and merges.txt (defaults to cfg.TOKENIZER_DIR)
        vocab_size: BPE vocabulary size (defaults to cfg.VOCAB_SIZE)

    Returns:
        trained ByteLevelBPETokenizer instance
    """
    train_file = train_file or cfg.TRAIN_TXT
    save_dir   = save_dir   or cfg.TOKENIZER_DIR
    vocab_size = vocab_size or cfg.VOCAB_SIZE

    os.makedirs(save_dir, exist_ok=True)

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[train_file],
        vocab_size=vocab_size,
        min_frequency=2,          # ignore pairs that appear fewer than 2 times
        special_tokens=[
            "<s>",    # id=0
            "<pad>",  # id=1
            "</s>",   # id=2
            "<unk>",  # id=3
            "<mask>", # id=4
        ],
    )

    tokenizer.save_model(save_dir)
    print(f"Tokenizer saved to {save_dir}  (vocab size: {tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(vocab_path: str = None, merges_path: str = None) -> ByteLevelBPETokenizer:
    """
    Load a previously trained tokenizer from disk.

    Args:
        vocab_path:   path to vocab.json (defaults to cfg.TOKENIZER_VOCAB)
        merges_path:  path to merges.txt (defaults to cfg.TOKENIZER_MERGES)

    Returns:
        ByteLevelBPETokenizer instance
    """
    vocab_path  = vocab_path  or cfg.TOKENIZER_VOCAB
    merges_path = merges_path or cfg.TOKENIZER_MERGES
    return ByteLevelBPETokenizer(vocab_path, merges_path)


def tokenize_and_save(
    split:      str,
    tokenizer:  ByteLevelBPETokenizer,
    batch_size: int = 2000,
) -> None:
    """
    Tokenize a text split and write the token ids to a binary .npy file.

    Reads the text file line by line in batches, tokenizes each batch with
    the fast Rust-backed encode_batch, and appends the raw uint16 ids to the
    output file.  The output is a flat array — sequences are not separated by
    any delimiter, which is the standard for causal LM pretraining.

    Args:
        split:      "train", "val", or "test"
        tokenizer:  trained ByteLevelBPETokenizer
        batch_size: lines to tokenize at once (larger = faster, more RAM)

    Output files:
        train → cfg.TRAIN_TOKENS
        val   → cfg.VAL_TOKENS
        test  → cfg.TEST_TOKENS
    """
    txt_paths = {
        "train": cfg.TRAIN_TXT,
        "val":   cfg.VAL_TXT,
        "test":  cfg.TEST_TXT,
    }
    out_paths = {
        "train": cfg.TRAIN_TOKENS,
        "val":   cfg.VAL_TOKENS,
        "test":  cfg.TEST_TOKENS,
    }

    if split not in txt_paths:
        raise ValueError(f"split must be one of {list(txt_paths)}; got '{split}'")

    input_file  = txt_paths[split]
    output_file = out_paths[split]

    if not os.path.exists(input_file):
        print(f"Skipping {split}: {input_file} not found")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Tokenising {split}: {input_file} → {output_file}")

    with open(output_file, "wb") as out_f:
        batch = []
        with open(input_file, "r", encoding="utf-8") as in_f:
            for line in tqdm(in_f, desc=split):
                line = line.strip()
                if not line:
                    continue
                batch.append(line)
                if len(batch) == batch_size:
                    _flush_batch(batch, tokenizer, out_f)
                    batch = []
            if batch:
                _flush_batch(batch, tokenizer, out_f)  # final partial batch

    print(f"Done: {output_file}")


def _flush_batch(
    batch: list,
    tokenizer: ByteLevelBPETokenizer,
    out_f,
) -> None:
    """Encode a batch of strings and write the uint16 token ids to out_f."""
    encodings = tokenizer.encode_batch(batch)
    tokens = []
    for enc in encodings:
        tokens.extend(enc.ids)
    np.array(tokens, dtype=np.uint16).tofile(out_f)
