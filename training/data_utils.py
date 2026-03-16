"""
training/data_utils.py — Data loading utilities for pretraining.

Token files are stored as flat uint16 numpy arrays (one token per element).
We use np.memmap instead of np.load so the data lives on disk and is only
paged into RAM as needed — critical when the full dataset exceeds Colab RAM.

A training 'batch' is sampled by drawing B random starting positions, then
extracting B overlapping windows of length block_size from the token array.
The target sequence is the input shifted by one position (next-token prediction).
"""

import numpy as np
import torch
import config as cfg


def load_tokens(split: str) -> np.memmap:
    """
    Memory-map the tokenised dataset for a given split.

    Using memmap means the file is read from disk on-demand rather than
    loaded all at once — safe even when the file is larger than available RAM.

    Args:
        split: one of "train", "val", or "test"

    Returns:
        np.memmap of shape (N,) and dtype uint16

    Raises:
        FileNotFoundError if the token file does not exist.
        ValueError if split is not recognised.
    """
    path_map = {
        "train": cfg.TRAIN_TOKENS,
        "val":   cfg.VAL_TOKENS,
        "test":  cfg.TEST_TOKENS,
    }
    if split not in path_map:
        raise ValueError(f"split must be one of {list(path_map)}; got '{split}'")

    path = path_map[split]
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple:
    """
    Sample a random mini-batch of (input, target) sequence pairs.

    Each sample is a window of length `block_size` drawn from `data`.
    The target is the same window shifted right by one position so the model
    learns to predict the next token at every position simultaneously.

    Args:
        data:       flat token array (train or val tokens from load_tokens)
        block_size: context length — window size drawn from data
        batch_size: number of sequences per batch
        device:     "cuda" or "cpu"

    Returns:
        x: (batch_size, block_size)   input token ids (int64)
        y: (batch_size, block_size)   target token ids, x shifted by 1

    Example:
        data = load_tokens("train")
        x, y = get_batch(data, cfg.BLOCK_SIZE, cfg.PRETRAIN_BATCH_SIZE, device)
    """
    # Draw B random start indices; ensure the window + 1 target fits in data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([
        torch.from_numpy(data[i     : i + block_size    ].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
        for i in ix
    ])

    return x.to(device), y.to(device)
