"""
training/sft_data_utils.py — Data preparation for Supervised Fine-Tuning (SFT).

SFT overview:
    We fine-tune the pretrained GPT to follow instructions by training on
    (instruction, response) pairs formatted as:
        "Instruction: <prompt> Response: <story>"

    The key difference from pretraining is *response masking*: we only compute
    loss on the response tokens, not the instruction tokens. This teaches the
    model to generate good responses given a prompt, without penalising it for
    not perfectly reproducing the instruction prefix.

    Masking is done by setting instruction token labels to -100, which
    F.cross_entropy ignores via the `ignore_index` parameter.

Dataset:
    We use TinyStories (roneneldan/TinyStories) — a corpus of short children's
    stories. We wrap each story in a random instruction template so the model
    learns to associate the instruction with the story style.
"""

import os
import pickle
import random
import torch
import numpy as np
from tqdm import tqdm
from typing import List

import config as cfg


# ---------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------

# Each story is wrapped in a randomly chosen instruction.
# Using multiple phrasings makes the model more robust — it won't only respond
# to one exact phrasing of the same intent.
INSTRUCTION_TEMPLATES = [
    "Write a short children's story.",
    "Tell a bedtime story.",
    "Write a story for kids.",
    "Create a short fairy tale.",
    "Tell a story about friendship.",
]

# The response marker — we locate this in the token sequence to find where
# the response starts and apply the loss mask.
RESPONSE_MARKER = " Response:"


def create_instruction(story: str) -> str:
    """
    Wrap a story in a randomly chosen instruction template.

    Args:
        story: raw story text

    Returns:
        formatted string: "Instruction: <prompt> Response: <story>"
    """
    instruction = random.choice(INSTRUCTION_TEMPLATES)
    return f"Instruction: {instruction} Response: {story}"


# ---------------------------------------------------------------------------
# Dataset tokenization
# ---------------------------------------------------------------------------

def tokenize_sft_dataset(tokenizer, dataset_texts: List[str]) -> List[List[int]]:
    """
    Tokenize a list of story texts into SFT-formatted token sequences.

    Each story is:
      1. Wrapped in a random instruction template
      2. Tokenised with the BPE tokenizer
      3. Appended with the EOS token

    The result is a list of variable-length token id lists (not padded here —
    padding happens in get_batch to avoid wasting memory in storage).

    Args:
        tokenizer:      ByteLevelBPETokenizer instance (loaded from TOKENIZER_DIR)
        dataset_texts:  list of raw story strings

    Returns:
        samples: list of token id lists (one per story), each ending with EOS
    """
    eos_id = tokenizer.token_to_id("</s>")
    samples = []

    for story in tqdm(dataset_texts, desc="Tokenising SFT data"):
        story = story.replace("\n", " ")  # normalise whitespace
        text  = create_instruction(story)
        tokens = tokenizer.encode(text).ids
        tokens.append(eos_id)             # mark the end of each story
        samples.append(tokens)

    return samples


def save_sft_tokens(samples: List[List[int]], path: str = None) -> None:
    """
    Pickle the tokenised SFT samples to disk.

    Args:
        samples: output of tokenize_sft_dataset
        path:    save path (defaults to cfg.SFT_TOKENS)
    """
    path = path or cfg.SFT_TOKENS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(samples, f)
    print(f"Saved {len(samples)} SFT samples to {path}")


def load_sft_tokens(path: str = None) -> List[List[int]]:
    """
    Load previously pickled SFT token samples from disk.

    Args:
        path: load path (defaults to cfg.SFT_TOKENS)

    Returns:
        list of token id lists
    """
    path = path or cfg.SFT_TOKENS
    with open(path, "rb") as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} SFT samples from {path}")
    return samples


# ---------------------------------------------------------------------------
# Batch sampling with response masking
# ---------------------------------------------------------------------------

def get_sft_batch(
    samples: List[List[int]],
    tokenizer,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple:
    """
    Sample a mini-batch for SFT training with response masking.

    Each sequence is padded to block_size with the PAD token (id=1).
    The target tensor has -100 for all instruction tokens and PAD tokens
    so that the loss is computed only on response tokens.

    Why mask the instruction?
        We want the model to learn *what to generate*, not to memorise the
        exact phrasing of the instruction. If we included instruction tokens
        in the loss, the model would waste capacity learning to predict the
        instruction given the beginning of the instruction — which is not
        what we want.

    Args:
        samples:    list of token id lists (from load_sft_tokens)
        tokenizer:  ByteLevelBPETokenizer (needed to locate the response marker)
        block_size: context length — sequences are truncated/padded to this
        batch_size: number of sequences per batch
        device:     "cuda" or "cpu"

    Returns:
        x: (batch_size, block_size)  input token ids
        y: (batch_size, block_size)  target token ids; -100 for masked positions
    """
    PAD_ID = 1   # <pad> token id in our BPE vocab

    # Token ids for the response marker " Response:" (used to locate mask boundary)
    response_marker_ids = tokenizer.encode(RESPONSE_MARKER).ids
    response_marker_len = len(response_marker_ids)

    idx = torch.randint(len(samples), (batch_size,))
    x_batch, y_batch = [], []

    for i in idx:
        tokens = list(samples[i])

        # Truncate if longer than block_size, pad if shorter
        if len(tokens) > block_size:
            tokens = tokens[:block_size]
        else:
            tokens = tokens + [PAD_ID] * (block_size - len(tokens))

        x = torch.tensor(tokens[:-1], dtype=torch.long)  # input:  t[0..T-2]
        y = torch.tensor(tokens[1:],  dtype=torch.long)  # target: t[1..T-1]

        # Find where the response starts and mask all positions before it
        response_start = None
        for j in range(len(tokens) - response_marker_len):
            if tokens[j : j + response_marker_len] == response_marker_ids:
                response_start = j + response_marker_len
                break

        if response_start is not None:
            # Mask instruction tokens: set their targets to -100 (ignored in loss)
            y[: response_start - 1] = -100

        x_batch.append(x)
        y_batch.append(y)

    x = torch.stack(x_batch).to(device)
    y = torch.stack(y_batch).to(device)

    return x, y
