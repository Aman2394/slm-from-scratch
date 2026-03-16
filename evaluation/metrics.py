"""
evaluation/metrics.py — Evaluation metrics and diagnostic tests.

Metrics implemented:
  - evaluate_perplexity : cross-entropy loss + perplexity on val/test splits
  - repetition_score    : fraction of n-grams that are repeated (lower is better)
  - prefix_completion_test : feed a real prefix, compare model continuation vs. truth
  - nonsense_prompt_test   : generate on out-of-distribution prompts (sanity check)
  - longest_token_match    : longest token sequence shared with training data
                             (checks for memorisation)

All functions accept a model and tokenizer rather than capturing them from
a notebook scope — this makes them importable and testable in isolation.

Usage:
    from evaluation.metrics import evaluate_perplexity, repetition_score
    import config as cfg

    val_loss, val_ppl = evaluate_perplexity(model, device, split="val")
    print(f"val perplexity: {val_ppl:.2f}")
"""

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

import config as cfg
from training.data_utils import load_tokens, get_batch


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    device: str,
    split: str    = "val",
    num_batches:  int = None,
    batch_size:   int = None,
) -> Tuple[float, float]:
    """
    Compute mean cross-entropy loss and perplexity on a token split.

    Perplexity (PPL) = exp(loss).  Lower is better.
    A PPL of k roughly means the model is as uncertain as a uniform distribution
    over k tokens at each step.

    Additional metric: bits-per-byte (BPB) = loss / log(2).
    BPB is hardware-independent and commonly reported for LLM pretraining.

    Args:
        model:       GPT model, already on device
        device:      "cuda" or "cpu"
        split:       "train", "val", or "test"
        num_batches: batches to average over (defaults to cfg.EVAL_NUM_BATCHES)
        batch_size:  sequences per batch (defaults to cfg.EVAL_BATCH_SIZE)

    Returns:
        (loss, perplexity) as floats
    """
    num_batches = num_batches or cfg.EVAL_NUM_BATCHES
    batch_size  = batch_size  or cfg.EVAL_BATCH_SIZE
    block_size  = model.config.block_size

    data = load_tokens(split)
    model.eval()

    losses = []
    for _ in range(num_batches):
        x, y   = get_batch(data, block_size, batch_size, device)
        logits  = model(x)
        loss    = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        losses.append(loss.item())

    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    bpb        = mean_loss / math.log(2)

    print(f"[{split}] loss: {mean_loss:.4f} | perplexity: {perplexity:.2f} | bpb: {bpb:.4f}")
    return mean_loss, perplexity


# ---------------------------------------------------------------------------
# Repetition score
# ---------------------------------------------------------------------------

def repetition_score(tokens: list, n: int = 3) -> float:
    """
    Measure how repetitive a generated sequence is.

    Computes the fraction of n-grams that are *not* unique.
    Score ∈ [0, 1]; 0 means perfectly diverse, 1 means completely repetitive.

    Formula:  1 - (unique_ngrams / total_ngrams)

    Args:
        tokens: list of integer token ids
        n:      n-gram size (default 3 — trigrams capture phrase-level repetition)

    Returns:
        repetition score in [0, 1]
    """
    if len(tokens) <= n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n)]
    unique_ratio = len(set(ngrams)) / len(ngrams)
    return 1.0 - unique_ratio


# ---------------------------------------------------------------------------
# Prefix completion test
# ---------------------------------------------------------------------------

def prefix_completion_test(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    generate_fn,
    prefix_len:       int = None,
    continuation_len: int = None,
    split:            str = "train",
) -> None:
    """
    Pick a random window from the token data, feed the first prefix_len tokens
    as a prompt, and compare the model's continuation with the true continuation.

    This test shows whether the model has learned plausible continuations —
    not memorisation (the completion will differ from the truth) but
    coherent continuation in a similar style.

    Args:
        model:            GPT model, already on device
        tokenizer:        ByteLevelBPETokenizer
        device:           "cuda" or "cpu"
        generate_fn:      generation function (generate or generate_kv from generation/sampler.py)
        prefix_len:       tokens to use as prompt (defaults to cfg.EVAL_PREFIX_LEN)
        continuation_len: tokens to generate (defaults to cfg.EVAL_CONTINUATION_LEN)
        split:            which token file to sample from (default "train")
    """
    prefix_len       = prefix_len       or cfg.EVAL_PREFIX_LEN
    continuation_len = continuation_len or cfg.EVAL_CONTINUATION_LEN
    block_size       = model.config.block_size

    data  = load_tokens(split)
    start = random.randint(0, len(data) - prefix_len - continuation_len)

    prefix_tokens = data[start : start + prefix_len]
    true_tokens   = data[start : start + prefix_len + continuation_len]

    prefix_text = tokenizer.decode(prefix_tokens.tolist())
    true_text   = tokenizer.decode(true_tokens.tolist())

    x      = torch.from_numpy(prefix_tokens.astype(np.int64)).unsqueeze(0).to(device)
    output = generate_fn(model, x, tokenizer, block_size, max_new_tokens=continuation_len)
    gen_text = tokenizer.decode(output[0].tolist())

    print("\n===== PREFIX COMPLETION TEST =====")
    print("\nPREFIX:")
    print(prefix_text)
    print("\nMODEL COMPLETION:")
    print(gen_text)
    print("\nTRUE CONTINUATION:")
    print(true_text)


# ---------------------------------------------------------------------------
# Nonsense prompt test
# ---------------------------------------------------------------------------

def nonsense_prompt_test(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    generate_fn,
    max_new_tokens: int = None,
) -> None:
    """
    Generate completions for deliberately odd, out-of-distribution prompts.

    The goal is to check whether the model degrades gracefully on unusual input
    or collapses into repetition / incoherence.  Good models should produce
    fluent (if surprising) continuations.

    Args:
        model:          GPT model, already on device
        tokenizer:      ByteLevelBPETokenizer
        device:         "cuda" or "cpu"
        generate_fn:    generate or generate_kv
        max_new_tokens: tokens to generate per prompt (defaults to cfg.EVAL_CONTINUATION_LEN)
    """
    max_new_tokens = max_new_tokens or cfg.EVAL_CONTINUATION_LEN
    block_size     = model.config.block_size

    prompts = [
        "Once upon a time there was a flying sandwich",
        "A dinosaur who loved pizza lived in a tiny house",
        "A robot went to school for the first time",
        "A magical banana started talking to a cat",
    ]

    print("\n===== NONSENSE PROMPT TEST =====")
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        x   = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        out = generate_fn(model, x, tokenizer, block_size, max_new_tokens=max_new_tokens)
        print(f"\nPROMPT: {prompt}")
        print(tokenizer.decode(out[0].tolist()))


# ---------------------------------------------------------------------------
# Memorisation check — longest token match
# ---------------------------------------------------------------------------

def longest_token_match(
    gen_tokens: list,
    train_tokens: np.ndarray,
    window: int = 100_000,
) -> int:
    """
    Find the longest contiguous token sequence in gen_tokens that also
    appears in the first `window` tokens of the training set.

    A long match (> ~10 tokens) is a warning sign of memorisation.
    Short matches (< 5) are expected even for non-memorising models because
    common phrases will naturally recur.

    Args:
        gen_tokens:   list of generated token ids
        train_tokens: numpy array of training token ids (from load_tokens("train"))
        window:       how much of the training data to search (default 100k tokens)

    Returns:
        length of the longest matching subsequence
    """
    train_subset = train_tokens[:window]
    max_match    = 0

    for i in range(len(gen_tokens)):
        for j in range(len(train_subset)):
            k = 0
            while (
                i + k < len(gen_tokens)
                and j + k < len(train_subset)
                and gen_tokens[i + k] == train_subset[j + k]
            ):
                k += 1
            max_match = max(max_match, k)

    return max_match
