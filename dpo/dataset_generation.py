"""
dpo/dataset_generation.py — DPO preference pair dataset generation.

Direct Preference Optimization (DPO) trains the model to prefer "chosen"
responses over "rejected" responses, without needing an explicit reward model.

To create the preference dataset we need (prompt, chosen, rejected) triples.
We generate them automatically by:
  1. Extracting a prompt from each training story (first sentence).
  2. Sampling N candidate responses from the SFT model.
  3. Scoring each candidate with a heuristic quality function.
  4. Taking the best-scoring response as "chosen" and worst as "rejected".

Scoring heuristic (lower score = better):
  - Perplexity  — model's own confidence in the text (lower = more fluent)
  - Repetition  — penalty for exact phrase repetition
  - Length      — penalty for very short responses

Note: This is an approximation. For higher quality DPO data you'd use a
separate reward model or human annotations. For a learning project this
heuristic produces usable training signal.

Usage:
    from dpo.dataset_generation import generate_dpo_dataset
    import config as cfg

    pairs = generate_dpo_dataset(model, tokenizer, stories, device)
    # pairs is a list of {"prompt": ..., "chosen": ..., "rejected": ...} dicts
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Dict

import config as cfg


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

def extract_prompt(story: str) -> str:
    """
    Extract an instruction prompt from a story.

    We take the first sentence of the story as a narrative seed and wrap it
    in a standard instruction template.  This ensures the prompt is grounded
    in the story distribution the model was trained on.

    Args:
        story: raw story text

    Returns:
        formatted prompt string
    """
    sentences = story.split(".")
    first_sentence = sentences[0].strip() + "."
    return f"Write a short story for kids.\n\n{first_sentence}\n\nResponse:\n"


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def perplexity_score(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: str,
) -> float:
    """
    Compute the model's perplexity on a text string.

    Lower perplexity = the model finds the text more probable = more fluent.
    We use this as a proxy for response quality.

    Args:
        model:     GPT model
        tokenizer: ByteLevelBPETokenizer
        text:      the candidate response text
        device:    "cuda" or "cpu"

    Returns:
        perplexity (float, ≥ 1.0)
    """
    ids    = tokenizer.encode(text).ids
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(tokens)  # (1, T, vocab_size)

    # Shift: predict each token from the previous one
    shift_logits = logits[:, :-1, :]   # (1, T-1, vocab_size)
    shift_labels = tokens[:, 1:]       # (1, T-1)

    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
    )
    return torch.exp(loss).item()


def repetition_score(text: str, ngram_len: int = None) -> int:
    """
    Count exact phrase repetitions in the text.

    Counts occurrences where a sequence of `ngram_len` words appears
    immediately again (i.e., [A, B, C, A, B, C]).  More repetitions → worse.

    Args:
        text:      candidate response text
        ngram_len: number of words in the repeated phrase (defaults to cfg.DPO_REPETITION_NGRAM_LEN)

    Returns:
        number of exact phrase repetitions found
    """
    ngram_len = ngram_len or cfg.DPO_REPETITION_NGRAM_LEN
    words     = text.split()
    repeats   = 0

    for i in range(len(words) - ngram_len * 2):
        if words[i : i + ngram_len] == words[i + ngram_len : i + ngram_len * 2]:
            repeats += 1

    return repeats


def length_penalty(text: str, min_words: int = None) -> float:
    """
    Penalise responses that are too short.

    Very short responses tend to be low quality — the model gave up early.
    We linearly penalise responses shorter than min_words.

    Args:
        text:      candidate response text
        min_words: minimum acceptable word count (defaults to cfg.DPO_LENGTH_PENALTY_MIN)

    Returns:
        penalty in [0, min_words]; 0 if the response is long enough
    """
    min_words = min_words or cfg.DPO_LENGTH_PENALTY_MIN
    return max(0, min_words - len(text.split()))


def score_response(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    device: str,
) -> float:
    """
    Combined quality score for a candidate response.  LOWER = BETTER.

    score = perplexity + 2 × repetitions + 0.5 × length_penalty

    The weights (2, 0.5) are heuristic — repetition is penalised more heavily
    than length because repetitive output is more obviously bad.

    Args:
        model:     GPT model
        tokenizer: ByteLevelBPETokenizer
        text:      candidate response text
        device:    "cuda" or "cpu"

    Returns:
        scalar quality score (lower = better)
    """
    ppl    = perplexity_score(model, tokenizer, text, device)
    rep    = repetition_score(text)
    length = length_penalty(text)
    return ppl + 2.0 * rep + 0.5 * length


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    device: str,
    generate_fn,
    n: int = None,
) -> List[str]:
    """
    Sample N candidate responses for a given prompt.

    Uses high temperature (cfg.DPO_GEN_TEMPERATURE) and top-p sampling to get
    diverse candidates — we need variety so the best and worst are meaningfully
    different.

    Args:
        model:       SFT GPT model
        tokenizer:   ByteLevelBPETokenizer
        prompt:      formatted instruction prompt (from extract_prompt)
        device:      "cuda" or "cpu"
        generate_fn: generation function (e.g. generation.sampler.generate)
        n:           number of candidates to generate (defaults to cfg.DPO_NUM_CANDIDATES)

    Returns:
        list of n decoded text strings
    """
    n            = n or cfg.DPO_NUM_CANDIDATES
    ids          = tokenizer.encode(prompt).ids
    x            = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    candidates   = []

    for _ in range(n):
        out  = generate_fn(
            model, x, tokenizer, cfg.BLOCK_SIZE,
            max_new_tokens=cfg.DPO_GEN_MAX_TOKENS,
            temperature=cfg.DPO_GEN_TEMPERATURE,
            top_p=cfg.DPO_GEN_TOP_P,
            top_k=cfg.DPO_GEN_TOP_K,
        )
        text = tokenizer.decode(out[0].tolist())
        candidates.append(text)

    return candidates


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dpo_dataset(
    model: torch.nn.Module,
    tokenizer,
    stories: List[str],
    device: str,
    generate_fn,
    num_candidates: int = None,
) -> List[Dict]:
    """
    Build a DPO preference dataset from a list of stories.

    For each story:
      1. Extract a prompt (first sentence + instruction wrapper).
      2. Generate num_candidates responses from the SFT model.
      3. Score each response with score_response().
      4. chosen  = lowest-score  (best) response
         rejected = highest-score (worst) response

    Args:
        model:          SFT GPT model (fine-tuned, not the base pretrained model)
        tokenizer:      ByteLevelBPETokenizer
        stories:        list of raw story text strings
        device:         "cuda" or "cpu"
        generate_fn:    generation function (generation.sampler.generate)
        num_candidates: responses per prompt (defaults to cfg.DPO_NUM_CANDIDATES)

    Returns:
        list of dicts, each with keys "prompt", "chosen", "rejected"
    """
    num_candidates = num_candidates or cfg.DPO_NUM_CANDIDATES
    model.eval()
    pairs = []

    for story in tqdm(stories, desc="Generating DPO pairs"):
        prompt     = extract_prompt(story)
        candidates = generate_candidates(model, tokenizer, prompt, device, generate_fn, n=num_candidates)
        scores     = [score_response(model, tokenizer, c, device) for c in candidates]

        chosen   = candidates[int(np.argmin(scores))]   # best (lowest score)
        rejected = candidates[int(np.argmax(scores))]   # worst (highest score)

        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return pairs


def save_dpo_dataset(pairs: List[Dict], path: str = None) -> None:
    """
    Append DPO pairs to a JSON Lines file.

    Using append mode ('a') lets you resume generation if it was interrupted —
    re-run and new pairs are added to the existing file.

    Args:
        pairs: list of {"prompt", "chosen", "rejected"} dicts
        path:  output file path (defaults to cfg.DPO_DATASET)
    """
    path = path or cfg.DPO_DATASET
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"Saved {len(pairs)} DPO pairs to {path}")
