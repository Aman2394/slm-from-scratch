"""
generation/sampler.py — Autoregressive text generation with sampling strategies.

Two generation functions are provided:

  generate()     — standard generation, reprocesses the full context every step.
                   Simple and correct; O(T²) compute per sequence.

  generate_kv()  — generation with KV cache; only processes the newest token
                   each step after the first.  O(T) compute per sequence.
                   Use this for inference once the model is trained.

Sampling strategies (applied in order within each step):
  1. Temperature scaling — divides logits by T before softmax.
                           T → 0: greedy (argmax).  T > 1: more random.
  2. Repetition penalty  — divides logits of already-seen tokens by a factor > 1,
                           making the model less likely to repeat itself.
  3. Top-k filtering     — zeros out all but the top-k probability mass,
                           preventing very low-probability tokens from being sampled.
  4. Top-p (nucleus)     — zeros out tokens outside the smallest set whose
                           cumulative probability exceeds p.  More adaptive than top-k.

Usage:
    from generation.sampler import generate, generate_kv
    from model import GPT, GPTWithKVCache, GPTConfig
    import config as cfg

    config   = GPTConfig()
    model    = GPT(config).to(device)
    model_kv = GPTWithKVCache(config).to(device)
    # ... load weights ...

    x = encode_prompt("Once upon a time", tokenizer, device)

    out    = generate(model,    x, tokenizer, cfg.BLOCK_SIZE)
    out_kv = generate_kv(model_kv, x, tokenizer, cfg.BLOCK_SIZE)

    print(tokenizer.decode(out[0].tolist()))
"""

import torch
import torch.nn.functional as F

import config as cfg


def _apply_sampling_filters(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    temperature: float,
    repetition_penalty: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """
    Apply temperature scaling, repetition penalty, top-k, and top-p in order.

    Args:
        logits:           (1, vocab_size) raw logits for the next token
        generated_ids:    (1, T_so_far) all token ids generated so far
        temperature:      > 0; lower → sharper distribution
        repetition_penalty: > 1 penalises repeated tokens; 1.0 = no penalty
        top_k:            keep only the top-k tokens; None = disabled
        top_p:            nucleus probability threshold; None = disabled

    Returns:
        filtered logits (1, vocab_size), ready to pass to softmax
    """
    # --- Greedy shortcut: skip filtering when temperature is 0 ---
    if temperature == 0:
        return logits  # caller uses argmax directly

    logits = logits / temperature

    # --- Repetition penalty: discourage repeating tokens already in context ---
    # We divide the logit (not multiply) so that high-logit repeated tokens
    # are penalised more aggressively than low-logit ones.
    if repetition_penalty != 1.0:
        for token_id in set(generated_ids[0].tolist()):
            logits[:, token_id] /= repetition_penalty

    # --- Top-k: zero out all tokens below the k-th highest logit ---
    if top_k is not None:
        top_values, _ = torch.topk(logits, top_k)
        logits[logits < top_values[:, [-1]]] = float("-inf")

    # --- Top-p (nucleus sampling): keep the smallest set of tokens whose
    #     cumulative probability mass exceeds p ---
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens after the nucleus threshold is crossed.
        # Shift right by one so we always keep at least one token.
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0]  = False

        # Scatter the removal mask back to the original (unsorted) order
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = float("-inf")

    return logits


def generate(
    model: torch.nn.Module,
    x: torch.Tensor,
    tokenizer,
    block_size: int = None,
    max_new_tokens: int     = None,
    temperature:    float   = None,
    top_k:          int     = None,
    top_p:          float   = None,
    repetition_penalty: float = None,
) -> torch.Tensor:
    """
    Autoregressive generation (no KV cache).

    At each step the full context is re-processed by the model. Simple but
    O(T²) in compute — use generate_kv() for faster inference.

    Args:
        model:              GPT model (from model/gpt.py), already on device
        x:                  (1, T_prompt) initial token ids
        tokenizer:          ByteLevelBPETokenizer — used to get the EOS token id
        block_size:         maximum context length (defaults to cfg.BLOCK_SIZE)
        max_new_tokens:     tokens to generate (defaults to cfg.GEN_MAX_NEW_TOKENS)
        temperature:        sampling temperature (defaults to cfg.GEN_TEMPERATURE)
        top_k:              top-k filter (defaults to cfg.GEN_TOP_K)
        top_p:              nucleus threshold (defaults to cfg.GEN_TOP_P)
        repetition_penalty: repetition penalty (defaults to cfg.GEN_REPETITION_PENALTY)

    Returns:
        (1, T_prompt + max_new_tokens) token id tensor (generation stops early at EOS)
    """
    block_size          = block_size          or cfg.BLOCK_SIZE
    max_new_tokens      = max_new_tokens      or cfg.GEN_MAX_NEW_TOKENS
    temperature         = temperature         if temperature is not None else cfg.GEN_TEMPERATURE
    top_k               = top_k               if top_k is not None       else cfg.GEN_TOP_K
    top_p               = top_p               if top_p is not None       else cfg.GEN_TOP_P
    repetition_penalty  = repetition_penalty  if repetition_penalty is not None else cfg.GEN_REPETITION_PENALTY

    eos_id = tokenizer.token_to_id("</s>")
    model.eval()

    for _ in range(max_new_tokens):

        # Truncate context to block_size if it has grown beyond it
        x_cond = x[:, -block_size:]

        with torch.inference_mode():
            logits = model(x_cond)          # (1, T, vocab_size)
        logits = logits[:, -1, :]           # logits for the next token: (1, vocab_size)

        logits = _apply_sampling_filters(
            logits, x, temperature, repetition_penalty, top_k, top_p
        )

        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eos_id:
            break

        x = torch.cat([x, next_token], dim=1)

    return x


def generate_kv(
    model: torch.nn.Module,
    x: torch.Tensor,
    tokenizer,
    block_size: int = None,
    max_new_tokens: int     = None,
    temperature:    float   = None,
    top_k:          int     = None,
    top_p:          float   = None,
    repetition_penalty: float = None,
) -> torch.Tensor:
    """
    Autoregressive generation with KV cache (fast inference).

    On the first step the full prompt is processed to populate the cache.
    On every subsequent step only the single newest token is passed through
    the model — making generation significantly faster for long sequences.

    Requires a model from model/gpt_kv.py (GPTWithKVCache).

    Args:
        model:              GPTWithKVCache model, already on device
        x:                  (1, T_prompt) initial token ids
        tokenizer:          ByteLevelBPETokenizer
        block_size:         maximum context length (defaults to cfg.BLOCK_SIZE)
        max_new_tokens:     tokens to generate (defaults to cfg.GEN_MAX_NEW_TOKENS)
        temperature:        sampling temperature (defaults to cfg.GEN_TEMPERATURE)
        top_k:              top-k filter (defaults to cfg.GEN_TOP_K)
        top_p:              nucleus threshold (defaults to cfg.GEN_TOP_P)
        repetition_penalty: repetition penalty (defaults to cfg.GEN_REPETITION_PENALTY)

    Returns:
        (1, T_prompt + T_generated) token id tensor
    """
    block_size          = block_size          or cfg.BLOCK_SIZE
    max_new_tokens      = max_new_tokens      or cfg.GEN_MAX_NEW_TOKENS
    temperature         = temperature         if temperature is not None else cfg.GEN_TEMPERATURE
    top_k               = top_k               if top_k is not None       else cfg.GEN_TOP_K
    top_p               = top_p               if top_p is not None       else cfg.GEN_TOP_P
    repetition_penalty  = repetition_penalty  if repetition_penalty is not None else cfg.GEN_REPETITION_PENALTY

    eos_id   = tokenizer.token_to_id("</s>")
    past_kv  = None
    model.eval()

    for _ in range(max_new_tokens):

        # First step: pass the full prompt to populate the KV cache.
        # Subsequent steps: pass only the newest single token.
        x_input = x if past_kv is None else x[:, -1:]

        with torch.inference_mode():
            logits, past_kv = model(x_input, past_kv)
        logits = logits[:, -1, :]  # (1, vocab_size)

        logits = _apply_sampling_filters(
            logits, x, temperature, repetition_penalty, top_k, top_p
        )

        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eos_id:
            break

        x = torch.cat([x, next_token], dim=1)

    return x


def encode_prompt(prompt: str, tokenizer, device: str) -> torch.Tensor:
    """
    Convenience helper: encode a text prompt into a (1, T) token tensor.

    Args:
        prompt:    raw text string
        tokenizer: ByteLevelBPETokenizer
        device:    "cuda" or "cpu"

    Returns:
        (1, T) int64 tensor on the specified device
    """
    ids = tokenizer.encode(prompt).ids
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
