"""
model/gpt_kv.py — GPT with KV cache for efficient autoregressive inference.

Why KV cache?
    In standard generation (model/gpt.py), every new token requires a full
    forward pass over the entire sequence so far — O(T²) work per step.
    With a KV cache, we store the key/value tensors from previous steps and
    only process the *new* token at each step — O(T) work per step.

    This gives a significant speedup for long sequences. Memory cost is
    O(n_layer × T × n_embd) for the cache, which is well within T4 VRAM for
    our block_size of 256.

Interface difference vs. gpt.py:
    - forward(idx, past_kv) returns (logits, new_past_kv) instead of just logits
    - During generation, pass idx[:, -1:] (just the last token) after the first step

Usage:
    from model.gpt_kv import GPT as GPTWithKVCache
    from model.gpt   import GPTConfig

    config   = GPTConfig()
    model_kv = GPTWithKVCache(config).to(device)

    # Load weights trained with the base GPT (architectures are identical)
    model_kv.load_state_dict(torch.load(cfg.PRETRAIN_FINAL_CKPT))

    # Generation loop (see generation/sampler.py for the full implementation)
    past_kv = None
    for _ in range(max_new_tokens):
        x_in = x if past_kv is None else x[:, -1:]  # full prompt first, then one token
        logits, past_kv = model_kv(x_in, past_kv)
        next_token = sample(logits[:, -1, :])
        x = torch.cat([x, next_token], dim=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from model.gpt import GPTConfig, MLP  # MLP is identical in both variants


# ---------------------------------------------------------------------------
# Attention with KV cache
# ---------------------------------------------------------------------------

class CausalSelfAttentionKV(nn.Module):
    """
    Multi-head causal self-attention with KV cache support.

    At each generation step, instead of recomputing keys/values for the entire
    sequence, we:
      1. Compute Q, K, V only for the *new* token(s).
      2. Concatenate the new K, V with the cached keys/values from past steps.
      3. Run attention over the full (cached + new) sequence.
      4. Return the updated KV pair so the caller can cache it.

    Args:
        config: GPTConfig instance.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:        (B, T_new, C)   — T_new=1 during generation, T_new=T at prefill
            past_kv:  tuple(k, v) where k, v are (B, n_head, T_past, head_dim)
                      or None on the first call.

        Returns:
            out:     (B, T_new, C)
            present: tuple(k, v) with shape (B, n_head, T_past + T_new, head_dim)
                     — store this for the next step
        """
        B, T_new, C = x.size()

        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_new, self.n_head, self.head_dim).transpose(1, 2)

        # Append current K, V to the cache (if any)
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, n_head, T_past + T_new, head_dim)
            v = torch.cat([past_v, v], dim=2)

        # present holds the *full* K, V sequence for caching
        present = (k, v)

        # Total sequence length in K/V (past + new)
        T_full = k.size(2)

        # Scaled dot-product attention over the full sequence
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, n_head, T_new, T_full)

        # Causal mask: each of the T_new queries can attend to all T_full keys
        # up to and including its own position.
        # We take the bottom-right T_new × T_full slice of the full lower-triangular mask.
        mask = torch.tril(torch.ones(T_full, T_full, device=x.device))
        mask = mask[-T_new:, :].view(1, 1, T_new, T_full)
        att  = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        y = att @ v                                     # (B, n_head, T_new, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T_new, C)

        return self.c_proj(y), present


# ---------------------------------------------------------------------------
# Block with KV cache
# ---------------------------------------------------------------------------

class BlockKV(nn.Module):
    """
    Transformer block that threads a KV cache through attention.

    Args:
        config: GPTConfig instance.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttentionKV(config)

        self.ln2  = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:       (B, T_new, C)
            past_kv: cached (k, v) from previous steps, or None

        Returns:
            x:       (B, T_new, C)
            present: updated (k, v) cache
        """
        attn_out, present = self.attn(self.ln1(x), past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


# ---------------------------------------------------------------------------
# Full GPT model with KV cache
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT language model with KV cache.

    The model weights are identical to model/gpt.py — you can load a
    checkpoint trained with the base GPT directly into this model.

    Args:
        config: GPTConfig instance.

    Inputs:
        idx:     (B, T) integer token ids
        past_kv: list of per-layer (k, v) caches, or None for the first call

    Outputs:
        logits:  (B, T, vocab_size)
        new_past: updated list of per-layer (k, v) caches — pass back in on
                  the next generation step
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb   = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList(
            [BlockKV(config) for _ in range(config.n_layer)]
        )

        self.ln_f    = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        past_kv: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            idx:     (B, T) — T=full sequence on first call, T=1 on subsequent calls
            past_kv: list of length n_layer; each element is (k, v) or None

        Returns:
            logits:   (B, T, vocab_size)
            new_past: list of length n_layer, each element is updated (k, v)
        """
        B, T = idx.shape

        # Initialise cache list if this is the first call
        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        tok = self.token_emb(idx)

        # Compute the correct absolute positions.
        # If the cache is populated, our new tokens start at past_len.
        past_len = past_kv[0][0].size(2) if past_kv[0] is not None else 0
        positions = torch.arange(past_len, past_len + T, device=idx.device)
        # Wrap around block_size to stay within the positional embedding table
        pos = self.pos_emb(positions % self.config.block_size)

        x = tok + pos

        new_past = []
        for block, layer_past in zip(self.blocks, past_kv):
            x, present = block(x, layer_past)
            new_past.append(present)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_past
