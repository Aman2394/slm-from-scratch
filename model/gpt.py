"""
model/gpt.py — GPT-style transformer (base model, no KV cache).

Architecture overview:
    token embedding + positional embedding
    → N transformer blocks (LayerNorm → Attention → residual,
                            LayerNorm → MLP → residual)
    → final LayerNorm
    → linear LM head (projects back to vocab_size)

Each transformer block uses pre-norm (LayerNorm before attention/MLP),
which is more stable to train than the original post-norm GPT-1 design.

Usage:
    from model.gpt import GPT, GPTConfig
    config = GPTConfig()
    model = GPT(config).to(device)
    logits = model(idx)          # idx: (B, T) token ids → logits: (B, T, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class GPTConfig:
    """
    Holds all model architecture hyperparameters.

    Reads defaults from config.py so there is a single source of truth.
    Override individual attributes after construction for one-off experiments
    (but never edit config.py for that — override in the notebook instead).

    Example:
        c = GPTConfig()
        c.n_layer = 4   # smaller model for quick testing
    """

    def __init__(self):
        self.vocab_size  = cfg.VOCAB_SIZE   # number of BPE tokens
        self.block_size  = cfg.BLOCK_SIZE   # maximum context length (tokens)
        self.n_layer     = cfg.N_LAYER      # number of transformer blocks
        self.n_head      = cfg.N_HEAD       # number of attention heads
        self.n_embd      = cfg.N_EMBD       # embedding / hidden dimension
        self.dropout     = cfg.DROPOUT      # dropout probability


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    'Causal' means each token can only attend to itself and tokens that came
    before it — never to future tokens. This is enforced with a lower-triangular
    mask applied before the softmax.

    The QKV projection is fused into a single Linear(n_embd, 3*n_embd) for
    efficiency, then split into three (B, n_head, T, head_dim) tensors.

    Args:
        config: GPTConfig instance.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Fused QKV projection: one matrix instead of three saves memory & compute
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection: brings the concatenated heads back to n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)  — batch, sequence length, embedding dim

        Returns:
            out: (B, T, C) — same shape as input, after self-attention
        """
        B, T, C = x.size()  # e.g. (4, 256, 512)

        # Split the fused QKV projection → three (B, T, C) tensors
        q, k, v = self.c_attn(x).split(C, dim=2)

        # Reshape to (B, n_head, T, head_dim) for batched attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_head, T, T)
        # Dividing by sqrt(head_dim) prevents gradients from vanishing at init
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask: set future positions to -inf so softmax gives them 0 weight
        mask = torch.tril(torch.ones(T, T, device=x.device))
        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)  # (B, n_head, T, T)

        # Weighted sum of values: (B, n_head, T, head_dim)
        y = att @ v

        # Merge heads back: (B, n_head, T, head_dim) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)  # final linear projection: (B, T, C)


# ---------------------------------------------------------------------------
# Feed-forward (MLP)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear layers with a GELU activation in between.
    The inner dimension is 4× the embedding dimension — this ratio comes from
    the original Transformer paper and has remained standard since.

    Input shape:  (B, T, C)
    Output shape: (B, T, C)  — same as input (MLP acts independently per token)

    Args:
        config: GPTConfig instance.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # Expand to 4× then contract back — the "bottleneck" structure
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # expand
            nn.GELU(),                                      # smooth non-linearity
            nn.Linear(4 * config.n_embd, config.n_embd),  # contract
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    One transformer block = pre-norm + attention + residual,
                            followed by pre-norm + MLP + residual.

    Pre-norm (LayerNorm applied before the sub-layer, not after) is the
    standard in modern LLMs because it makes training more stable — gradients
    flow more easily through the residual stream.

    Args:
        config: GPTConfig instance.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1  = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2  = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
        Returns:
            x: (B, T, C) — residual stream after attention + MLP
        """
        # Attention sub-layer: normalize → attend → add residual
        x = x + self.attn(self.ln1(x))
        # MLP sub-layer: normalize → transform → add residual
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full GPT model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT-style language model.

    Embeds token ids and positions, passes them through N transformer blocks,
    applies a final LayerNorm, then projects to vocabulary logits.

    The model predicts the next token at every position, so during training
    we compute cross-entropy loss against the shifted target sequence:
        input:  [t0, t1, t2, ..., t_{T-1}]
        target: [t1, t2, t3, ..., t_T    ]

    Args:
        config: GPTConfig instance.

    Inputs:
        idx: (B, T) — integer token ids

    Outputs:
        logits: (B, T, vocab_size)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token embedding: maps each token id to a dense vector
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Positional embedding: learned position encodings (absolute, like GPT-2)
        self.pos_emb   = nn.Embedding(config.block_size, config.n_embd)

        # Stack of N transformer blocks
        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )

        # Final layer norm before the LM head (pre-norm style)
        self.ln_f = nn.LayerNorm(config.n_embd)

        # LM head: project from n_embd → vocab_size (no bias, following GPT-2)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (B, T) integer token ids; T must be ≤ block_size

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        # Token embeddings: (B, T) → (B, T, n_embd)
        tok = self.token_emb(idx)
        # Positional embeddings: positions [0, 1, ..., T-1] → (T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))

        # Add token + position embeddings (broadcasted over batch)
        x = tok + pos  # (B, T, n_embd)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
