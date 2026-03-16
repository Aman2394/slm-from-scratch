import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):

        B, T, C = x.size() # Input: (batch_size, sequence_length, embedding_dimension) e.g., (4, 256, 512)

        q, k, v = self.c_attn(x).split(C, dim=2) # Input: (B, T, C), Output: three tensors of (B, T, C) eg., (4, 256, 512)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # Input: (B, T, C), Output: (B, n_head, T, head_dimension) e.g., (4, 8, 256, 64)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # Input: (B, T, C), Output: (B, n_head, T, head_dimension) e.g., (4, 8, 256, 64)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # Input: (B, T, C), Output: (B, n_head, T, head_dimension) e.g., (4, 8, 256, 64)

        att = (q @ k.transpose(-2,-1)) / (k.size(-1) ** 0.5) # Input q: (B, n_head, T, head_dimension), Input k.transpose: (B, n_head, head_dimension, T), Output: (B, n_head, T, T) (4, 8, 256, 256)

        mask = torch.tril(torch.ones(T, T, device=x.device)) # Output: (T, T) lower triangular mask

        att = att.masked_fill(mask == 0, float('-inf')) # Input/Output: (B, n_head, T, T) with masked values set to -inf

        att = F.softmax(att, dim=-1) # Input/Output: (B, n_head, T, T) with softmax applied on the last dimension eg., (4, 8, 256, 256)

        y = att @ v # Input att: (B, n_head, T, T), Input v: (B, n_head, T, head_dimension), Output: (B, n_head, T, head_dimension) eg., (4, 8, 256, 64)

        y = y.transpose(1,2).contiguous().view(B, T, C) # Input: (B, n_head, T, head_dimension), Output: (B, T, C) eg., (4, 256, 64*8 = 512)

        return self.c_proj(y) # Input: (B, T, C), Output: (B, T, C)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        ln1_out = self.ln1(x) # (4,256, 512) -> (4,256, 512)
        attn_out = self.attn(ln1_out) # (4,256, 512) -> (4,256, 512)
        x = x + attn_out
        ln2_out = self.ln2(x)
        mlp_out = self.mlp(ln2_out)
        x = x + mlp_out

        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx):

        B, T = idx.shape # batch size, context length (4,256)

        tok = self.token_emb(idx) # (4,256) -> (4, 256, 512)
        pos = self.pos_emb(torch.arange(T, device=idx.device)) # [0,1,2,...,255] -> (256,512)

        x = tok + pos # (4, 256, 512) -> (4, 256, 512)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits