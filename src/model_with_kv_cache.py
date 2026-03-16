import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, past_kv=None):

        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ---- KV Cache Handling ----
        if past_kv is not None:

            past_k, past_v = past_kv

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present = (k, v)

        # ---- Attention ----
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        seq_len = k.size(-2)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        mask = mask[-T:, :].view(1, 1, T, seq_len)
        
        # mask = torch.tril(
        #     torch.ones(T, seq_len, device=x.device)
        # ).view(1, 1, T, seq_len)

        att = att.masked_fill(mask == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y, present


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
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

    def forward(self, x, past_kv=None):

        ln1_out = self.ln1(x)

        attn_out, present = self.attn(ln1_out, past_kv)

        x = x + attn_out

        ln2_out = self.ln2(x)

        mlp_out = self.mlp(ln2_out)

        x = x + mlp_out

        return x, present


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, past_kv=None):

        B, T = idx.shape

        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        tok = self.token_emb(idx)

        if past_kv[0] is not None:
            past_len = past_kv[0][0].size(2)
        else:
            past_len = 0

        pos = self.pos_emb(
            torch.arange(past_len, past_len + T, device=idx.device) % self.config.block_size
        )
        # pos = self.pos_emb(
        #     torch.arange(T, device=idx.device)
        # )

        x = tok + pos

        new_past = []

        for block, past in zip(self.blocks, past_kv):

            x, present = block(x, past)

            new_past.append(present)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        return logits, new_past