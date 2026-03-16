"""
model/ — GPT architecture module.

Exports:
    GPTConfig         — model hyperparameters (reads defaults from config.py)
    GPT               — base GPT model (no KV cache), returns logits
    GPTWithKVCache    — GPT with KV cache for fast inference, returns (logits, past_kv)
"""

from model.gpt    import GPT, GPTConfig
from model.gpt_kv import GPT as GPTWithKVCache

__all__ = ["GPT", "GPTConfig", "GPTWithKVCache"]
