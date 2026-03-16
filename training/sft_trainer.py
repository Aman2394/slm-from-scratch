"""
training/sft_trainer.py — Supervised Fine-Tuning (SFT) training loop.

SFT continues from the pretrained checkpoint and teaches the model to follow
instructions. The key differences from pretraining:

  1. Lower learning rate (3e-5 vs 3e-4) — we are fine-tuning, not training
     from scratch. Too high an LR would destroy the knowledge from pretraining.

  2. Response masking — loss is computed only on response tokens (see
     training/sft_data_utils.py for details).

  3. Loads the pretrained model as the starting point.

Usage (in a Colab notebook):
    from training.sft_trainer import train_sft
    import config as cfg

    train_sft(model, tokenizer, samples, device)
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
from training.sft_data_utils import get_sft_batch


def train_sft(
    model,
    tokenizer,
    samples: list,
    device: str,
    # ---- hyperparams (override in notebook; do NOT edit config.py) ----
    lr:            float = None,
    weight_decay:  float = None,
    batch_size:    int   = None,
    max_steps:     int   = None,
    eval_interval: int   = None,
    save_interval: int   = None,
    # ---- path overrides ----
    checkpoint_path: str = None,
    final_path:      str = None,
) -> None:
    """
    SFT training loop.

    Resumes from checkpoint_path if it exists.
    Saves a rolling checkpoint every save_interval steps.
    Saves final model weights to final_path at the end.

    Args:
        model:           pretrained GPT, already on device (load PRETRAIN_FINAL_CKPT first)
        tokenizer:       ByteLevelBPETokenizer (needed by get_sft_batch for masking)
        samples:         tokenised SFT samples from load_sft_tokens()
        device:          "cuda" or "cpu"
        lr:              learning rate (default: cfg.SFT_LR)
        weight_decay:    AdamW weight decay (default: cfg.SFT_WEIGHT_DECAY)
        batch_size:      sequences per step (default: cfg.SFT_BATCH_SIZE)
        max_steps:       total gradient updates (default: cfg.SFT_MAX_STEPS)
        eval_interval:   print loss every N steps (default: cfg.SFT_EVAL_INTERVAL)
        save_interval:   save checkpoint every N steps (default: cfg.SFT_SAVE_INTERVAL)
        checkpoint_path: rolling checkpoint path (default: cfg.SFT_CKPT)
        final_path:      final model save path (default: cfg.SFT_FINAL_CKPT)
    """
    # ---- resolve defaults ----
    lr              = lr             or cfg.SFT_LR
    weight_decay    = weight_decay   or cfg.SFT_WEIGHT_DECAY
    batch_size      = batch_size     or cfg.SFT_BATCH_SIZE
    max_steps       = max_steps      or cfg.SFT_MAX_STEPS
    eval_interval   = eval_interval  or cfg.SFT_EVAL_INTERVAL
    save_interval   = save_interval  or cfg.SFT_SAVE_INTERVAL
    checkpoint_path = checkpoint_path or cfg.SFT_CKPT
    final_path      = final_path      or cfg.SFT_FINAL_CKPT

    block_size = model.config.block_size

    # ---- optimizer ----
    # Lower LR than pretraining — we want to nudge the model toward instruction
    # following without catastrophically forgetting its language knowledge
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ---- resume from checkpoint ----
    start_step = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming SFT from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"  → resuming from step {start_step}")

    # ---- training loop ----
    model.train()
    print(f"SFT: training for {max_steps - start_step} steps "
          f"(steps {start_step} → {max_steps})")

    for step in tqdm(range(start_step, max_steps)):

        x, y = get_sft_batch(samples, tokenizer, block_size, batch_size, device)

        logits = model(x)  # (B, T, vocab_size)

        # ignore_index=-100 skips instruction tokens in the loss computation
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:>6} | sft loss {loss.item():.4f}")

        # ---- rolling checkpoint ----
        if step % save_interval == 0:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step":      step,
                },
                checkpoint_path,
            )

    # ---- save final weights only (no optimizer state needed for inference) ----
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save({"model": model.state_dict()}, final_path)
    print(f"SFT complete. Final model saved to {final_path}")
