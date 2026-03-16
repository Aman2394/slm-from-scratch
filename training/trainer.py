"""
training/trainer.py — Pretraining loop for the GPT language model.

Implements a simple but complete training loop:
  - AdamW optimizer with configurable lr and weight decay
  - Checkpoint saving/resuming (to Drive, via config.py paths)
  - Periodic validation loss estimation
  - Logging to stdout every 50 steps

All hyperparameters default to values in config.py. Override them by passing
keyword arguments to `train()` — do this in notebooks, not here.

Usage (in a Colab notebook):
    from training.trainer import train
    import config as cfg

    train(model, device)                          # use all config.py defaults
    train(model, device, max_steps=5000, lr=1e-4) # override for one run
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import config as cfg
from training.data_utils import load_tokens, get_batch


# ---------------------------------------------------------------------------
# Loss estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    device: str,
    block_size: int,
    batch_size: int,
    num_batches: int = None,
) -> float:
    """
    Estimate mean validation loss over several random batches.

    We temporarily set the model to eval mode (disables dropout) and use
    torch.no_grad() to skip building the computation graph — saving memory
    and time during evaluation.

    Args:
        model:       the GPT model
        device:      "cuda" or "cpu"
        block_size:  context length
        batch_size:  sequences per batch
        num_batches: how many batches to average over (default: cfg.PRETRAIN_EVAL_BATCHES)

    Returns:
        mean cross-entropy loss over num_batches validation batches
    """
    if num_batches is None:
        num_batches = cfg.PRETRAIN_EVAL_BATCHES

    val_data = load_tokens("val")

    model.eval()
    losses = []

    for _ in range(num_batches):
        x, y = get_batch(val_data, block_size, batch_size, device)
        logits = model(x)                          # (B, T, vocab_size)
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),      # (B*T, vocab_size)
            y.view(-1),                            # (B*T,)
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    device: str,
    # ---- training hyperparams (override in notebook; do NOT edit config.py) ----
    lr:             float = None,
    weight_decay:   float = None,
    batch_size:     int   = None,
    max_steps:      int   = None,
    eval_interval:  int   = None,
    save_interval:  int   = None,
    # ---- path overrides (rarely needed) ----
    checkpoint_path: str  = None,
    final_path:      str  = None,
) -> None:
    """
    Pretraining loop: train the model on the pretraining token file.

    Resumes automatically if a checkpoint exists at checkpoint_path.

    Saves a rolling checkpoint every save_interval steps (overwrites the
    previous checkpoint — this is intentional to save Drive space).
    Saves the final model weights to final_path at the end of training.

    Args:
        model:           GPT model instance, already moved to device
        device:          "cuda" or "cpu"
        lr:              peak learning rate for AdamW
        weight_decay:    L2 regularisation coefficient
        batch_size:      sequences per gradient step
        max_steps:       total number of gradient updates
        eval_interval:   evaluate validation loss every N steps
        save_interval:   save rolling checkpoint every N steps
        checkpoint_path: where to save/load the rolling checkpoint
        final_path:      where to save the final model weights

    Returns:
        None  (model is modified in-place)
    """
    # ---- resolve defaults from config.py ----
    lr              = lr             or cfg.PRETRAIN_LR
    weight_decay    = weight_decay   or cfg.PRETRAIN_WEIGHT_DECAY
    batch_size      = batch_size     or cfg.PRETRAIN_BATCH_SIZE
    max_steps       = max_steps      or cfg.PRETRAIN_MAX_STEPS
    eval_interval   = eval_interval  or cfg.PRETRAIN_EVAL_INTERVAL
    save_interval   = save_interval  or cfg.PRETRAIN_SAVE_INTERVAL
    checkpoint_path = checkpoint_path or cfg.PRETRAIN_CKPT
    final_path      = final_path      or cfg.PRETRAIN_FINAL_CKPT

    block_size = model.config.block_size

    # ---- load training data ----
    train_data = load_tokens("train")

    # ---- optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # ---- resume from checkpoint if one exists ----
    start_step = 0
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"  → resuming from step {start_step}")

    # ---- training loop ----
    model.train()
    print(f"Training for {max_steps - start_step} steps "
          f"(steps {start_step} → {max_steps})")

    for step in tqdm(range(start_step, max_steps)):

        x, y = get_batch(train_data, block_size, batch_size, device)

        logits = model(x)
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- stdout logging ----
        if step % 50 == 0:
            print(f"step {step:>6} | train loss {loss.item():.4f}")

        # ---- validation ----
        if step % eval_interval == 0:
            val_loss = estimate_loss(model, device, block_size, batch_size)
            print(f"step {step:>6} | val loss   {val_loss:.4f}")

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

    # ---- save final weights ----
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")
