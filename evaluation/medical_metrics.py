"""
evaluation/medical_metrics.py — Domain-specific evaluation metrics for the Medical SLM.

Why domain-specific evaluation?
    Perplexity measures how well the model predicts the next token, but for a
    medical Q&A model we care about a different question: does the model select
    the *correct* answer when given a multiple-choice question?  This file
    provides accuracy-based evaluation tailored to the USMLE / MedQA format.

Metrics implemented:
    evaluate_mcq_accuracy   : multiple-choice accuracy on a list of medqa examples
    evaluate_medical_perplexity : perplexity on the medical token split
                                  (wraps the general evaluate_perplexity)
    run_usmle_benchmark     : end-to-end benchmark that prints a comparison table
                              across model checkpoints (base → DAPT → SFT → DPO)

How multiple-choice scoring works:
    Given a question + options A/B/C/D, we feed each full "question + option"
    string through the model and compare the model's log-probability for each
    option.  The model picks whichever option has the highest log-probability.
    This is called *likelihood scoring* and avoids needing the model to generate
    free text — useful for small models that are not yet good at instruction following.

Usage:
    from evaluation.medical_metrics import evaluate_mcq_accuracy, run_usmle_benchmark
    acc = evaluate_mcq_accuracy(model, tokenizer, device, examples)
    print(f"MCQ accuracy: {acc:.1%}")
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

import config as cfg


# ---------------------------------------------------------------------------
# Helper: log-probability scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def _score_text(
    model: torch.nn.Module,
    token_ids: List[int],
    device: str,
) -> float:
    """
    Compute the mean per-token log-probability of a sequence under the model.

    We use mean (not sum) so that shorter and longer options are comparable
    on the same scale — without this, the model always prefers shorter answers.

    Args:
        model:     GPT model, already on device
        token_ids: List of integer token IDs representing the full text
        device:    "cuda" or "cpu"

    Returns:
        mean log-probability (higher = model considers the text more likely)
    """
    if len(token_ids) < 2:
        return -float("inf")

    block_size = model.config.block_size
    ids = token_ids[:block_size]  # truncate to context window

    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)  # (1, T-1)
    y = torch.tensor(ids[1:],  dtype=torch.long, device=device)                # (T-1,)

    logits = model(x)                                                          # (1, T-1, V)
    log_probs = F.log_softmax(logits[0], dim=-1)                               # (T-1, V)

    # Gather the log-prob of each actual next token
    token_log_probs = log_probs[range(len(y)), y]                              # (T-1,)
    return token_log_probs.mean().item()


# ---------------------------------------------------------------------------
# Multiple-choice accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mcq_accuracy(
    model: torch.nn.Module,
    tokenizer,
    device: str,
    examples: List[Dict],
    max_examples: Optional[int] = None,
    verbose: bool = True,
) -> float:
    """
    Evaluate multiple-choice accuracy using likelihood scoring.

    For each example we score every answer option (A–E) by computing the mean
    log-probability of (question + options + answer) and pick the highest.
    We then compare the predicted letter to the ground-truth letter.

    This approach does NOT require the model to generate free text — it only
    requires a forward pass per option, which works even for base models that
    have not been SFT-trained yet.

    Dataset format (medalpaca/medical_meadow_medqa):
        input  : "Q: <question>? {'A': 'opt1', ..., 'E': 'opt5'}"
        output : "C: Cortical laminar necrosis"
    MedQA has 5 options (A–E) → random baseline = 20%.

    Args:
        model:        GPT model, already on device and in eval mode
        tokenizer:    ByteLevelBPETokenizer loaded with the medical vocab
        device:       "cuda" or "cpu"
        examples:     List of raw medqa example dicts
        max_examples: If set, evaluate on only the first N examples (faster)
        verbose:      If True, print per-example details for first 5 examples

    Returns:
        accuracy in [0, 1]
    """
    from loaders.medical import _parse_medqa_input

    model.eval()
    correct = 0
    total   = 0

    if max_examples is not None:
        examples = examples[:max_examples]

    for i, ex in enumerate(examples):
        raw_input = ex.get("input", "")
        answer    = ex.get("output", "").strip()   # e.g. "C: Cortical laminar necrosis"

        # Extract the correct answer letter ("C" from "C: Cortical laminar necrosis")
        correct_letter = answer.split(":")[0].strip().upper() if ":" in answer else ""
        if not correct_letter:
            continue

        # Parse question text and options dict from the input field
        question, options = _parse_medqa_input(raw_input)
        if not options:
            continue

        # Score each option: log-prob of the full (question + formatted answer) text
        # MedQA has 5 options (A-E), so random baseline = 20%
        option_letters = sorted(options.keys())
        scores = []
        for letter in option_letters:
            opt_text  = options[letter]
            full_text = (
                f"### Question:\n{question}\n\n"
                f"### Options:\n"
                + "\n".join(f"{l}. {options[l]}" for l in option_letters)
                + f"\n\n### Answer:\n{letter}: {opt_text}"
            )
            ids   = tokenizer.encode(full_text).ids
            score = _score_text(model, ids, device)
            scores.append(score)

        # Pick the letter with the highest log-probability
        pred_idx    = int(torch.tensor(scores).argmax().item())
        pred_letter = option_letters[pred_idx]
        is_correct  = (pred_letter == correct_letter)

        if is_correct:
            correct += 1
        total += 1

        if verbose and i < 5:
            print(f"\nExample {i+1}:")
            print(f"  Q: {question[:100].strip()}...")
            for j, letter in enumerate(option_letters):
                pred_mark    = "<-- predicted" if j == pred_idx else ""
                correct_mark = "(correct)"     if letter == correct_letter else ""
                print(f"  {letter}. {options[letter][:55]:<55}  score={scores[j]:.3f} {pred_mark}{correct_mark}")
            print(f"  Correct: {is_correct}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nMCQ Accuracy: {correct}/{total} = {accuracy:.1%}  (random baseline = 20.0%)")
    return accuracy


# ---------------------------------------------------------------------------
# Medical perplexity (thin wrapper around medical token split)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_medical_perplexity(
    model: torch.nn.Module,
    device: str,
    split: str = "val",
    num_batches: int = 50,
    batch_size: int = 16,
) -> Tuple[float, float]:
    """
    Compute perplexity on the medical textbook token splits.

    Uses memory-mapped numpy arrays at cfg.MED_VAL_TOKENS / cfg.MED_TEST_TOKENS.
    This measures how well the model predicts medical *text* (domain fit),
    which is complementary to MCQ accuracy (task performance).

    Args:
        model:       GPT model, already on device
        device:      "cuda" or "cpu"
        split:       "val" or "test" (must have been tokenized in notebook 11)
        num_batches: number of random batches to average
        batch_size:  sequences per batch

    Returns:
        (mean_loss, perplexity)
    """
    token_path = cfg.MED_VAL_TOKENS if split == "val" else cfg.MED_TEST_TOKENS
    data       = np.memmap(token_path, dtype=np.uint16, mode="r")
    block_size = model.config.block_size

    model.eval()
    losses = []

    for _ in range(num_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x  = torch.stack([
            torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
        ]).to(device)
        y  = torch.stack([
            torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix
        ]).to(device)

        logits = model(x)
        loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

    mean_loss  = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss)
    bpb        = mean_loss / math.log(2)

    print(f"[medical {split}] loss: {mean_loss:.4f} | ppl: {perplexity:.2f} | bpb: {bpb:.4f}")
    return mean_loss, perplexity


# ---------------------------------------------------------------------------
# End-to-end USMLE benchmark across checkpoints
# ---------------------------------------------------------------------------

def run_usmle_benchmark(
    checkpoints: Dict[str, str],
    model_factory,
    tokenizer,
    device: str,
    examples: List[Dict],
    max_examples: int = 200,
) -> Dict[str, float]:
    """
    Run MCQ accuracy evaluation across multiple model checkpoints and print a
    comparison table.

    This is the main result to include in your HuggingFace model card and resume —
    it shows how each training stage (DAPT, SFT, DPO) improves medical Q&A accuracy
    compared to a random 25% baseline.

    Args:
        checkpoints:   Dict mapping stage name → checkpoint path.
                       E.g. {"tinystories": cfg.PRETRAIN_FINAL_CKPT,
                              "dapt":       cfg.MED_DAPT_FINAL_CKPT,
                              "sft":        cfg.MED_SFT_FINAL_CKPT,
                              "dpo":        cfg.MED_DPO_FINAL_CKPT}
        model_factory: Callable() → GPT model instance (uninitialised weights).
                       We call this fresh for each checkpoint to avoid leaking state.
        tokenizer:     ByteLevelBPETokenizer loaded with the medical vocab.
        device:        "cuda" or "cpu"
        examples:      List of raw medqa example dicts (held-out test set).
        max_examples:  How many examples to evaluate per checkpoint (for speed).

    Returns:
        Dict mapping stage name → accuracy float
    """
    results: Dict[str, float] = {}

    print("=" * 55)
    print(f"  USMLE Benchmark  (random baseline = 20.0%)")
    print("=" * 55)

    for stage, ckpt_path in checkpoints.items():
        print(f"\n[{stage}]  loading {ckpt_path} ...")
        try:
            model = model_factory().to(device)
            state = torch.load(ckpt_path, map_location=device)
            # Handle both raw state_dict and checkpoint dicts with a "model" key
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            model.load_state_dict(state, strict=False)
            model.eval()

            acc = evaluate_mcq_accuracy(
                model, tokenizer, device, examples,
                max_examples=max_examples,
                verbose=False,
            )
            results[stage] = acc
        except FileNotFoundError:
            print(f"  checkpoint not found — skipping {stage}")
            results[stage] = float("nan")
        finally:
            # Free GPU memory before loading the next checkpoint
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    # Print comparison table
    print("\n" + "=" * 55)
    print(f"  {'Stage':<15}  {'Accuracy':>10}  {'vs baseline':>12}")
    print("  " + "-" * 40)
    baseline = 0.20   # 5 options (A-E)
    for stage, acc in results.items():
        if math.isnan(acc):
            print(f"  {stage:<15}  {'N/A':>10}  {'N/A':>12}")
        else:
            delta = acc - baseline
            sign  = "+" if delta >= 0 else ""
            print(f"  {stage:<15}  {acc:>10.1%}  {sign}{delta:>11.1%}")
    print("=" * 55)

    return results
