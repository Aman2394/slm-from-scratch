"""
datasets/medical.py — Dataset classes and instruction formatters for the Medical SLM.

Supported datasets:
    - medalpaca/medical_meadow_medqa  : USMLE-style multiple-choice Q&A
    - pubmed_qa (labeled split)       : Biomedical research yes/no/maybe Q&A

Both datasets are converted to a shared instruction-response format so the SFT
trainer can consume them without knowing which source an example came from.

Instruction template (MedQA):
    ### Question:
    {question}

    ### Options:
    A. {option_a}
    B. {option_b}
    ...

    ### Answer:
    {letter}: {answer_text}

Instruction template (PubMedQA):
    ### Question:
    {question}

    ### Context:
    {context}

    ### Answer:
    {final_decision}. {long_answer}
"""

from __future__ import annotations

import ast
import random
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Instruction formatters
# ---------------------------------------------------------------------------

def _parse_medqa_input(raw_input: str) -> Tuple[str, Dict[str, str]]:
    """
    Split the medqa 'input' field into question text and an options dict.

    The raw input looks like:
        "Q: <question text>? {'A': 'opt1', 'B': 'opt2', ...}"

    We locate the last occurrence of a dict literal (starting with '{') and
    treat everything before it as the question.

    Args:
        raw_input: The full 'input' string from the medqa example.

    Returns:
        (question, options) where options is a dict mapping letter -> text.
        Returns (raw_input, {}) if no dict literal is found.
    """
    # Find the last '{' that starts an options dict
    brace_pos = raw_input.rfind("{")
    if brace_pos == -1:
        return raw_input.strip(), {}

    question_part = raw_input[:brace_pos].strip()
    options_str   = raw_input[brace_pos:].strip()

    try:
        parsed = ast.literal_eval(options_str)
        # A trailing comma after the dict (e.g. "{'A': ...},") makes ast return a tuple
        if isinstance(parsed, tuple) and len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            return raw_input.strip(), {}
        options = parsed
    except (ValueError, SyntaxError):
        return raw_input.strip(), {}

    return question_part, options


def format_medqa_instruction(example: Dict) -> Tuple[str, str]:
    """
    Convert a medalpaca/medical_meadow_medqa example into a (prompt, response) pair.

    Actual dataset fields:
        input       : question text followed by an options dict literal, e.g.
                      "Q: <question>? {'A': 'opt1', 'B': 'opt2', ...}"
        instruction : "Please answer with one of the option in the bracket" (ignored)
        output      : correct answer in "LETTER: text" format, e.g. "E: Palpable purpura"

    Args:
        example: A dict with at least 'input' and 'output' keys.

    Returns:
        (prompt, response) where:
            prompt   = everything up to and including "### Answer:\n"
            response = the full output string, e.g. "E: Palpable purpura"
    """
    raw_input = example.get("input", "")
    answer    = example.get("output", "").strip()

    question, options = _parse_medqa_input(raw_input)

    if options:
        # Sort by key (A, B, C, ...) to guarantee consistent ordering
        options_lines = "\n".join(
            f"{letter}. {text}" for letter, text in sorted(options.items())
        )
        prompt = (
            f"### Question:\n{question}\n\n"
            f"### Options:\n{options_lines}\n\n"
            f"### Answer:\n"
        )
    else:
        # Fallback: no options dict found — plain Q&A format
        prompt = (
            f"### Question:\n{question}\n\n"
            f"### Answer:\n"
        )

    return prompt, answer


def format_pubmedqa_instruction(example: Dict) -> Tuple[str, str]:
    """
    Convert a pubmed_qa (labeled split) example into a (prompt, response) pair.

    The pubmed_qa labeled split has fields:
        pubid         : PubMed article ID (ignored)
        question      : the research question
        context       : dict with 'contexts' (list of strings) and 'labels' (list of strings)
        long_answer   : a paragraph-length answer
        final_decision: yes / no / maybe

    We truncate context to the first 3 sentences to stay within block_size.

    Args:
        example: A dict from the pubmed_qa labeled split.

    Returns:
        (prompt, response) where response = "{final_decision}. {long_answer}"
    """
    question       = example.get("question", "").strip()
    final_decision = example.get("final_decision", "").strip()
    long_answer    = example.get("long_answer", "").strip()

    # Flatten the context list and take the first few sentences to cap length
    context_parts = example.get("context", {}).get("contexts", [])
    context_text  = " ".join(context_parts)
    # Limit context to ~300 chars to leave room for question + answer in block_size
    if len(context_text) > 300:
        context_text = context_text[:300].rsplit(" ", 1)[0] + "..."

    prompt = (
        f"### Question:\n{question}\n\n"
        f"### Context:\n{context_text}\n\n"
        f"### Answer:\n"
    )
    response = f"{final_decision}. {long_answer}" if long_answer else final_decision

    return prompt, response


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class MedQADataset(Dataset):
    """
    PyTorch Dataset wrapping medalpaca/medical_meadow_medqa for SFT training.

    Each item is a dict with:
        input_ids     : (seq_len,) int64 tensor — full tokenized [prompt + response]
        prompt_len    : int — number of tokens in the prompt (used to mask loss)

    Loss masking: during training the SFT trainer sets labels for prompt tokens
    to -100 so the model only learns to predict the response tokens.

    Args:
        examples  : List of raw example dicts from HuggingFace datasets.
        tokenizer : A loaded ByteLevelBPETokenizer instance.
        max_len   : Maximum total sequence length (prompt + response). Longer
                    examples are truncated at the response end.
    """

    def __init__(self, examples: List[Dict], tokenizer, max_len: int = 512) -> None:
        self.items     = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        for ex in examples:
            prompt, response = format_medqa_instruction(ex)
            prompt_ids   = tokenizer.encode(prompt).ids
            response_ids = tokenizer.encode(response).ids

            # Truncate response if combined length exceeds max_len
            total = len(prompt_ids) + len(response_ids)
            if total > max_len:
                response_ids = response_ids[: max_len - len(prompt_ids)]

            input_ids = prompt_ids + response_ids
            if len(input_ids) < 2:
                continue   # skip degenerate examples

            self.items.append({
                "input_ids":  torch.tensor(input_ids, dtype=torch.long),
                "prompt_len": len(prompt_ids),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


class PubMedQADataset(Dataset):
    """
    PyTorch Dataset wrapping pubmed_qa (labeled split) for SFT training.

    Identical interface to MedQADataset — same dict shape so the SFT trainer
    can mix examples from both datasets transparently.

    Args:
        examples  : List of raw example dicts from HuggingFace datasets.
        tokenizer : A loaded ByteLevelBPETokenizer instance.
        max_len   : Maximum total sequence length. Defaults to 512.
    """

    def __init__(self, examples: List[Dict], tokenizer, max_len: int = 512) -> None:
        self.items     = []
        self.tokenizer = tokenizer
        self.max_len   = max_len

        for ex in examples:
            prompt, response = format_pubmedqa_instruction(ex)
            prompt_ids   = tokenizer.encode(prompt).ids
            response_ids = tokenizer.encode(response).ids

            total = len(prompt_ids) + len(response_ids)
            if total > max_len:
                response_ids = response_ids[: max_len - len(prompt_ids)]

            input_ids = prompt_ids + response_ids
            if len(input_ids) < 2:
                continue

            self.items.append({
                "input_ids":  torch.tensor(input_ids, dtype=torch.long),
                "prompt_len": len(prompt_ids),
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


# ---------------------------------------------------------------------------
# SFT batch utilities
# ---------------------------------------------------------------------------

def collate_sft_batch(
    batch: List[Dict],
    pad_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate a list of MedQADataset / PubMedQADataset items into padded tensors.

    Labels are set to -100 for:
        - Prompt tokens (we don't want the model to predict the question)
        - Padding tokens (masked in cross-entropy loss by PyTorch)

    Args:
        batch  : List of dicts, each with 'input_ids' and 'prompt_len'.
        pad_id : Token ID used for padding (should match tokenizer's <pad> id).

    Returns:
        x      : (batch_size, max_len) int64 — input token ids
        labels : (batch_size, max_len) int64 — target ids with -100 for masked positions
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    x_list, label_list = [], []
    for item in batch:
        ids        = item["input_ids"]
        prompt_len = item["prompt_len"]
        seq_len    = ids.size(0)
        pad_len    = max_len - seq_len

        # Pad input ids
        padded_ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])

        # Build labels: -100 for prompt + padding, real ids for response
        labels = torch.full((max_len,), -100, dtype=torch.long)
        labels[prompt_len:seq_len] = ids[prompt_len:]

        x_list.append(padded_ids)
        label_list.append(labels)

    return torch.stack(x_list), torch.stack(label_list)


def build_dpo_pairs_from_medqa(
    examples: List[Dict],
    max_pairs: Optional[int] = None,
) -> List[Dict]:
    """
    Build DPO preference pairs directly from MedQA labeled data.

    Key insight: MedQA already provides the correct answer and three wrong options.
    We don't need to generate candidates — the wrong options ARE the rejected responses.
    This gives a cleaner supervision signal than heuristic ranking of model samples.

    For each example we produce one pair:
        chosen   = correct answer text
        rejected = a randomly sampled wrong option

    Args:
        examples  : List of raw medqa example dicts.
        max_pairs : If set, return at most this many pairs (useful for quick runs).

    Returns:
        List of dicts: [{"prompt": str, "chosen": str, "rejected": str}, ...]
    """
    pairs = []

    for ex in examples:
        prompt, correct = format_medqa_instruction(ex)

        # Parse options dict from the input field; correct answer is "LETTER: text"
        _, options = _parse_medqa_input(ex.get("input", ""))
        # Extract just the answer text from "E: Palpable purpura" → "Palpable purpura"
        correct_text = correct.split(":", 1)[-1].strip().lower() if ":" in correct else correct.strip().lower()
        wrong_options = [
            text for text in options.values()
            if text.strip() and text.strip().lower() != correct_text
        ]

        if not wrong_options:
            # If we can't identify wrong options, skip this example
            continue

        rejected = random.choice(wrong_options)
        pairs.append({"prompt": prompt, "chosen": correct, "rejected": rejected})

        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    return pairs
