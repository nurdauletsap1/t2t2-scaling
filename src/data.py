"""Dataset loading and preprocessing for T2T² experiments."""

import hashlib
import os
from typing import Dict, Tuple

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm
from transformers import GPT2Tokenizer


def _format_example(question: str, answer: str) -> str:
    """Format a QA pair into the training string format."""
    return f"Problem: {question}\nSolution: {answer}<|endoftext|>"


def _hash_text(text: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MathDataset(TorchDataset):
    """Tokenized dataset returning input_ids, attention_mask, and labels."""

    def __init__(self, texts: list, tokenizer: GPT2Tokenizer, max_length: int = 2048) -> None:
        """Initialize and tokenize all examples.

        Args:
            texts: List of formatted text strings.
            tokenizer: GPT-2 tokenizer instance.
            max_length: Maximum token sequence length.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    def __len__(self) -> int:
        """Return number of examples."""
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return tokenized example with labels (padding = -100).

        Args:
            idx: Index of the example.

        Returns:
            Dict with 'input_ids', 'attention_mask', 'labels'.
        """
        input_ids = self.encodings["input_ids"][idx]
        attention_mask = self.encodings["attention_mask"][idx]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_datasets(config: Dict) -> Tuple[TorchDataset, TorchDataset, TorchDataset]:
    """Load, format, interleave, and tokenize GSM8K + MATH datasets.

    Args:
        config: Full experiment config dict (must include 'model.max_seq_len', 'seed').

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as MathDataset instances.
    """
    seed = config["seed"]
    max_seq_len = config["model"]["max_seq_len"]
    cache_dir = config.get("data", {}).get("cache_dir", "/workspace/.cache/huggingface")
    max_samples = config.get("data", {}).get("max_samples", None)
    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load raw datasets ──────────────────────────────────────────────────
    gsm8k = load_dataset("gsm8k", "main", cache_dir=cache_dir)
    try:
        _full = load_dataset("AI-MO/NuminaMath-CoT", split="train", cache_dir=cache_dir)
        print(f"Loaded NuminaMath-CoT: {len(_full)} problems")
    except Exception as _e1:
        try:
            _full = load_dataset("nvidia/OpenMathReasoning", split="train", cache_dir=cache_dir)
            print(f"Loaded OpenMathReasoning: {len(_full)} problems")
        except Exception as _e2:
            raise RuntimeError(f"Could not load math dataset: {_e1}, fallback: {_e2}")

    if max_samples:
        _full = _full.select(range(min(max_samples, len(_full))))
        print(f"Debug mode: using {len(_full)} samples")

    # Deduplicate by problem text (NuminaMath-CoT has same problem with multiple solutions)
    _unique: dict = {}
    for _item in tqdm(_full, desc="Deduplicating"):
        _h = _hash_text(_item["problem"])
        if _h not in _unique:
            _unique[_h] = _item
    unique_list = list(_unique.values())
    print(f"Deduplicated to {len(unique_list)} unique problems")

    # Split: 80% train, 10% val, 10% test — no overlap by construction
    _n = len(unique_list)
    _train_end = int(0.8 * _n)
    _val_end = int(0.9 * _n)
    math_train = unique_list[:_train_end]
    math_test = unique_list[_val_end:]

    _train_hashes = {_hash_text(x["problem"]) for x in math_train}
    _test_hashes = {_hash_text(x["problem"]) for x in math_test}
    assert len(_train_hashes & _test_hashes) == 0, \
        f"Leakage: {len(_train_hashes & _test_hashes)} problems in both sets"

    # ── Build test set (math test split) ─────────────────────────────────
    test_texts = [
        _format_example(ex["problem"], ex["solution"])
        for ex in math_test
    ]

    # ── Format training data ───────────────────────────────────────────────
    gsm8k_texts = [
        _format_example(ex["question"], ex["answer"])
        for ex in gsm8k["train"]
    ]
    math_train_texts = [
        _format_example(ex["problem"], ex["solution"])
        for ex in math_train
    ]

    # 50/50 interleaved by index
    n = min(len(gsm8k_texts), len(math_train_texts))
    interleaved: list = []
    for i in range(n):
        interleaved.append(gsm8k_texts[i])
        interleaved.append(math_train_texts[i])
    # Append remaining examples from the longer list
    interleaved.extend(gsm8k_texts[n:])
    interleaved.extend(math_train_texts[n:])

    # ── Validation split (10% stratified, seeded) ─────────────────────────
    import random
    rng = random.Random(seed)
    indices = list(range(len(interleaved)))
    rng.shuffle(indices)
    val_n = max(1, int(0.1 * len(interleaved)))
    val_indices = set(indices[:val_n])

    train_texts = [t for i, t in enumerate(interleaved) if i not in val_indices]
    val_texts = [t for i, t in enumerate(interleaved) if i in val_indices]

    # ── Tokenize and wrap ─────────────────────────────────────────────────
    train_dataset = MathDataset(train_texts, tokenizer, max_length=max_seq_len)
    val_dataset = MathDataset(val_texts, tokenizer, max_length=max_seq_len)
    test_dataset = MathDataset(test_texts, tokenizer, max_length=max_seq_len)

    return train_dataset, val_dataset, test_dataset
