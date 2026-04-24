"""Evaluation with majority voting and FLOP tracking for T2T² experiments."""

import re
from collections import Counter
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, LlamaForCausalLM


def _extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from a model-generated solution string.

    Tries LaTeX boxed, display math, inline math, then last numeric value.

    Args:
        text: Raw model output string.

    Returns:
        Extracted answer string, or None if no answer found.
    """
    patterns = [
        r'\\boxed\{([^}]+)\}',
        r'\$\$([^$]+)\$\$',
        r'\$([^$]+)\$',
        r'[-+]?\d*\.?\d+',
    ]
    for i, pat in enumerate(patterns):
        matches = re.findall(pat, text)
        if matches:
            return matches[-1]
    return None


def _normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Args:
        answer: Raw extracted answer.

    Returns:
        Normalized answer string.
    """
    answer = answer.strip().lower()
    # Remove trailing zeros after decimal
    try:
        f = float(answer)
        if f == int(f):
            return str(int(f))
        # Remove trailing zeros
        return f"{f:.10f}".rstrip("0").rstrip(".")
    except (ValueError, OverflowError):
        return answer


def _answers_match(pred: Optional[str], gold: str) -> bool:
    """Check if predicted answer matches the gold answer.

    Args:
        pred: Predicted answer (possibly None).
        gold: Gold answer string.

    Returns:
        True if the answers match within tolerance.
    """
    if pred is None:
        return False
    pred_norm = _normalize_answer(pred)
    gold_norm = _normalize_answer(gold)

    # Numeric comparison with tolerance
    try:
        return abs(float(pred_norm) - float(gold_norm)) < 1e-3
    except (ValueError, OverflowError):
        return pred_norm == gold_norm


def _majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Select the majority answer; break ties alphabetically.

    Args:
        answers: List of extracted answer strings (may include None).

    Returns:
        The majority answer string.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    counts = Counter(valid)
    max_count = max(counts.values())
    candidates = sorted(a for a, c in counts.items() if c == max_count)
    return candidates[0]


@torch.no_grad()
def evaluate(
    config: Dict,
    model: LlamaForCausalLM,
    test_dataset,
    k_values: List[int] = None,
) -> Dict:
    """Evaluate a model on the test set with pass@k and maj@k metrics.

    Args:
        config: Full experiment config dict.
        model: Trained LlamaForCausalLM instance.
        test_dataset: MathDataset instance (test split).
        k_values: List of k values to evaluate; defaults to [1, 4, 8, 16].

    Returns:
        Dict with pass@k and maj@k for each k, plus compute_flops sub-dict.
    """
    if k_values is None:
        k_values = [1, 4, 8, 16]

    eval_cfg = config.get("evaluation", {})
    temperature = eval_cfg.get("temperature", 0.7)
    top_p = eval_cfg.get("top_p", 0.95)
    max_new_tokens = eval_cfg.get("max_new_tokens", 512)
    device = next(model.parameters()).device

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    max_k = max(k_values)
    n_problems = len(test_dataset)

    # Per-problem: list of (generated_text, n_tokens_generated)
    all_preds: List[List[Optional[str]]] = []
    all_gold: List[str] = []
    total_tokens_generated: List[int] = []

    model.eval()

    for problem_idx in range(n_problems):
        try:
            item = test_dataset[problem_idx]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention_mask = item["attention_mask"].unsqueeze(0).to(device)

            # Trim padding from the right to find actual prompt length
            actual_len = int(attention_mask.sum().item())
            input_ids = input_ids[:, :actual_len]
            attention_mask = attention_mask[:, :actual_len]

            # Extract gold answer from the raw text
            raw_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            gold_extracted = _extract_answer(raw_text)
            # Fallback: use last number in the "Solution:" portion
            if gold_extracted is None:
                nums = re.findall(r'[-+]?\d*\.?\d+', raw_text.split("Solution:")[-1])
                gold_extracted = nums[-1] if nums else ""
            all_gold.append(gold_extracted or "")

            problem_preds: List[Optional[str]] = []
            problem_token_counts: List[int] = []
            generated_k = 0

            while generated_k < max_k:
                batch_size = min(max_k - generated_k, 4)
                is_greedy = (generated_k == 0 and 1 in k_values and max_k == 1)

                try:
                    if batch_size > 1 or (1 not in k_values):
                        # Sampling
                        outputs = model.generate(
                            input_ids.expand(batch_size, -1),
                            attention_mask=attention_mask.expand(batch_size, -1),
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        # Greedy for k=1
                        outputs = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    for out in outputs:
                        new_tokens = out[actual_len:]
                        n_new = int((new_tokens != tokenizer.eos_token_id).sum().item())
                        problem_token_counts.append(n_new)
                        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
                        problem_preds.append(_extract_answer(decoded))

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        print(
                            f"OOM during eval on problem {problem_idx} batch_size={batch_size} — "
                            "reducing to batch_size=1."
                        )
                        batch_size = 1
                        continue
                    raise

                generated_k += batch_size

            all_preds.append(problem_preds)
            total_tokens_generated.extend(problem_token_counts)

        except Exception as e:
            print(f"Skipping problem {problem_idx} due to error: {e}")
            all_preds.append([None] * max_k)
            all_gold.append("") if len(all_gold) <= problem_idx else None
            continue

    # ── Compute metrics ────────────────────────────────────────────────────
    results: Dict = {}

    for k in k_values:
        pass_correct = 0
        maj_correct = 0

        for i in range(n_problems):
            gold = all_gold[i] if i < len(all_gold) else ""
            preds = (all_preds[i][:k] if i < len(all_preds) else [None] * k)

            # pass@k: at least one correct
            if any(_answers_match(p, gold) for p in preds):
                pass_correct += 1

            # maj@k: majority vote is correct
            mv = _majority_vote(preds)
            if _answers_match(mv, gold):
                maj_correct += 1

        results[f"pass@{k}"] = pass_correct / n_problems if n_problems > 0 else 0.0
        results[f"maj@{k}"] = maj_correct / n_problems if n_problems > 0 else 0.0

    # ── FLOP tracking ──────────────────────────────────────────────────────
    from src.model import count_parameters
    param_counts = count_parameters(model)
    N = param_counts["non_embedding"]
    D_tokens = config["training"]["D_tokens"]
    mean_tokens_generated = (
        sum(total_tokens_generated) / len(total_tokens_generated)
        if total_tokens_generated else max_new_tokens
    )

    train_flops = 6 * N * D_tokens
    infer_flops = 2 * N * max_k * mean_tokens_generated * n_problems
    total_flops = train_flops + infer_flops

    results["compute_flops"] = {
        "train_flops": train_flops,
        "infer_flops": infer_flops,
        "total_flops": total_flops,
        "mean_tokens_generated": mean_tokens_generated,
    }

    return results
