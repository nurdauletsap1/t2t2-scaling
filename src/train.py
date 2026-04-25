"""Training loop for T2T² using HuggingFace Accelerate."""

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup


def _get_lr(config: Dict) -> float:
    """Return learning rate based on model size.

    Args:
        config: Full experiment config dict.

    Returns:
        Learning rate float.
    """
    size = config["model"].get("size", "tiny")
    if size in ("tiny", "small"):
        return float(config["training"].get("learning_rate", 5e-4))
    return float(config["training"].get("learning_rate", 3e-4))


def _save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    step: int,
    consumed_tokens: int,
    best_val_loss: float,
    config: Dict,
    checkpoint_dir: Path,
) -> Path:
    """Save full training checkpoint.

    Args:
        accelerator: Accelerate instance.
        model: Unwrapped or wrapped model.
        optimizer: AdamW optimizer.
        scheduler: LR scheduler.
        step: Current global step.
        consumed_tokens: Total tokens seen.
        best_val_loss: Best validation loss so far.
        config: Full experiment config.
        checkpoint_dir: Directory for this run's checkpoints.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"step_{step:08d}.pt"

    unwrapped_model = accelerator.unwrap_model(model)

    torch.save(
        {
            "model_state_dict": unwrapped_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": step,
            "consumed_tokens": consumed_tokens,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        ckpt_path,
    )
    return ckpt_path


def _load_latest_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_dir: Path,
    device,
) -> Tuple[int, int, float]:
    """Auto-load the latest checkpoint; fall back to second-latest or scratch.

    Args:
        model: Model whose state dict will be updated.
        optimizer: Optimizer whose state dict will be updated.
        scheduler: Scheduler whose state dict will be updated.
        checkpoint_dir: Directory to scan for checkpoint files.
        device: Target device.

    Returns:
        Tuple of (global_step, consumed_tokens, best_val_loss).
    """
    if not checkpoint_dir.exists():
        return 0, 0, float("inf")

    ckpts = sorted(checkpoint_dir.glob("step_*.pt"))
    if not ckpts:
        return 0, 0, float("inf")

    for ckpt_path in reversed(ckpts):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"Resumed from checkpoint: {ckpt_path}")
            return ckpt["global_step"], ckpt["consumed_tokens"], ckpt["best_val_loss"]
        except Exception as e:
            print(f"Warning: failed to load checkpoint {ckpt_path}: {e}. Trying previous.")

    print("Warning: all checkpoints failed to load — starting from scratch.")
    return 0, 0, float("inf")


def _validate(model, val_loader, accelerator) -> float:
    """Compute mean validation loss.

    Args:
        model: The (possibly accelerate-wrapped) model.
        val_loader: Validation DataLoader.
        accelerator: Accelerate instance.

    Returns:
        Mean validation loss float.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def train(
    config: Dict,
    model,
    train_dataset,
    val_dataset,
) -> Dict:
    """Train a model using HuggingFace Accelerate.

    Args:
        config: Full experiment config dict.
        model: LlamaForCausalLM instance.
        train_dataset: Training MathDataset.
        val_dataset: Validation MathDataset.

    Returns:
        Dict with keys: best_val_loss, total_steps, consumed_tokens, checkpoint_path.
    """
    train_cfg = config["training"]
    run_id = config["run_id"]
    checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / run_id
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    wandb_fallback_path = log_dir / "wandb_fallback.jsonl"

    accelerator = Accelerator(
        mixed_precision="bf16" if train_cfg.get("bf16", True) else "no",
        gradient_accumulation_steps=1,  # Updated below after we know batch size
    )

    device = accelerator.device
    n_gpus = max(accelerator.num_processes, 1)

    # ── Batch size / gradient accumulation ────────────────────────────────
    per_gpu_batch_size = train_cfg.get("per_gpu_batch_size", 8)
    max_seq_len = config["model"]["max_seq_len"]
    target_tokens = 65536
    grad_accum = math.ceil(target_tokens / (per_gpu_batch_size * n_gpus * max_seq_len))
    grad_accum = max(grad_accum, 1)

    # Rebuild accelerator with correct grad_accum
    accelerator = Accelerator(
        mixed_precision="bf16" if train_cfg.get("bf16", True) else "no",
        gradient_accumulation_steps=grad_accum,
    )

    # ── Optional torch.compile ─────────────────────────────────────────────
    from src.model import count_parameters
    params = count_parameters(model)
    if train_cfg.get("compile", True) and params["total"] <= 40_000_000:
        try:
            model = torch.compile(model, mode="default")
            if accelerator.is_main_process:
                print("torch.compile enabled.")
        except Exception as e:
            print(f"Warning: torch.compile failed ({e}) — continuing without compilation.")

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    lr = _get_lr(config)
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg.get("weight_decay", 0.1),
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        eps=1e-8,
    )

    D_tokens = train_cfg["D_tokens"]
    tokens_per_step = per_gpu_batch_size * n_gpus * max_seq_len * grad_accum
    total_steps = D_tokens // tokens_per_step
    warmup_steps = train_cfg.get("warmup_steps", 100)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )
    # Cosine decays to ~10% of max lr: handled by the default implementation
    # (min_lr = 0 by default; we set eta_min via a custom lambda if needed)
    # HuggingFace cosine schedule decays to 0; we patch to 10% of max_lr.
    base_schedule = scheduler
    min_lr_ratio = 0.1

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ── DataLoaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    # ── Resume from checkpoint ────────────────────────────────────────────
    global_step, consumed_tokens, best_val_loss = _load_latest_checkpoint(
        accelerator.unwrap_model(model), optimizer, scheduler, checkpoint_dir, device
    )

    # ── WandB setup ───────────────────────────────────────────────────────
    wandb_ok = False
    wandb_log_count = 0
    if accelerator.is_main_process and config.get("wandb", {}).get("project"):
        try:
            import wandb
            wandb_cfg = config.get("wandb", {})
            wandb.init(
                project=wandb_cfg.get("project", "t2t2-scaling"),
                entity=wandb_cfg.get("entity") or None,
                name=run_id,
                config=config,
                resume="allow",
                id=run_id,
            )
            wandb_ok = True
        except Exception as e:
            print(f"Warning: WandB init failed ({e}) — falling back to local logging.")

    def log_metrics(metrics: Dict) -> None:
        nonlocal wandb_ok, wandb_log_count
        if accelerator.is_main_process:
            if wandb_ok:
                try:
                    import wandb as _wandb
                    _wandb.log(metrics, step=global_step)
                except Exception as e:
                    wandb_ok = False
                    if wandb_log_count % 100 == 0:
                        print(f"Warning: WandB logging failed ({e}) — writing to local fallback.")
            if not wandb_ok:
                with open(wandb_fallback_path, "a") as f:
                    f.write(json.dumps({**metrics, "step": global_step}) + "\n")
            wandb_log_count += 1

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    val_every = train_cfg.get("val_every_steps", 500)
    patience = train_cfg.get("early_stopping_patience", 200)
    steps_without_improvement = 0
    last_checkpoint_path = None

    step_in_epoch = 0
    train_iter = iter(train_loader)
    t_start = time.time()
    tokens_at_t_start = consumed_tokens

    while global_step < total_steps:
        # ── Get next batch ────────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # ── Forward + backward with OOM retry ────────────────────────────
        oom_retry = True
        while oom_retry:
            oom_retry = False
            try:
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                    accelerator.backward(loss)
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    new_bs = max(1, per_gpu_batch_size // 2)
                    print(
                        f"OOM at batch_size={per_gpu_batch_size} on {device} — "
                        f"reduced to batch_size={new_bs}. "
                        "Consider reducing max_seq_len if OOM persists."
                    )
                    log_metrics({
                        "event": "oom_batch_reduction",
                        "old_batch_size": per_gpu_batch_size,
                        "new_batch_size": new_bs,
                    })
                    per_gpu_batch_size = new_bs
                    # Rebuild loader with new batch size
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=per_gpu_batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                    )
                    train_loader = accelerator.prepare(train_loader)
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                    oom_retry = True
                    continue
                raise

        # ── Update counters ───────────────────────────────────────────────
        global_step += 1
        batch_tokens = per_gpu_batch_size * max_seq_len * n_gpus
        consumed_tokens += batch_tokens

        loss_val = loss.item()

        # ── NaN check ─────────────────────────────────────────────────────
        if math.isnan(loss_val) or math.isinf(loss_val):
            last_checkpoint_path = _save_checkpoint(
                accelerator, model, optimizer, scheduler,
                global_step, consumed_tokens, best_val_loss, config, checkpoint_dir
            )
            log_metrics({
                "event": "nan_loss",
                "step": global_step,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
            })
            raise RuntimeError(
                f"NaN loss detected at step {global_step} — reduce lr "
                f"(current: {scheduler.get_last_lr()[0]:.2e}). "
                f"Checkpoint saved to {last_checkpoint_path}."
            )

        # ── Tokens/sec ────────────────────────────────────────────────────
        elapsed = time.time() - t_start
        tokens_per_sec = (consumed_tokens - tokens_at_t_start) / max(elapsed, 1e-6)
        current_lr = scheduler.get_last_lr()[0]
        grad_norm_val = float(grad_norm) if grad_norm is not None else 0.0

        log_metrics({
            "train/loss": loss_val,
            "train/lr": current_lr,
            "train/tokens_per_sec": tokens_per_sec,
            "train/grad_norm": grad_norm_val,
            "train/global_step": global_step,
            "train/consumed_tokens": consumed_tokens,
        })

        # ── Periodic checkpoint ───────────────────────────────────────────
        if global_step % 1000 == 0:
            last_checkpoint_path = _save_checkpoint(
                accelerator, model, optimizer, scheduler,
                global_step, consumed_tokens, best_val_loss, config, checkpoint_dir
            )

        # ── Validation ────────────────────────────────────────────────────
        if global_step % val_every == 0:
            val_loss = _validate(model, val_loader, accelerator)
            log_metrics({"val/loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                steps_without_improvement = 0
                last_checkpoint_path = _save_checkpoint(
                    accelerator, model, optimizer, scheduler,
                    global_step, consumed_tokens, best_val_loss, config, checkpoint_dir
                )
                if accelerator.is_main_process:
                    print(f"Step {global_step}: new best val_loss={best_val_loss:.4f} → saved.")
            else:
                steps_without_improvement += val_every
                if steps_without_improvement >= patience:
                    if accelerator.is_main_process:
                        print(
                            f"Early stopping at step {global_step}: "
                            f"no improvement for {steps_without_improvement} steps."
                        )
                    break

        # ── Progress print ────────────────────────────────────────────────
        if accelerator.is_main_process and global_step % 100 == 0:
            print(
                f"Step {global_step}/{total_steps} | loss={loss_val:.4f} | "
                f"lr={current_lr:.2e} | tok/s={tokens_per_sec:.0f}"
            )

    # ── Final checkpoint ──────────────────────────────────────────────────
    if last_checkpoint_path is None or global_step % 1000 != 0:
        last_checkpoint_path = _save_checkpoint(
            accelerator, model, optimizer, scheduler,
            global_step, consumed_tokens, best_val_loss, config, checkpoint_dir
        )

    if accelerator.is_main_process and wandb_ok:
        try:
            import wandb as _wandb
            _wandb.finish()
        except Exception:
            pass

    return {
        "best_val_loss": best_val_loss,
        "total_steps": global_step,
        "consumed_tokens": consumed_tokens,
        "checkpoint_path": str(last_checkpoint_path),
    }
