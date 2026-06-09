"""Training loop for CNNBoundaryDetector.

Supports three class-balance strategies:
  none           -- standard unweighted BCE loss
  weighted_loss  -- per-sample BCE weighted by inverse class frequency
  resample       -- WeightedRandomSampler to balance batches

Includes:
  - Trackio integration for live experiment tracking
  - JSON history logging to results/logs/<run_name>.json
  - Checkpoint/resume system at results/checkpoint.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import trackio

from src.utils import get_device, seed_everything, save_json

BalanceStrategy = Literal["none", "weighted_loss", "weighted_loss_capped", "resample"]


# ---------------------------------------------------------------------------
# Training step helpers
# ---------------------------------------------------------------------------

def _bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    loss_fn = nn.BCELoss(weight=weights)
    return loss_fn(logits, labels)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    run_name: str,
    output_dir: str | Path = "results",
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    balance: BalanceStrategy = "none",
    seed: int = 42,
    patience: int = 10,
) -> dict:
    """Train the boundary detector and return best validation metrics."""
    seed_everything(seed)
    device = get_device()
    output_dir = Path(output_dir)
    log_dir = output_dir / "logs"
    ckpt_dir = output_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # Build dataloader
    if balance == "resample":
        sampler = WeightedRandomSampler(
            weights=train_dataset.class_weights(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Pre-compute class weights for weighted_loss strategies
    if balance == "weighted_loss":
        class_w = train_dataset.class_weights().to(device)
    elif balance == "weighted_loss_capped":
        raw_w = train_dataset.class_weights()
        # Cap the ratio between max and min weight at 10× to prevent instability
        w_min = raw_w[raw_w > 0].min()
        class_w = torch.clamp(raw_w, max=10.0 * w_min).to(device)
    else:
        class_w = None

    trackio.init(project="segment-consistency", name=run_name)

    history: dict = {
        "run_name": run_name,
        "balance": balance,
        "seed": seed,
        "epochs": [],
        "best_val_loss": float("inf"),
        "best_epoch": -1,
    }
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for patches, labels in train_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = model(patches)
            w = class_w[: labels.shape[0]] if class_w is not None else None
            loss = _bce_loss(preds, labels, w)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= max(len(train_dataset), 1)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for patches, labels in val_loader:
                patches = patches.to(device)
                labels = labels.to(device)
                preds = model(patches)
                loss = nn.BCELoss()(preds, labels)
                val_loss += loss.item() * len(labels)
        val_loss /= max(len(val_dataset), 1)

        scheduler.step(val_loss)
        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_dir / f"{run_name}_best.pt")
        else:
            epochs_no_improve += 1

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
            "is_best": is_best,
        }
        history["epochs"].append(epoch_data)
        save_json(history, log_dir / f"{run_name}.json")

        trackio.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "lr": current_lr,
            "epoch": epoch + 1,
        })

        print(
            f"[{run_name}] epoch {epoch+1:3d}/{epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"lr={current_lr:.2e} | {'*' if is_best else ''}",
            flush=True,
        )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}", flush=True)
            break

    history["total_epochs"] = len(history["epochs"])
    save_json(history, log_dir / f"{run_name}.json")
    trackio.finish()
    return history


# ---------------------------------------------------------------------------
# Checkpoint/resume for experiment grids
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Run a grid of experiments with checkpoint/resume support."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self._results: dict = {}
        self._completed: set[str] = self._load_checkpoint()

    def _load_checkpoint(self) -> set[str]:
        if not self.checkpoint_file.exists():
            return set()
        data = json.loads(self.checkpoint_file.read_text())
        self._results = data.get("results", {})
        return set(data.get("completed_experiments", []))

    def _save_checkpoint(self, exp_id: str, result: dict) -> None:
        self._completed.add(exp_id)
        self._results[exp_id] = result
        save_json(
            {
                "completed_experiments": sorted(self._completed),
                "results": self._results,
                "last_updated": time.time(),
            },
            self.checkpoint_file,
        )

    def is_done(self, exp_id: str) -> bool:
        return exp_id in self._completed

    def save(self, exp_id: str, result: dict) -> None:
        self._save_checkpoint(exp_id, result)

    def all_results(self) -> dict:
        return self._results
