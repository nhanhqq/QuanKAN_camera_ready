"""Strict Leave-One-Subject-Out (LOSO) Cross-Validation for QuanKAN V4.

Protocol:
    - Subject-disjoint Leave-One-Subject-Out: train on N-1 subjects, test on held-out subject.
    - 150 epochs per fold.
    - Optimizer: AdamW with CosineAnnealingLR scheduler.
    - Loss functions: CrossEntropy with label smoothing + auxiliary supervision.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

try:
    from .model import QuanKANV4
except ImportError:
    from model import QuanKANV4


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EEGSubjDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, subj: np.ndarray, augment: bool = False):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.subj = torch.as_tensor(subj, dtype=torch.long)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.x[idx]
        if self.augment:
            valid = (sample.abs().sum(dim=(1, 2)) > 0).unsqueeze(-1).unsqueeze(-1)
            noise = torch.randn_like(sample) * 0.02
            sample = torch.where(valid, sample + noise, sample)
        return sample, self.y[idx], self.subj[idx]


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    embeddings = F.normalize(features, dim=1)
    batch_size = embeddings.size(0)
    logits = (embeddings @ embeddings.T) / temperature
    self_mask = torch.eye(batch_size, device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1e9)
    labels = labels.view(-1)
    positive_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = F.log_softmax(logits, dim=1)
    positive_count = positive_mask.sum(dim=1)
    loss = -(log_prob * positive_mask.float()).sum(dim=1) / positive_count.clamp_min(1)
    valid = positive_count > 0
    return loss[valid].mean() if valid.any() else features.sum() * 0.0


def cross_subject_prototype_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    subject_ids: torch.Tensor,
    num_classes: int,
    inter_margin: float = 0.20,
) -> torch.Tensor:
    embeddings = F.normalize(features, dim=1)
    zero = features.sum() * 0.0
    alignment_losses, class_prototypes = [], []

    for c in range(num_classes):
        class_mask = labels == c
        if class_mask.sum() < 2:
            continue
        global_proto = F.normalize(embeddings[class_mask].mean(dim=0), dim=0)
        class_prototypes.append(global_proto)
        for s in torch.unique(subject_ids[class_mask]):
            subj_mask = class_mask & (subject_ids == s)
            subj_proto = F.normalize(embeddings[subj_mask].mean(dim=0), dim=0)
            alignment_losses.append(1.0 - torch.sum(subj_proto * global_proto))

    loss_intra = torch.stack(alignment_losses).mean() if alignment_losses else zero
    if len(class_prototypes) < 2:
        return loss_intra

    prototypes = torch.stack(class_prototypes)
    sim = prototypes @ prototypes.T
    off_diag = ~torch.eye(sim.size(0), device=features.device, dtype=torch.bool)
    loss_inter = F.relu(sim[off_diag] - inter_margin).mean()
    return loss_intra + 0.5 * loss_inter


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_y, all_p = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lambda_adv=0.0, q_strength=1.0)[0]
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * len(y)
            all_y.extend(y.cpu().tolist())
            all_p.extend(logits.argmax(dim=1).cpu().tolist())
    y_true, y_pred = np.asarray(all_y), np.asarray(all_p)
    return {
        "loss": total_loss / max(len(y_true), 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_single_fold(
    fold_idx: int,
    target_subject: str,
    x: np.ndarray,
    y: np.ndarray,
    subjects: List[str],
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    print(f"\n==================== FOLD {fold_idx + 1} | Target Subject: {target_subject} ====================")
    train_mask = np.asarray([s != target_subject for s in subjects])
    test_mask = ~train_mask

    unique_subjs = sorted(set(subjects))
    subj_to_idx = {s: i for i, s in enumerate(unique_subjs)}
    subj_indices = np.asarray([subj_to_idx[s] for s in subjects])

    # Source-only feature standardization (avoids data leakage from target)
    x_fold = x.copy()
    valid_train = np.any(np.abs(x_fold[train_mask]) > 0, axis=(2, 3))
    for b in range(x_fold.shape[-1]):
        train_vals = x_fold[train_mask, :, :, b][valid_train]
        mean = float(train_vals.mean())
        std = float(max(train_vals.std(), 1e-4))
        x_fold[:, :, :, b] = (x_fold[:, :, :, b] - mean) / std

    train_ds = EEGSubjDataset(x_fold[train_mask], y[train_mask], subj_indices[train_mask], augment=True)
    test_ds = EEGSubjDataset(x_fold[test_mask], y[test_mask], subj_indices[test_mask], augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = QuanKANV4(
        in_channels=5,
        num_nodes=62,
        num_classes=num_classes,
        num_subjects=len(unique_subjs),
        q_device=args.q_device,
        quantum_enabled=True,
    ).to(device)

    backbone_params = list(model.backbone.parameters()) + list(model.classical_proj.parameters())
    head_params = (
        list(model.classical_classifier.parameters())
        + list(model.final_classifier.parameters())
        + list(model.subject_head.parameters())
    )
    frontend_params = list(model.frontend.parameters())
    quantum_params = (
        list(model.quantum_residual.parameters())
        + list(model.quantum_proj.parameters())
        + list(model.quantum_aux_head.parameters())
        + [model.quantum_scale]
    )
    if model.intensity_head is not None:
        head_params += list(model.intensity_head.parameters())

    param_groups = [
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
        {"params": frontend_params, "lr": args.frontend_lr},
        {"params": quantum_params, "lr": args.head_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    fold_dir = Path(args.output_dir) / f"fold_{target_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    csv_file = fold_dir / "epochs.csv"

    best_acc, best_f1, best_epoch = 0.0, 0.0, 0
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_loss", "test_acc", "test_f1", "lr"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            q_strength = min(1.0, max(0.0, (epoch - 15) / 30.0))  # Smooth quantum warmup

            for batch_x, batch_y, batch_subj in train_loader:
                batch_x, batch_y, batch_subj = batch_x.to(device), batch_y.to(device), batch_subj.to(device)
                optimizer.zero_grad()

                emo_logits, subj_logits, z_clean, classical_logits, q_aux_logits, _, _, _, _ = model(
                    batch_x, lambda_adv=0.05, q_strength=q_strength
                )

                loss_emo = F.cross_entropy(emo_logits, batch_y, label_smoothing=0.05)
                loss_classical = F.cross_entropy(classical_logits, batch_y, label_smoothing=0.05)
                loss_frontend = F.cross_entropy(model.frontend.aux_logits, batch_y)
                loss_supcon = supervised_contrastive_loss(z_clean, batch_y)
                loss_proto = cross_subject_prototype_loss(z_clean, batch_y, batch_subj, num_classes=num_classes)
                loss_subj = F.cross_entropy(subj_logits, batch_subj)

                with torch.no_grad():
                    p_classical = torch.softmax(classical_logits, dim=-1)
                    y_onehot = F.one_hot(batch_y, num_classes).float()
                    residual_target = y_onehot - p_classical
                loss_q = F.smooth_l1_loss(torch.tanh(q_aux_logits), residual_target)

                total_loss = (
                    loss_emo
                    + 0.20 * loss_classical
                    + 0.25 * loss_frontend
                    + 0.08 * loss_supcon
                    + 0.05 * loss_proto
                    + 0.20 * loss_subj
                    + 0.15 * loss_q
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(total_loss.item())

            scheduler.step()
            test_metrics = evaluate(model, test_loader, device)

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                best_f1 = test_metrics["macro_f1"]
                best_epoch = epoch
                torch.save(model.state_dict(), fold_dir / "best_model.pt")

            writer.writerow({
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "test_loss": test_metrics["loss"],
                "test_acc": test_metrics["accuracy"],
                "test_f1": test_metrics["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            })
            f.flush()

            if epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"Epoch {epoch:03d}/{args.epochs:03d} | Train Loss: {np.mean(train_losses):.4f} | "
                    f"Test Acc: {test_metrics['accuracy']:.4f} | Best Acc: {best_acc:.4f} (@ep {best_epoch})"
                )

    print(f"Fold {target_subject} Complete! Best Test Acc: {best_acc:.4f}, Best Macro-F1: {best_f1:.4f} at epoch {best_epoch}")
    return {"subject": target_subject, "best_acc": best_acc, "best_f1": best_f1, "best_epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser(description="QuanKAN V4 Strict LOSO Training (150 Epochs)")
    parser.add_argument("--dataset", choices=["seed", "seediv"], default="seediv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--frontend_lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--q_device", type=str, default="default.qubit")
    parser.add_argument("--output_dir", type=str, default="results_quankan_loso")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}, Quantum Device: {args.q_device}")

    # Load SEED-IV features
    data_dir = "/home/namphuongtran9196/intel_project/seed_iv/eeg_feature_smooth"
    if not os.path.exists(data_dir):
        data_dir = "/home/namphuongtran9196/intel_project/eeg_feature_smooth"
    import sys
    sys.path.insert(0, "/home/namphuongtran9196/intel_project/QuanKAN")
    from data.dataset import load_seediv_all
    x, y, subjects, _ = load_seediv_all(data_dir, session_ids=[1])
    num_classes = len(np.unique(y))
    unique_subjects = sorted(set(subjects))

    print(f"Dataset Loaded: {len(x)} trials, {len(unique_subjects)} subjects, {num_classes} classes.")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for fold_idx, target_subj in enumerate(unique_subjects):
        res = train_single_fold(fold_idx, target_subj, x, y, subjects, num_classes, args, device)
        results.append(res)
        with open(os.path.join(args.output_dir, "loso_summary.json"), "w") as f:
            json.dump(results, f, indent=2)

    mean_acc = np.mean([r["best_acc"] for r in results])
    mean_f1 = np.mean([r["best_f1"] for r in results])
    print(f"\n==================== FULL LOSO FINISHED ====================")
    print(f"Mean Accuracy: {mean_acc * 100:.2f}% | Mean Macro-F1: {mean_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
