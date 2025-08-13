from __future__ import annotations
import argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import TrainConfig
from .data import load_and_preprocess, pca_to_qubits, to_angles
from .qfs import qfs_select
from .model import VQC
from .utils import set_seed, iterate_batches, bin_metrics

def build_argparser():
    p = argparse.ArgumentParser(description="Train VQC for Asthma Diagnosis")
    p.add_argument("--csv", required=True, type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--test-size", default=0.2, type=float)

    p.add_argument("--n-qubits", default=6, type=int)
    p.add_argument("--vqc-layers", default=3, type=int)
    p.add_argument("--epochs", default=60, type=int)
    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--lr", default=0.02, type=float)
    p.add_argument("--patience", default=8, type=int)

    p.add_argument("--enable-qfs", default=False, type=lambda s: s.lower() in {"1","true","yes","y"})
    p.add_argument("--qfs-max-features", default=64, type=int)
    return p

def main() -> int:
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        csv_path=args.csv, seed=args.seed, test_size=args.test_size,
        n_qubits=args.n_qubits, vqc_layers=args.vqc_layers,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience,
        enable_qfs=args.enable_qfs, qfs_max_features=args.qfs_max_features
    )
    set_seed(cfg.seed)

    # 1) preprocess
    Xtr, Xte, ytr, yte, _ = load_and_preprocess(
        cfg.csv_path, cfg.seed, cfg.test_size, cfg.target_col, cfg.drop_id_cols
    )

    # 2) (optional) QFS (placeholder impl; swap with QAOA-based selector)
    if cfg.enable_qfs:
        Xtr, idx = qfs_select(Xtr, ytr, max_features=cfg.qfs_max_features)
        Xte = Xte[:, idx]

    # 3) PCA → qubits → angle embedding
    Xtr_p, Xte_p, _ = pca_to_qubits(Xtr, Xte, cfg.n_qubits, cfg.seed)
    Xtr_a = to_angles(Xtr_p)
    Xte_a = to_angles(Xte_p)

    Xtr_t = torch.tensor(Xtr_a, dtype=torch.float32)
    Xte_t = torch.tensor(Xte_a, dtype=torch.float32)
    ytr_t = torch.tensor(ytr.reshape(-1,1), dtype=torch.float32)
    yte_t = torch.tensor(yte.reshape(-1,1), dtype=torch.float32)

    # 4) model
    model = VQC(n_qubits=cfg.n_qubits, n_layers=cfg.vqc_layers)
    pos_weight = torch.tensor(
        ((ytr == 0).sum() + 1e-9) / ((ytr == 1).sum() + 1e-9), dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimz = optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc, best_state, bad = -1.0, None, 0

    # 5) train
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in iterate_batches(Xtr_t, ytr_t, cfg.batch_size):
            optimz.zero_grad()
            logits = model(xb)              # [B,1]
            loss = criterion(logits, yb)
            loss.backward()
            optimz.step()
            running += loss.item() * len(xb)

        # val
        model.eval()
        with torch.no_grad():
            logits = model(Xte_t)           # [N,1]
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            acc, _, _, _, _, _ = bin_metrics(yte, probs)

        print(f"Epoch {epoch:03d} | loss={running/len(Xtr_t):.4f} | val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    # 6) test metrics
    model.eval()
    with torch.no_grad():
        logits = model(Xte_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    acc, prec, rec, f1, auc, report = bin_metrics(yte, probs)
    print("\n=== Test Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nClassification report:\n")
    print(report)
    return 0
