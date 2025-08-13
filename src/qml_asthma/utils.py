from __future__ import annotations
import random, numpy as np, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def iterate_batches(X, y, bs: int):
    idx = np.arange(len(X)); np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i+bs]
        yield X[j], y[j]

def bin_metrics(y_true, probs, threshold: float = 0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(y_true, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception:
        auc = float("nan")
    report = classification_report(y_true, preds, digits=4)
    return acc, prec, rec, f1, auc, report
