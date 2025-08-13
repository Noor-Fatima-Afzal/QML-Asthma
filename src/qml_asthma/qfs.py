"""
Quantum Feature Selection (QFS) placeholder.

This module exposes a unified interface you can replace with a proper
QAOA/Ising-based solver. For now, we provide a *classical* sparse selector
to keep code runnable and aligned with the paper section.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def qfs_select(X: np.ndarray, y: np.ndarray, max_features: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (X_reduced, selected_idx)

    Current strategy (placeholder):
      - rank features by mutual information with y
      - keep top-k
    Swap this with your QAOA-based bitstring solver to match the Ising formulation.
    """
    k = min(max_features, X.shape[1])
    scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
    idx = np.argsort(scores)[::-1][:k]
    Xr = X[:, idx]
    return Xr, idx
