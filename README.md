# QML-Asthma: Variational Quantum Classifier (PennyLane + PyTorch)

End-to-end binary asthma diagnosis pipeline:
1) Preprocess (impute, one-hot, scale)  
2) (Optional) Quantum Feature Selection (QAOA placeholder interface)  
3) PCA → angles → VQC (PennyLane)  
4) Train with BCEWithLogitsLoss + class imbalance handling  
5) Report Accuracy/Precision/Recall/F1/ROC-AUC

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Run training
python scripts/run_train.py \
  --csv /path/to/asthma_disease_data.csv \
  --seed 42 \
  --test-size 0.2 \
  --n-qubits 6 \
  --epochs 60 \
  --batch-size 32 \
  --lr 0.02 \
  --enable-qfs false     # set true when you plug in a real QAOA/Ising QFS
