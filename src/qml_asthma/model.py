from __future__ import annotations
import torch
import torch.nn as nn
import pennylane as qml

class VQC(nn.Module):
    """
    Variational quantum classifier using:
      - AngleEmbedding (RY) on n_qubits
      - StronglyEntanglingLayers(weights)
      - Z expectation on wire 0
      - Linear map to logit via learnable scale/bias
    """
    def __init__(self, n_qubits: int, n_layers: int = 3, shots: int | None = None):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias  = nn.Parameter(torch.tensor(0.0))

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
        self.qnode = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, n_qubits] OR [n_qubits]
        returns logits: [B, 1]
        """
        if x.dim() == 1:
            expval = self.qnode(x, self.weights)
            if expval.dim() == 0:
                expval = expval.unsqueeze(0)
            return (self.scale * expval + self.bias).unsqueeze(1)

        # batched (loop — simple & robust)
        outs = []
        for i in range(x.size(0)):
            ev = self.qnode(x[i], self.weights)
            if ev.dim() == 0:
                ev = ev.unsqueeze(0)
            outs.append(self.scale * ev + self.bias)
        return torch.stack(outs, dim=0).unsqueeze(1).squeeze(-1)
