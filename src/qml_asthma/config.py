from dataclasses import dataclass

@dataclass
class TrainConfig:
    csv_path: str
    seed: int = 42
    test_size: float = 0.2

    # columns
    target_col: str = "Diagnosis"
    drop_id_cols: tuple[str, ...] = ("PatientID",)

    # model / encoding
    n_qubits: int = 6
    vqc_layers: int = 3

    # training
    batch_size: int = 32
    epochs: int = 60
    lr: float = 0.02
    patience: int = 8

    # QFS
    enable_qfs: bool = False
    qfs_max_features: int = 64  
