"""
DE-FAKE-style attribution head trained on frozen CLIP features (Phase E).

We freeze the CLIP backbone (we only ever consume precomputed embeddings) and learn a small
MLP head over the image embedding - the same recipe the GOLD review endorsed for adding
new generator classes (FLUX, StyleGAN3) without retraining CLIP.

System interpreter only (torch lives there). ASCII-only; Python 3.9.
"""
from typing import Dict, List, Tuple

import numpy as np


class _MLPHead:
    """Thin wrapper around a torch MLP so callers do not import torch directly."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256,
                 dropout: float = 0.3, device: str = "cuda", seed: int = 42):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None,
            epochs: int = 60, lr: float = 1e-3, batch_size: int = 64,
            weight_decay: float = 1e-4, class_weights=None, logger=None):
        import torch
        import torch.nn as nn

        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        cw = None
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        n = len(Xt)
        best_val = -1.0
        best_state = None
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            total = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optim.zero_grad()
                logits = self.model(Xt[idx])
                loss = criterion(logits, yt[idx])
                loss.backward()
                optim.step()
                total += float(loss.item())
            if X_val is not None and len(X_val) > 0:
                val_acc = (self.predict(X_val) == y_val).mean()
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if logger and (epoch % 10 == 0 or epoch == epochs - 1):
                    logger.info("epoch %d loss=%.4f val_acc=%.3f", epoch, total, val_acc)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.model(Xt)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def save(self, path: str, classes: List[str]):
        import torch
        torch.save({"state_dict": self.model.state_dict(), "classes": classes}, path)


def encode_labels(generators: np.ndarray, classes: List[str]) -> np.ndarray:
    """Map generator strings to integer class indices given an ordered class list."""
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[g] for g in generators], dtype=np.int64)


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights (handles small/imbalanced generator sets)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def stratified_split(y: np.ndarray, test_size: float, val_size: float, seed: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays stratified by class label."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                           stratify=y, random_state=seed)
    if val_size > 0:
        rel_val = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(train_idx, test_size=rel_val,
                                              stratify=y[train_idx], random_state=seed)
    else:
        val_idx = np.array([], dtype=int)
    return train_idx, val_idx, test_idx
