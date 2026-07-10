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
            nb = 0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                optim.zero_grad()
                logits = self.model(Xt[idx])
                loss = criterion(logits, yt[idx])
                loss.backward()
                optim.step()
                total += float(loss.item())
                nb += 1
            if X_val is not None and len(X_val) > 0:
                # Select on BALANCED accuracy, not plain accuracy: the loss is class-weighted
                # for imbalance, so checkpoint selection must be too, or the majority (real)
                # class dominates the choice and undoes the weighting.
                val_pred = self.predict(X_val)
                val_bal = balanced_accuracy(np.asarray(y_val), val_pred)
                if val_bal > best_val:
                    best_val = val_bal
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                if logger and (epoch % 10 == 0 or epoch == epochs - 1):
                    val_acc = float((val_pred == y_val).mean())
                    logger.info("epoch %d loss=%.4f val_balAcc=%.3f val_acc=%.3f",
                                epoch, total / max(1, nb), val_bal, val_acc)
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


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall (numpy-only, no sklearn) over the classes present in y_true.

    Kept dependency-free so the head stays torch/numpy-only; matches the definition used by
    sklearn.balanced_accuracy_score when every true class is present.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    if classes.size == 0:
        return 0.0
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in classes]
    return float(np.mean(recalls))


def encode_labels(generators: np.ndarray, classes: List[str]) -> np.ndarray:
    """Map generator strings to integer class indices given an ordered class list."""
    index = {c: i for i, c in enumerate(classes)}
    return np.array([index[g] for g in generators], dtype=np.int64)


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency class weights (handles small/imbalanced generator sets)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = counts.sum()                 # true sample count, before empty-class substitution
    counts[counts == 0] = 1.0            # avoid divide-by-zero for classes with no samples
    weights = total / (num_classes * counts)
    return weights.astype(np.float32)


def _hash_unit(key: str, seed: int) -> float:
    """Deterministic pseudo-random value in [0, 1) from a string key + seed (SHA-256)."""
    import hashlib
    h = hashlib.sha256(("%d:%s" % (seed, key)).encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def _hash_stratified_split(y: np.ndarray, keys: np.ndarray, test_size: float,
                           val_size: float, seed: int, groups: np.ndarray = None
                           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Content-stable stratified split: each sample's bucket depends only on a hash of its own
    key + seed, so adding/removing OTHER samples (e.g. a few images failing to load) never
    reshuffles the rest. Within each class, samples are ranked by hash and the lowest
    test_size fraction -> test, next val_size -> val, remainder -> train (per-class counts kept).

    GROUP-AWARE (optional): `groups` is a per-sample group id (e.g. OpenForensics source
    image_id) such that all samples sharing a group id MUST land on the same split side - the
    same-source-photo real/fake coupling fix. Samples whose group id is unique to them
    ("singleton" groups - every non-OpenForensics row, and any OpenForensics crop whose source
    photo contributed only one sampled crop) are split EXACTLY as before via the per-class hash
    ranking above (byte-identical when `groups` is None, since every key is then its own
    singleton group). Samples in a multi-member group are assigned WHOLE-GROUP via a hash of the
    group id against the same test_size/val_size thresholds, trading exact per-class
    stratification (for that small coupled subset only) for a hard guarantee against splitting
    one source photo's real and fake crops onto different sides.

    IMPORTANT: "grouped" is determined by whether a row has an EXPLICIT group id different from
    its own key (i.e. `groups[i] != keys[i]`, meaning `io_utils.apply_group_map` found a sidecar
    entry for it) - NOT by counting how many rows of that group happen to be present in THIS
    call's arrays. Counting co-occurrence within the call would make a row's bucket depend on
    which OTHER rows the caller happened to include: e.g. finetune_defake_head.py restricts to
    TRAINED classes before splitting, which removes the out-of-set sibling from an
    OpenForensics-real/OpenForensics-fake coupled pair (OpenForensics-fake is out-of-set) - so
    the surviving OpenForensics-real row would look like a lone singleton there but a 2-member
    group in make_split.py's UNRESTRICTED call, and the two functions would (and did, until this
    fix) disagree on that row's split side. Keying "grouped" off the id itself makes the decision
    depend only on (group_id, seed), identical across every caller regardless of population
    filtering.
    """
    if groups is None:
        groups = keys
    groups = np.asarray([str(g) for g in groups])
    keys = np.asarray(keys)
    is_grouped = np.array([g != k for g, k in zip(groups, keys)])

    tr, va, te = [], [], []

    # Ungrouped rows: the ORIGINAL per-class hash-ranked split, scoped to ungrouped rows only
    # (identical output to the pre-group-aware function whenever every row is ungrouped).
    s_idx = np.where(~is_grouped)[0]
    if s_idx.size:
        s_y = y[s_idx]
        for c in np.unique(s_y):
            cls_idx = s_idx[s_y == c]
            scores = np.array([_hash_unit(str(keys[i]), seed) for i in cls_idx])
            order = cls_idx[np.argsort(scores, kind="stable")]
            n = len(order)
            n_test = int(round(n * test_size))
            n_val = int(round(n * val_size))
            te.extend(order[:n_test].tolist())
            va.extend(order[n_test:n_test + n_val].tolist())
            tr.extend(order[n_test + n_val:].tolist())

    # Grouped rows (explicit sidecar-assigned id): assign the WHOLE group by a hash of the group
    # id (not per-sample, not per-call-population), so no group can straddle train/val/test AND
    # the decision is stable regardless of which other rows a particular caller filtered out
    # first. Only the OpenForensics coupled subset (or any other future grouped dataset) takes
    # this path; a row here may be the ONLY member of its group present in this call (e.g. its
    # sibling was filtered out as out-of-set) and that is fine - the hash still only depends on
    # the group id + seed.
    g_idx = np.where(is_grouped)[0]
    if g_idx.size:
        for g in sorted(set(groups[g_idx].tolist())):
            members = g_idx[groups[g_idx] == g]
            score = _hash_unit("GROUP:%s" % g, seed)
            if score < test_size:
                te.extend(members.tolist())
            elif score < test_size + val_size:
                va.extend(members.tolist())
            else:
                tr.extend(members.tolist())

    return (np.array(sorted(tr), dtype=int), np.array(sorted(va), dtype=int),
            np.array(sorted(te), dtype=int))


def stratified_split(y: np.ndarray, test_size: float, val_size: float, seed: int,
                     keys: np.ndarray = None, groups: np.ndarray = None
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays stratified by class label.

    If `keys` (one stable identifier per sample, e.g. full_path) is given, the split is
    content-stable via per-key hashing (recommended - reproducible regardless of row order or
    a few dropped images). Otherwise falls back to sklearn's positional stratified split.

    `groups` (optional, requires `keys`): a per-sample group id; every sample sharing a group id
    is kept on the same split side (see `_hash_stratified_split`). Pass this whenever a dataset
    can contribute multiple correlated samples from one source (currently: OpenForensics
    same-source-photo real/fake crops via `source_image_id`).
    """
    y = np.asarray(y)
    if keys is not None:
        return _hash_stratified_split(y, np.asarray(keys), test_size, val_size, seed,
                                      groups=groups)
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
