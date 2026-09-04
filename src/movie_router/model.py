"""A small two-head PyTorch model for evidence and payoff prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class RouterTrainConfig:
    """Conservative defaults suitable for one 24GB GPU or CPU smoke tests."""

    hidden_dim: int = 64
    depth: int = 2
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 256
    payoff_weight: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.hidden_dim, self.depth, self.epochs, self.batch_size) < 1:
            raise ValueError("hidden_dim, depth, epochs, and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0 or self.dropout < 0.0:
            raise ValueError("invalid optimization or dropout parameter")
        if self.dropout >= 1.0 or self.payoff_weight < 0.0:
            raise ValueError("dropout must be below one and payoff_weight nonnegative")


class BFMovieRouter(nn.Module):
    """Shared MLP trunk with an evidence logit and signed payoff head."""

    def __init__(self, feature_dim: int, config: RouterTrainConfig | None = None) -> None:
        super().__init__()
        cfg = config or RouterTrainConfig()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        layers: list[nn.Module] = []
        input_dim = feature_dim
        for _ in range(cfg.depth):
            layers.extend([nn.Linear(input_dim, cfg.hidden_dim), nn.GELU()])
            if cfg.dropout:
                layers.append(nn.Dropout(cfg.dropout))
            input_dim = cfg.hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.evidence_head = nn.Linear(input_dim, 1)
        self.payoff_head = nn.Linear(input_dim, 1)
        self.feature_dim = feature_dim

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError("features must have shape (batch, feature_dim)")
        hidden = self.trunk(features)
        evidence = self.evidence_head(hidden).squeeze(-1)
        payoff = torch.tanh(self.payoff_head(hidden).squeeze(-1))
        return evidence, payoff


def _validate_training_arrays(
    features: Any, relation: Any, payoff: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(relation, dtype=np.float32)
    d = np.asarray(payoff, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 1 or d.ndim != 1 or x.shape[0] != y.size or y.size != d.size:
        raise ValueError("features, relation, and payoff have incompatible shapes")
    if x.shape[0] == 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(d)):
        raise ValueError("training arrays must be nonempty and finite")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("relation must be binary")
    if np.unique(y).size < 2:
        raise ValueError("balanced relation training requires both classes")
    return x, y, d


def train_router(
    features: Any,
    relation: Any,
    payoff: Any,
    *,
    config: RouterTrainConfig | None = None,
    seed: int = 0,
) -> BFMovieRouter:
    """Fit the balanced evidence/payoff router and return an eval-mode model."""

    cfg = config or RouterTrainConfig()
    x, y, d = _validate_training_arrays(features, relation, payoff)
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {cfg.device}, but CUDA is unavailable")
    if np.any(np.abs(d) > 1.0 + 1e-6):
        raise ValueError("payoff targets must lie in [-1, 1] for the signed payoff head")
    torch.manual_seed(seed)
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = BFMovieRouter(x.shape[1], cfg).to(cfg.device)
    dataset = TensorDataset(
        torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(d)
    )
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    bce = nn.BCEWithLogitsLoss()
    huber = nn.SmoothL1Loss()
    for _ in range(cfg.epochs):
        model.train()
        for batch_x, batch_y, batch_d in loader:
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)
            batch_d = batch_d.to(cfg.device)
            evidence, predicted_payoff = model(batch_x)
            loss = bce(evidence, batch_y) + cfg.payoff_weight * huber(predicted_payoff, batch_d)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict_router(
    model: BFMovieRouter,
    features: Any,
    *,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return evidence logits and signed payoff predictions as NumPy arrays."""

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != model.feature_dim:
        raise ValueError("features have the wrong shape for this router")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = next(model.parameters()).device
    evidence_parts: list[np.ndarray] = []
    payoff_parts: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        evidence, payoff = model(batch)
        evidence_parts.append(evidence.detach().cpu().numpy())
        payoff_parts.append(payoff.detach().cpu().numpy())
    return np.concatenate(evidence_parts), np.concatenate(payoff_parts)
