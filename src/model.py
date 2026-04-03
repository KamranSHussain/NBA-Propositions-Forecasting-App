"""Model definitions for NBA player-prop quantile regression."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

DEFAULT_QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)


class PinballLoss(nn.Module):
    """Pinball loss for multi-quantile regression.

    The module expects predictions shaped as ``(batch_size, num_quantiles)`` and
    labels shaped as ``(batch_size, 1)`` (or broadcast-compatible).
    """

    def __init__(self, quantiles: Sequence[float] = DEFAULT_QUANTILES) -> None:
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute average pinball loss across all configured quantiles."""
        total_loss = 0.0
        for i, q in enumerate(self.quantiles):
            pred_q = y_pred[:, i : i + 1]
            error = y_true - pred_q
            loss_q = torch.maximum(q * error, (q - 1.0) * error)
            total_loss += torch.mean(loss_q)

        return total_loss / len(self.quantiles)


class PlayerPropNN(nn.Module):
    """Feedforward network that predicts floor/median/ceiling outcomes."""

    #Look into: transformer, set transformer

    def __init__(self, input_size: int = 33) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(DEFAULT_QUANTILES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a forward pass over feature tensors."""
        return self.network(features)


__all__ = ["DEFAULT_QUANTILES", "PinballLoss", "PlayerPropNN"]
    