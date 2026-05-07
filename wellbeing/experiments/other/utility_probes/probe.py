"""Heteroscedastic linear probe for predicting pairwise utility preferences."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroscedasticProbe(nn.Module):
    """Linear probe that predicts a mean utility and learned uncertainty (std).

    For a pair of activations (h_a, h_b), the predicted preference probability is:
        P(A > B) = sigmoid((mu_a - mu_b) / sqrt(sigma_a^2 + sigma_b^2))
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        mean = self.mean_head(x).squeeze(-1)
        std = F.softplus(self.log_std_head(x).squeeze(-1)) + 1e-6
        return mean, std

    def predict_preference(self, h_a, h_b):
        """Predict P(A preferred over B)."""
        mu_a, sig_a = self.forward(h_a)
        mu_b, sig_b = self.forward(h_b)
        return torch.sigmoid((mu_a - mu_b) / torch.sqrt(sig_a**2 + sig_b**2))
