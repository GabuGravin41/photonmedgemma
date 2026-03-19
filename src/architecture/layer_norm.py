"""
Electronic Layer Normalization
================================

RMSNorm — the normalization used in Gemma-3/MedGemma.
This runs on the electronic co-processor (not photonic).

Placed here for completeness in the hybrid electro-optic architecture.
RMSNorm is applied:
- Before each attention block (pre-norm)
- Before each FFN block (pre-norm)
"""

import numpy as np
from typing import Optional


class ElectronicLayerNorm:
    """
    RMSNorm: Root Mean Square Layer Normalization.

    RMSNorm(x) = x / RMS(x) × γ

    where RMS(x) = √(mean(x²) + ε)
    and γ is a learned scale parameter.

    Unlike standard LayerNorm, RMSNorm does NOT subtract the mean.
    This was shown to perform comparably with ~7% speed improvement.

    Reference:
        Zhang & Sennrich, "Root Mean Square Layer Normalization," NeurIPS 2019.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        weight: Optional[np.ndarray] = None,
    ):
        """
        Args:
            dim: Dimension of the input vectors
            eps: Small constant for numerical stability
            weight: Learned scale parameter γ. If None, uses ones.
        """
        self.dim = dim
        self.eps = eps
        self.weight = weight if weight is not None else np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply RMSNorm to input.

        Args:
            x: Input, shape (..., dim)

        Returns:
            Normalized output, same shape
        """
        # Compute RMS along last dimension
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

        # Normalize and scale
        return (x / rms) * self.weight

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"RMSNorm(dim={self.dim}, eps={self.eps})"


class ElectronicLayerNormFactory:
    """Factory for creating RMSNorm layers from model weights."""

    @staticmethod
    def from_weights(
        weight: np.ndarray,
        eps: float = 1e-6,
    ) -> ElectronicLayerNorm:
        """Create RMSNorm from a weight array."""
        dim = len(weight)
        return ElectronicLayerNorm(dim=dim, eps=eps, weight=weight.astype(np.float32))
