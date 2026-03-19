"""
Photonic Feed-Forward Network (FFN)
=====================================

Implements the GeGLU FFN used in Gemma-3 (and thus MedGemma):

    FFN(x) = down_proj( GELU(gate_proj(x)) × up_proj(x) )

where:
    gate_proj: d_model → d_ffn  (2048 → 16384)
    up_proj:   d_model → d_ffn  (2048 → 16384)
    down_proj: d_ffn → d_model  (16384 → 2048)

Photonic vs Electronic:
    PHOTONIC:
    - gate_proj: d_model → d_ffn projection
    - up_proj:   d_model → d_ffn projection
    - down_proj: d_ffn → d_model projection

    ELECTRONIC:
    - GELU activation on gate output
    - Element-wise multiplication: gate × up
    - These are nonlinear operations; pure optical nonlinearity
      is an active research area but not yet practical at scale.

Note on FFN scale:
    The FFN inner dimension (16384) is 8× the model dimension (2048).
    This makes FFN weight matrices much larger than attention projections.
    A rank-64 SVD of 16384×2048 still requires 16384×64 + 2048×64 MZIs.
    This is the largest component of the chip area budget.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from photonic.mesh import SVDLayer


@dataclass
class GemmaFFNConfig:
    """Configuration matching Gemma-3 4B FFN parameters."""
    d_model: int = 2048
    d_ffn: int = 16384
    activation: str = "gelu_pytorch_tanh"  # GeGLU gate activation
    layer_idx: int = 0


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation (PyTorch tanh approximation).

    GELU(x) = 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))

    This is the electronic nonlinearity applied after the photonic
    gate_proj matrix multiplication.

    Note: For future all-optical implementations, this can be approximated
    by semiconductor optical amplifier (SOA) gain saturation, which has
    a sigmoid-like response that approximates GELU for small signals.
    """
    return 0.5 * x * (1.0 + np.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    ))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x × sigmoid(x)."""
    return x * (1.0 / (1.0 + np.exp(-x)))


class PhotonicFeedForward:
    """
    GeGLU Feed-Forward Network with photonic weight projections.

    The gate and up projections run in PARALLEL (both read the same input x).
    This is naturally suited to WDM: gate and up can be processed simultaneously
    on different wavelengths, then combined electronically.

    Power advantage: Parallel processing means the laser power for gate_proj
    and up_proj can be shared via WDM — two computations for the price of one laser.
    """

    def __init__(
        self,
        config: GemmaFFNConfig,
        gate_proj: Optional[SVDLayer] = None,
        up_proj: Optional[SVDLayer] = None,
        down_proj: Optional[SVDLayer] = None,
    ):
        self.config = config
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(
        self,
        x: np.ndarray,
        include_loss: bool = False,
    ) -> np.ndarray:
        """
        GeGLU FFN forward pass.

        Args:
            x: Input activations, shape (..., d_model)
            include_loss: Apply photonic propagation loss

        Returns:
            y: Output activations, shape (..., d_model)
        """
        # ── Photonic projections (parallel, can be WDM-parallelized) ───────
        if self.gate_proj is not None:
            gate = self.gate_proj.forward(x)
        else:
            gate = x  # fallback

        if self.up_proj is not None:
            up = self.up_proj.forward(x)
        else:
            up = x  # fallback

        # ── Electronic: activation and element-wise multiply ────────────────
        # GELU gate — runs on electronic co-processor
        gate_activated = gelu(gate)

        # Element-wise product (GeGLU gating mechanism)
        hidden = gate_activated * up

        # ── Photonic down projection ─────────────────────────────────────────
        if self.down_proj is not None:
            output = self.down_proj.forward(hidden)
        else:
            output = hidden

        return output

    def photonic_flops(self) -> dict:
        """
        Estimate photonic vs electronic FLOP breakdown.

        Returns:
            Dict with FLOP counts for photonic and electronic operations
        """
        d_m = self.config.d_model
        d_f = self.config.d_ffn

        # Matrix multiply FLOPs: 2×m×n per vector (multiply + add)
        photonic_flops = {
            "gate_proj": 2 * d_m * d_f,
            "up_proj": 2 * d_m * d_f,
            "down_proj": 2 * d_f * d_m,
        }

        electronic_flops = {
            "gelu": 5 * d_f,        # ~5 ops for GELU approximation
            "element_mul": d_f,      # element-wise product
        }

        total_photonic = sum(photonic_flops.values())
        total_electronic = sum(electronic_flops.values())
        total = total_photonic + total_electronic

        return {
            "photonic": photonic_flops,
            "electronic": electronic_flops,
            "photonic_fraction": total_photonic / total,
            "total": total,
        }

    def resource_summary(self) -> dict:
        """Get MZI resource usage."""
        summary = {}
        for name, layer in [
            ("gate_proj", self.gate_proj),
            ("up_proj", self.up_proj),
            ("down_proj", self.down_proj),
        ]:
            if layer is not None:
                summary[name] = {
                    "in": layer.input_dim,
                    "out": layer.output_dim,
                    "rank": layer.sigma_stage.rank,
                    "n_mzis": layer.total_mzis(),
                }
        return summary

    def __repr__(self) -> str:
        gate_rank = self.gate_proj.sigma_stage.rank if self.gate_proj else "?"
        return (
            f"PhotonicFFN("
            f"d_model={self.config.d_model}, "
            f"d_ffn={self.config.d_ffn}, "
            f"rank={gate_rank})"
        )
