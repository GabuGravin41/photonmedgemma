"""
Error Analysis — Fabrication and Numerical Error Bounds
========================================================

Analyzes errors arising from:
1. SVD truncation (rank approximation of weight matrices)
2. Phase quantization (DAC resolution)
3. Fabrication imperfections (MZI phase errors, coupler imbalance)
4. Propagation loss (optical power decay)
5. Thermal drift (temperature changes shifting phases)

These error models inform:
- Minimum required rank for acceptable accuracy
- Required DAC resolution
- Fabrication tolerance specifications for the foundry
- Calibration procedure design
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ErrorBudget:
    """Complete error budget for one weight matrix."""

    # SVD truncation error
    svd_error: float = 0.0
    svd_energy_retained: float = 0.0
    svd_rank: int = 0

    # Phase quantization error
    phase_quantization_error: float = 0.0
    dac_bits: int = 12

    # Fabrication errors
    mzi_phase_error: float = 0.0      # from phase shifter imprecision
    coupler_error: float = 0.0        # from directional coupler imbalance
    fabrication_total: float = 0.0

    # Propagation loss
    propagation_loss_error: float = 0.0

    # Combined error estimate
    total_error_estimate: float = 0.0
    total_error_bound: float = 0.0     # worst-case upper bound

    def report(self) -> str:
        """Human-readable error budget."""
        lines = [
            "=== Error Budget ===",
            f"  SVD truncation:        {self.svd_error:.4f} (rank={self.svd_rank}, energy={self.svd_energy_retained:.4f})",
            f"  Phase quantization:    {self.phase_quantization_error:.4f} ({self.dac_bits}-bit DAC)",
            f"  Fabrication (MZI):     {self.mzi_phase_error:.4f}",
            f"  Fabrication (coupler): {self.coupler_error:.4f}",
            f"  Propagation loss:      {self.propagation_loss_error:.4f}",
            f"  ─────────────────────────────────────",
            f"  Total (RMS estimate):  {self.total_error_estimate:.4f}",
            f"  Total (worst-case):    {self.total_error_bound:.4f}",
            "=" * 40,
        ]
        return "\n".join(lines)


class ErrorAnalyzer:
    """
    Analyzes and combines error sources for photonic neural network layers.

    Usage:
        analyzer = ErrorAnalyzer(N=2048, rank=64, dac_bits=12)
        budget = analyzer.full_error_budget(W)
        print(budget.report())
    """

    def __init__(
        self,
        N: int = 2048,
        rank: int = 64,
        dac_bits: int = 12,
        chip_length_cm: float = 1.0,
        loss_db_per_cm: float = 0.2,
        mzi_phase_std: float = 0.01,    # rad, typical fabrication error
        coupler_imbalance: float = 0.01,  # fractional deviation from 0.5
    ):
        """
        Args:
            N: Matrix dimension
            rank: SVD truncation rank
            dac_bits: DAC resolution in bits
            chip_length_cm: Total optical path length [cm]
            loss_db_per_cm: Propagation loss [dB/cm]
            mzi_phase_std: Std dev of MZI phase fabrication error [rad]
            coupler_imbalance: Std dev of coupler splitting ratio error
        """
        self.N = N
        self.rank = rank
        self.dac_bits = dac_bits
        self.chip_length_cm = chip_length_cm
        self.loss_db_per_cm = loss_db_per_cm
        self.mzi_phase_std = mzi_phase_std
        self.coupler_imbalance = coupler_imbalance

    def svd_truncation_error(self, W: np.ndarray) -> Tuple[float, float]:
        """
        Compute SVD truncation error for a weight matrix.

        Args:
            W: Weight matrix (m × n)

        Returns:
            (relative_error, energy_retained)
        """
        _, s, _ = np.linalg.svd(W.astype(np.float64), full_matrices=False)
        total = np.sum(s ** 2)
        retained = np.sum(s[:self.rank] ** 2)
        energy = retained / (total + 1e-30)
        error = np.sqrt(1.0 - energy)
        return float(error), float(energy)

    def phase_quantization_error_bound(self) -> float:
        """
        Upper bound on matrix reconstruction error from phase quantization.

        For N-mode Clements mesh with B-bit DAC:
        ε_quant ≤ π × √(N(N-1)/2) / 2^B

        Returns:
            Error bound (relative Frobenius norm)
        """
        n_mzis = self.N * (self.N - 1) // 2
        lsb = np.pi / (1 << self.dac_bits)
        return float(lsb * np.sqrt(2 * n_mzis) / self.N)

    def fabrication_error_bound(self) -> Tuple[float, float]:
        """
        Estimate matrix reconstruction errors from fabrication imperfections.

        Models:
        - MZI phase errors: Gaussian noise on θ and φ with std mzi_phase_std
        - Coupler imbalance: Gaussian noise on coupling ratio κ

        Returns:
            (mzi_phase_error, coupler_error) — relative Frobenius norm estimates
        """
        n_mzis = self.N * (self.N - 1) // 2

        # Phase error: each MZI error contributes ~phase_std/√N to matrix error
        mzi_phase_error = self.mzi_phase_std * np.sqrt(2 * n_mzis) / self.N

        # Coupler error: similar scaling
        # For 50:50 coupler with δκ = 0.01: matrix element error ≈ δκ/2
        coupler_error = self.coupler_imbalance * np.sqrt(n_mzis) / self.N

        return float(mzi_phase_error), float(coupler_error)

    def propagation_loss_error(self) -> float:
        """
        Estimate accuracy impact of optical propagation loss.

        Loss causes the output amplitudes to be uniformly attenuated.
        This introduces a systematic scaling error that can be compensated
        by adjusting the laser power or detector gain.

        After compensation, the residual error is due to non-uniform loss
        (different modes see slightly different loss due to routing).

        Returns:
            Estimated relative error from non-uniform loss
        """
        total_loss_db = self.loss_db_per_cm * self.chip_length_cm
        loss_linear = 10 ** (-total_loss_db / 20)

        # Non-uniform loss: assume ±10% variation in path lengths
        path_variation = 0.1  # 10% variation
        nonuniform_loss_db = total_loss_db * path_variation

        # Convert to amplitude error
        error = 1.0 - 10 ** (-nonuniform_loss_db / 20)
        return float(error)

    def full_error_budget(self, W: Optional[np.ndarray] = None) -> ErrorBudget:
        """
        Compute the complete error budget for a weight matrix.

        Args:
            W: Optional weight matrix for SVD error computation.
               If None, SVD error is estimated from rank/N.

        Returns:
            ErrorBudget with all components
        """
        budget = ErrorBudget(svd_rank=self.rank, dac_bits=self.dac_bits)

        # SVD truncation
        if W is not None:
            budget.svd_error, budget.svd_energy_retained = self.svd_truncation_error(W)
        else:
            # Estimate: for typical transformer weights, energy ≈ 1 - (rank/N)^0.5
            budget.svd_energy_retained = 1.0 - (1.0 - self.rank / self.N) ** 0.5
            budget.svd_error = np.sqrt(1.0 - budget.svd_energy_retained)

        # Phase quantization
        budget.phase_quantization_error = self.phase_quantization_error_bound()

        # Fabrication
        budget.mzi_phase_error, budget.coupler_error = self.fabrication_error_bound()
        budget.fabrication_total = np.sqrt(
            budget.mzi_phase_error ** 2 + budget.coupler_error ** 2
        )

        # Propagation loss
        budget.propagation_loss_error = self.propagation_loss_error()

        # Combined error (RMS of independent sources)
        budget.total_error_estimate = np.sqrt(
            budget.svd_error ** 2
            + budget.phase_quantization_error ** 2
            + budget.fabrication_total ** 2
            + budget.propagation_loss_error ** 2
        )

        # Worst-case bound (sum of all errors)
        budget.total_error_bound = (
            budget.svd_error
            + budget.phase_quantization_error
            + budget.fabrication_total
            + budget.propagation_loss_error
        )

        return budget

    def minimum_rank_for_accuracy(
        self,
        target_accuracy: float = 0.95,
        singular_values: Optional[np.ndarray] = None,
    ) -> int:
        """
        Find minimum SVD rank to achieve target task accuracy.

        The relationship between SVD reconstruction error and task accuracy
        is task-dependent. We use a simple linear approximation:
            task_accuracy ≈ 1 - k × reconstruction_error

        where k ≈ 2–5 for transformer models (empirical).

        Args:
            target_accuracy: Minimum acceptable task accuracy [0, 1]
            singular_values: Optional singular values for exact computation

        Returns:
            Minimum required rank
        """
        # Total non-SVD error floor
        quant_error = self.phase_quantization_error_bound()
        fab_error_mzi, fab_error_coupler = self.fabrication_error_bound()
        error_floor = np.sqrt(quant_error**2 + fab_error_mzi**2 + fab_error_coupler**2)

        target_svd_error = (1.0 - target_accuracy) / 3.0  # allow SVD 1/3 of error budget
        target_energy = max(0.0, 1.0 - target_svd_error ** 2)

        if singular_values is not None:
            from .svd_utils import rank_for_energy
            return rank_for_energy(singular_values, target_energy)

        # Rough estimate without actual singular values
        # For typical transformer weights: rank ≈ N × (1 - target_energy)^2
        return max(1, int(self.N * (1.0 - target_energy) ** 2))


def fabrication_error_bound(
    N: int,
    mzi_phase_std: float = 0.01,
    coupler_error: float = 0.01,
) -> float:
    """
    Standalone function: compute fabrication error bound for N-mode Clements mesh.

    Args:
        N: Matrix dimension
        mzi_phase_std: Phase error std dev [rad]
        coupler_error: Coupler imbalance std dev

    Returns:
        Relative matrix error estimate
    """
    n_mzis = N * (N - 1) // 2
    phase_err = mzi_phase_std * np.sqrt(2 * n_mzis) / N
    coup_err = coupler_error * np.sqrt(n_mzis) / N
    return float(np.sqrt(phase_err**2 + coup_err**2))
