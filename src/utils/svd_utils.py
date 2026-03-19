"""
SVD Utilities
=============
Helper functions for Singular Value Decomposition operations
used throughout the compilation pipeline.
"""

import numpy as np
from typing import Tuple, Optional


def truncated_svd(
    W: np.ndarray,
    rank: Optional[int] = None,
    energy_threshold: float = 0.99,
    return_full_unitaries: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute truncated SVD of weight matrix W.

    W ≈ U_r × diag(σ_r) × V_r†

    Args:
        W: Weight matrix (m × n)
        rank: Truncation rank. If None, determined by energy_threshold.
        energy_threshold: Fraction of total energy to retain (0 < t ≤ 1).
        return_full_unitaries: If True, return full m×m and n×n unitaries
                               instead of truncated m×r and n×r.

    Returns:
        (U, sigma, Vh): SVD factors
        - If return_full_unitaries=True: U(m,m), sigma(r,), Vh(n,n)
        - If return_full_unitaries=False: U(m,r), sigma(r,), Vh(r,n)
    """
    W_f64 = W.astype(np.float64)

    # Full SVD always for numerical stability
    U, s, Vh = np.linalg.svd(W_f64, full_matrices=True)

    # Determine rank
    if rank is None:
        rank = rank_for_energy(s, energy_threshold)
    rank = min(rank, W.shape[0], W.shape[1])

    if return_full_unitaries:
        return U.astype(np.float32), s[:rank].astype(np.float32), Vh.astype(np.float32)
    else:
        return (
            U[:, :rank].astype(np.float32),
            s[:rank].astype(np.float32),
            Vh[:rank, :].astype(np.float32),
        )


def energy_at_rank(singular_values: np.ndarray, rank: int) -> float:
    """
    Compute fraction of total energy captured by top-r singular values.

    Args:
        singular_values: Full set of singular values (descending)
        rank: Number of singular values to include

    Returns:
        Energy fraction ∈ [0, 1]
    """
    total = np.sum(singular_values ** 2)
    if total < 1e-30:
        return 1.0
    return float(np.sum(singular_values[:rank] ** 2) / total)


def rank_for_energy(
    singular_values: np.ndarray,
    target_energy: float = 0.99,
) -> int:
    """
    Find minimum rank to capture target_energy fraction of total energy.

    Args:
        singular_values: Full set of singular values (must be descending)
        target_energy: Target fraction of total energy ∈ (0, 1]

    Returns:
        Minimum rank r such that energy_at_rank(s, r) >= target_energy
    """
    total = np.sum(singular_values ** 2)
    if total < 1e-30:
        return 1

    cumulative = np.cumsum(singular_values ** 2)
    threshold = target_energy * total

    # Find first index where cumulative energy exceeds threshold
    indices = np.where(cumulative >= threshold)[0]
    if len(indices) == 0:
        return len(singular_values)
    return int(indices[0]) + 1


def reconstruction_error(
    W: np.ndarray,
    U: np.ndarray,
    sigma: np.ndarray,
    Vh: np.ndarray,
    relative: bool = True,
) -> float:
    """
    Compute reconstruction error ‖W - U Σ Vh‖_F (/ ‖W‖_F if relative).

    Args:
        W: Original matrix
        U: Left singular vectors (m×r or m×m)
        sigma: Singular values (r,)
        Vh: Right singular vectors (r×n or n×n)
        relative: If True, normalize by ‖W‖_F

    Returns:
        Frobenius norm error
    """
    r = len(sigma)
    if U.shape[1] > r:
        U = U[:, :r]
    if Vh.shape[0] > r:
        Vh = Vh[:r, :]

    W_approx = U @ np.diag(sigma.astype(np.float64)) @ Vh
    error = np.linalg.norm(W.astype(np.float64) - W_approx, 'fro')

    if relative:
        norm_W = np.linalg.norm(W, 'fro')
        error = error / (norm_W + 1e-15)

    return float(error)


def singular_value_spectrum(W: np.ndarray) -> np.ndarray:
    """
    Compute the singular value spectrum of a matrix.

    Args:
        W: Input matrix

    Returns:
        Singular values in descending order
    """
    _, s, _ = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    return s


def effective_rank(W: np.ndarray, threshold: float = 0.01) -> int:
    """
    Compute the effective rank of a matrix.

    Effective rank = number of singular values above threshold × max_sv.

    Args:
        W: Input matrix
        threshold: Relative threshold (fraction of max singular value)

    Returns:
        Effective rank
    """
    s = singular_value_spectrum(W)
    cutoff = threshold * s[0]
    return int(np.sum(s > cutoff))


def condition_number(W: np.ndarray) -> float:
    """Compute condition number σ_max / σ_min."""
    s = singular_value_spectrum(W)
    if s[-1] < 1e-15:
        return float('inf')
    return float(s[0] / s[-1])
