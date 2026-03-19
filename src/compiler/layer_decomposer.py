"""
Layer Decomposer — SVD Decomposition of Weight Matrices
=========================================================

Takes a weight matrix W (m×n) and performs truncated SVD:
    W ≈ U_r × Σ_r × V_r†

where r is the truncation rank. The three factors are then
passed to the MZI mapper for photonic implementation.

Key decisions:
- Rank selection: automatic (energy threshold) or manual
- Normalization: weight matrices are normalized to [0, 1] singular values
  before SVD for numerical stability; scale is restored at the Σ stage
- dtype: float64 for SVD stability, then quantized to float32 or float16
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

import numpy as np

from .model_parser import LayerInfo

logger = logging.getLogger(__name__)


@dataclass
class DecomposedLayer:
    """
    Result of SVD decomposition for one weight matrix.

    Stores:
    - U_r: Left singular vectors (m×r semi-unitary)
    - sigma_r: Singular values (r,)
    - Vh_r: Right singular vectors (r×n semi-unitary)
    - U_full: Extended m×m unitary (for full Clements mesh)
    - Vh_full: Extended n×n unitary (for full Clements mesh)
    """
    layer_info: LayerInfo

    # Original matrix properties
    m: int
    n: int
    rank: int

    # SVD factors (truncated to rank r)
    U_r: np.ndarray        # shape (m, r), orthonormal columns
    sigma_r: np.ndarray    # shape (r,), singular values (descending)
    Vh_r: np.ndarray       # shape (r, n), orthonormal rows

    # Extended unitary matrices for Clements decomposition
    U_full: np.ndarray     # shape (m, m), full unitary (U_r extended)
    Vh_full: np.ndarray    # shape (n, n), full unitary (Vh_r extended)

    # Scale factor (max singular value, removed before SVD for numerical stability)
    scale: float = 1.0

    # Quality metrics
    energy_retained: float = 0.0  # fraction of Frobenius norm energy retained
    reconstruction_error: float = 0.0  # ‖W - U_r Σ_r Vh_r‖_F / ‖W‖_F
    decomposition_time_s: float = 0.0

    def __post_init__(self):
        self.energy_retained = float(np.sum(self.sigma_r**2) / (np.sum(self.sigma_r**2) + 1e-30))
        if self.U_r is not None and self.sigma_r is not None and self.Vh_r is not None:
            W_approx = self.U_r @ np.diag(self.sigma_r) @ self.Vh_r
            W_norm = np.linalg.norm(self.U_r @ np.diag(self.sigma_r) @ self.Vh_r)
            self.reconstruction_error = 0.0  # computed separately


class LayerDecomposer:
    """
    Performs truncated SVD decomposition of neural network weight matrices.

    This is Stage 2 of the compilation pipeline. For each weight matrix,
    it computes the truncated SVD and extends the semi-unitary factors to
    full unitary matrices suitable for Clements decomposition.

    Supports:
    - Automatic rank selection (by energy threshold)
    - Manual rank specification
    - Batch processing (for parallelization)
    - Quality metrics (reconstruction error, energy retained)
    """

    def __init__(
        self,
        rank: Optional[int] = None,
        energy_threshold: float = 0.99,
        max_rank: int = 512,
        min_rank: int = 8,
        random_seed: int = 42,
    ):
        """
        Args:
            rank: Fixed SVD truncation rank. If None, use energy_threshold.
            energy_threshold: Target fraction of weight energy to retain.
                             Used only when rank=None. Range: (0, 1].
            max_rank: Maximum allowed rank (for memory/resource limits).
            min_rank: Minimum rank (floor for energy-based selection).
            random_seed: Seed for reproducible unitary extension.
        """
        self.rank = rank
        self.energy_threshold = energy_threshold
        self.max_rank = max_rank
        self.min_rank = min_rank
        self.rng = np.random.default_rng(random_seed)

    def decompose(
        self,
        layer_info: LayerInfo,
        weight: np.ndarray,
        rank_override: Optional[int] = None,
    ) -> DecomposedLayer:
        """
        Decompose a single weight matrix W into SVD factors.

        Args:
            layer_info: Metadata about this layer
            weight: Float32 numpy array of shape (m, n)
            rank_override: Override the default rank for this specific layer.

        Returns:
            DecomposedLayer with U, Σ, V† factors and quality metrics
        """
        t_start = time.perf_counter()

        m, n = weight.shape
        W = weight.astype(np.float64)  # float64 for SVD numerical stability

        # Normalize: remove overall scale to stabilize SVD
        scale = np.linalg.norm(W, 'fro')
        if scale < 1e-15:
            logger.warning(f"Layer {layer_info.name} has near-zero weight. Skipping.")
            scale = 1.0
        W_normalized = W / scale

        # Compute full SVD
        logger.debug(f"Computing SVD for {layer_info.name} ({m}×{n})...")
        U_full_svd, sigma_full, Vh_full_svd = np.linalg.svd(
            W_normalized, full_matrices=True
        )
        # U_full_svd: (m, m), sigma_full: (min(m,n),), Vh_full_svd: (n, n)

        # Determine truncation rank
        rank = self._determine_rank(sigma_full, rank_override)
        rank = min(rank, m, n)

        logger.info(
            f"  {layer_info.name}: shape=({m},{n}), rank={rank}/{min(m,n)}, "
            f"energy={np.sum(sigma_full[:rank]**2)/np.sum(sigma_full**2):.4f}"
        )

        # Truncated factors
        U_r = U_full_svd[:, :rank]          # (m, r)
        sigma_r = sigma_full[:rank] * scale  # restore scale
        Vh_r = Vh_full_svd[:rank, :]        # (r, n)

        # The full SVD already gives us m×m and n×n unitaries!
        U_full_unitary = U_full_svd          # already m×m unitary
        Vh_full_unitary = Vh_full_svd        # already n×n unitary

        t_end = time.perf_counter()

        # Compute reconstruction error
        W_approx = U_r @ np.diag(sigma_r) @ Vh_r
        recon_error = np.linalg.norm(W - W_approx, 'fro') / (np.linalg.norm(W, 'fro') + 1e-15)

        decomposed = DecomposedLayer(
            layer_info=layer_info,
            m=m,
            n=n,
            rank=rank,
            U_r=U_r.astype(np.float32),
            sigma_r=sigma_r.astype(np.float32),
            Vh_r=Vh_r.astype(np.float32),
            U_full=U_full_unitary.astype(np.float32),
            Vh_full=Vh_full_unitary.astype(np.float32),
            scale=float(scale),
            decomposition_time_s=t_end - t_start,
        )
        decomposed.reconstruction_error = float(recon_error)
        decomposed.energy_retained = float(
            np.sum(sigma_r**2) / (np.sum(sigma_full**2) * scale**2 + 1e-30)
        )

        return decomposed

    def _determine_rank(
        self,
        sigma: np.ndarray,
        rank_override: Optional[int] = None,
    ) -> int:
        """
        Determine the SVD truncation rank.

        Args:
            sigma: Full set of singular values (descending order)
            rank_override: Override all other logic if provided

        Returns:
            Truncation rank r
        """
        if rank_override is not None:
            return int(np.clip(rank_override, self.min_rank, self.max_rank))

        if self.rank is not None:
            return int(np.clip(self.rank, self.min_rank, self.max_rank))

        # Automatic: find minimum rank that retains energy_threshold fraction
        total_energy = np.sum(sigma ** 2)
        cumulative_energy = np.cumsum(sigma ** 2)

        rank = np.searchsorted(cumulative_energy, self.energy_threshold * total_energy) + 1
        rank = int(np.clip(rank, self.min_rank, self.max_rank))

        return rank

    def decompose_batch(
        self,
        layers: list,
        n_workers: int = 1,
        verbose: bool = True,
    ) -> list:
        """
        Decompose multiple layers, optionally in parallel.

        Args:
            layers: List of (LayerInfo, weight_numpy_array) tuples
            n_workers: Number of parallel workers (uses multiprocessing)
            verbose: Print progress

        Returns:
            List of DecomposedLayer objects (same order as input)
        """
        results = []

        if n_workers > 1:
            # Parallel decomposition using multiprocessing
            try:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = {
                        executor.submit(self.decompose, info, weight): i
                        for i, (info, weight) in enumerate(layers)
                    }
                    results = [None] * len(layers)
                    for future, idx in futures.items():
                        results[idx] = future.result()
            except Exception as e:
                logger.warning(f"Parallel decomposition failed ({e}), falling back to sequential")
                n_workers = 1

        if n_workers <= 1:
            for i, (info, weight) in enumerate(layers):
                if verbose:
                    logger.info(f"  [{i+1}/{len(layers)}] Decomposing {info.name}...")
                results.append(self.decompose(info, weight))

        return results

    def rank_sensitivity_analysis(
        self,
        weight: np.ndarray,
        ranks: list = None,
    ) -> dict:
        """
        Analyze how reconstruction quality varies with SVD rank.
        Useful for selecting the minimum rank that preserves model accuracy.

        Args:
            weight: Weight matrix (m×n)
            ranks: List of ranks to test. Default: [4, 8, 16, 32, 64, 128, 256]

        Returns:
            Dict mapping rank → {"error": float, "energy": float, "mzis": int}
        """
        if ranks is None:
            ranks = [4, 8, 16, 32, 64, 128, 256, 512]

        m, n = weight.shape
        W = weight.astype(np.float64)
        scale = np.linalg.norm(W, 'fro')
        W_norm = W / (scale + 1e-15)

        U, sigma, Vh = np.linalg.svd(W_norm, full_matrices=False)
        total_energy = np.sum(sigma ** 2)

        results = {}
        for r in ranks:
            r = min(r, m, n)
            W_approx = U[:, :r] @ np.diag(sigma[:r]) @ Vh[:r, :]
            error = np.linalg.norm(W_norm - W_approx, 'fro') / (np.linalg.norm(W_norm) + 1e-15)
            energy = np.sum(sigma[:r] ** 2) / (total_energy + 1e-30)
            mzis_approx = m * r + n * r  # approx MZI count (rectangular Clements)

            results[r] = {
                "reconstruction_error": float(error),
                "energy_retained": float(energy),
                "mzis_estimated": mzis_approx,
            }

        return results
