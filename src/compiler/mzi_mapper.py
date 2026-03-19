"""
MZI Mapper — Map Decomposed Layers to Photonic MZI Meshes
===========================================================

Takes the SVD-decomposed weight matrices (U, Σ, V†) and:
1. Runs Clements decomposition on U and V† to get MZI phase angles
2. Creates SVDLayer objects representing the full photonic layer
3. Tracks chip assignments and resource usage

This is Stage 3 of the compilation pipeline.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from .layer_decomposer import DecomposedLayer
from .clements import clements_decompose, ClementsResult
from photonic.mesh import ClementsMesh, SigmaStage, SVDLayer, MeshConfig

logger = logging.getLogger(__name__)


@dataclass
class CompiledLayer:
    """
    A fully compiled photonic layer, ready for netlist generation.

    Contains:
    - U_mesh: Clements mesh for the U factor
    - sigma_stage: Attenuator stage for Σ
    - Vh_mesh: Clements mesh for V† factor
    - Resource estimates
    """
    layer_name: str
    component: str         # "language_model" | "vision_encoder"
    projection_type: str   # "q", "k", "v", etc.
    transformer_layer_idx: Optional[int]

    U_mesh: ClementsMesh
    sigma_stage: SigmaStage
    Vh_mesh: ClementsMesh

    # Resource tracking
    n_mzis_U: int = 0
    n_mzis_Vh: int = 0
    chip_id_U: int = 0
    chip_id_Vh: int = 0

    # Quality
    reconstruction_error_svd: float = 0.0  # from SVD truncation
    reconstruction_error_clements_U: float = 0.0  # from Clements decomp
    reconstruction_error_clements_Vh: float = 0.0
    compilation_time_s: float = 0.0

    @property
    def total_mzis(self) -> int:
        return self.n_mzis_U + self.n_mzis_Vh

    @property
    def rank(self) -> int:
        return self.sigma_stage.rank


class MZIMapper:
    """
    Maps SVD-decomposed weight matrices to photonic MZI meshes.

    Runs the Clements decomposition on U and V† unitary matrices,
    creating ClementsMesh objects that encode the phase settings
    for all MZIs on the chip.

    Resource tracking:
    - Assigns chip IDs to each mesh (based on MZI count per chip)
    - Tracks total MZI count across the full model

    Note on computation time:
    - Clements decomposition of N×N matrix: O(N³) — slow for large N
    - For N=2048, each decomposition takes ~30-120 seconds
    - Use n_workers > 1 for parallel decomposition
    - For prototyping, use small N (e.g., N=64) with high rank
    """

    def __init__(
        self,
        mzis_per_chip: int = 4096,
        dac_bits: int = 12,
        wavelength: float = 1310e-9,
    ):
        """
        Args:
            mzis_per_chip: Number of MZIs that fit on one chip (for resource tracking)
            dac_bits: DAC resolution for phase quantization
            wavelength: Operating wavelength [m]
        """
        self.mzis_per_chip = mzis_per_chip
        self.dac_bits = dac_bits
        self.wavelength = wavelength

        # Resource tracking state
        self._total_mzis = 0
        self._chip_id = 0
        self._mzis_on_current_chip = 0

    def map_layer(
        self,
        decomposed: DecomposedLayer,
        verbose: bool = True,
    ) -> CompiledLayer:
        """
        Map one decomposed weight matrix to a photonic SVD layer.

        Args:
            decomposed: DecomposedLayer from LayerDecomposer
            verbose: Print progress

        Returns:
            CompiledLayer with compiled U and V† Clements meshes
        """
        t_start = time.perf_counter()

        info = decomposed.layer_info
        if verbose:
            logger.info(
                f"Mapping {info.name} ({decomposed.m}×{decomposed.n}, "
                f"rank={decomposed.rank}) to MZI meshes..."
            )

        # Decompose V† (n×n unitary)
        if verbose:
            logger.info(f"  Clements decomposition of V† ({decomposed.n}×{decomposed.n})...")
        Vh_full = decomposed.Vh_full.astype(np.float64)

        # Ensure Vh_full is unitary
        Vh_full = _orthogonalize(Vh_full)
        Vh_clements = clements_decompose(Vh_full)

        # Decompose U (m×m unitary)
        if verbose:
            logger.info(f"  Clements decomposition of U ({decomposed.m}×{decomposed.m})...")
        U_full = decomposed.U_full.astype(np.float64)
        U_full = _orthogonalize(U_full)
        U_clements = clements_decompose(U_full)

        # Create mesh configs
        chip_id_Vh, self._chip_id, self._mzis_on_current_chip = self._assign_chip(
            Vh_clements.n_mzis
        )
        chip_id_U, self._chip_id, self._mzis_on_current_chip = self._assign_chip(
            U_clements.n_mzis
        )

        Vh_config = MeshConfig(
            N=decomposed.n,
            rank=decomposed.rank,
            chip_id=chip_id_Vh,
            layer_name=info.name,
            matrix_type="Vh",
            dac_bits=self.dac_bits,
            wavelength=self.wavelength,
        )
        U_config = MeshConfig(
            N=decomposed.m,
            rank=decomposed.rank,
            chip_id=chip_id_U,
            layer_name=info.name,
            matrix_type="U",
            dac_bits=self.dac_bits,
            wavelength=self.wavelength,
        )

        Vh_mesh = ClementsMesh(Vh_clements, Vh_config)
        U_mesh = ClementsMesh(U_clements, U_config)

        # Create sigma stage
        sigma_stage = SigmaStage(
            singular_values=decomposed.sigma_r.astype(np.float64),
            full_dim=decomposed.m,
        )

        t_end = time.perf_counter()
        self._total_mzis += Vh_clements.n_mzis + U_clements.n_mzis

        compiled = CompiledLayer(
            layer_name=info.name,
            component=info.component,
            projection_type=info.projection_type,
            transformer_layer_idx=info.transformer_layer_idx,
            U_mesh=U_mesh,
            sigma_stage=sigma_stage,
            Vh_mesh=Vh_mesh,
            n_mzis_U=U_clements.n_mzis,
            n_mzis_Vh=Vh_clements.n_mzis,
            chip_id_U=chip_id_U,
            chip_id_Vh=chip_id_Vh,
            reconstruction_error_svd=decomposed.reconstruction_error,
            reconstruction_error_clements_U=float(U_clements.reconstruction_error),
            reconstruction_error_clements_Vh=float(Vh_clements.reconstruction_error),
            compilation_time_s=t_end - t_start,
        )

        if verbose:
            logger.info(
                f"  Done: {compiled.total_mzis:,} MZIs, "
                f"chips U={chip_id_U}, Vh={chip_id_Vh}, "
                f"error_SVD={decomposed.reconstruction_error:.4f}, "
                f"error_U={U_clements.reconstruction_error:.2e}, "
                f"error_Vh={Vh_clements.reconstruction_error:.2e}, "
                f"time={t_end - t_start:.1f}s"
            )

        return compiled

    def _assign_chip(self, n_mzis: int):
        """
        Assign MZIs to chips. Returns (chip_id, new_chip_id, new_mzis_count).
        Moves to next chip if current one is full.
        """
        if self._mzis_on_current_chip + n_mzis > self.mzis_per_chip:
            # Move to next chip
            self._chip_id += 1
            self._mzis_on_current_chip = 0

        chip_id = self._chip_id
        self._mzis_on_current_chip += n_mzis
        return chip_id, self._chip_id, self._mzis_on_current_chip

    def total_mzis(self) -> int:
        """Total MZI count compiled so far."""
        return self._total_mzis

    def total_chips(self) -> int:
        """Total chips used so far."""
        return self._chip_id + 1

    def resource_report(self) -> str:
        """Generate a human-readable resource usage report."""
        lines = [
            "=== PhotoMedGemma Compilation Resource Report ===",
            f"Total MZIs compiled: {self._total_mzis:,}",
            f"Chips used: {self.total_chips()}",
            f"MZIs per chip: {self.mzis_per_chip:,}",
            f"Chip size (10mm×10mm): {self.total_chips() * 100} mm²",
            "=" * 50,
        ]
        return "\n".join(lines)


def _orthogonalize(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Re-orthogonalize a matrix using SVD to correct numerical drift.
    Replaces the matrix with the nearest unitary matrix.

    Args:
        M: Square matrix (approximately unitary)
        eps: Tolerance for unitarity check

    Returns:
        Nearest unitary matrix (via polar decomposition)
    """
    U, s, Vh = np.linalg.svd(M)
    # Nearest unitary = U @ Vh (polar decomposition, orthogonal factor)
    return (U @ Vh).astype(np.float64)
