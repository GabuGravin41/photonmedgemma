"""
MZI Mesh — Full Photonic Matrix Processor
==========================================

A Clements mesh implements an N×N unitary matrix using N(N-1)/2 MZIs
arranged in a rectangular grid. This module provides the high-level
mesh abstraction used by the architecture modules.

The mesh accepts N optical input modes, processes them through all MZIs
(each implementing a 2×2 rotation), and produces N output modes.

Physical layout:
    Inputs → [Stage 0] → [Stage 1] → ... → [Stage N-1] → Phase screen → Outputs

Each stage contains N/2 MZIs acting on non-overlapping mode pairs.
Stages alternate between even pairs (0,1), (2,3), ... and odd pairs (1,2), (3,4), ...
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from compiler.clements import (
    ClementsResult,
    MZISpec,
    clements_decompose,
    clements_simulate,
    clements_reconstruct,
)
from .mzi import MZI, MZIParameters


@dataclass
class MeshConfig:
    """Configuration for a Clements MZI mesh."""
    N: int                              # number of optical modes
    rank: int                           # effective rank (for rectangular meshes)
    chip_id: int = 0                    # which chip this mesh is on
    layer_name: str = ""               # originating neural network layer
    matrix_type: str = "U"             # "U" or "Vh" (which SVD factor)
    dac_bits: int = 12                  # phase DAC resolution
    loss_per_mzi_db: float = 0.01      # insertion loss per MZI
    wavelength: float = 1310e-9        # operating wavelength


class ClementsMesh:
    """
    Complete Clements MZI mesh implementing an N×N unitary transformation.

    This is the physical photonic "matrix multiplication unit." It holds
    the phase settings for all MZIs derived from the Clements decomposition
    of a compiled neural network weight matrix.

    Usage:
        # From a compiled weight matrix
        mesh = ClementsMesh.from_matrix(W, rank=64)

        # Simulate optical inference
        output = mesh.forward(input_field)

        # Get all MZI phase settings for chip programming
        phase_map = mesh.get_phase_map()

        # Get total power consumption
        power_W = mesh.static_power()
    """

    def __init__(
        self,
        clements_result: ClementsResult,
        config: Optional[MeshConfig] = None,
    ):
        self.result = clements_result
        self.config = config or MeshConfig(N=clements_result.N, rank=clements_result.N)
        self.N = clements_result.N

        # Build MZI objects from specs
        self._mzis: List[MZI] = self._build_mzis()

    @classmethod
    def from_matrix(
        cls,
        U: np.ndarray,
        config: Optional[MeshConfig] = None,
        eps: float = 1e-12,
    ) -> "ClementsMesh":
        """
        Compile a unitary matrix U into a Clements mesh.

        Args:
            U: N×N unitary matrix to compile
            config: Mesh configuration
            eps: Numerical tolerance for decomposition

        Returns:
            ClementsMesh with programmed phase settings
        """
        result = clements_decompose(U, eps=eps)
        N = U.shape[0]
        if config is None:
            config = MeshConfig(N=N, rank=N)
        return cls(result, config)

    def _build_mzis(self) -> List[MZI]:
        """Instantiate MZI objects from Clements decomposition specs."""
        mzis = []
        for spec in self.result.mzis:
            params = MZIParameters(
                theta=spec.theta,
                phi=spec.phi,
                mesh_row=spec.row,
                mesh_col=spec.col,
                chip_id=self.config.chip_id,
                dac_bits=self.config.dac_bits,
            )
            mzis.append(MZI(params))
        return mzis

    def forward(
        self,
        input_field: np.ndarray,
        include_loss: bool = False,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Compute optical forward pass through the mesh.

        Simulates what happens physically: light enters the mesh,
        propagates through each MZI (rotating the mode pair), and
        exits carrying the result of the matrix multiplication U·x.

        Args:
            input_field: Complex array of shape (N,) or (batch, N)
            include_loss: If True, apply per-MZI propagation loss
            add_noise: If True, add fabrication noise to MZI phases

        Returns:
            output_field: Complex array of same shape as input
        """
        if input_field.ndim == 1:
            return self._forward_single(input_field, include_loss, add_noise)
        else:
            # Batch processing
            return np.stack([
                self._forward_single(x, include_loss, add_noise)
                for x in input_field
            ])

    def _forward_single(
        self, field: np.ndarray, include_loss: bool, add_noise: bool
    ) -> np.ndarray:
        """Forward pass for a single input vector."""
        if len(field) != self.N:
            raise ValueError(f"Input dimension {len(field)} != mesh size {self.N}")

        mzi_specs = self.result.mzis
        phase_screen = self.result.phase_screen

        if add_noise:
            # Perturb phases with fabrication noise
            mzi_specs = [
                MZISpec(
                    row=m.row, col=m.col,
                    theta=m.theta + np.random.normal(0, 0.01),
                    phi=m.phi + np.random.normal(0, 0.01),
                    mode_i=m.mode_i, mode_j=m.mode_j,
                )
                for m in mzi_specs
            ]
            phase_screen = phase_screen + np.random.normal(0, 0.01, size=self.N)

        return clements_simulate(
            field, mzi_specs, phase_screen, self.N,
            include_loss=include_loss,
            loss_per_mzi_db=self.config.loss_per_mzi_db,
        )

    def reconstruct_matrix(self) -> np.ndarray:
        """
        Reconstruct the full N×N unitary matrix from mesh settings.
        Used for verification and debugging.

        Returns:
            U_reconstructed: N×N complex unitary matrix
        """
        return clements_reconstruct(
            self.result.mzis, self.result.phase_screen, self.N
        )

    def get_phase_map(self) -> Dict[str, np.ndarray]:
        """
        Get all phase settings as a dictionary for chip programming.

        Returns:
            Dict with keys:
            - 'theta': array of theta values for all MZIs
            - 'phi': array of phi values for all MZIs
            - 'theta_dac': integer DAC codes for theta
            - 'phi_dac': integer DAC codes for phi
            - 'phase_screen': diagonal phase screen values
            - 'phase_screen_dac': DAC codes for phase screen
            - 'mode_i', 'mode_j': mode indices per MZI
            - 'row', 'col': mesh positions
        """
        mzis = self.result.mzis
        bits = self.config.dac_bits
        max_code = (1 << bits) - 1

        def quantize(angle):
            return np.round(
                (np.array(angle) % (2 * np.pi)) / (2 * np.pi) * max_code
            ).astype(int) % (max_code + 1)

        return {
            "theta": np.array([m.theta for m in mzis]),
            "phi": np.array([m.phi for m in mzis]),
            "theta_dac": quantize([m.theta for m in mzis]),
            "phi_dac": quantize([m.phi for m in mzis]),
            "phase_screen": self.result.phase_screen,
            "phase_screen_dac": quantize(self.result.phase_screen),
            "mode_i": np.array([m.mode_i for m in mzis]),
            "mode_j": np.array([m.mode_j for m in mzis]),
            "row": np.array([m.row for m in mzis]),
            "col": np.array([m.col for m in mzis]),
        }

    def static_power(self, include_phase_screen: bool = True) -> float:
        """
        Estimate total static heater power during inference [W].

        For static weights, this is the power needed to maintain
        the phase settings. With thermal phase shifters:
        - Power is ~0 in steady state (no switching)
        - Initialization power (one-time): P_pi × θ/π per MZI

        Returns:
            Estimated power [W] for initialization
        """
        # Power to set each MZI phase (one-time initialization)
        P_pi = 15e-3  # 15 mW per π shift

        total_power = 0.0
        for mzi in self.result.mzis:
            total_power += abs(mzi.theta) / np.pi * P_pi
            total_power += abs(mzi.phi) / np.pi * P_pi

        if include_phase_screen:
            for phase in self.result.phase_screen:
                total_power += abs(phase) / np.pi * P_pi

        return total_power

    def num_mzis(self) -> int:
        """Return the number of MZIs in this mesh."""
        return len(self.result.mzis)

    def expected_num_mzis(self) -> int:
        """Expected number of MZIs for N×N Clements mesh: N(N-1)/2."""
        return self.N * (self.N - 1) // 2

    def reconstruction_error(self) -> float:
        """Relative Frobenius norm reconstruction error."""
        return self.result.reconstruction_error

    def footprint_mm2(
        self,
        mzi_width_um: float = 200.0,   # single MZI footprint along light propagation
        mzi_height_um: float = 4.0,    # spacing between waveguides
    ) -> float:
        """
        Estimate total chip footprint of this mesh [mm²].

        Args:
            mzi_width_um: Width (along propagation direction) of single MZI [μm]
            mzi_height_um: Height (perpendicular to propagation) of single MZI [μm]

        Returns:
            Estimated footprint [mm²]
        """
        # Clements mesh: N stages, each N/2 MZIs wide
        n_stages = self.N
        mzis_per_stage = self.N // 2

        width_um = n_stages * mzi_width_um
        height_um = mzis_per_stage * 2 * mzi_height_um  # ×2 for waveguide pairs

        return width_um * height_um * 1e-6  # convert μm² → mm²

    def __repr__(self) -> str:
        return (
            f"ClementsMesh(N={self.N}, "
            f"n_MZIs={self.num_mzis()}, "
            f"error={self.reconstruction_error():.2e}, "
            f"layer='{self.config.layer_name}')"
        )


class SigmaStage:
    """
    Diagonal singular value (Σ) stage.

    Implements the middle scaling stage of the SVD decomposition:
        y = Σ × x

    Physically realized as optical attenuators/amplifiers on each mode.
    For modes beyond the rank, the signal is blocked (attenuator = 0).

    In hardware:
    - Attenuators: VOAs (variable optical attenuators), ring resonators
    - Amplifiers: SOAs (semiconductor optical amplifiers) — for σ > 1
    - Blocker: opaque absorber or path redirect
    """

    def __init__(self, singular_values: np.ndarray, full_dim: int):
        """
        Args:
            singular_values: Array of singular values (rank-r, not normalized)
            full_dim: Full dimension of the mode space (N)
        """
        self.singular_values = singular_values
        self.full_dim = full_dim
        self.rank = len(singular_values)

        # Normalize singular values for optical implementation
        # (maximum singular value determines required laser power)
        self.max_sv = float(np.max(singular_values)) if len(singular_values) > 0 else 1.0
        self.normalized_sv = singular_values / (self.max_sv + 1e-15)

    def forward(self, field: np.ndarray) -> np.ndarray:
        """
        Apply singular value scaling to input field.

        Args:
            field: Complex array of shape (full_dim,)

        Returns:
            Scaled field of shape (full_dim,), modes beyond rank zeroed
        """
        output = np.zeros(self.full_dim, dtype=complex)
        r = min(self.rank, len(field), self.full_dim)
        output[:r] = self.normalized_sv[:r] * field[:r]
        return output

    def attenuation_db(self) -> np.ndarray:
        """Required attenuation in dB for each mode."""
        with np.errstate(divide='ignore'):
            att = -20 * np.log10(self.normalized_sv + 1e-15)
        return np.clip(att, 0, 60)  # max 60dB attenuation

    def __repr__(self) -> str:
        return (
            f"SigmaStage(rank={self.rank}/{self.full_dim}, "
            f"σ_max={self.max_sv:.4f}, σ_min={self.singular_values[-1]:.4f})"
        )


class SVDLayer:
    """
    Full SVD photonic layer: V† mesh + Σ stage + U mesh.

    This implements one compiled weight matrix W ≈ U_r × Σ_r × V_r†
    as a sequence of three photonic stages.

    The forward pass computes y = W·x optically:
        1. x → V†_mesh → (rotated)
        2. (rotated) → Σ_stage → (scaled)
        3. (scaled) → U_mesh → y
    """

    def __init__(
        self,
        U_mesh: ClementsMesh,
        sigma_stage: SigmaStage,
        Vh_mesh: ClementsMesh,
        layer_name: str = "",
        scale_factor: float = 1.0,
    ):
        self.U_mesh = U_mesh
        self.sigma_stage = sigma_stage
        self.Vh_mesh = Vh_mesh
        self.layer_name = layer_name
        self.scale_factor = scale_factor  # overall scale from weight matrix norm

    @property
    def input_dim(self) -> int:
        return self.Vh_mesh.N

    @property
    def output_dim(self) -> int:
        return self.U_mesh.N

    def forward(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Photonic forward pass: y = W·x

        Args:
            x: Input field amplitudes, shape (input_dim,)
            **kwargs: Passed to mesh.forward() (include_loss, add_noise)

        Returns:
            y: Output field amplitudes, shape (output_dim,)
        """
        # Stage 1: Apply V†
        after_Vh = self.Vh_mesh.forward(x, **kwargs)

        # Stage 2: Apply Σ (truncate to rank, scale)
        after_sigma = self.sigma_stage.forward(after_Vh)

        # Stage 3: Apply U (output rotation)
        y = self.U_mesh.forward(after_sigma, **kwargs)

        # Restore overall scale
        y = y * self.scale_factor

        return y

    def reconstruction_error(self, W_original: np.ndarray, n_samples: int = 100) -> float:
        """
        Estimate the reconstruction error of this layer compared to the original weight matrix.

        Args:
            W_original: Original weight matrix
            n_samples: Number of random vectors to test

        Returns:
            Mean relative error ‖W·x - W_photonic·x‖ / ‖W·x‖
        """
        rng = np.random.default_rng(42)
        errors = []

        for _ in range(n_samples):
            x = rng.standard_normal(self.input_dim) + 1j * rng.standard_normal(self.input_dim)
            x = x / (np.linalg.norm(x) + 1e-15)

            y_ref = W_original @ x.real  # reference (real weights, real input)
            y_photonic = self.forward(x.real)

            err = np.linalg.norm(y_ref - y_photonic.real) / (np.linalg.norm(y_ref) + 1e-15)
            errors.append(err)

        return float(np.mean(errors))

    def total_mzis(self) -> int:
        """Total MZI count for this SVD layer."""
        return self.U_mesh.num_mzis() + self.Vh_mesh.num_mzis()

    def static_power(self) -> float:
        """Total static heater power for initialization [W]."""
        return self.U_mesh.static_power() + self.Vh_mesh.static_power()

    def __repr__(self) -> str:
        return (
            f"SVDLayer('{self.layer_name}', "
            f"in={self.input_dim}, out={self.output_dim}, "
            f"rank={self.sigma_stage.rank}, "
            f"n_MZIs={self.total_mzis()})"
        )
