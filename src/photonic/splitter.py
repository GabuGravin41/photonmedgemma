"""
Optical Splitters and Couplers
================================

Implements directional couplers and Y-splitters — the fundamental
"beamsplitter" devices used as the input/output stages of each MZI.

Directional Coupler (DC):
    Two waveguides brought close together (gap ~200nm) over a coupling length L.
    Power coupling ratio κ depends on gap and length.
    A 50:50 coupler (κ = 0.5) splits light equally.

Y-Splitter (1×2):
    A single waveguide splits into two — always 50:50.
    Simpler to fabricate, less tunable.

Transfer matrices follow the convention in Clements et al. (2016).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DirectionalCouplerSpec:
    """Physical parameters for a directional coupler."""

    coupling_length: float = 10e-6    # coupling region length [m]
    gap: float = 0.2e-6              # waveguide separation in coupling region [m]
    waveguide_width: float = 0.45e-6  # waveguide width [m]

    # Effective coupling coefficient [rad/m] for Si at 1310nm
    # κ = π / (2 * L_pi) where L_pi is the coupling length for full transfer
    kappa: float = 7.85e4  # rad/m, gives L_pi ≈ 20μm for 220nm SOI at 1310nm

    # Loss
    insertion_loss_db: float = 0.1  # insertion loss per coupler [dB]
    excess_loss_db: float = 0.05    # excess loss from coupling [dB]

    # Fabrication error
    coupling_error_std: float = 0.01  # std dev of coupling ratio error


class DirectionalCoupler:
    """
    Evanescent directional coupler — the beamsplitter of integrated photonics.

    The transfer matrix for a directional coupler with coupling ratio κ is:
        DC(κ) = [√(1-κ)    i√κ   ]
                [i√κ       √(1-κ)]

    For a 50:50 coupler (κ = 0.5):
        DC(0.5) = (1/√2) · [1  i]
                            [i  1]

    This is equivalent to a Hadamard gate (up to phases).

    Reference:
        L. Chrostowski & M. Hochberg, "Silicon Photonics Design," CUP (2015)
    """

    def __init__(
        self,
        coupling_ratio: float = 0.5,
        spec: Optional[DirectionalCouplerSpec] = None,
    ):
        """
        Args:
            coupling_ratio: Power coupling ratio κ ∈ [0, 1].
                           κ=0 → no coupling (straight through)
                           κ=0.5 → 50:50 split
                           κ=1 → full coupling (cross)
            spec: Physical coupler parameters.
        """
        if not 0 <= coupling_ratio <= 1:
            raise ValueError(f"coupling_ratio must be in [0,1], got {coupling_ratio}")

        self.coupling_ratio = coupling_ratio
        self.spec = spec or DirectionalCouplerSpec()

    @classmethod
    def from_length(cls, length: float, spec: Optional[DirectionalCouplerSpec] = None) -> "DirectionalCoupler":
        """
        Create a coupler with a specific physical coupling length.

        Args:
            length: Coupling length [m]
            spec: Physical parameters

        Returns:
            DirectionalCoupler with computed coupling ratio
        """
        spec = spec or DirectionalCouplerSpec()
        # κ = sin²(κ_coeff * L)
        coupling_ratio = np.sin(spec.kappa * length) ** 2
        coupling_ratio = np.clip(coupling_ratio, 0.0, 1.0)
        coupler = cls(coupling_ratio=coupling_ratio, spec=spec)
        coupler.spec.coupling_length = length
        return coupler

    def transfer_matrix(
        self,
        include_loss: bool = False,
        add_fabrication_noise: bool = False,
    ) -> np.ndarray:
        """
        Compute the 2×2 transfer matrix of the directional coupler.

        Args:
            include_loss: If True, apply insertion loss.
            add_fabrication_noise: If True, perturb coupling ratio.

        Returns:
            T: 2×2 complex numpy array
        """
        kappa = self.coupling_ratio

        if add_fabrication_noise:
            kappa += np.random.normal(0, self.spec.coupling_error_std)
            kappa = np.clip(kappa, 0.0, 1.0)

        sqrt_kappa = np.sqrt(kappa)
        sqrt_1_kappa = np.sqrt(1 - kappa)

        T = np.array([
            [sqrt_1_kappa, 1j * sqrt_kappa],
            [1j * sqrt_kappa, sqrt_1_kappa],
        ], dtype=complex)

        if include_loss:
            loss_linear = 10 ** (-(self.spec.insertion_loss_db + self.spec.excess_loss_db) / 20)
            T = T * loss_linear

        return T

    def apply(self, E_in: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply coupler transformation to input fields.

        Args:
            E_in: Complex array of shape (2,) — [E_in1, E_in2]

        Returns:
            E_out: Complex array of shape (2,)
        """
        return self.transfer_matrix(**kwargs) @ E_in

    def coupling_length_for_ratio(self, target_ratio: float) -> float:
        """
        Compute coupling length needed for a target power coupling ratio.

        Args:
            target_ratio: Desired coupling ratio κ

        Returns:
            Required coupling length [m]
        """
        if target_ratio <= 0:
            return 0.0
        if target_ratio >= 1:
            return np.pi / (2 * self.spec.kappa)

        return np.arcsin(np.sqrt(target_ratio)) / self.spec.kappa

    def is_balanced(self, tolerance: float = 0.01) -> bool:
        """Check if this is a 50:50 coupler within tolerance."""
        return abs(self.coupling_ratio - 0.5) < tolerance

    def __repr__(self) -> str:
        return f"DirectionalCoupler(κ={self.coupling_ratio:.3f}, L={self.spec.coupling_length*1e6:.1f}μm)"


class YSplitter:
    """
    Y-junction 1×2 optical splitter.

    Simpler than a directional coupler — always 50:50.
    The Y-branch adiabatically splits a single waveguide into two.

    Transfer matrix (1×2):
        Input → [1/√2, 1/√2]  (power split)

    Note: For 2×2 applications (MZIs), directional couplers are preferred
    because they are reversible (2→2). Y-splitters are used for power tapping,
    monitoring, and 1-to-N fanout.
    """

    def __init__(
        self,
        split_ratio: float = 0.5,
        insertion_loss_db: float = 0.2,
    ):
        """
        Args:
            split_ratio: Power fraction to port 1. Port 2 gets (1 - split_ratio).
            insertion_loss_db: Insertion loss [dB]
        """
        self.split_ratio = split_ratio
        self.insertion_loss_db = insertion_loss_db

    def apply(self, E_in: complex, include_loss: bool = True) -> Tuple[complex, complex]:
        """
        Split input field into two outputs.

        Args:
            E_in: Input complex field amplitude

        Returns:
            (E_out1, E_out2): Tuple of output complex amplitudes
        """
        loss = 10 ** (-self.insertion_loss_db / 20) if include_loss else 1.0

        E_out1 = np.sqrt(self.split_ratio) * E_in * loss
        E_out2 = np.sqrt(1 - self.split_ratio) * E_in * loss * 1j  # phase from splitting

        return E_out1, E_out2

    def __repr__(self) -> str:
        return f"YSplitter(ratio={self.split_ratio:.2f}, loss={self.insertion_loss_db:.1f}dB)"


class MultiModeCoupler:
    """
    Multi-mode interference (MMI) coupler.

    An MMI coupler uses the self-imaging effect in a wide multimode
    waveguide to implement arbitrary N×N splitters. More fabrication-tolerant
    than directional couplers.

    Commonly used as the coupler stage in MZIs for robustness.
    """

    def __init__(
        self,
        n_inputs: int = 2,
        n_outputs: int = 2,
        wavelength: float = 1310e-9,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.wavelength = wavelength

    def transfer_matrix(self) -> np.ndarray:
        """
        Compute ideal N×M transfer matrix for MMI coupler.
        Assumes equal splitting and flat phase response.
        """
        # Ideal MMI: equal power splitting with 1/√N amplitude per output
        # with a specific phase pattern determined by self-imaging theory
        N = self.n_inputs
        M = self.n_outputs

        T = np.zeros((M, N), dtype=complex)
        for i in range(M):
            for j in range(N):
                # Phase from MMI self-imaging: φ_{i,j} = π(i+1)(j+1)/(N+1)
                phase = np.pi * (i + 1) * (j + 1) / (N + 1)
                T[i, j] = np.exp(1j * phase) / np.sqrt(N)

        return T
