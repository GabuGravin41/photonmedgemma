"""
Mach-Zehnder Interferometer (MZI) — Core Photonic Compute Unit
===============================================================

An MZI is the fundamental building block of photonic matrix multiplication.
It implements a 2×2 unitary transformation on two optical modes using two
directional couplers and two phase shifters.

Physical structure:
    Input 1 ──┬── [DC_in] ──[θ arm]── [DC_out] ──┬── Output 1
              │                                    │
              └─────────────[φ arm]────────────────┘── Output 2

Transfer matrix:
    T(θ, φ) = i · e^(iφ/2) · [[-sin(θ/2)·e^(iφ),  cos(θ/2)],
                                [ cos(θ/2)·e^(iφ),  sin(θ/2)]]

    (Convention following Clements et al. 2016)

Reference:
    W. R. Clements et al., "Optimal design for universal multiport
    interferometers," Optica 3, 1460 (2016).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
import warnings


@dataclass
class MZIParameters:
    """Physical and fabrication parameters for a single MZI."""

    # Phase shifter angles (radians)
    theta: float = 0.0       # inner phase shift (controls splitting ratio)
    phi: float = 0.0         # outer phase shift (controls output phase)

    # Physical dimensions (meters)
    arm_length: float = 100e-6        # phase shifter arm length [m]
    coupling_length: float = 10e-6   # directional coupler length [m]
    coupler_gap: float = 0.2e-6      # directional coupler gap [m]
    waveguide_width: float = 0.45e-6 # waveguide core width [m]
    waveguide_height: float = 0.22e-6 # waveguide core height [m]

    # Wavelength
    wavelength: float = 1310e-9  # operating wavelength [m]

    # Loss parameters
    propagation_loss_db_per_m: float = 200.0  # 2 dB/cm = 200 dB/m
    insertion_loss_db: float = 0.1           # coupler insertion loss [dB]

    # Phase shifter type
    phase_shifter_type: str = "thermal"  # "thermal" or "electro_optic"

    # DAC resolution for phase control
    dac_bits: int = 12

    # Fabrication error model
    theta_std: float = 0.01   # std dev of theta fabrication error [rad]
    phi_std: float = 0.01     # std dev of phi fabrication error [rad]
    coupling_error: float = 0.01  # std dev of coupler splitting ratio error

    # Chip position (for layout generation)
    mesh_row: int = 0
    mesh_col: int = 0
    chip_id: int = 0

    def __post_init__(self):
        if not (-np.pi <= self.theta <= np.pi):
            warnings.warn(f"theta={self.theta:.3f} outside [-π, π]. Wrapping.")
            self.theta = self.theta % (2 * np.pi)
        if not (-np.pi <= self.phi <= 2 * np.pi):
            warnings.warn(f"phi={self.phi:.3f} outside expected range.")


class MZI:
    """
    Mach-Zehnder Interferometer — 2×2 photonic unitary gate.

    This class implements both the transfer matrix computation and
    the physical property calculations for a single MZI.

    The MZI implements the transformation:
        [E_out1]   [T00  T01] [E_in1]
        [E_out2] = [T10  T11] [E_in2]

    where T is the 2×2 unitary transfer matrix T(θ, φ).
    """

    # Clements convention: T(θ, φ) = i·e^(iφ/2) · [[-sin(θ/2)·e^iφ, cos(θ/2)],
    #                                                 [ cos(θ/2)·e^iφ, sin(θ/2)]]
    # This convention places phase φ on the first input port.

    def __init__(self, params: Optional[MZIParameters] = None):
        self.params = params or MZIParameters()

    @property
    def theta(self) -> float:
        return self.params.theta

    @theta.setter
    def theta(self, value: float):
        self.params.theta = float(value)

    @property
    def phi(self) -> float:
        return self.params.phi

    @phi.setter
    def phi(self, value: float):
        self.params.phi = float(value)

    def transfer_matrix(
        self,
        include_loss: bool = False,
        add_fabrication_noise: bool = False,
    ) -> np.ndarray:
        """
        Compute the 2×2 complex transfer matrix of the MZI.

        Args:
            include_loss: If True, apply propagation and insertion loss.
            add_fabrication_noise: If True, add Gaussian noise to theta and phi
                                   to simulate fabrication imperfections.

        Returns:
            T: 2×2 complex numpy array, unitary (if no loss).
        """
        theta = self.params.theta
        phi = self.params.phi

        if add_fabrication_noise:
            theta += np.random.normal(0, self.params.theta_std)
            phi += np.random.normal(0, self.params.phi_std)

        # Clements 2016 convention
        # T = i * e^(i*phi/2) * [[-sin(theta/2)*e^(i*phi), cos(theta/2)],
        #                         [ cos(theta/2)*e^(i*phi), sin(theta/2)]]
        cos_h = np.cos(theta / 2)
        sin_h = np.sin(theta / 2)
        e_phi = np.exp(1j * phi)
        e_phi_half = np.exp(1j * phi / 2)

        T = 1j * e_phi_half * np.array([
            [-sin_h * e_phi, cos_h],
            [ cos_h * e_phi, sin_h],
        ], dtype=complex)

        if include_loss:
            T = self._apply_loss(T)

        return T

    def _apply_loss(self, T: np.ndarray) -> np.ndarray:
        """Apply propagation and insertion loss to transfer matrix."""
        p = self.params

        # Total optical path length through MZI (approximate)
        path_length = 2 * p.arm_length + 2 * p.coupling_length

        # Propagation loss
        loss_linear = 10 ** (-p.propagation_loss_db_per_m * path_length / 10)

        # Insertion loss (two couplers)
        insertion = 10 ** (-2 * p.insertion_loss_db / 10)

        total_amplitude_loss = np.sqrt(loss_linear * insertion)
        return T * total_amplitude_loss

    def apply(self, E_in: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply MZI transformation to input field amplitudes.

        Args:
            E_in: Complex array of shape (2,) — [E_in1, E_in2]
            **kwargs: Passed to transfer_matrix()

        Returns:
            E_out: Complex array of shape (2,) — [E_out1, E_out2]
        """
        T = self.transfer_matrix(**kwargs)
        return T @ E_in

    @classmethod
    def from_matrix(cls, U: np.ndarray) -> "MZI":
        """
        Fit MZI parameters to best approximate a given 2×2 unitary matrix.

        This finds (θ, φ) such that T(θ, φ) ≈ U (up to a global phase).

        Args:
            U: 2×2 complex unitary matrix

        Returns:
            MZI instance with fitted parameters
        """
        if U.shape != (2, 2):
            raise ValueError(f"Expected 2×2 matrix, got {U.shape}")

        # Normalize to remove global phase
        # Extract theta from the magnitude of off-diagonal elements
        # |T[0,1]| = cos(θ/2), |T[1,1]| = sin(θ/2)
        # (using Clements convention)

        # Compute theta from column amplitudes
        theta = 2 * np.arctan2(np.abs(U[1, 1]), np.abs(U[0, 1]))

        # Compute phi from phases
        # T[0,1] = i * e^(i*phi/2) * cos(theta/2)  →  angle(T[0,1]) = pi/2 + phi/2
        # T[1,1] = i * e^(i*phi/2) * sin(theta/2)  →  angle(T[1,1]) = pi/2 + phi/2
        phi = 2 * (np.angle(U[0, 1]) - np.pi / 2)

        params = MZIParameters(theta=float(theta), phi=float(phi))
        return cls(params)

    def phase_power(self) -> Tuple[float, float]:
        """
        Estimate heater power required for current (θ, φ) settings.

        For thermal phase shifters, power scales as P ∝ Δφ.
        Returns (P_theta, P_phi) in milliwatts.

        Reference: ~10–20 mW for π phase shift in 220nm SOI thermal PS.
        """
        P_pi = 15e-3  # 15 mW for π phase shift (nominal)

        P_theta = abs(self.params.theta) / np.pi * P_pi
        P_phi = abs(self.params.phi) / np.pi * P_pi

        return P_theta * 1e3, P_phi * 1e3  # return in mW

    def quantize_phases(self, bits: Optional[int] = None) -> Tuple[int, int]:
        """
        Quantize (θ, φ) to integer DAC codes.

        Args:
            bits: DAC resolution in bits. Defaults to self.params.dac_bits.

        Returns:
            (code_theta, code_phi) as integers in [0, 2^bits - 1]
        """
        bits = bits or self.params.dac_bits
        max_code = (1 << bits) - 1

        def quantize(angle):
            normalized = (angle % (2 * np.pi)) / (2 * np.pi)
            return int(round(normalized * max_code)) % (max_code + 1)

        return quantize(self.params.theta), quantize(self.params.phi)

    def footprint(self) -> Tuple[float, float]:
        """
        Estimate physical footprint of MZI (width, height) in meters.

        Returns:
            (width_m, height_m)
        """
        p = self.params
        width = 2 * p.coupling_length + p.arm_length
        height = 4e-6  # ~4 μm between waveguide centers (typical)
        return width, height

    def __repr__(self) -> str:
        return (
            f"MZI(θ={self.params.theta:.4f} rad, "
            f"φ={self.params.phi:.4f} rad, "
            f"chip={self.params.chip_id}, "
            f"pos=({self.params.mesh_row},{self.params.mesh_col}))"
        )


def mzi_transfer_matrix(theta: float, phi: float) -> np.ndarray:
    """
    Fast computation of MZI transfer matrix without creating an MZI object.
    Used in inner loops of the Clements decomposition.

    Args:
        theta: Inner phase shift (radians)
        phi: Outer phase shift (radians)

    Returns:
        T: 2×2 complex numpy array
    """
    cos_h = np.cos(theta / 2)
    sin_h = np.sin(theta / 2)
    e_phi = np.exp(1j * phi)
    e_phi_half = np.exp(1j * phi / 2)

    return 1j * e_phi_half * np.array([
        [-sin_h * e_phi, cos_h],
        [ cos_h * e_phi, sin_h],
    ], dtype=complex)


def inverse_mzi_transfer_matrix(theta: float, phi: float) -> np.ndarray:
    """Compute the inverse (conjugate transpose) of the MZI transfer matrix."""
    T = mzi_transfer_matrix(theta, phi)
    return T.conj().T


def find_mzi_nulling_params(a: complex, b: complex) -> Tuple[float, float]:
    """
    Find (θ, φ) such that the MZI zeros out the 'b' input mode.

    Given input amplitudes a and b on two modes, find MZI parameters
    that route all power to one output port (nulling operation).
    Used in the Clements decomposition algorithm.

    Args:
        a: Complex amplitude on mode 1
        b: Complex amplitude on mode 2 (to be nulled)

    Returns:
        (theta, phi) in radians
    """
    if np.abs(b) < 1e-15:
        return 0.0, 0.0

    phi = np.angle(b) - np.angle(a) + np.pi
    theta = 2 * np.arctan2(np.abs(b), np.abs(a))

    return float(theta), float(phi)
