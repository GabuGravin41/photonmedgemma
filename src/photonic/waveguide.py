"""
Waveguide — Optical "Wire" Primitive
=====================================

Models a silicon photonic waveguide segment. Waveguides are the fundamental
passive routing element in photonic circuits — equivalent to wires in electronics,
but for light.

Physical parameters are based on the 220nm SOI (Silicon-on-Insulator) platform,
the most widely available silicon photonics process node.

Key specs (220nm SOI):
    Core: Si, width=450nm, height=220nm
    Cladding: SiO₂ (buried oxide + top oxide)
    Wavelength: 1310nm or 1550nm
    Effective index (TE mode): ~2.4 at 1310nm
    Group index: ~4.2 at 1310nm
    Propagation loss: ~2–3 dB/cm
    Bend loss: negligible for R > 5 μm at 1310nm
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WaveguideSpec:
    """
    Specification for a silicon photonic waveguide.
    All dimensions in meters, wavelengths in meters.
    """

    # Geometry
    width: float = 450e-9       # core width [m]
    height: float = 220e-9      # core height [m]

    # Material (Si core, SiO₂ cladding)
    n_core: float = 3.47        # Si refractive index at 1310nm
    n_clad: float = 1.44        # SiO₂ refractive index

    # Operating wavelength
    wavelength: float = 1310e-9  # [m]

    # Loss (measured/typical for process)
    propagation_loss_db_per_m: float = 200.0  # 2 dB/cm = 200 dB/m

    # Effective indices (TE fundamental mode, computed or measured)
    n_eff: float = 2.40         # effective index (TE00)
    n_group: float = 4.20       # group index (TE00)

    # Thermo-optic coefficient
    dn_dT: float = 1.84e-4      # dn/dT for Si at 1310nm [K⁻¹]

    # Platform name for documentation
    platform: str = "220nm_SOI"


# Standard platform presets
SOI_220NM_1310 = WaveguideSpec(
    width=450e-9, height=220e-9,
    wavelength=1310e-9,
    n_core=3.47, n_clad=1.44,
    n_eff=2.40, n_group=4.20,
    propagation_loss_db_per_m=200.0,
    platform="220nm_SOI_1310nm",
)

SOI_220NM_1550 = WaveguideSpec(
    width=450e-9, height=220e-9,
    wavelength=1550e-9,
    n_core=3.47, n_clad=1.44,
    n_eff=2.34, n_group=4.35,
    propagation_loss_db_per_m=250.0,  # slightly higher loss at 1550nm in Si
    platform="220nm_SOI_1550nm",
)

SIN_300NM = WaveguideSpec(
    width=800e-9, height=300e-9,
    wavelength=1550e-9,
    n_core=2.00, n_clad=1.44,  # Si₃N₄
    n_eff=1.68, n_group=1.90,
    propagation_loss_db_per_m=10.0,   # Si₃N₄ has much lower loss (~0.1 dB/cm)
    dn_dT=2.5e-5,                     # lower thermo-optic coefficient
    platform="SiN_300nm",
)


class Waveguide:
    """
    Photonic waveguide segment model.

    Computes optical field propagation, loss, and phase accumulation
    for a straight or bent waveguide segment.
    """

    def __init__(
        self,
        length: float,
        spec: Optional[WaveguideSpec] = None,
        bend_radius: Optional[float] = None,
    ):
        """
        Args:
            length: Waveguide length [m]
            spec: Waveguide specification. Defaults to SOI_220NM_1310.
            bend_radius: If not None, model as a bent waveguide [m].
                         Bend radius < 5μm may cause significant bend loss.
        """
        self.length = length
        self.spec = spec or SOI_220NM_1310
        self.bend_radius = bend_radius

    @property
    def propagation_loss_linear(self) -> float:
        """Linear (amplitude) propagation loss factor for this waveguide."""
        loss_db = self.spec.propagation_loss_db_per_m * self.length
        return 10 ** (-loss_db / 20)  # amplitude, not power

    @property
    def phase_accumulation(self) -> float:
        """Optical phase accumulated over the waveguide length [radians]."""
        k0 = 2 * np.pi / self.spec.wavelength
        return k0 * self.spec.n_eff * self.length

    @property
    def bend_loss_db(self) -> float:
        """
        Estimate bend loss for a curved waveguide.
        Uses a simple empirical model; for accurate results use FDTD.

        Returns 0 if waveguide is straight (bend_radius is None).
        """
        if self.bend_radius is None:
            return 0.0

        R = self.bend_radius
        w = self.spec.width
        n_eff = self.spec.n_eff
        n_clad = self.spec.n_clad

        # Simple radiation loss model (approximate)
        # Accurate for R >> λ/2π condition
        NA = np.sqrt(n_eff**2 - n_clad**2)
        V = np.pi * w / self.spec.wavelength * NA  # V-number

        if V < 1.5:
            # Weakly guided — significant bend loss
            loss_per_m = 1e3 * np.exp(-R / 1e-6)  # empirical, rough
        else:
            # Well-confined — low bend loss for R > 5μm
            loss_per_m = 0.1 * np.exp(-R / 5e-6)  # rough estimate

        return loss_per_m * self.length * 4.343  # convert to dB

    def transfer_matrix(
        self,
        include_loss: bool = True,
        delta_T: float = 0.0,
    ) -> np.ndarray:
        """
        Compute 1×1 transfer matrix (complex scalar) for this waveguide.

        For a waveguide, the output field E_out = T · E_in where
        T is a complex scalar encoding both phase and loss.

        Args:
            include_loss: Whether to include propagation loss.
            delta_T: Temperature change above nominal [K].
                     Used to model thermo-optic phase shift.

        Returns:
            T: Complex scalar (shape: scalar, not array)
        """
        # Phase accumulation (including thermo-optic effect)
        k0 = 2 * np.pi / self.spec.wavelength
        delta_n = self.spec.dn_dT * delta_T
        phase = k0 * (self.spec.n_eff + delta_n) * self.length

        T = np.exp(1j * phase)

        if include_loss:
            # Propagation loss
            T *= self.propagation_loss_linear
            # Bend loss
            if self.bend_radius is not None:
                bend_loss_linear = 10 ** (-self.bend_loss_db / 20)
                T *= bend_loss_linear

        return T

    def apply(self, E_in: complex, **kwargs) -> complex:
        """Propagate field E_in through this waveguide."""
        return self.transfer_matrix(**kwargs) * E_in

    def delay(self) -> float:
        """
        Propagation delay through this waveguide [seconds].
        Uses group velocity: v_g = c / n_group
        """
        c = 3e8  # speed of light [m/s]
        v_g = c / self.spec.n_group
        return self.length / v_g

    def phase_from_temperature(self, delta_T: float) -> float:
        """
        Compute additional phase shift from temperature change.

        Args:
            delta_T: Temperature change [K]

        Returns:
            Additional phase shift [radians]
        """
        k0 = 2 * np.pi / self.spec.wavelength
        delta_n = self.spec.dn_dT * delta_T
        return k0 * delta_n * self.length

    def required_temperature_for_phase(self, target_phase: float) -> float:
        """
        Compute temperature change needed to achieve a given phase shift.
        Used for thermal phase shifter design.

        Args:
            target_phase: Desired phase shift [radians]

        Returns:
            Required ΔT [K]
        """
        k0 = 2 * np.pi / self.spec.wavelength
        delta_n_needed = target_phase / (k0 * self.length)
        return delta_n_needed / self.spec.dn_dT

    def __repr__(self) -> str:
        bend_str = f", R={self.bend_radius*1e6:.1f}μm" if self.bend_radius else ""
        return (
            f"Waveguide(L={self.length*1e6:.1f}μm, "
            f"λ={self.spec.wavelength*1e9:.0f}nm"
            f"{bend_str}, "
            f"platform={self.spec.platform})"
        )
