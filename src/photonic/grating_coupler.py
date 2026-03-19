"""
Grating Coupler — Chip I/O Interface
=====================================

Grating couplers are the standard method for coupling light between
optical fibers and silicon photonic chips. They diffract light from
a fiber (propagating vertically) into a waveguide (propagating horizontally).

Typical specs (220nm SOI, 1310nm):
    Insertion loss: 2–4 dB per coupler
    Bandwidth: ~40–80 nm (3dB)
    Coupling angle: ~8–12° from vertical
    Polarization: typically single-polarization (TE)
    Footprint: ~12μm × 20μm

For the PhotoMedGemma chip, grating couplers provide the interface
between the laser sources (external) and the photonic compute mesh.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GratingCouplerSpec:
    """Physical specification for a grating coupler."""

    # Coupling efficiency
    insertion_loss_db: float = 3.0      # typical loss per coupler [dB]
    peak_wavelength: float = 1310e-9    # peak coupling wavelength [m]
    bandwidth_3db_nm: float = 50.0      # 3dB bandwidth [nm]

    # Geometry
    grating_period: float = 590e-9     # grating period [m] (for 1310nm, 10° angle)
    fill_factor: float = 0.5           # ratio of etched to unetched region
    etch_depth: float = 70e-9          # grating etch depth [m]
    num_periods: int = 20              # number of grating periods
    width: float = 12e-6              # coupler width [m]

    # Fiber coupling
    fiber_mode_field_diameter: float = 10.4e-6  # SMF-28 MFD [m]
    coupling_angle_deg: float = 10.0            # fiber tilt angle [degrees]

    # Polarization
    polarization: str = "TE"   # "TE" or "TM"
    polarization_dependent_loss_db: float = 0.5


class GratingCoupler:
    """
    Model for a standard silicon photonic grating coupler.

    Used for:
    - Fiber-to-chip input coupling (laser into waveguide)
    - Chip-to-fiber output coupling (waveguide to photodetector fiber)

    In the PhotoMedGemma chip:
    - Input couplers: receive modulated laser signal (encoded input activations)
    - Output couplers: send output to external photodetector array (if off-chip)
    - On-chip photodetectors are preferred to avoid off-chip coupling loss
    """

    def __init__(self, spec: Optional[GratingCouplerSpec] = None):
        self.spec = spec or GratingCouplerSpec()

    def coupling_efficiency(self, wavelength: float = None) -> float:
        """
        Compute coupling efficiency at a given wavelength.

        Uses a Gaussian wavelength response model:
            η(λ) = η_peak × exp(-0.5 × ((λ - λ_peak) / σ)²)

        Args:
            wavelength: Optical wavelength [m]. Defaults to peak wavelength.

        Returns:
            Coupling efficiency (linear, not dB) in [0, 1]
        """
        wavelength = wavelength or self.spec.peak_wavelength

        # Peak efficiency from insertion loss
        eta_peak = 10 ** (-self.spec.insertion_loss_db / 10)

        # Gaussian bandwidth model
        sigma_m = (self.spec.bandwidth_3db_nm * 1e-9) / (2 * np.sqrt(2 * np.log(2)))
        eta = eta_peak * np.exp(-0.5 * ((wavelength - self.spec.peak_wavelength) / sigma_m) ** 2)

        return eta

    def coupling_loss_db(self, wavelength: float = None) -> float:
        """Coupling loss at given wavelength in dB."""
        eta = self.coupling_efficiency(wavelength)
        if eta <= 0:
            return float('inf')
        return -10 * np.log10(eta)

    def apply(self, E_in: complex, wavelength: float = None) -> complex:
        """
        Apply coupling to input field amplitude.

        Args:
            E_in: Input complex field amplitude (fiber mode)
            wavelength: Operating wavelength [m]

        Returns:
            E_out: Coupled field amplitude (waveguide mode)
        """
        eta = self.coupling_efficiency(wavelength)
        # Amplitude coupling = √(power coupling efficiency)
        return np.sqrt(eta) * E_in

    def bandwidth_hz(self) -> float:
        """3dB bandwidth in Hz."""
        # Convert nm bandwidth to Hz using λ² / c
        c = 3e8
        lambda_peak = self.spec.peak_wavelength
        delta_lambda = self.spec.bandwidth_3db_nm * 1e-9
        return c * delta_lambda / lambda_peak ** 2

    def wdm_channels(self, channel_spacing_ghz: float = 400.0) -> int:
        """
        Number of WDM channels that fit within the 3dB bandwidth.

        Args:
            channel_spacing_ghz: Channel spacing [GHz]

        Returns:
            Number of WDM channels
        """
        bw_hz = self.bandwidth_hz()
        spacing_hz = channel_spacing_ghz * 1e9
        return max(1, int(bw_hz / spacing_hz))

    def footprint(self):
        """Return (length, width) in meters."""
        length = self.spec.num_periods * self.spec.grating_period + 10e-6  # routing taper
        width = self.spec.width
        return length, width

    def __repr__(self) -> str:
        return (
            f"GratingCoupler("
            f"λ={self.spec.peak_wavelength*1e9:.0f}nm, "
            f"loss={self.spec.insertion_loss_db:.1f}dB, "
            f"BW={self.spec.bandwidth_3db_nm:.0f}nm)"
        )
