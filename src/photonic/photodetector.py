"""
Photodetector — Optical-to-Electrical Converter
=================================================

Converts optical power (|E|²) back to an electrical signal.
Used at the output of each photonic layer to read out results
and feed into the electronic nonlinearity / next layer encoder.

For silicon photonics at 1310nm/1550nm:
    - Germanium (Ge) photodiode — most common
    - Responsivity: ~0.9 A/W at 1310nm, ~1.0 A/W at 1550nm
    - Bandwidth: >40 GHz (sufficient for our application)
    - Dark current: ~1–10 nA
    - Noise: shot noise limited at high power

The O→E conversion step is the major hybrid interface between the
optical compute domain and the electronic control domain.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class PhotoDetectorSpec:
    """Physical parameters for a Ge-on-Si photodetector."""

    # Responsivity
    responsivity: float = 0.9       # [A/W] at 1310nm
    quantum_efficiency: float = 0.85 # ηQE

    # Bandwidth
    bandwidth_hz: float = 40e9      # 3dB bandwidth [Hz]
    rc_bandwidth_hz: float = 40e9   # RC-limited bandwidth [Hz]

    # Noise
    dark_current_A: float = 5e-9    # dark current [A]
    noise_figure_db: float = 3.0    # transimpedance amplifier noise

    # Saturation
    saturation_power_W: float = 10e-3  # optical saturation power [W = 10mW]

    # Physical
    area_m2: float = (10e-6) ** 2      # active area [m²] — 10μm × 10μm

    # Transimpedance amplifier
    transimpedance_ohm: float = 1000.0  # TIA gain [Ω]

    wavelength: float = 1310e-9


class PhotoDetector:
    """
    Germanium-on-Silicon photodetector model.

    Converts optical field amplitude (complex) to:
    1. Photocurrent [A]
    2. Voltage (after transimpedance amplifier) [V]
    3. Digital code (after ADC) [int]

    Includes noise models: shot noise, thermal noise, dark current.
    """

    def __init__(self, spec: Optional[PhotoDetectorSpec] = None):
        self.spec = spec or PhotoDetectorSpec()

    def detect(
        self,
        E_in: complex,
        include_noise: bool = False,
        integration_time: float = 1e-9,  # 1 ns per sample
    ) -> float:
        """
        Convert optical field to photocurrent.

        Args:
            E_in: Complex optical field amplitude (normalized units)
            include_noise: If True, add shot noise and dark current noise
            integration_time: Integration time [s] for noise calculation

        Returns:
            Photocurrent [A]
        """
        # Optical power from field amplitude (assuming normalized units)
        power_W = np.abs(E_in) ** 2

        # Clip at saturation
        power_W = min(power_W, self.spec.saturation_power_W)

        # Photocurrent
        I_signal = self.spec.responsivity * power_W

        if not include_noise:
            return I_signal

        # Shot noise: σ² = 2qI·Δf
        q = 1.6e-19  # electron charge [C]
        bandwidth = 0.5 / integration_time
        I_shot_std = np.sqrt(2 * q * (I_signal + self.spec.dark_current_A) * bandwidth)

        # Dark current noise contribution
        I_dark = self.spec.dark_current_A

        # Total noise
        I_noise = np.random.normal(0, I_shot_std)

        return I_signal + I_dark + I_noise

    def detect_field_array(
        self,
        E_array: np.ndarray,
        include_noise: bool = False,
    ) -> np.ndarray:
        """
        Convert array of complex field amplitudes to photocurrents.

        Args:
            E_array: Complex array of shape (N,)
            include_noise: If True, add noise

        Returns:
            Photocurrents: Float array of shape (N,)
        """
        powers = np.abs(E_array) ** 2
        powers = np.clip(powers, 0, self.spec.saturation_power_W)
        I_signal = self.spec.responsivity * powers

        if not include_noise:
            return I_signal

        q = 1.6e-19
        bandwidth = self.spec.bandwidth_hz
        I_shot_std = np.sqrt(2 * q * (I_signal + self.spec.dark_current_A) * bandwidth / 2)
        I_noise = np.random.normal(0, I_shot_std, size=I_signal.shape)

        return I_signal + self.spec.dark_current_A + I_noise

    def to_voltage(self, photocurrent: float) -> float:
        """Convert photocurrent to voltage via transimpedance amplifier."""
        return photocurrent * self.spec.transimpedance_ohm

    def to_adc_code(
        self,
        photocurrent: float,
        adc_bits: int = 12,
        v_ref: float = 1.0,
    ) -> int:
        """
        Convert photocurrent to ADC digital code.

        Args:
            photocurrent: Input current [A]
            adc_bits: ADC resolution in bits
            v_ref: ADC full-scale voltage [V]

        Returns:
            Integer ADC code in [0, 2^adc_bits - 1]
        """
        voltage = self.to_voltage(photocurrent)
        normalized = np.clip(voltage / v_ref, 0, 1)
        max_code = (1 << adc_bits) - 1
        return int(round(normalized * max_code))

    def snr_db(self, optical_power_W: float, bandwidth_hz: float = None) -> float:
        """
        Compute signal-to-noise ratio in dB.

        Args:
            optical_power_W: Optical input power [W]
            bandwidth_hz: Detection bandwidth [Hz]

        Returns:
            SNR [dB]
        """
        bandwidth_hz = bandwidth_hz or self.spec.bandwidth_hz
        q = 1.6e-19
        kB = 1.38e-23
        T = 300  # room temperature [K]
        R_L = self.spec.transimpedance_ohm

        I_signal = self.spec.responsivity * optical_power_W

        # Shot noise power spectral density
        S_shot = 2 * q * I_signal * bandwidth_hz

        # Thermal noise power
        S_thermal = 4 * kB * T * bandwidth_hz / R_L

        # Signal power
        I_sig_power = I_signal ** 2

        snr_linear = I_sig_power / (S_shot + S_thermal + 1e-30)
        return 10 * np.log10(snr_linear)

    def minimum_detectable_power(self, bandwidth_hz: float = None) -> float:
        """
        Minimum detectable optical power (SNR = 1) [W].

        Args:
            bandwidth_hz: Detection bandwidth

        Returns:
            Minimum power [W]
        """
        bandwidth_hz = bandwidth_hz or self.spec.bandwidth_hz
        q = 1.6e-19
        # Shot-noise limited: I_min = sqrt(2q·Δf) / R
        I_min = np.sqrt(2 * q * self.spec.dark_current_A * bandwidth_hz)
        return I_min / self.spec.responsivity

    def __repr__(self) -> str:
        return (
            f"PhotoDetector(R={self.spec.responsivity:.2f}A/W, "
            f"BW={self.spec.bandwidth_hz/1e9:.0f}GHz, "
            f"I_dark={self.spec.dark_current_A*1e9:.1f}nA)"
        )
