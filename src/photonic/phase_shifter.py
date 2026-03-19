"""
Phase Shifters — Controllable Phase Elements
=============================================

Phase shifters are the "programmable" elements of photonic circuits.
They shift the phase of light passing through a waveguide, controlled
by an external electrical signal (voltage or current).

Two main types for silicon photonics:

1. Thermal Phase Shifter (TPS):
   - Mechanism: Heater changes waveguide temperature → thermo-optic effect
   - Speed: ~1–10 kHz (thermal time constant)
   - Power: 10–20 mW for π shift (steady state), ~0 in static mode
   - Used for: STATIC weight setting (set once, zero steady-state power)

2. Electro-Optic Phase Shifter (EOPS):
   - Mechanism: Carrier injection/depletion → plasma dispersion effect
   - Speed: 1–40 GHz
   - Power: 1–5 mW (continuous)
   - Used for: Dynamic modulation (vision encoder pixel input encoding)

For PhotoMedGemma, we use THERMAL phase shifters for all weight matrices
(static compilation) and electro-optic modulators only for input encoding.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ThermalPhaseShifterSpec:
    """
    Specification for a titanium nitride (TiN) thermal phase shifter
    on 220nm SOI platform.
    """

    # Geometry
    heater_length: float = 100e-6    # heater element length [m]
    heater_width: float = 2.5e-6    # heater width [m]
    heater_height: float = 0.12e-6  # TiN heater layer height [m]
    oxide_thickness: float = 1.0e-6 # SiO₂ between heater and waveguide [m]

    # Waveguide
    waveguide_length: float = 100e-6  # length of waveguide under heater [m]

    # Material properties
    resistivity: float = 2.5e-6      # TiN resistivity [Ω·m]
    dn_dT: float = 1.84e-4           # Si thermo-optic coefficient [K⁻¹]
    thermal_resistance: float = 2e4  # K/W — heater to waveguide thermal resistance

    # Electrical
    resistance: float = 1000.0       # heater resistance [Ω]
    max_voltage: float = 5.0         # maximum drive voltage [V]
    max_power: float = 25e-3         # maximum heater power [W = 25mW]

    # Performance
    pi_power: float = 15e-3          # power required for π phase shift [W]
    bandwidth_hz: float = 1e4        # thermal bandwidth [Hz] (~10 kHz)
    wavelength: float = 1310e-9      # operating wavelength [m]

    # Fabrication error
    resistance_tolerance: float = 0.05  # ±5% resistance variation


@dataclass
class ElectroOpticPhaseShifterSpec:
    """
    Specification for a carrier-depletion silicon electro-optic phase shifter.
    Based on a reverse-biased PN junction in the silicon waveguide.
    """

    # Geometry
    length: float = 1e-3            # modulator length [m] (1mm for ~20dB extinction)
    width: float = 0.45e-6         # waveguide width [m]

    # Electrical
    v_pi: float = 5.0               # voltage for π phase shift [V] (Vπ)
    v_pi_L: float = 5e-3            # Vπ·L product [V·m] (process figure of merit)
    max_voltage: float = -5.0       # maximum reverse bias [V]
    capacitance_per_meter: float = 300e-12  # [F/m]

    # Performance
    bandwidth_ghz: float = 25.0     # 3dB bandwidth [GHz]
    insertion_loss_db: float = 3.0  # modulator insertion loss [dB]

    # Operating mode: "depletion" (fast) or "injection" (slower, higher dn)
    mode: str = "depletion"

    wavelength: float = 1310e-9


class ThermalPhaseShifter:
    """
    Thermal phase shifter model for silicon photonics.

    Uses the thermo-optic effect: heating the waveguide changes its
    refractive index, accumulating phase on the optical mode.

    Δφ = (2π/λ) · (dn/dT) · ΔT · L

    where ΔT is the temperature increase from the heater.

    For STATIC weight compilation:
    - Phase is set once during chip initialization
    - Heater power during inference: ZERO (no switching needed)
    - Latency: ~0.1ms for thermal settling during initialization
    """

    def __init__(
        self,
        target_phase: float = 0.0,
        spec: Optional[ThermalPhaseShifterSpec] = None,
    ):
        """
        Args:
            target_phase: Desired optical phase shift [radians]
            spec: Physical parameters of the phase shifter
        """
        self.spec = spec or ThermalPhaseShifterSpec()
        self._target_phase = target_phase
        self._current_phase = 0.0  # actual phase (settles to target after τ)

    @property
    def target_phase(self) -> float:
        return self._target_phase

    @target_phase.setter
    def target_phase(self, value: float):
        self._target_phase = float(value) % (2 * np.pi)

    @property
    def current_phase(self) -> float:
        return self._current_phase

    def required_power(self, phase: Optional[float] = None) -> float:
        """
        Compute heater power required for a given phase shift.

        P = P_π × (|Δφ| / π)

        Args:
            phase: Phase shift [rad]. Defaults to target_phase.

        Returns:
            Required heater power [W]
        """
        phase = phase if phase is not None else self._target_phase
        phase = phase % (2 * np.pi)

        # Minimum phase direction (shorter path around circle)
        if phase > np.pi:
            phase = 2 * np.pi - phase  # use negative phase shift instead

        return self.spec.pi_power * abs(phase) / np.pi

    def required_voltage(self, phase: Optional[float] = None) -> float:
        """
        Compute heater drive voltage for a given phase shift.

        P = V² / R  →  V = √(P × R)

        Args:
            phase: Phase shift [rad]. Defaults to target_phase.

        Returns:
            Required drive voltage [V]
        """
        P = self.required_power(phase)
        V = np.sqrt(P * self.spec.resistance)
        return min(V, self.spec.max_voltage)

    def required_temperature(self, phase: Optional[float] = None) -> float:
        """
        Compute required waveguide temperature increase for given phase.

        Args:
            phase: Phase shift [rad]

        Returns:
            Required ΔT [K]
        """
        phase = phase if phase is not None else self._target_phase
        k0 = 2 * np.pi / self.spec.wavelength
        delta_n_needed = abs(phase) / (k0 * self.spec.waveguide_length)
        return delta_n_needed / self.spec.dn_dT

    def transfer_matrix(self, add_noise: bool = False) -> np.ndarray:
        """
        Compute 1×1 transfer matrix (phase-only, ideal lossless).

        Returns:
            T: Complex scalar exp(i·phase)
        """
        phase = self._current_phase
        if add_noise:
            # Phase noise from temperature fluctuations
            dT_noise = np.random.normal(0, 0.01)  # ±0.01K thermal noise
            k0 = 2 * np.pi / self.spec.wavelength
            phase += k0 * self.spec.dn_dT * dT_noise * self.spec.waveguide_length

        return np.exp(1j * phase)

    def settle(self):
        """Update current phase to target (simulates thermal settling)."""
        self._current_phase = self._target_phase

    def dac_code(self, bits: int = 12) -> int:
        """
        Convert target phase to DAC integer code.

        Args:
            bits: DAC resolution in bits

        Returns:
            Integer code in [0, 2^bits - 1]
        """
        max_code = (1 << bits) - 1
        normalized = (self._target_phase % (2 * np.pi)) / (2 * np.pi)
        return int(round(normalized * max_code)) % (max_code + 1)

    @classmethod
    def from_dac_code(cls, code: int, bits: int = 12, **spec_kwargs) -> "ThermalPhaseShifter":
        """Create a phase shifter from a DAC code."""
        phase = (code / (1 << bits)) * 2 * np.pi
        ps = cls(target_phase=phase, **spec_kwargs)
        ps.settle()
        return ps

    def footprint(self) -> Tuple[float, float]:
        """Return (length, width) in meters."""
        return self.spec.heater_length, self.spec.heater_width * 3  # routing overhead

    def __repr__(self) -> str:
        return (
            f"ThermalPS(φ={self._target_phase:.4f}rad, "
            f"P={self.required_power()*1e3:.2f}mW, "
            f"V={self.required_voltage():.2f}V)"
        )


class ElectroOpticPhaseShifter:
    """
    Electro-optic (carrier depletion) phase shifter model.

    Used for high-speed input encoding (modulating the input signal
    into the optical domain at GHz speeds). Not used for static weight
    storage — use ThermalPhaseShifter for that.

    Phase shift: Δφ = π × V / Vπ  (for reverse-biased PN junction)
    Bandwidth: up to 40 GHz
    """

    def __init__(
        self,
        voltage: float = 0.0,
        spec: Optional[ElectroOpticPhaseShifterSpec] = None,
    ):
        """
        Args:
            voltage: Drive voltage [V]. Positive = more reverse bias.
            spec: Physical parameters
        """
        self.spec = spec or ElectroOpticPhaseShifterSpec()
        self.voltage = voltage

    @property
    def phase_shift(self) -> float:
        """Current phase shift [radians]."""
        return np.pi * self.voltage / self.spec.v_pi

    def transfer_matrix(self, include_loss: bool = True) -> complex:
        """Complex transmission factor."""
        T = np.exp(1j * self.phase_shift)
        if include_loss:
            loss = 10 ** (-self.spec.insertion_loss_db / 20)
            T *= loss
        return T

    def modulation_bandwidth(self) -> float:
        """3dB modulation bandwidth [Hz]."""
        return self.spec.bandwidth_ghz * 1e9

    def encode_amplitude(self, amplitude: float) -> float:
        """
        Set voltage to encode a real-valued amplitude as optical phase.
        Used for input data encoding in inference.

        Maps amplitude ∈ [-1, 1] to voltage that produces phase ∈ [0, π].

        Args:
            amplitude: Normalized signal amplitude in [-1, 1]

        Returns:
            Required voltage [V]
        """
        # Map amplitude to phase: amplitude → θ where cos(θ) = amplitude
        theta = np.arccos(np.clip(amplitude, -1, 1))
        voltage = theta / np.pi * self.spec.v_pi
        self.voltage = voltage
        return voltage

    def __repr__(self) -> str:
        return (
            f"EOPS(V={self.voltage:.2f}V, "
            f"Δφ={self.phase_shift:.4f}rad, "
            f"BW={self.spec.bandwidth_ghz:.0f}GHz)"
        )
