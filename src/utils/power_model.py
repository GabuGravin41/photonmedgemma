"""
Power Model — Energy Consumption Estimation
============================================

Estimates the energy consumption of the PhotoMedGemma photonic chip system
for inference workloads.

Compared to GPU inference:
    - A100 GPU: ~300–400W TDP, ~200W during inference
    - H100 GPU: ~700W TDP, ~400W during inference
    - PhotoMedGemma target: ~5–20W total system

Components modeled:
1. Laser sources (CW, always on during inference)
2. Heater power for phase settings (zero in static mode after init)
3. Photodetector readout electronics
4. Electronic co-processor (FPGA/ASIC for softmax, LayerNorm, attention)
5. Memory (KV cache, embeddings — LPDDR5)
6. Thermal control (TEC for temperature stability)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class PhotonicSystemPower:
    """Power breakdown for the complete photonic MedGemma system."""

    # Laser sources
    laser_power_mW: float = 0.0       # CW laser array power
    laser_efficiency: float = 0.15    # Wall-plug efficiency of laser diodes

    # Phase shifters
    heater_init_power_mW: float = 0.0   # One-time initialization power
    heater_static_power_mW: float = 0.0 # Steady-state heater power (near 0 for thermal PS)

    # Photodetectors + readout
    detector_power_mW: float = 0.0

    # Electronic co-processor
    fpga_power_mW: float = 0.0         # Softmax, LayerNorm, attention scores

    # Memory
    memory_power_mW: float = 0.0       # KV cache, embeddings

    # Thermal control
    tec_power_mW: float = 0.0          # TEC for chip temperature stabilization

    # Overhead
    control_mcu_power_mW: float = 50.0  # Phase DAC control MCU

    @property
    def total_inference_power_mW(self) -> float:
        """Total power during steady-state inference."""
        return (
            self.laser_power_mW / self.laser_efficiency  # wall-plug laser power
            + self.heater_static_power_mW
            + self.detector_power_mW
            + self.fpga_power_mW
            + self.memory_power_mW
            + self.tec_power_mW
            + self.control_mcu_power_mW
        )

    @property
    def total_inference_power_W(self) -> float:
        return self.total_inference_power_mW / 1000.0

    def report(self) -> str:
        """Human-readable power breakdown."""
        lines = [
            "=== PhotoMedGemma System Power Budget ===",
            f"  Laser (optical):      {self.laser_power_mW:.1f} mW optical",
            f"  Laser (wall-plug):    {self.laser_power_mW / self.laser_efficiency:.1f} mW",
            f"  Phase heaters (init): {self.heater_init_power_mW:.1f} mW (one-time)",
            f"  Phase heaters (run):  {self.heater_static_power_mW:.1f} mW",
            f"  Photodetectors:       {self.detector_power_mW:.1f} mW",
            f"  FPGA co-processor:    {self.fpga_power_mW:.1f} mW",
            f"  Memory (LPDDR5):      {self.memory_power_mW:.1f} mW",
            f"  TEC stabilizer:       {self.tec_power_mW:.1f} mW",
            f"  Control MCU:          {self.control_mcu_power_mW:.1f} mW",
            f"  ─────────────────────────────────────",
            f"  TOTAL (inference):    {self.total_inference_power_W:.2f} W",
            "=" * 45,
        ]
        return "\n".join(lines)


@dataclass
class GPUInferencePower:
    """Reference GPU power for comparison."""
    model: str = "A100 SXM4"
    tdp_W: float = 400.0
    inference_power_W: float = 200.0  # typical during inference
    tokens_per_second: float = 50.0   # MedGemma inference speed
    energy_per_token_mJ: float = 4000.0  # 200W / 50 tok/s × 1000 = 4000 mJ/tok


class PowerModel:
    """
    Estimates power consumption for the PhotoMedGemma photonic chip system.

    Models:
    - Optical power budget (laser → chip → detector)
    - Electrical power budget (heaters, DACs, FPGA, memory)
    - Energy per token at various inference speeds
    """

    # Physical constants and process parameters (220nm SOI)
    PI_POWER_mW = 15.0           # mW for π phase shift (thermal PS)
    PHOTON_ENERGY_eV = 0.95     # at 1310nm
    DETECTOR_RESPONSIVITY = 0.9  # A/W
    DETECTOR_POWER_PER_CHANNEL_mW = 0.1  # readout electronics per detector
    LPDDR5_POWER_mW_per_GB = 100.0  # mW per GB of active LPDDR5

    def __init__(
        self,
        n_mzis: int,
        n_chips: int,
        n_output_channels: int = 2048,
        n_wavelengths: int = 8,
        inference_tokens_per_sec: float = 100.0,
        kv_cache_size_GB: float = 4.0,
    ):
        """
        Args:
            n_mzis: Total number of MZIs in the system
            n_chips: Number of photonic chips
            n_output_channels: Number of output optical modes
            n_wavelengths: WDM wavelength count (parallelism multiplier)
            inference_tokens_per_sec: Target inference throughput
            kv_cache_size_GB: KV cache memory size [GB]
        """
        self.n_mzis = n_mzis
        self.n_chips = n_chips
        self.n_output_channels = n_output_channels
        self.n_wavelengths = n_wavelengths
        self.inference_tokens_per_sec = inference_tokens_per_sec
        self.kv_cache_size_GB = kv_cache_size_GB

    def estimate(self) -> PhotonicSystemPower:
        """
        Compute complete power estimate for the system.

        Returns:
            PhotonicSystemPower with all component power values
        """
        power = PhotonicSystemPower()

        # ── Laser power ─────────────────────────────────────────────────────
        # Need enough optical power so that signal after loss is above detector floor
        # Assume 20dB total optical budget (propagation + coupling)
        # Target optical power at detector: 0.1 mW per channel
        # After 20dB loss: need 10 mW per channel at laser
        power.laser_power_mW = (
            self.n_output_channels  # parallel modes
            * self.n_wavelengths    # WDM channels
            * 0.1                   # mW per channel at detector
            * 100                   # 20dB loss compensation factor
        )

        # ── Phase heater power ──────────────────────────────────────────────
        # STATIC mode: heaters consume power only during initialization
        # Average phase ≈ π/2 per MZI (uniform distribution assumption)
        avg_phase_per_mzi = np.pi / 2
        power.heater_init_power_mW = self.n_mzis * 2 * (avg_phase_per_mzi / np.pi) * self.PI_POWER_mW
        power.heater_static_power_mW = 0.0  # zero in static mode (TPS latches phase)

        # ── Photodetector power ─────────────────────────────────────────────
        # One detector per output channel, per wavelength
        n_detectors = self.n_output_channels * self.n_wavelengths
        power.detector_power_mW = n_detectors * self.DETECTOR_POWER_PER_CHANNEL_mW

        # ── FPGA co-processor ───────────────────────────────────────────────
        # Handles: softmax, LayerNorm, attention scores, routing
        # Estimate: Xilinx KV260 FPGA at ~5W for attention-scale compute
        power.fpga_power_mW = 5000.0  # 5W FPGA

        # ── Memory (LPDDR5 for KV cache and embeddings) ─────────────────────
        power.memory_power_mW = self.kv_cache_size_GB * self.LPDDR5_POWER_mW_per_GB

        # ── TEC temperature stabilizer ──────────────────────────────────────
        # One TEC per chip, ~2W per TEC at ΔT=10°C
        power.tec_power_mW = self.n_chips * 2000.0 / self.n_chips  # shared TEC control
        # More realistic: global temperature chamber at ~5W for small module
        power.tec_power_mW = min(power.tec_power_mW, 5000.0)

        return power

    def energy_per_token(
        self,
        system_power: Optional[PhotonicSystemPower] = None,
        tokens_per_sec: Optional[float] = None,
    ) -> float:
        """
        Compute energy per generated token [mJ].

        Args:
            system_power: Pre-computed power. If None, calls estimate().
            tokens_per_sec: Inference speed. Defaults to self.inference_tokens_per_sec.

        Returns:
            Energy per token in millijoules
        """
        if system_power is None:
            system_power = self.estimate()

        tps = tokens_per_sec or self.inference_tokens_per_sec
        return system_power.total_inference_power_W * 1000 / tps  # mJ/token

    def compare_to_gpu(
        self,
        gpu_ref: Optional[GPUInferencePower] = None,
    ) -> dict:
        """
        Compare power efficiency to GPU baseline.

        Args:
            gpu_ref: GPU reference. Defaults to A100.

        Returns:
            Comparison dictionary
        """
        gpu = gpu_ref or GPUInferencePower()
        photonic = self.estimate()

        photonic_energy = self.energy_per_token(photonic)
        gpu_energy = gpu.energy_per_token_mJ

        return {
            "photonic_power_W": photonic.total_inference_power_W,
            "gpu_power_W": gpu.inference_power_W,
            "power_reduction_factor": gpu.inference_power_W / photonic.total_inference_power_W,
            "photonic_energy_mJ_per_token": photonic_energy,
            "gpu_energy_mJ_per_token": gpu_energy,
            "energy_reduction_factor": gpu_energy / (photonic_energy + 1e-6),
            "gpu_reference": gpu.model,
        }


def estimate_system_power(
    total_mzis: int,
    n_chips: int,
    **kwargs,
) -> PhotonicSystemPower:
    """
    Convenience function to estimate system power from basic parameters.

    Args:
        total_mzis: Total MZI count in the system
        n_chips: Number of photonic chips
        **kwargs: Additional arguments for PowerModel

    Returns:
        PhotonicSystemPower breakdown
    """
    model = PowerModel(n_mzis=total_mzis, n_chips=n_chips, **kwargs)
    return model.estimate()
