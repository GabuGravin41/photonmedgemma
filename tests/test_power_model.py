"""
Power Model Tests
==================
Tests for the power estimation models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from utils.power_model import PowerModel, PhotonicSystemPower, GPUInferencePower


class TestPowerModel:

    def test_basic_estimate(self):
        """Power model should return positive values with meaningful components."""
        # Use a single-chip, single-wavelength system for this unit test
        model = PowerModel(n_mzis=4_000, n_chips=1, n_output_channels=64, n_wavelengths=1)
        power = model.estimate()

        assert power.total_inference_power_W > 0.0
        assert power.laser_power_mW > 0.0
        assert power.heater_static_power_mW == 0.0  # static mode: zero steady-state heater power
        assert power.detector_power_mW > 0.0

    def test_gpu_comparison(self):
        """Comparison dict should contain all required keys."""
        model = PowerModel(n_mzis=1_000_000, n_chips=250)
        comparison = model.compare_to_gpu()

        assert "photonic_power_W" in comparison
        assert "gpu_power_W" in comparison
        assert "power_reduction_factor" in comparison
        assert comparison["photonic_power_W"] > 0
        assert comparison["gpu_power_W"] > 0

    def test_energy_per_token(self):
        """Energy per token should be a positive finite number."""
        model = PowerModel(n_mzis=100_000, n_chips=25)
        power = model.estimate()
        energy = model.energy_per_token(power)

        assert energy > 0
        assert energy < 1e8  # should be some finite number

    def test_static_heater_power_zero(self):
        """Thermal phase shifters should have zero steady-state power."""
        model = PowerModel(n_mzis=1_000_000, n_chips=250)
        power = model.estimate()

        assert power.heater_static_power_mW == 0.0

    def test_power_report(self):
        """Power report should be a non-empty string."""
        model = PowerModel(n_mzis=10_000, n_chips=3)
        power = model.estimate()
        report = power.report()

        assert len(report) > 0
        assert "TOTAL" in report
        assert "Laser" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
