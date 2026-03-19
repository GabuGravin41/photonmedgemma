"""
PhotoMedGemma Photonic Primitives Library
==========================================
Core photonic building blocks for simulating and designing
MZI-based optical neural network hardware.

All physical quantities in SI units unless otherwise noted:
- Lengths: meters (m)
- Wavelengths: meters (m)
- Power: watts (W)
- Phase: radians (rad)
- Loss: dB/m
"""

from photonic.mzi import MZI, MZIParameters
from photonic.waveguide import Waveguide, WaveguideSpec
from photonic.mesh import ClementsMesh, SigmaStage, SVDLayer, MeshConfig
from photonic.splitter import DirectionalCoupler, YSplitter
from photonic.phase_shifter import ThermalPhaseShifter, ElectroOpticPhaseShifter
from photonic.photodetector import PhotoDetector
from photonic.grating_coupler import GratingCoupler

__version__ = "0.1.0"

__all__ = [
    "MZI",
    "MZIParameters",
    "Waveguide",
    "WaveguideSpec",
    "ClementsMesh",
    "SigmaStage",
    "SVDLayer",
    "MeshConfig",
    "DirectionalCoupler",
    "YSplitter",
    "ThermalPhaseShifter",
    "ElectroOpticPhaseShifter",
    "PhotoDetector",
    "GratingCoupler",
]
