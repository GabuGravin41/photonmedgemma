"""
Phase Encoder — Convert Phase Angles to Chip Configuration
============================================================

Takes compiled Clements phase angles and converts them to:
1. Integer DAC codes for physical chip programming
2. Binary configuration files (.phcfg) for deployment
3. Chip initialization scripts (SPI bitstream for the control MCU)

The phase encoder is the final step before the chip configuration
is shipped to hardware. It handles:
- Phase quantization (float64 → N-bit integer)
- Chip address mapping (which register on which chip gets which code)
- Checksum generation (for data integrity verification)
- Phase calibration metadata (for post-fabrication correction)
"""

import json
import struct
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np

from .mzi_mapper import CompiledLayer

logger = logging.getLogger(__name__)


@dataclass
class MZIPhaseEntry:
    """Single MZI phase configuration entry."""
    chip_id: int
    mzi_row: int
    mzi_col: int
    mode_i: int
    mode_j: int
    theta_rad: float
    phi_rad: float
    theta_dac: int
    phi_dac: int
    layer_name: str
    matrix_type: str  # "U" or "Vh"


@dataclass
class PhaseMap:
    """
    Complete phase configuration for the photonic chip system.

    Contains all MZI phase assignments for a compiled model,
    ready for physical chip programming.
    """
    model_id: str
    rank: int
    dac_bits: int
    n_chips: int
    n_mzis: int
    entries: List[MZIPhaseEntry] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "model_id": self.model_id,
            "rank": self.rank,
            "dac_bits": self.dac_bits,
            "n_chips": self.n_chips,
            "n_mzis": self.n_mzis,
            "metadata": self.metadata,
            "entries": [
                {
                    "chip_id": e.chip_id,
                    "mzi_row": e.mzi_row,
                    "mzi_col": e.mzi_col,
                    "mode_i": e.mode_i,
                    "mode_j": e.mode_j,
                    "theta_rad": e.theta_rad,
                    "phi_rad": e.phi_rad,
                    "theta_dac": e.theta_dac,
                    "phi_dac": e.phi_dac,
                    "layer_name": e.layer_name,
                    "matrix_type": e.matrix_type,
                }
                for e in self.entries
            ],
        }

    def save_json(self, path: str):
        """Save phase map to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Phase map saved to {path} ({len(self.entries):,} entries)")

    @classmethod
    def load_json(cls, path: str) -> "PhaseMap":
        """Load phase map from JSON file."""
        with open(path) as f:
            data = json.load(f)

        entries = [MZIPhaseEntry(**e) for e in data.pop("entries")]
        pm = cls(**{k: v for k, v in data.items() if k != "entries"})
        pm.entries = entries
        return pm

    def save_binary(self, path: str):
        """
        Save phase map in compact binary format (.phcfg).

        Binary format:
            Header: magic(4) + version(2) + n_entries(4) + dac_bits(1) + pad(5) = 16 bytes
            Entry:  chip_id(2) + row(2) + col(2) + theta_dac(2) + phi_dac(2) = 10 bytes
            Footer: SHA256 checksum (32 bytes)

        Total: 16 + n_entries × 10 + 32 bytes
        """
        MAGIC = b'PMGM'  # PhotoMedGemma Magic
        VERSION = 1

        header = struct.pack(
            '>4sHIBxxx',  # big-endian: magic(4), version(2), n_entries(4), dac_bits(1), pad(3) = 14 bytes
            MAGIC, VERSION, len(self.entries), self.dac_bits
        )

        body = bytearray()
        for e in self.entries:
            body += struct.pack(
                '>HHHHHHH',
                e.chip_id,
                e.mzi_row & 0xFFFF,
                e.mzi_col & 0xFFFF,
                e.theta_dac & 0xFFFF,
                e.phi_dac & 0xFFFF,
                e.mode_i & 0xFFFF,
                e.mode_j & 0xFFFF,
            )

        checksum = hashlib.sha256(header + bytes(body)).digest()

        with open(path, 'wb') as f:
            f.write(header)
            f.write(body)
            f.write(checksum)

        size_kb = (len(header) + len(body) + len(checksum)) / 1024
        logger.info(f"Binary phase config saved to {path} ({size_kb:.1f} KB)")

    def entries_for_chip(self, chip_id: int) -> List[MZIPhaseEntry]:
        """Get all entries assigned to a specific chip."""
        return [e for e in self.entries if e.chip_id == chip_id]

    def generate_spi_bitstream(self, chip_id: int) -> bytes:
        """
        Generate SPI command bitstream for programming a single chip.

        Format: [register_addr(16bit) | dac_value(16bit)] per phase register
        Each command is 4 bytes: addr(16bit) + data(16bit)

        Phase screen entries (mzi_row == -1) are assigned to a dedicated
        register bank starting at base address 0xF000.

        Args:
            chip_id: Target chip ID

        Returns:
            Raw bytes for SPI transmission
        """
        entries = self.entries_for_chip(chip_id)
        bitstream = bytearray()

        PHASE_SCREEN_BASE = 0xF000  # dedicated register bank for phase screen

        for e in entries:
            if e.mzi_row == -1:
                # Phase screen register: addr = PHASE_SCREEN_BASE + col*2
                base_addr = PHASE_SCREEN_BASE + e.mzi_col * 2
                # Only theta is meaningful for phase screen (phi=0)
                theta_cmd = struct.pack('>HH', base_addr & 0xFFFF, e.theta_dac & 0xFFFF)
                bitstream += theta_cmd
            else:
                # MZI register: addr = row * 256 + col * 2
                base_addr = e.mzi_row * 256 + e.mzi_col * 2

                # Theta register
                theta_cmd = struct.pack('>HH', base_addr & 0xFFFF, e.theta_dac & 0xFFFF)
                bitstream += theta_cmd

                # Phi register
                phi_cmd = struct.pack('>HH', (base_addr + 1) & 0xFFFF, e.phi_dac & 0xFFFF)
                bitstream += phi_cmd

        return bytes(bitstream)


class PhaseEncoder:
    """
    Encodes compiled photonic phase angles into chip configuration.

    Takes a list of CompiledLayer objects and produces:
    1. A PhaseMap (complete phase configuration)
    2. Per-chip SPI bitstreams for hardware programming
    3. Calibration metadata for post-fabrication adjustment
    """

    def __init__(self, dac_bits: int = 12):
        """
        Args:
            dac_bits: DAC resolution in bits. 12 bits → 0.088° resolution.
        """
        self.dac_bits = dac_bits
        self.max_code = (1 << dac_bits) - 1

    def encode(
        self,
        compiled_layers: List[CompiledLayer],
        model_id: str = "medgemma-4b-it",
        rank: int = 64,
    ) -> PhaseMap:
        """
        Encode all compiled layers into a PhaseMap.

        Args:
            compiled_layers: List of CompiledLayer from MZIMapper
            model_id: Model identifier string
            rank: SVD rank used in compilation

        Returns:
            PhaseMap with all MZI phase assignments
        """
        entries = []
        all_chip_ids = set()

        for layer in compiled_layers:
            # Encode V† mesh
            entries += self._encode_mesh_entries(
                layer.Vh_mesh.get_phase_map(),
                layer.Vh_mesh.result.mzis,
                chip_id=layer.chip_id_Vh,
                layer_name=layer.layer_name,
                matrix_type="Vh",
            )
            all_chip_ids.add(layer.chip_id_Vh)

            # Encode U mesh
            entries += self._encode_mesh_entries(
                layer.U_mesh.get_phase_map(),
                layer.U_mesh.result.mzis,
                chip_id=layer.chip_id_U,
                layer_name=layer.layer_name,
                matrix_type="U",
            )
            all_chip_ids.add(layer.chip_id_U)

        phase_map = PhaseMap(
            model_id=model_id,
            rank=rank,
            dac_bits=self.dac_bits,
            n_chips=len(all_chip_ids),
            n_mzis=len(entries),
            entries=entries,
            metadata={
                "n_layers": len(compiled_layers),
                "total_mzis": len(entries) // 2,  # ÷2 because each MZI has θ and φ
                "dac_resolution_rad": 2 * 3.14159265 / self.max_code,
                "phase_rms_error_rad": 3.14159265 / (self.max_code * 1.732),  # π/(√3 × 2^B)
            },
        )

        logger.info(
            f"Encoded {len(entries):,} phase entries across "
            f"{len(all_chip_ids)} chips."
        )

        return phase_map

    def _encode_mesh_entries(
        self,
        phase_map: dict,
        mzi_specs,
        chip_id: int,
        layer_name: str,
        matrix_type: str,
    ) -> List[MZIPhaseEntry]:
        """Encode MZI specs from a mesh into PhaseMap entries."""
        entries = []

        for i, mzi in enumerate(mzi_specs):
            theta_rad = float(mzi.theta)
            phi_rad = float(mzi.phi)

            theta_dac = self._quantize(theta_rad)
            phi_dac = self._quantize(phi_rad)

            entry = MZIPhaseEntry(
                chip_id=chip_id,
                mzi_row=int(mzi.row),
                mzi_col=int(mzi.col),
                mode_i=int(mzi.mode_i),
                mode_j=int(mzi.mode_j),
                theta_rad=theta_rad,
                phi_rad=phi_rad,
                theta_dac=theta_dac,
                phi_dac=phi_dac,
                layer_name=layer_name,
                matrix_type=matrix_type,
            )
            entries.append(entry)

        # Also encode phase screen entries (using special row=-1 convention)
        if "phase_screen" in phase_map:
            for j, phase in enumerate(phase_map["phase_screen"]):
                dac_code = self._quantize(float(phase))
                entry = MZIPhaseEntry(
                    chip_id=chip_id,
                    mzi_row=-1,   # convention: -1 = phase screen register
                    mzi_col=j,
                    mode_i=j,
                    mode_j=j,
                    theta_rad=float(phase),
                    phi_rad=0.0,
                    theta_dac=dac_code,
                    phi_dac=0,
                    layer_name=layer_name,
                    matrix_type=f"{matrix_type}_phase_screen",
                )
                entries.append(entry)

        return entries

    def _quantize(self, angle: float) -> int:
        """Quantize a phase angle to DAC code."""
        normalized = (angle % (2 * np.pi)) / (2 * np.pi)
        code = int(round(normalized * self.max_code))
        return code % (self.max_code + 1)

    def phase_error_analysis(self) -> dict:
        """
        Compute theoretical phase encoding errors.

        Returns:
            Dict with error statistics
        """
        lsb_rad = 2 * np.pi / (self.max_code + 1)
        max_error_rad = lsb_rad / 2
        rms_error_rad = lsb_rad / (2 * np.sqrt(3))  # uniform quantization

        return {
            "dac_bits": self.dac_bits,
            "lsb_rad": lsb_rad,
            "lsb_deg": np.degrees(lsb_rad),
            "max_phase_error_rad": max_error_rad,
            "rms_phase_error_rad": rms_error_rad,
            "rms_phase_error_deg": np.degrees(rms_error_rad),
        }
