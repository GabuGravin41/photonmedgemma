"""
End-to-End Compiler Tests
==========================
Tests for the complete compilation pipeline:
    Weight matrix → SVD → Clements → Phase encoding → Netlist
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import tempfile
import os

from compiler.model_parser import LayerInfo
from compiler.layer_decomposer import LayerDecomposer
from compiler.mzi_mapper import MZIMapper
from compiler.phase_encoder import PhaseEncoder
from compiler.netlist_generator import NetlistGenerator


def make_test_weight(m: int, n: int, rank: int = 5, seed: int = 42) -> np.ndarray:
    """Create a synthetic weight matrix with controlled rank structure."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((m, rank))
    V = rng.standard_normal((rank, n))
    S = np.exp(-np.arange(rank) / 2.0)
    W = (U * S) @ V
    return W.astype(np.float32)


def make_layer_info(name: str, shape, proj_type: str = "q") -> LayerInfo:
    return LayerInfo(
        name=name,
        shape=shape,
        module_type="attention",
        transformer_layer_idx=0,
        projection_type=proj_type,
        component="language_model",
    )


class TestLayerDecomposer:

    def test_basic_svd(self):
        """SVD factors should reconstruct to original matrix within rank error."""
        W = make_test_weight(32, 32, rank=8)
        info = make_layer_info("test.weight", W.shape)
        decomposer = LayerDecomposer(rank=8)

        decomposed = decomposer.decompose(info, W)

        assert decomposed.rank == 8
        assert decomposed.reconstruction_error < 0.5  # rank-8 of rank-8 matrix should be exact-ish
        assert decomposed.energy_retained > 0.8

    def test_energy_threshold(self):
        """Energy-based rank selection should respect threshold."""
        W = make_test_weight(32, 32, rank=16)
        info = make_layer_info("test.weight", W.shape)
        decomposer = LayerDecomposer(rank=None, energy_threshold=0.95)

        decomposed = decomposer.decompose(info, W)

        assert decomposed.energy_retained >= 0.95 or decomposed.rank == min(32, 32)

    def test_full_unitaries_are_unitary(self):
        """U_full and Vh_full should be unitary matrices."""
        W = make_test_weight(16, 16, rank=4)
        info = make_layer_info("test.weight", W.shape)
        decomposer = LayerDecomposer(rank=4)

        decomposed = decomposer.decompose(info, W)

        N_U = decomposed.U_full.shape[0]
        N_Vh = decomposed.Vh_full.shape[0]

        err_U = np.linalg.norm(
            decomposed.U_full.astype(np.float64) @ decomposed.U_full.astype(np.float64).T
            - np.eye(N_U)
        )
        err_Vh = np.linalg.norm(
            decomposed.Vh_full.astype(np.float64) @ decomposed.Vh_full.astype(np.float64).T
            - np.eye(N_Vh)
        )

        assert err_U < 1e-4, f"U_full not unitary: {err_U:.2e}"
        assert err_Vh < 1e-4, f"Vh_full not unitary: {err_Vh:.2e}"


class TestMZIMapper:

    def test_compile_small_layer(self):
        """Compile a small weight matrix end-to-end."""
        N = 8
        rank = 4
        W = make_test_weight(N, N, rank=rank)
        info = make_layer_info("test.q_proj.weight", W.shape, "q")

        # min_rank=1 so the requested rank is not clipped for this small test
        decomposer = LayerDecomposer(rank=rank, min_rank=1)
        decomposed = decomposer.decompose(info, W)

        mapper = MZIMapper(mzis_per_chip=10000)
        compiled = mapper.map_layer(decomposed, verbose=False)

        assert compiled.n_mzis_U > 0
        assert compiled.n_mzis_Vh > 0
        assert compiled.rank == rank
        assert compiled.reconstruction_error_clements_U < 1e-4
        assert compiled.reconstruction_error_clements_Vh < 1e-4

    def test_forward_pass_accuracy(self):
        """Photonic forward pass should approximate matrix multiplication."""
        N = 8
        rank = 6
        W = make_test_weight(N, N, rank=rank)
        info = make_layer_info("test.q_proj.weight", W.shape, "q")

        # min_rank=1 so the requested rank is not clipped for this small test
        decomposer = LayerDecomposer(rank=rank, min_rank=1)
        decomposed = decomposer.decompose(info, W)

        mapper = MZIMapper()
        compiled = mapper.map_layer(decomposed, verbose=False)

        # Test forward pass
        from photonic.mesh import SVDLayer
        svd_layer = SVDLayer(
            U_mesh=compiled.U_mesh,
            sigma_stage=compiled.sigma_stage,
            Vh_mesh=compiled.Vh_mesh,
            layer_name="test",
        )

        x = np.random.randn(N).astype(np.float32)
        y_photonic = svd_layer.forward(x)

        # Compare to SVD approximation using normalized singular values
        # (SVDLayer uses normalized_sv internally; use the same for ground truth)
        actual_rank = compiled.rank
        U_r = decomposed.U_full[:, :actual_rank]
        Vh_r = decomposed.Vh_full[:actual_rank, :]
        sigma_n = compiled.sigma_stage.normalized_sv  # normalized singular values
        y_svd = U_r @ (sigma_n[:actual_rank] * (Vh_r @ x))

        error = np.linalg.norm(y_svd - y_photonic.real) / (np.linalg.norm(y_svd) + 1e-10)
        assert error < 0.01, f"Forward pass error too large: {error:.4f}"


class TestPhaseEncoder:

    def test_phase_quantization_roundtrip(self):
        """Quantized and dequantized phases should be close to originals."""
        from utils.quantization import quantize_phase, dequantize_phase

        test_phases = np.linspace(0, 2 * np.pi - 0.001, 100)
        for bits in [8, 10, 12]:
            codes = quantize_phase(test_phases, bits=bits)
            recovered = dequantize_phase(codes, bits=bits)
            lsb = 2 * np.pi / (1 << bits)
            errors = np.abs(test_phases - recovered)
            assert np.max(errors) <= lsb, f"Max error {np.max(errors):.4f} > LSB {lsb:.4f} at {bits} bits"

    def test_phase_map_generation(self):
        """Phase map should be generated correctly from compiled layers."""
        N = 6
        rank = 3
        W = make_test_weight(N, N, rank=rank)
        info = make_layer_info("model.layers.0.q_proj.weight", W.shape, "q")

        decomposer = LayerDecomposer(rank=rank)
        decomposed = decomposer.decompose(info, W)
        mapper = MZIMapper()
        compiled = mapper.map_layer(decomposed, verbose=False)

        encoder = PhaseEncoder(dac_bits=12)
        phase_map = encoder.encode([compiled], model_id="test", rank=rank)

        assert phase_map.n_mzis > 0
        assert phase_map.dac_bits == 12
        assert len(phase_map.entries) > 0

        # All DAC codes should be in valid range
        max_code = (1 << 12) - 1
        for entry in phase_map.entries:
            assert 0 <= entry.theta_dac <= max_code
            assert 0 <= entry.phi_dac <= max_code


class TestNetlistGenerator:

    def test_netlist_generation(self):
        """Netlist should be generated without errors."""
        N = 6
        rank = 3
        W = make_test_weight(N, N, rank=rank)
        info = make_layer_info("model.layers.0.q_proj.weight", W.shape, "q")

        decomposer = LayerDecomposer(rank=rank)
        decomposed = decomposer.decompose(info, W)
        mapper = MZIMapper()
        compiled = mapper.map_layer(decomposed, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = NetlistGenerator(output_dir=tmpdir)
            netlist = generator.generate_layer_netlist(compiled)

            # Netlist should contain key sections
            assert ".subckt" in netlist
            assert ".ends" in netlist
            assert "MZI" in netlist
            assert "ATTENUATOR" in netlist
            assert "PHASE_SCREEN" in netlist

    def test_json_manifest(self):
        """JSON manifest should be valid and complete."""
        N = 6
        rank = 3
        W = make_test_weight(N, N, rank=rank)
        info = make_layer_info("model.layers.0.q_proj.weight", W.shape, "q")

        decomposer = LayerDecomposer(rank=rank, min_rank=1)
        decomposed = decomposer.decompose(info, W)
        mapper = MZIMapper()
        compiled = mapper.map_layer(decomposed, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = NetlistGenerator(output_dir=tmpdir)
            manifest = generator.generate_json_manifest([compiled])

        assert manifest["n_layers"] == 1
        assert manifest["total_mzis"] > 0
        assert len(manifest["layers"]) == 1
        assert manifest["layers"][0]["rank"] == rank


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
