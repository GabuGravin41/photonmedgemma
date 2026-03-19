"""
Tests for the Clements Decomposition Algorithm
===============================================

These tests verify that:
1. The decomposition produces the correct number of MZIs
2. The reconstructed matrix matches the original (within numerical tolerance)
3. The simulation of the mesh gives correct matrix-vector products
4. Edge cases (N=1, N=2, identity) are handled correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from compiler.clements import (
    clements_decompose,
    clements_reconstruct,
    clements_simulate,
    decomposition_stats,
)


def random_unitary(N: int, seed: int = 42) -> np.ndarray:
    """Generate a random N×N unitary matrix using QR decomposition."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(A)
    return Q


class TestClementsDecomposition:

    def test_mzi_count_small(self):
        """Verify N(N-1)/2 MZIs for small matrices."""
        for N in [2, 3, 4, 5, 6]:
            U = random_unitary(N, seed=N)
            result = clements_decompose(U)
            expected_mzis = N * (N - 1) // 2
            assert result.n_mzis == expected_mzis, (
                f"N={N}: expected {expected_mzis} MZIs, got {result.n_mzis}"
            )

    def test_reconstruction_identity(self):
        """Identity matrix should decompose and reconstruct exactly."""
        N = 4
        U = np.eye(N, dtype=complex)
        result = clements_decompose(U)

        U_recon = clements_reconstruct(result.mzis, result.phase_screen, N)
        error = np.linalg.norm(U - U_recon)

        assert error < 1e-8, f"Identity reconstruction error: {error:.2e}"

    def test_reconstruction_accuracy_N4(self):
        """Test reconstruction accuracy for N=4."""
        N = 4
        U = random_unitary(N, seed=1)
        result = clements_decompose(U)

        U_recon = clements_reconstruct(result.mzis, result.phase_screen, N)
        error = np.linalg.norm(U - U_recon, 'fro') / np.linalg.norm(U, 'fro')

        assert error < 1e-10, f"N={N} reconstruction error: {error:.2e}"

    def test_reconstruction_accuracy_N8(self):
        """Test reconstruction accuracy for N=8."""
        N = 8
        U = random_unitary(N, seed=2)
        result = clements_decompose(U)

        U_recon = clements_reconstruct(result.mzis, result.phase_screen, N)
        error = np.linalg.norm(U - U_recon, 'fro') / np.linalg.norm(U, 'fro')

        assert error < 1e-8, f"N={N} reconstruction error: {error:.2e}"

    def test_reconstruction_accuracy_N16(self):
        """Test reconstruction accuracy for N=16."""
        N = 16
        U = random_unitary(N, seed=3)
        result = clements_decompose(U)

        error = result.reconstruction_error
        assert error < 1e-6, f"N={N} reconstruction error: {error:.2e}"

    def test_simulation_matches_matrix_product(self):
        """
        Test that simulating the mesh gives the same result as
        multiplying by the reconstructed matrix.
        """
        N = 6
        U = random_unitary(N, seed=10)
        result = clements_decompose(U)

        # Random input
        x = np.random.randn(N) + 1j * np.random.randn(N)
        x = x / np.linalg.norm(x)

        # Method 1: simulate the mesh
        y_sim = clements_simulate(x, result.mzis, result.phase_screen, N)

        # Method 2: multiply by reconstructed matrix
        U_recon = clements_reconstruct(result.mzis, result.phase_screen, N)
        y_mat = U_recon @ x

        error = np.linalg.norm(y_sim - y_mat) / np.linalg.norm(y_mat)
        assert error < 1e-8, f"Simulation vs. matrix error: {error:.2e}"

    def test_unitarity_preserved(self):
        """Reconstructed matrix should be unitary."""
        N = 8
        U = random_unitary(N, seed=5)
        result = clements_decompose(U)

        U_recon = clements_reconstruct(result.mzis, result.phase_screen, N)
        unitarity_error = np.linalg.norm(U_recon @ U_recon.conj().T - np.eye(N))

        assert unitarity_error < 1e-8, (
            f"Reconstructed matrix not unitary: error={unitarity_error:.2e}"
        )

    def test_decomposition_stats(self):
        """Test that decomposition stats are computed correctly."""
        N = 6
        U = random_unitary(N, seed=7)
        result = clements_decompose(U)
        stats = decomposition_stats(result)

        assert stats["N"] == N
        assert stats["n_mzis_actual"] == N * (N - 1) // 2
        assert stats["n_mzis_expected"] == N * (N - 1) // 2
        assert 0 <= stats["theta_mean"] <= 2 * np.pi
        assert 0 <= stats["phi_mean"] <= 2 * np.pi

    def test_phase_range(self):
        """All phase angles should be in [0, 2π)."""
        N = 6
        U = random_unitary(N, seed=8)
        result = clements_decompose(U)

        for mzi in result.mzis:
            assert 0 <= mzi.theta <= 2 * np.pi + 1e-10, f"theta={mzi.theta} out of range"
            assert 0 <= mzi.phi <= 2 * np.pi + 1e-10, f"phi={mzi.phi} out of range"

    def test_real_matrix_decomposition(self):
        """Real-valued unitary (orthogonal) matrices should also work."""
        N = 5
        # Random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(N, N))

        result = clements_decompose(Q.astype(complex))
        error = result.reconstruction_error

        assert error < 1e-8, f"Real matrix reconstruction error: {error:.2e}"

    @pytest.mark.parametrize("N", [2, 3, 4, 6, 8])
    def test_multiple_sizes(self, N):
        """Parametrized test for multiple matrix sizes."""
        U = random_unitary(N, seed=N * 100)
        result = clements_decompose(U)

        assert result.n_mzis == N * (N - 1) // 2
        assert result.reconstruction_error < 1e-6, (
            f"N={N}: reconstruction error {result.reconstruction_error:.2e}"
        )


class TestMZIMesh:
    """Tests for the ClementsMesh wrapper."""

    def test_mesh_forward_pass(self):
        """Mesh forward pass should compute U×x correctly."""
        from photonic.mesh import ClementsMesh, MeshConfig

        N = 8
        U = random_unitary(N, seed=42)
        config = MeshConfig(N=N, rank=N, layer_name="test")
        mesh = ClementsMesh.from_matrix(U, config)

        x = np.random.randn(N)
        y_mesh = np.real(mesh.forward(x))
        y_ref = np.real(U @ x)

        error = np.linalg.norm(y_mesh - y_ref) / (np.linalg.norm(y_ref) + 1e-15)
        assert error < 0.01, f"Mesh forward error: {error:.4f}"

    def test_mesh_reconstruction_error(self):
        """Reconstruction error should be small."""
        from photonic.mesh import ClementsMesh, MeshConfig

        N = 6
        U = random_unitary(N, seed=99)
        config = MeshConfig(N=N, rank=N)
        mesh = ClementsMesh.from_matrix(U, config)

        assert mesh.reconstruction_error() < 1e-6

    def test_phase_map_dac_codes(self):
        """DAC codes should be in valid range."""
        from photonic.mesh import ClementsMesh, MeshConfig

        N = 4
        U = random_unitary(N, seed=77)
        config = MeshConfig(N=N, rank=N, dac_bits=12)
        mesh = ClementsMesh.from_matrix(U, config)

        phase_map = mesh.get_phase_map()
        max_code = (1 << 12) - 1

        assert all(0 <= c <= max_code for c in phase_map["theta_dac"])
        assert all(0 <= c <= max_code for c in phase_map["phi_dac"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
