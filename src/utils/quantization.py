"""
Phase Quantization Utilities
==============================
Functions for converting between floating-point phase angles
and integer DAC codes for chip programming.
"""

import numpy as np
from typing import Union


def quantize_phase(angle: Union[float, np.ndarray], bits: int = 12) -> Union[int, np.ndarray]:
    """
    Quantize phase angle(s) to integer DAC code(s).

    Maps angle ∈ [0, 2π) → integer code ∈ [0, 2^bits - 1]

    Args:
        angle: Phase angle in radians (scalar or array)
        bits: DAC resolution in bits

    Returns:
        Integer DAC code(s)
    """
    max_code = (1 << bits) - 1
    normalized = (np.asarray(angle) % (2 * np.pi)) / (2 * np.pi)
    codes = np.round(normalized * max_code).astype(int) % (max_code + 1)
    return int(codes) if codes.ndim == 0 else codes


def dequantize_phase(code: Union[int, np.ndarray], bits: int = 12) -> Union[float, np.ndarray]:
    """
    Convert integer DAC code(s) back to phase angle(s).

    Args:
        code: Integer code(s) ∈ [0, 2^bits - 1]
        bits: DAC resolution in bits

    Returns:
        Phase angle(s) in radians ∈ [0, 2π)
    """
    max_code = (1 << bits) - 1
    return (np.asarray(code) / max_code) * 2 * np.pi


def phase_quantization_error(bits: int = 12) -> dict:
    """
    Compute phase quantization error statistics for given DAC resolution.

    Args:
        bits: DAC resolution in bits

    Returns:
        Dict with error metrics
    """
    lsb = 2 * np.pi / (1 << bits)
    max_err = lsb / 2
    rms_err = lsb / (2 * np.sqrt(3))  # uniform quantization RMS

    return {
        "bits": bits,
        "n_levels": 1 << bits,
        "lsb_rad": lsb,
        "lsb_deg": np.degrees(lsb),
        "max_error_rad": max_err,
        "max_error_deg": np.degrees(max_err),
        "rms_error_rad": rms_err,
        "rms_error_deg": np.degrees(rms_err),
    }


def phase_matrix_reconstruction_error(
    phases: np.ndarray,
    bits: int = 12,
    N: int = 64,
) -> float:
    """
    Estimate matrix reconstruction error from phase quantization.

    For an N-mode Clements mesh with B-bit phase DACs, the matrix
    reconstruction error scales approximately as:
        ε ≈ π√N / (2^B × √3)

    Args:
        phases: Array of phase angles
        bits: DAC resolution
        N: Matrix dimension

    Returns:
        Estimated relative matrix error
    """
    rms_phase_err = phase_quantization_error(bits)["rms_error_rad"]
    n_mzis = N * (N - 1) // 2
    # Each MZI has 2 phase errors; errors accumulate as √(2×n_mzis)
    matrix_error = rms_phase_err * np.sqrt(2 * n_mzis) / N
    return float(matrix_error)
