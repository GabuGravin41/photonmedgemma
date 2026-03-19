"""
Reck / Clements Decomposition — MZI Mesh Synthesis from Unitary Matrices
==========================================================================

Decomposes any N×N unitary matrix into N(N-1)/2 Mach-Zehnder Interferometers.

We use the **column-by-column triangular (Reck) decomposition** with LEFT
multiplications (T† acts on ROWS, not columns):

    D = T†_M × ... × T†_2 × T†_1 × U
    ⇒ U = T_1 × T_2 × ... × T_M × D

where D = diag(e^{iφ_0}, ..., e^{iφ_{N-1}}) is the phase screen.

MZI convention (standard beamsplitter basis):
    T(θ, φ) = [cos(θ/2)          -e^{-iφ} sin(θ/2)]
               [e^{iφ} sin(θ/2)   cos(θ/2)          ]

    T†(θ, φ) = [cos(θ/2)            e^{-iφ} sin(θ/2)]
                [-e^{iφ} sin(θ/2)   cos(θ/2)          ]

Nulling condition: Apply T†(θ, φ) from LEFT on rows (i, j) to zero W[j, col]:
    T† @ [W[i,col]; W[j,col]] = [c; 0]
    Second element: -e^{iφ}·sin(θ/2)·W[i,col] + cos(θ/2)·W[j,col] = 0
    → tan(θ/2) = W[j,col] / (e^{iφ} · W[i,col])
    Choose φ = angle(W[j,col]) - angle(W[i,col]) to make the ratio real:
    → θ = 2·arctan2(|W[j,col]|, |W[i,col]|)

Reconstruction: U = T_1 × T_2 × ... × T_M × D
    Build T_1 × ... × T_M by iterating reversed(mzis) with left multiplication,
    then post-multiply by D.

Simulation: U×x = T_1 × ... × T_M × D × x
    Apply D first, then iterate reversed(mzis) left-multiplying each T_k.

Reference:
    M. Reck et al., Phys. Rev. Lett. 73, 58 (1994).
    W. R. Clements et al., Optica 3, 1460 (2016).
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings


@dataclass
class MZISpec:
    """Specifies one MZI in the decomposition mesh."""
    row: int       # mesh row (0 = top)
    col: int       # mesh column (stage index)
    theta: float   # splitting angle [rad]
    phi: float     # phase angle [rad]
    mode_i: int    # first mode (upper)
    mode_j: int    # second mode (lower, gets nulled)


@dataclass
class ClementsResult:
    """Output of the unitary decomposition."""
    N: int
    mzis: List[MZISpec]
    phase_screen: np.ndarray      # length-N diagonal phases [rad]
    reconstruction_error: float   # relative Frobenius error
    n_mzis: int


# ─── MZI math ─────────────────────────────────────────────────────────────────

def _T(theta: float, phi: float) -> np.ndarray:
    """2×2 MZI transfer matrix T(θ, φ)."""
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    ep = np.exp(1j * phi)
    return np.array([
        [c,        -np.conj(ep) * s],
        [ep * s,    c              ],
    ], dtype=complex)


def _Tdag(theta: float, phi: float) -> np.ndarray:
    """Conjugate transpose (inverse) of T(θ, φ)."""
    return _T(theta, phi).conj().T


def _null_params(a: complex, b: complex) -> Tuple[float, float]:
    """
    Find (θ, φ) so that T†(θ, φ) @ [a; b] = [c; 0]  (LEFT multiplication).

    T†[1,0]·a + T†[1,1]·b = 0
    → -e^{iφ}·sin(θ/2)·a + cos(θ/2)·b = 0
    → tan(θ/2) = b / (e^{iφ}·a)

    Choose φ = angle(b) - angle(a) so the ratio is real positive:
        e^{iφ}·a = |a|·e^{iangle(b)}
        b / (e^{iφ}·a) = |b|/|a|  (real, positive)
    → θ = 2·arctan2(|b|, |a|)

    Verification: second element of T† @ [a; b]:
        -e^{iφ}·sin(θ/2)·a + cos(θ/2)·b
        = -e^{i(angle(b)-angle(a))}·|a|·sin(θ/2)·e^{iangle(a)} + |b|·cos(θ/2)·e^{iangle(b)}
        = e^{iangle(b)}·(-|a|·sin(θ/2) + |b|·cos(θ/2))
        With tan(θ/2) = |b|/|a|: sin = |b|/r, cos = |a|/r → = e^{iangle(b)}·0 = 0 ✓

    Args:
        a: W[row-1, col]  (upper element)
        b: W[row, col]    (lower element, to be nulled)

    Returns:
        (theta, phi) in [0, 2π)
    """
    if np.abs(b) < 1e-15:
        return 0.0, 0.0

    phi = np.angle(b) - np.angle(a)
    theta = 2.0 * np.arctan2(np.abs(b), np.abs(a))

    return float(theta % (2 * np.pi)), float(phi % (2 * np.pi))


# ─── Decomposition ────────────────────────────────────────────────────────────

def clements_decompose(U: np.ndarray, eps: float = 1e-12) -> ClementsResult:
    """
    Decompose unitary matrix U into N(N-1)/2 MZI phase settings.

    Computes: U = T_1 × T_2 × ... × T_M × D
    where D = diag(e^{iφ_0}, ..., e^{iφ_{N-1}}) is the phase screen.

    Algorithm (LEFT-elimination, column by column):
        W = U.copy()
        For col = 0, 1, ..., N-2:
            For row = N-1, N-2, ..., col+1:
                Find (θ, φ) to null W[row, col] by LEFT multiply
                W ← T†(θ, φ) × W   applied to rows (row-1, row)
        → W becomes diagonal D

    The left multiplications accumulate as:
        D = T†_M × ... × T†_1 × U  →  U = T_1 × ... × T_M × D

    Args:
        U: N×N complex unitary matrix
        eps: Tolerance for zero-element detection

    Returns:
        ClementsResult with MZI phases and diagonal phase screen
    """
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f"Expected square 2D matrix, got shape {U.shape}")

    N = U.shape[0]

    # Ensure matrix is unitary (re-orthogonalize if needed)
    err = np.linalg.norm(U @ U.conj().T - np.eye(N))
    if err > 1e-4:
        warnings.warn(f"Input matrix deviation from unitary: {err:.2e}. Re-orthogonalizing.")
        Uu, _, Uvh = np.linalg.svd(U)
        U = (Uu @ Uvh).astype(complex)

    W = U.astype(complex).copy()
    mzis: List[MZISpec] = []

    # Column-by-column left elimination: null W[row, col] for row > col
    # Process within each column from bottom (row=N-1) up to row=col+1.
    # Using T†_{row-1, row} from the LEFT modifies rows row-1 and row for ALL columns.
    # Previously-zeroed elements W[r', c'] (c' < col, r' > c') stay zero because:
    #   new W[row, c'] = T†[1,1]*W[row,c'] + T†[1,0]*W[row-1,c'] = 0 (both already 0)
    #
    # MZISpec.col = physical mesh column (0 to N-2), matching the loop variable.
    # MZISpec.row = physical mesh row = mode_i (upper mode, 0 to N-2).
    # Together (row, col) uniquely identify each MZI in the triangular mesh.
    # SPI address: mzi_row * 256 + mzi_col * 2  (fits in 16 bits for N ≤ 256).
    for col in range(N - 1):
        for row in range(N - 1, col, -1):
            # Null W[row, col] using T†(θ, φ) acting on rows (row-1, row)
            a = W[row - 1, col]
            b = W[row, col]

            theta, phi = _null_params(a, b)

            # Apply T† to rows (row-1, row): W[[row-1, row], :] = T† @ W[[row-1, row], :]
            Td = _Tdag(theta, phi)
            rows_ij = W[[row - 1, row], :].copy()
            W[[row - 1, row], :] = Td @ rows_ij

            mzis.append(MZISpec(
                row=row - 1,         # mode_i: upper mode (0 to N-2)
                col=col,             # physical mesh column (0 to N-2)
                theta=float(theta),
                phi=float(phi),
                mode_i=row - 1,
                mode_j=row,
            ))

    # W should now be diagonal
    phase_screen = np.angle(np.diag(W))

    # Verify
    off_diag = np.linalg.norm(W - np.diag(np.diag(W)))
    if off_diag > 1e-4:
        warnings.warn(
            f"Off-diagonal residual after elimination: {off_diag:.2e}. "
            "Numerical accumulation for large N is expected."
        )

    # Compute reconstruction error
    U_recon = _reconstruct(mzis, phase_screen, N)
    recon_err = np.linalg.norm(U - U_recon, 'fro') / (np.linalg.norm(U, 'fro') + 1e-15)

    return ClementsResult(
        N=N,
        mzis=mzis,
        phase_screen=phase_screen,
        reconstruction_error=float(recon_err),
        n_mzis=len(mzis),
    )


def _reconstruct(
    mzis: List[MZISpec],
    phase_screen: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    Reconstruct U from Reck decomposition.

    During decomposition (right multiply):
        W = U × T_1† × T_2† × ... × T_M† = D
    So:
        U = D × T_M × ... × T_2 × T_1

    Build this product by applying T_k from the LEFT in reversed order:
        R = I
        for k = M, M-1, ..., 1:
            R[[mode_i, mode_j], :] = T_k @ R[[mode_i, mode_j], :]
        then: R = D @ R

    Args:
        mzis: MZI list in decomposition order (T_1 was applied first)
        phase_screen: N-element diagonal phases
        N: Dimension

    Returns:
        Reconstructed N×N unitary
    """
    R = np.eye(N, dtype=complex)

    # U = T_1 × T_2 × ... × T_M × D
    # Build T_1 × ... × T_M by left-multiplying in REVERSED order:
    #   iter 1 (T_M): R = T_M
    #   iter 2 (T_{M-1}): R = T_{M-1} × T_M
    #   ...
    #   iter M (T_1): R = T_1 × T_2 × ... × T_M
    # Then U = R @ diag(exp(i*phase_screen))
    for mzi in reversed(mzis):
        T = _T(mzi.theta, mzi.phi)
        i, j = mzi.mode_i, mzi.mode_j
        rows_ij = R[[i, j], :].copy()
        R[[i, j], :] = T @ rows_ij

    # Post-multiply diagonal phase screen (U = [T_1...T_M] × D)
    R = R @ np.diag(np.exp(1j * phase_screen))

    return R


def clements_reconstruct(
    mzis: List[MZISpec],
    phase_screen: np.ndarray,
    N: int,
) -> np.ndarray:
    """Public API for reconstruction."""
    return _reconstruct(mzis, phase_screen, N)


def clements_simulate(
    input_field: np.ndarray,
    mzis: List[MZISpec],
    phase_screen: np.ndarray,
    N: int,
    include_loss: bool = False,
    loss_per_mzi_db: float = 0.01,
) -> np.ndarray:
    """
    Simulate the photonic forward pass: compute U × input_field.

    U = D × T_M × ... × T_1
    → output = D × (T_M × ... × (T_1 × input))

    Apply T_1 first (T_1 = mzis[0] in reversed list → mzis[-1] in forward),
    then T_2, ..., T_M, then D.

    In practice, we reverse the MZI list and apply from the left.

    Args:
        input_field: Complex input, length N
        mzis: MZI specs in decomposition order
        phase_screen: Diagonal phase screen
        N: Dimension
        include_loss: Apply per-MZI optical loss
        loss_per_mzi_db: MZI insertion loss [dB]

    Returns:
        Output field, length N
    """
    field = input_field.astype(complex).copy()
    if len(field) != N:
        raise ValueError(f"Input field length {len(field)} != N={N}")

    loss_amp = 10 ** (-loss_per_mzi_db / 20) if include_loss else 1.0

    # U = T_1 × T_2 × ... × T_M × D
    # U × x = T_1 × (T_2 × ... × (T_M × (D × x)))
    # Apply D first (innermost), then T_M, T_{M-1}, ..., T_1 (outermost).
    # Iterate reversed(mzis) = [T_M, ..., T_1] applying each from left.
    field *= np.exp(1j * phase_screen)  # apply D first

    for mzi in reversed(mzis):
        i, j = mzi.mode_i, mzi.mode_j
        T = _T(mzi.theta, mzi.phi)
        f_ij = field[[i, j]].copy()
        field[[i, j]] = T @ f_ij
        if include_loss:
            field[[i, j]] *= loss_amp

    return field


def clements_decompose_rectangular(
    A: np.ndarray,
    rank: int,
    eps: float = 1e-12,
) -> Tuple[ClementsResult, np.ndarray, ClementsResult]:
    """
    Full SVD + unitary decomposition of a rectangular matrix.

    A ≈ U_full × diag(σ_1,...,σ_r,0,...,0) × Vh_full
    → Decompose U_full (m×m) and Vh_full (n×n) separately.

    Args:
        A: Weight matrix (m × n)
        rank: Truncation rank
        eps: Numerical tolerance

    Returns:
        (U_decomp, sigma_r, Vh_decomp)
    """
    m, n = A.shape
    rank = min(rank, m, n)

    U_full, sigma, Vh_full = np.linalg.svd(A.astype(np.float64), full_matrices=True)

    print(f"  Decomposing U ({m}×{m})...")
    U_decomp = clements_decompose(U_full.astype(complex), eps=eps)

    print(f"  Decomposing Vh ({n}×{n})...")
    Vh_decomp = clements_decompose(Vh_full.astype(complex), eps=eps)

    return U_decomp, sigma[:rank].astype(np.float32), Vh_decomp


def decomposition_stats(result: ClementsResult) -> dict:
    """Statistics summary for a decomposition result."""
    thetas = np.array([m.theta for m in result.mzis]) if result.mzis else np.array([0.0])
    phis = np.array([m.phi for m in result.mzis]) if result.mzis else np.array([0.0])
    return {
        "N": result.N,
        "n_mzis_actual": result.n_mzis,
        "n_mzis_expected": result.N * (result.N - 1) // 2,
        "reconstruction_error": result.reconstruction_error,
        "theta_mean": float(np.mean(thetas)),
        "theta_std": float(np.std(thetas)),
        "phi_mean": float(np.mean(phis)),
        "phi_std": float(np.std(phis)),
        "max_theta": float(np.max(thetas)),
        "max_phi": float(np.max(phis)),
        "phase_screen_rms": float(np.sqrt(np.mean(result.phase_screen ** 2))),
    }
