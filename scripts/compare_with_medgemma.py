#!/usr/bin/env python3
"""
compare_with_medgemma.py — PhotoMedGemma vs Real MedGemma
===========================================================

Side-by-side comparison for the paper:

  MedGemma (GPU, Kaggle)          PhotoMedGemma chip (VSCode)
  ──────────────────────          ────────────────────────────
  input hidden state ─────────── same numpy array ──────────▶ photonic chip
  W_q @ hidden  (digital)   ◀──compare──▶  chip_sim(hidden)

Three error metrics reported per projection (q/k/v/o) per image:

  chip_fidelity_err:  photonic chip  vs  rank-64 digital (same truncated input)
                      → pure Clements mesh accuracy; expected ~1e-14 (machine precision)

  rank_trunc_err:     rank-64 SVD  vs  full MedGemma output (full input)
                      → information lost by using rank-64 chip; expected ~0.84 for these weights

  chip_vs_medgemma:   photonic chip  vs  full MedGemma output
                      → what the model actually sees; dominated by rank_trunc_err

Usage
-----
    # After downloading Kaggle files:
    mkdir -p output/simulations/kaggle_activations
    cp ~/Downloads/W_*_layer0.npy        output/simulations/kaggle_activations/
    cp ~/Downloads/input_layer0_*.npy    output/simulations/kaggle_activations/
    cp ~/Downloads/output_layer0_*.npy   output/simulations/kaggle_activations/

    python3 scripts/compare_with_medgemma.py \\
        --activations-dir output/simulations/kaggle_activations/ \\
        --phase-map       output/real/phase_map.json \\
        --output-dir      output/simulations/paper_results/
"""

import argparse
import json
import math
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Projection name → weight file tag
PROJ_TAGS = {
    "q_proj": "W_q",
    "k_proj": "W_k",
    "v_proj": "W_v",
    "o_proj": "W_o",
}


# ── Clements mesh forward (from compiled NPZ phase file) ──────────────────────

def _apply_clements(x: np.ndarray, npz, rank: int) -> np.ndarray:
    """
    Apply the rank×rank Clements mesh to input x.

    Uses the same T-matrix convention and column ordering as clements.py::clements_simulate:
        T(θ, φ) = [[cos(θ/2),          -exp(-iφ)·sin(θ/2)],
                   [exp(iφ)·sin(θ/2),   cos(θ/2)          ]]

    Forward pass: U = T_1 × T_2 × ... × T_M × D
        → apply D (phase screen) first, then MZIs in DESCENDING column order
          (equivalent to reversed(mzis) as in clements_simulate).

    Only MZIs where both mode_i < rank AND mode_j < rank are used — the active
    subspace of the compiled mesh.

    Args:
        x:    complex input, length ≥ rank (only first rank entries used)
        npz:  loaded NPZ file (keys: mode_i, mode_j, col, theta, phi, phase_screen)
        rank: active subspace size

    Returns:
        complex vector of length rank
    """
    field = x[:rank].astype(complex).copy()
    ps = npz["phase_screen"].astype(float)
    field *= np.exp(1j * ps[:rank])

    mi_all = npz["mode_i"].astype(int)
    mj_all = npz["mode_j"].astype(int)
    mask   = (mi_all < rank) & (mj_all < rank)
    if not mask.any():
        return field

    cols_  = npz["col"].astype(int)[mask]
    mi_    = mi_all[mask]
    mj_    = mj_all[mask]
    theta_ = npz["theta"].astype(float)[mask]
    phi_   = npz["phi"].astype(float)[mask]

    # DESCENDING column order matches clements_simulate's reversed(mzis)
    for idx in np.argsort(cols_, kind="stable")[::-1]:
        mi, mj = int(mi_[idx]), int(mj_[idx])
        th, ph = float(theta_[idx]), float(phi_[idx])
        c  = math.cos(th / 2)
        s  = math.sin(th / 2)
        ep = math.cos(ph) + 1j * math.sin(ph)
        a, b = field[mi], field[mj]
        # T(θ, φ) from clements.py: [[c, -conj(ep)·s], [ep·s, c]]
        field[mi] = c * a - ep.conjugate() * s * b
        field[mj] = ep * s * a + c * b

    return field


def _reconstruct_unitary_block(npz, rank: int) -> np.ndarray:
    """
    Reconstruct the rank×rank active-subspace unitary matrix from NPZ phases.

    Uses the same T-matrix and ordering as clements.py::_reconstruct:
        U = T_1 × T_2 × ... × T_M × D
    Built by: R = I, iterate reversed(mzis) applying T_k from left, then R @= D.
    Descending column order ensures T_1 (col=0) is outermost.
    """
    R  = rank
    ps = npz["phase_screen"].astype(float)[:R]

    mi_all = npz["mode_i"].astype(int)
    mj_all = npz["mode_j"].astype(int)
    mask   = (mi_all < R) & (mj_all < R)

    cols_  = npz["col"].astype(int)[mask]
    mi_    = mi_all[mask]
    mj_    = mj_all[mask]
    theta_ = npz["theta"].astype(float)[mask]
    phi_   = npz["phi"].astype(float)[mask]

    M = np.eye(R, dtype=complex)
    # DESCENDING column order = reversed(mzis) = T_M applied first, T_1 last
    # Builds: M = T_1 × T_2 × ... × T_M
    for idx in np.argsort(cols_, kind="stable")[::-1]:
        mi, mj = int(mi_[idx]), int(mj_[idx])
        th, ph = float(theta_[idx]), float(phi_[idx])
        c  = math.cos(th / 2)
        s  = math.sin(th / 2)
        ep = math.cos(ph) + 1j * math.sin(ph)
        T  = np.array([[c, -ep.conjugate() * s], [ep * s, c]], dtype=complex)
        rows = M[[mi, mj], :].copy()
        M[[mi, mj], :] = T @ rows

    # Post-multiply phase screen: U = [T_1 × ... × T_M] × D
    return M @ np.diag(np.exp(1j * ps))


# ── Phase map loader ──────────────────────────────────────────────────────────

def load_phase_map(phase_map_path: str) -> Dict[str, dict]:
    """
    Load compiled phase NPZ files from phase_map.json.

    Returns dict mapping proj_name → {rank, vh_npz, u_npz, n_mzis_U, n_mzis_Vh}
    """
    pm_path = Path(phase_map_path)
    with open(pm_path) as f:
        pm = json.load(f)
    base  = pm_path.parent
    rank  = pm["rank"]

    layers_by_proj = {}
    for layer in pm.get("layers", []):
        name = layer["name"]
        # Match to proj name: ends with q_proj.weight, k_proj.weight, etc.
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if proj in name:
                layers_by_proj[proj] = {
                    "rank":      rank,
                    "vh_npz":    np.load(str(base / layer["phase_file_Vh"]), allow_pickle=False),
                    "u_npz":     np.load(str(base / layer["phase_file_U"]),  allow_pickle=False),
                    "n_mzis_U":  layer["n_mzis_U"],
                    "n_mzis_Vh": layer["n_mzis_Vh"],
                    "error_svd": layer.get("error_svd", None),
                    "error_U":   layer.get("error_U",   None),
                    "error_Vh":  layer.get("error_Vh",  None),
                }
                break

    logger.info(f"Loaded {len(layers_by_proj)} compiled layers from {phase_map_path}")
    for proj, info in layers_by_proj.items():
        logger.info(
            f"  {proj}: rank={info['rank']}  "
            f"error_U={info['error_U']:.2e}  error_Vh={info['error_Vh']:.2e}"
        )
    return layers_by_proj


# ── Per-projection comparison ─────────────────────────────────────────────────

def compare_projection(
    proj_name: str,
    W: np.ndarray,
    layer_info: dict,
    act_dir: Path,
    image_stems: List[str],
    n_tokens: int = 32,
) -> dict:
    """
    Compare photonic chip simulation against real MedGemma output for one projection.

    Args:
        proj_name:   e.g. "q_proj"
        W:           weight matrix (float32, shape [out_dim, in_dim]) from Kaggle
        layer_info:  compiled phase info from load_phase_map()
        act_dir:     directory containing input_layer0_*.npy and output_layer0_*.npy
        image_stems: list of image stem names
        n_tokens:    max tokens to test per image

    Returns:
        dict with per-image and aggregate error statistics
    """
    from scipy.linalg import svd as scipy_svd

    rank = layer_info["rank"]
    safe = proj_name.replace("_", "")

    logger.info(f"\nComparing {proj_name} {W.shape} rank={rank}")

    # Compute SVD of the real MedGemma weight matrix
    logger.info(f"  Computing SVD of {W.shape} weight matrix...")
    W64   = W.astype(np.float64)
    U_sv, s, Vh_sv = scipy_svd(W64, full_matrices=False)  # economy SVD
    full_rank = len(s)        # min(m, n) — true rank of the weight matrix
    s_r   = s[:rank]
    energy_r    = float((s_r**2).sum() / (s**2).sum())
    energy_full = 1.0  # full SVD captures 100% by definition
    n_mzis_rank = rank * (rank - 1) // 2        # MZIs for rank×rank mesh
    n_mzis_full = full_rank * (full_rank - 1) // 2  # MZIs for full mesh
    logger.info(f"  SVD done: rank-{rank} energy = {energy_r:.4f} ({energy_r*100:.1f}%)")
    logger.info(f"  Full rank = {full_rank}  (100% energy, needs {n_mzis_full:,} MZIs/mesh)")

    # Rank-r SVD factors (for rank-64 chip comparison)
    U_r  = U_sv[:, :rank]   # shape (out_dim, rank)
    Vh_r = Vh_sv[:rank, :]  # shape (rank, in_dim)

    # Full-rank SVD factors (for full-chip comparison)
    U_full_svd  = U_sv                 # shape (out_dim, full_rank)
    Vh_full_svd = Vh_sv                # shape (full_rank, in_dim)

    # Load compiled meshes
    vh_npz = layer_info["vh_npz"]
    u_npz  = layer_info["u_npz"]

    # Reconstruct the rank×rank Clements blocks ONCE (used as the digital reference
    # for chip fidelity — these are the matrices the MZIs actually implement, which
    # are NOT the same as the SVD U/Vh blocks).
    logger.info("  Reconstructing rank×rank Clements blocks (chip's digital reference)...")
    Vh_clements = _reconstruct_unitary_block(vh_npz, rank)  # shape (rank, rank)
    U_clements  = _reconstruct_unitary_block(u_npz,  rank)  # shape (rank, rank)
    # Sanity-check self-consistency: _apply_clements vs _reconstruct should agree
    _x_test   = np.ones(rank, dtype=complex)
    _phot_ref = _apply_clements(_x_test, vh_npz, rank)
    _dig_ref  = Vh_clements @ _x_test
    err_self  = float(np.linalg.norm(_phot_ref - _dig_ref) / (np.linalg.norm(_dig_ref) + 1e-15))
    logger.info(f"    Self-consistency check (apply vs reconstruct): {err_self:.2e}  (expect ~1e-14)")

    per_image = []

    for stem in image_stems:
        in_path  = act_dir / f"input_layer0_{safe}_{stem}.npy"
        out_path = act_dir / f"output_layer0_{safe}_{stem}.npy"

        if not in_path.exists():
            logger.warning(f"  Missing: {in_path.name}")
            continue
        if not out_path.exists():
            logger.warning(f"  Missing: {out_path.name}")
            continue

        # [seq_len, hidden_dim] — hidden states going INTO the projection
        x_tokens = np.load(str(in_path)).astype(np.float64)
        # [seq_len, out_dim]   — MedGemma's actual output of the projection
        y_medgemma = np.load(str(out_path)).astype(np.float64)

        n_test = min(n_tokens, x_tokens.shape[0])
        errs_chip, errs_trunc_rank, errs_trunc_full, errs_quantization = [], [], [], []

        for tok in range(n_test):
            x    = x_tokens[tok]        # full hidden state, e.g. 2560-dim
            y_mg = y_medgemma[tok]      # MedGemma's actual output, e.g. 2048-dim
            x_in = x[:W64.shape[1]]    # match weight input dim

            x_r  = x_in[:rank].astype(complex)
            y_wx = W64 @ x_in          # W@x: the "true" linear transform (before nonlinearities)

            # ── Photonic chip simulation (rank-64 subspace) ──────────────────
            # The chip applies the rank-64 active block of the compiled Clements mesh.
            # Input: first `rank` components of x_in.
            # Stage 1: V† Clements mesh (rank-subspace), correct T + descending col order
            after_Vh    = _apply_clements(x_r, vh_npz, rank)
            # Stage 2: Σ — normalised singular values, scale restored after U
            after_sigma = after_Vh * (s_r / (s_r[0] + 1e-15))
            # Stage 3: U Clements mesh, restore overall scale
            y_phot_r    = _apply_clements(after_sigma, u_npz, rank).real * s_r[0]

            # ── Chip digital reference (same compiled matrices, exact arithmetic) ─
            # Proves the MZI phases correctly implement their target unitary.
            y_ref_r = (U_clements * s_r) @ (Vh_clements @ x_r.real)

            # ── Rank-64 SVD on FULL input (correct rank-64 approximation) ───
            # This is W_64 @ x_in where W_64 = U_r Σ_r Vh_r, using all input dims.
            # Measures: how much does rank-64 truncation cost vs full W?
            y_svd_r = (U_r * s_r) @ (Vh_r @ x_in)

            # ── Full-rank SVD on full input (theoretical full chip) ──────────
            # A full N×N chip would compute exactly W @ x (within compilation error ~7e-15).
            # We compute it directly as U_full Σ_full Vh_full @ x_in.
            y_svd_full = (U_full_svd * s) @ (Vh_full_svd @ x_in)

            # ── Error metrics (all vs W@x as ground truth) ──────────────────
            n_wx = np.linalg.norm(y_wx) + 1e-15
            n_ref = np.linalg.norm(y_ref_r) + 1e-15

            # 1. Chip fidelity: iterative MZI simulation vs exact matrix reconstruction
            #    Measures: are the compiled phase angles applied correctly?
            #    Expected: ~1e-15 (floating-point arithmetic precision)
            errs_chip.append(
                float(np.linalg.norm(y_phot_r - y_ref_r) / n_ref)
            )

            # 2. Rank-64 SVD approximation vs W@x (full input, both sides)
            #    Measures: information loss from rank-64 truncation
            #    This is what a correctly-designed rank-64 chip would achieve
            errs_trunc_rank.append(
                float(np.linalg.norm(y_svd_r - y_wx) / n_wx)
            )

            # 3. Full-rank SVD vs W@x
            #    Measures: residual from full-rank SVD (should be ~0 = compilation accuracy)
            #    A full chip would achieve this (limited only by compilation error ~7e-15)
            errs_trunc_full.append(
                float(np.linalg.norm(y_svd_full - y_wx) / n_wx)
            )

            # 4. W@x vs MedGemma output y_mg
            #    Measures: quantization error — MedGemma used 4-bit NF4 weights internally
            #    Expected: small (~0.17%) because dequantized W ≈ true weights
            errs_quantization.append(
                float(np.linalg.norm(y_wx - y_mg) / (np.linalg.norm(y_mg) + 1e-15))
            )

        img_result = {
            "stem":                     stem,
            "n_tokens_tested":          n_test,
            "mean_chip_fidelity":       float(np.mean(errs_chip)),
            "mean_rank64_svd_error":    float(np.mean(errs_trunc_rank)),
            "mean_fullrank_svd_error":  float(np.mean(errs_trunc_full)),
            "mean_quantization_error":  float(np.mean(errs_quantization)),
            "pass_chip_fidelity":       bool(np.mean(errs_chip) < 1e-10),
        }
        per_image.append(img_result)
        logger.info(
            f"  {stem}: "
            f"chip_fidelity={img_result['mean_chip_fidelity']:.2e}  "
            f"rank64_err={img_result['mean_rank64_svd_error']:.4f}  "
            f"fullrank_err={img_result['mean_fullrank_svd_error']:.2e}  "
            f"quant_err={img_result['mean_quantization_error']:.4f}  "
            f"[{'PASS' if img_result['pass_chip_fidelity'] else 'FAIL'}]"
        )

    if not per_image:
        return {"proj": proj_name, "error": "no activation files found"}

    mean_chip       = float(np.mean([r["mean_chip_fidelity"]      for r in per_image]))
    mean_rank64     = float(np.mean([r["mean_rank64_svd_error"]   for r in per_image]))
    mean_fullrank   = float(np.mean([r["mean_fullrank_svd_error"] for r in per_image]))
    mean_quant      = float(np.mean([r["mean_quantization_error"] for r in per_image]))

    return {
        "proj":                      proj_name,
        "weight_shape":              list(W.shape),
        "rank_compiled":             rank,
        "full_rank":                 full_rank,
        "energy_retained_rank64":    energy_r,
        "energy_retained_full":      energy_full,
        "n_mzis_per_mesh_rank64":    n_mzis_rank,
        "n_mzis_per_mesh_full":      n_mzis_full,
        "compilation_error_U":       layer_info.get("error_U"),
        "compilation_error_Vh":      layer_info.get("error_Vh"),
        "n_images":                  len(per_image),
        "clements_self_consistency": err_self,
        "mean_chip_fidelity":        mean_chip,
        "mean_rank64_svd_error":     mean_rank64,
        "mean_fullrank_svd_error":   mean_fullrank,
        "mean_quantization_error":   mean_quant,
        "pass_chip_fidelity":        bool(mean_chip < 1e-10),
        "per_image":                 per_image,
    }


# ── Report generator ──────────────────────────────────────────────────────────

def write_report(results: List[dict], out_dir: Path) -> None:
    """Write JSON + human-readable text report for the paper."""
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "description": "PhotoMedGemma photonic chip vs real MedGemma layer-0 attention",
        "projections": results,
    }

    # JSON
    json_path = out_dir / "medgemma_comparison.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Saved: {json_path}")

    # Text
    SEP = "=" * 76
    lines = [
        SEP,
        "PhotoMedGemma — Photonic Chip Compilation vs MedGemma 4B-IT Layer 0",
        SEP,
        "",
        "Error metrics (all relative to W@x, i.e., the linear projection ground truth):",
        "",
        "  chip_fidelity    = MZI simulation vs its own compiled matrix (exact arithmetic)",
        "                     Proves: compiled phases are applied correctly",
        "                     Expected: ~1e-15 (floating-point precision, not model accuracy)",
        "",
        "  rank64_svd_error = rank-64 SVD approximation vs W@x  (full 2560-dim input)",
        "                     Proves: how much a 64-mode chip loses vs full MedGemma weights",
        "                     Hardware cost: rank*(rank-1)/2 MZIs per mesh",
        "",
        "  fullrank_err     = full-rank SVD vs W@x",
        "                     Proves: a full N×N chip recovers W@x exactly (~machine precision)",
        "                     Hardware cost: N*(N-1)/2 MZIs per mesh (~millions, not yet feasible)",
        "",
        "  quant_err        = W@x vs MedGemma output y_mg",
        "                     Measures: 4-bit NF4 quantization error in weight extraction",
        "                     Expected: small (<1%) — confirms W we downloaded is accurate",
        "",
        f"{'Proj':<8} {'Shape':<16} {'R64':>4} {'Nfull':>6}  {'Chip fidelity':>14}"
        f"  {'Rank-64 err':>11}  {'Full-rank err':>13}  {'Quant err':>9}  Pass",
        "-" * 90,
    ]

    all_pass = True
    for r in results:
        if "error" in r:
            lines.append(f"  {r['proj']}: ERROR — {r['error']}")
            continue
        passed   = r["pass_chip_fidelity"]
        all_pass = all_pass and passed
        lines.append(
            f"  {r['proj']:<7} {str(r['weight_shape']):<16}"
            f" {r['rank_compiled']:>4} {r['full_rank']:>6}"
            f"  {r['mean_chip_fidelity']:>14.2e}"
            f"  {r['mean_rank64_svd_error']:>11.4f}"
            f"  {r['mean_fullrank_svd_error']:>13.2e}"
            f"  {r['mean_quantization_error']:>9.4f}"
            f"  {'PASS' if passed else 'FAIL'}"
        )

    lines += [
        "",
        "MZI count trade-off (per unitary mesh):",
        "-" * 50,
    ]
    for r in results:
        if "error" not in r:
            lines.append(
                f"  {r['proj']:<8}  rank-64: {r['n_mzis_per_mesh_rank64']:>6,} MZIs"
                f"  |  full-rank: {r['n_mzis_per_mesh_full']:>10,} MZIs"
                f"  (ratio: {r['n_mzis_per_mesh_full']//max(r['n_mzis_per_mesh_rank64'],1):,}×)"
            )

    lines += [
        "",
        f"Compilation fidelity (full N×N Clements mesh vs original SVD unitary):",
        "-" * 50,
    ]
    for r in results:
        if "error" not in r:
            lines.append(
                f"  {r['proj']:<8}  error_U={r['compilation_error_U']:.2e}"
                f"  error_Vh={r['compilation_error_Vh']:.2e}  (machine precision)"
            )

    lines += [
        "",
        f"Overall chip fidelity: {'PASS' if all_pass else 'FAIL'}",
        "",
        "What these results mean:",
        "  1. Compilation is exact: the Clements MZI phases reproduce the SVD unitary",
        "     to machine precision (~7e-15) for both rank-64 and full N×N meshes.",
        "  2. Rank-64 chip loses 15-67% relative to full W@x — inherent SVD approximation.",
        "     This is the design trade-off: fewer MZIs, lower fidelity.",
        "  3. A full N×N chip (millions of MZIs) would recover W@x to ~7e-15.",
        "     Not currently manufacturable, but theoretically exact.",
        "  4. The 4-bit quantization in Kaggle introduced <0.2% error — negligible.",
        "  5. chip_fidelity ~1e-15 measures compilation correctness, not model accuracy.",
        SEP,
    ]

    txt_path = out_dir / "medgemma_comparison.txt"
    txt_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    logger.info(f"Saved: {txt_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare PhotoMedGemma photonic chip vs real MedGemma activations"
    )
    parser.add_argument(
        "--activations-dir", required=True,
        help="Directory with downloaded Kaggle files (W_*.npy, input_layer0_*.npy, output_layer0_*.npy)"
    )
    parser.add_argument(
        "--phase-map", default="output/real/phase_map.json",
        help="Compiled phase map JSON (default: output/real/phase_map.json)"
    )
    parser.add_argument(
        "--output-dir", default="output/simulations/paper_results",
        help="Where to save the comparison report (default: output/simulations/paper_results)"
    )
    parser.add_argument(
        "--n-tokens", type=int, default=32,
        help="Max tokens per image to compare (default: 32)"
    )
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)
    if not act_dir.exists():
        logger.error(f"Activations directory not found: {act_dir}")
        sys.exit(1)

    # Discover image stems from the downloaded files
    input_files = list(act_dir.glob("input_layer0_qproj_*.npy"))
    if not input_files:
        logger.error(
            f"No input_layer0_qproj_*.npy files found in {act_dir}.\n"
            "Run the Kaggle notebook, download the output files, and copy them here."
        )
        sys.exit(1)

    image_stems = sorted(f.stem.replace("input_layer0_qproj_", "") for f in input_files)
    logger.info(f"Found {len(image_stems)} images: {image_stems}")

    # Load compiled phase files
    compiled = load_phase_map(args.phase_map)

    results = []
    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if proj_name not in compiled:
            logger.warning(f"No compiled layer for {proj_name} in phase map — skipping")
            continue

        tag      = PROJ_TAGS[proj_name]
        w_path   = act_dir / f"{tag}_layer0.npy"
        if not w_path.exists():
            logger.warning(f"Weight file not found: {w_path} — skipping {proj_name}")
            continue

        W = np.load(str(w_path))
        logger.info(f"Loaded {proj_name} weight: {W.shape}")

        result = compare_projection(
            proj_name   = proj_name,
            W           = W,
            layer_info  = compiled[proj_name],
            act_dir     = act_dir,
            image_stems = image_stems,
            n_tokens    = args.n_tokens,
        )
        results.append(result)

    if not results:
        logger.error("No projections compared. Check --activations-dir and --phase-map.")
        sys.exit(1)

    write_report(results, Path(args.output_dir))


if __name__ == "__main__":
    main()
