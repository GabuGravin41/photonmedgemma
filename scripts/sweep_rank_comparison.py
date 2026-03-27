#!/usr/bin/env python3
"""
sweep_rank_comparison.py — Full Rank Sweep: PhotoMedGemma Photonic Chip
========================================================================

Compares PhotoMedGemma photonic chip simulation against real MedGemma 4B-IT
activations at every meaningful rank (64, 128, 256, 512, 1024, 2048) for all
four layer-0 attention projections (q, k, v, o).

For each rank r and each projection, reports:

  chip_fidelity    — MZI simulation vs compiled matrix (exact arithmetic).
                     Proves the Clements phases are applied correctly.
                     Source: compiled NPZ phases (our work) only.

  svd_error        — rank-r SVD approximation of W vs full W@x.
                     Uses full input x_in (all input dimensions).
                     Source: real MedGemma hidden states (Kaggle) + real W.

  energy_pct       — fraction of Frobenius norm energy retained by rank-r SVD.

  n_mzis_per_mesh  — MZIs required for one r×r Clements unitary mesh.
                     Two meshes per projection (Vh and U) plus sigma stage.

  chip_area_mm2    — estimated die area per mesh in silicon photonics.
                     Uses two technology estimates (see below).

Physical area model (per unitary mesh of r modes):
  - Standard Si photonics (e.g. AIM Photonics, IMEC):
      MZI stage length: 150 μm, mode pitch: 8 μm
      Area ≈ r * 150 μm  ×  r * 8 μm  =  r² * 1200 μm²
  - Compact Si photonics / InP:
      MZI stage length: 80 μm, mode pitch: 5 μm
      Area ≈ r * 80 μm  ×  r * 5 μm  =  r² * 400 μm²

These are lower bounds (exclude I/O, routing, driver ICs).

Usage
-----
    python3 scripts/sweep_rank_comparison.py \\
        --activations-dir output/simulations/kaggle_activations/ \\
        --phase-map       output/real/phase_map.json \\
        --output-dir      output/simulations/paper_results/

    # Custom ranks:
    python3 scripts/sweep_rank_comparison.py \\
        --activations-dir output/simulations/kaggle_activations/ \\
        --phase-map       output/real/phase_map.json \\
        --ranks 32 64 128 256 512 1024 2048
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.linalg import svd as scipy_svd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Physical constants for chip area estimate
_STANDARD_SI = {"stage_um": 150, "pitch_um": 8,  "label": "Standard Si-photonics"}
_COMPACT_SI   = {"stage_um":  80, "pitch_um": 5,  "label": "Compact Si/InP"}

PROJ_TAGS = {"q_proj": "W_q", "k_proj": "W_k", "v_proj": "W_v", "o_proj": "W_o"}


# ── Correct Clements forward pass ─────────────────────────────────────────────

def _apply_clements(x: np.ndarray, npz, rank: int) -> np.ndarray:
    """
    Apply rank×rank active subspace of the compiled Clements mesh to x.

    Uses correct T(θ,φ) = [[c, -conj(ep)·s], [ep·s, c]] and descending
    column order, matching clements.py::clements_simulate exactly.
    """
    field = x[:rank].astype(complex).copy()
    field *= np.exp(1j * npz["phase_screen"][:rank])

    mi_all = npz["mode_i"].astype(int)
    mj_all = npz["mode_j"].astype(int)
    mask   = (mi_all < rank) & (mj_all < rank)
    if not mask.any():
        return field

    cols_  = npz["col"].astype(int)[mask]
    mi_    = mi_all[mask]; mj_    = mj_all[mask]
    theta_ = npz["theta"].astype(float)[mask]
    phi_   = npz["phi"].astype(float)[mask]

    for idx in np.argsort(cols_, kind="stable")[::-1]:   # descending col
        mi, mj = int(mi_[idx]), int(mj_[idx])
        c  = math.cos(theta_[idx] / 2)
        s  = math.sin(theta_[idx] / 2)
        ep = math.cos(phi_[idx]) + 1j * math.sin(phi_[idx])
        a, b = field[mi], field[mj]
        field[mi] = c * a - ep.conjugate() * s * b
        field[mj] = ep * s * a + c * b
    return field


def _reconstruct_block(npz, rank: int) -> np.ndarray:
    """Reconstruct the rank×rank compiled unitary block from NPZ phases."""
    ps    = npz["phase_screen"][:rank]
    mi_a  = npz["mode_i"].astype(int); mj_a = npz["mode_j"].astype(int)
    mask  = (mi_a < rank) & (mj_a < rank)
    cols_ = npz["col"].astype(int)[mask]
    mi_   = mi_a[mask]; mj_   = mj_a[mask]
    th_   = npz["theta"].astype(float)[mask]
    ph_   = npz["phi"].astype(float)[mask]

    M = np.eye(rank, dtype=complex)
    for idx in np.argsort(cols_, kind="stable")[::-1]:   # descending col
        mi, mj = int(mi_[idx]), int(mj_[idx])
        c  = math.cos(th_[idx] / 2); s = math.sin(th_[idx] / 2)
        ep = math.cos(ph_[idx]) + 1j * math.sin(ph_[idx])
        T  = np.array([[c, -ep.conjugate() * s], [ep * s, c]], dtype=complex)
        rows = M[[mi, mj], :].copy(); M[[mi, mj], :] = T @ rows
    return M @ np.diag(np.exp(1j * ps))


# ── Chip area model ───────────────────────────────────────────────────────────

def chip_area_mm2(rank: int, tech: dict) -> float:
    """
    Lower-bound area estimate for a single r-mode Clements mesh.

    Model: r columns of MZIs, each column has r/2 MZIs stacked vertically.
      Width  = r × stage_length
      Height = r × mode_pitch
      Area   = r² × stage_length × mode_pitch
    """
    area_um2 = (rank * tech["stage_um"]) * (rank * tech["pitch_um"])
    return area_um2 * 1e-6   # μm² → mm²


# ── Per-projection rank sweep ─────────────────────────────────────────────────

def sweep_projection(
    proj_name: str,
    W: np.ndarray,
    vh_npz,
    u_npz,
    compilation_error_U: float,
    compilation_error_Vh: float,
    act_dir: Path,
    image_stems: List[str],
    ranks: List[int],
) -> dict:
    """
    Run rank sweep for one projection.  Returns structured results dict.
    """
    safe = proj_name.replace("_", "")
    W64  = W.astype(np.float64)
    m, n = W64.shape

    logger.info(f"\n{'='*60}")
    logger.info(f"Projection: {proj_name}  shape=({m},{n})")
    logger.info(f"  Compilation error: error_U={compilation_error_U:.2e}  error_Vh={compilation_error_Vh:.2e}")

    # Determine max valid rank for this projection
    N_vh  = int(vh_npz["mode_j"].max()) + 1   # full Vh mesh size
    N_u   = int(u_npz["mode_j"].max()) + 1    # full U mesh size
    max_r = min(m, n, N_vh, N_u)
    valid_ranks = [r for r in ranks if r <= max_r]
    if not valid_ranks:
        logger.warning(f"  No valid ranks for {proj_name} (max_r={max_r})")
        return {}

    logger.info(f"  Mesh sizes: N_Vh={N_vh}, N_U={N_u}, max_rank={max_r}")
    logger.info(f"  Ranks to test: {valid_ranks}")

    # Load activation files
    x_list, y_list = [], []
    for stem in image_stems:
        in_p  = act_dir / f"input_layer0_{safe}_{stem}.npy"
        out_p = act_dir / f"output_layer0_{safe}_{stem}.npy"
        if in_p.exists() and out_p.exists():
            x_list.append(np.load(str(in_p)).astype(np.float64))
            y_list.append(np.load(str(out_p)).astype(np.float64))

    if not x_list:
        logger.error(f"  No activation files found for {proj_name}")
        return {}

    n_samples = sum(x.shape[0] for x in x_list)
    logger.info(f"  Activation samples: {n_samples} token(s) across {len(x_list)} image(s)")

    # Full SVD once (economy SVD)
    logger.info(f"  Computing economy SVD of ({m},{n})...")
    U_sv, s, Vh_sv = scipy_svd(W64, full_matrices=False)   # shapes: (m,k), (k,), (k,n)
    full_rank_svd  = len(s)
    total_energy   = float((s ** 2).sum())
    logger.info(f"  SVD done: full_rank={full_rank_svd}  s_max={s[0]:.4f}  s_min={s[-1]:.6f}")

    # Build ground truth W@x for all samples once
    wx_samples = []
    for x_tok in x_list:
        for tok in range(x_tok.shape[0]):
            x_in = x_tok[tok][:n]
            wx_samples.append(W64 @ x_in)

    rank_results = []

    for r in valid_ranks:
        # ── SVD approximation error ──────────────────────────────────────────
        U_r  = U_sv[:, :r]
        s_r  = s[:r]
        Vh_r = Vh_sv[:r, :]
        energy_r = float((s_r ** 2).sum() / total_energy)

        svd_errs = []
        for idx, x_tok in enumerate(x_list):
            for tok in range(x_tok.shape[0]):
                x_in   = x_tok[tok][:n]
                y_svd  = (U_r * s_r) @ (Vh_r @ x_in)
                y_wx   = wx_samples[idx * x_tok.shape[0] + tok]
                n_wx   = np.linalg.norm(y_wx) + 1e-15
                svd_errs.append(float(np.linalg.norm(y_svd - y_wx) / n_wx))

        mean_svd_err = float(np.mean(svd_errs))

        # ── Chip fidelity at rank r ──────────────────────────────────────────
        # Both chip and reference use compiled NPZ phases filtered to r×r subspace.
        Vh_block = _reconstruct_block(vh_npz, r)
        U_block  = _reconstruct_block(u_npz,  r)

        chip_errs = []
        for idx, x_tok in enumerate(x_list):
            for tok in range(x_tok.shape[0]):
                x_in = x_tok[tok][:n]
                x_r  = x_in[:r].astype(complex)

                # Photonic simulation: Clements mesh (iterative, mode by mode)
                after_Vh    = _apply_clements(x_r, vh_npz, r)
                after_sigma = after_Vh * (s_r / (s_r[0] + 1e-15))
                y_phot_r    = _apply_clements(after_sigma, u_npz, r).real * s_r[0]

                # Digital reference: exact matrix multiply (same compiled matrices)
                y_ref_r = (U_block * s_r) @ (Vh_block @ x_r.real)

                n_ref = np.linalg.norm(y_ref_r) + 1e-15
                chip_errs.append(float(np.linalg.norm(y_phot_r - y_ref_r) / n_ref))

        mean_chip_err = float(np.mean(chip_errs))

        # ── MZI count & chip area ────────────────────────────────────────────
        n_mzis_mesh  = r * (r - 1) // 2       # one unitary mesh (Vh or U)
        n_mzis_total = 2 * n_mzis_mesh + r    # Vh + U + sigma stage (r VOAs)

        area_std     = chip_area_mm2(r, _STANDARD_SI)   # per mesh
        area_cmp     = chip_area_mm2(r, _COMPACT_SI)    # per mesh
        area_std_2x  = 2 * area_std   # Vh + U meshes
        area_cmp_2x  = 2 * area_cmp

        logger.info(
            f"  rank={r:5d}: svd_err={mean_svd_err:.4f}  chip={mean_chip_err:.2e}"
            f"  energy={energy_r:.4f}  MZIs/mesh={n_mzis_mesh:,}"
            f"  area_std={area_std:.2f}mm²/mesh"
        )

        rank_results.append({
            "rank":               r,
            "energy_retained":    energy_r,
            "mean_svd_error":     mean_svd_err,
            "mean_chip_fidelity": mean_chip_err,
            "n_mzis_per_mesh":    n_mzis_mesh,
            "n_mzis_total":       n_mzis_total,
            "area_mm2_per_mesh_standard_si": round(area_std,  3),
            "area_mm2_per_mesh_compact_si":  round(area_cmp,  3),
            "area_mm2_full_layer_standard":  round(area_std_2x, 3),
            "area_mm2_full_layer_compact":   round(area_cmp_2x, 3),
        })

    return {
        "proj":                 proj_name,
        "weight_shape":         [m, n],
        "full_rank_svd":        full_rank_svd,
        "compilation_error_U":  compilation_error_U,
        "compilation_error_Vh": compilation_error_Vh,
        "n_samples":            n_samples,
        "ranks":                rank_results,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(all_results: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    (out_dir / "rank_sweep.json").write_text(
        json.dumps({"results": all_results}, indent=2)
    )

    SEP = "=" * 100

    lines = [
        SEP,
        "PhotoMedGemma — Full Rank Sweep: Photonic Chip vs MedGemma 4B-IT Layer 0",
        SEP,
        "",
        "All four layer-0 attention projections (q, k, v, o) compared at every",
        "manufacturable rank using real MedGemma activations from 5 breast cancer PNGs.",
        "",
        "Columns:",
        "  rank          — number of optical modes (SVD truncation rank)",
        "  energy        — fraction of weight Frobenius norm energy retained",
        "  svd_error     — ||W_r@x - W@x|| / ||W@x||  (full input, real MedGemma hidden states)",
        "  chip_fidelity — ||MZI_sim - compiled_matrix@x|| / ||compiled_matrix@x||",
        "                  (floating-point arithmetic precision of Clements compilation)",
        "  MZIs/mesh     — MZIs for one unitary (Vh or U); layer needs 2× + sigma",
        "  area/mesh     — estimated die area per unitary mesh",
        "                  Standard Si: 150μm stage × 8μm pitch",
        "                  Compact Si:   80μm stage × 5μm pitch",
        "",
    ]

    for proj_result in all_results:
        if not proj_result or "ranks" not in proj_result:
            continue
        m, n = proj_result["weight_shape"]
        lines += [
            f"── {proj_result['proj']}  shape=({m},{n})  full_rank={proj_result['full_rank_svd']}",
            f"   Compilation fidelity: error_U={proj_result['compilation_error_U']:.2e}"
            f"  error_Vh={proj_result['compilation_error_Vh']:.2e}",
            f"   Activation samples: {proj_result['n_samples']} token(s)",
            "",
            f"   {'rank':>6}  {'energy':>7}  {'svd_error':>9}  {'chip_fidelity':>13}"
            f"  {'MZIs/mesh':>12}  {'area/mesh (std)':>15}  {'area/mesh (cmp)':>15}",
            "   " + "-" * 84,
        ]

        for r in proj_result["ranks"]:
            lines.append(
                f"   {r['rank']:>6}  {r['energy_retained']:>6.1%}  {r['mean_svd_error']:>9.4f}"
                f"  {r['mean_chip_fidelity']:>13.2e}"
                f"  {r['n_mzis_per_mesh']:>12,}"
                f"  {r['area_mm2_per_mesh_standard_si']:>12.1f} mm²"
                f"  {r['area_mm2_per_mesh_compact_si']:>12.1f} mm²"
            )
        lines.append("")

    lines += [
        SEP,
        "Hardware constraints summary (one unitary mesh, per projection):",
        "",
        f"  {'Rank':>6}  {'MZIs/mesh':>12}  {'Area/mesh (std Si)':>20}  {'Area/mesh (compact)':>22}",
        "  " + "-" * 68,
    ]
    # Use q_proj ranks as the reference (largest full_rank)
    q_result = next((r for r in all_results if r and r.get("proj") == "q_proj"), None)
    if q_result:
        for r in q_result["ranks"]:
            lines.append(
                f"  {r['rank']:>6}  {r['n_mzis_per_mesh']:>12,}"
                f"  {r['area_mm2_per_mesh_standard_si']:>17.1f} mm²"
                f"  {r['area_mm2_per_mesh_compact_si']:>19.1f} mm²"
            )

    lines += [
        "",
        "Notes:",
        "  - Area estimates are lower bounds (mesh only; excludes I/O, drivers, routing).",
        "  - Each projection requires 2 unitary meshes (Vh + U) plus a sigma (VOA) stage.",
        "  - Layer 0 has 4 projections (q, k, v, o); full layer = 8 meshes + 4 sigma stages.",
        "  - svd_error is measured vs W@x (full input), not vs MedGemma output y_mg.",
        "  - chip_fidelity measures Clements compilation accuracy, not model approximation.",
        SEP,
    ]

    txt_path = out_dir / "rank_sweep.txt"
    txt_path.write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    logger.info(f"Saved: {out_dir / 'rank_sweep.json'}")
    logger.info(f"Saved: {txt_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rank sweep: PhotoMedGemma vs MedGemma")
    parser.add_argument("--activations-dir", required=True)
    parser.add_argument("--phase-map",        default="output/real/phase_map.json")
    parser.add_argument("--output-dir",       default="output/simulations/paper_results")
    parser.add_argument(
        "--ranks", type=int, nargs="+",
        default=[64, 128, 256, 512, 1024, 2048],
        help="Ranks to sweep (default: 64 128 256 512 1024 2048)",
    )
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)

    # Discover image stems
    stems = sorted(
        f.stem.replace("input_layer0_qproj_", "")
        for f in act_dir.glob("input_layer0_qproj_*.npy")
    )
    if not stems:
        logger.error("No activation files found")
        sys.exit(1)
    logger.info(f"Images: {stems}")

    # Load phase map
    with open(args.phase_map) as f:
        pm = json.load(f)
    base = Path(args.phase_map).parent

    layers_by_proj = {}
    for layer in pm["layers"]:
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if proj in layer["name"]:
                layers_by_proj[proj] = {
                    "vh_npz":    np.load(str(base / layer["phase_file_Vh"]), allow_pickle=False),
                    "u_npz":     np.load(str(base / layer["phase_file_U"]),  allow_pickle=False),
                    "error_U":   layer.get("error_U", 0.0),
                    "error_Vh":  layer.get("error_Vh", 0.0),
                }

    all_results = []
    for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if proj_name not in layers_by_proj:
            logger.warning(f"No compiled layer for {proj_name}")
            continue

        tag    = PROJ_TAGS[proj_name]
        w_path = act_dir / f"{tag}_layer0.npy"
        if not w_path.exists():
            logger.warning(f"Weight file not found: {w_path}")
            continue

        W    = np.load(str(w_path))
        info = layers_by_proj[proj_name]

        result = sweep_projection(
            proj_name           = proj_name,
            W                   = W,
            vh_npz              = info["vh_npz"],
            u_npz               = info["u_npz"],
            compilation_error_U = info["error_U"],
            compilation_error_Vh= info["error_Vh"],
            act_dir             = act_dir,
            image_stems         = stems,
            ranks               = args.ranks,
        )
        all_results.append(result)

    write_report(all_results, Path(args.output_dir))


if __name__ == "__main__":
    main()
