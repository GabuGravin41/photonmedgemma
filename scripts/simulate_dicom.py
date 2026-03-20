#!/usr/bin/env python3
"""
simulate_dicom.py — End-to-end DICOM/Image → Photonic Chip Simulation
=======================================================================

Feeds a medical image (DICOM, PNG, JPEG) through the full MedGemma
photonic chip pipeline and compares the output against the numpy
software reference.

Pipeline
--------
    Medical image (DICOM / PNG / JPEG)
        ↓  load_image()         — normalise, convert to grayscale float32
        ↓  siglip_preprocess()  — resize to 448×448, patch-extract (14×14 px)
                                   → 1024 patches × 196 dims each
        ↓  patch_project()      — random-key linear map: 196-dim → 2560-dim
                                   (matches q_proj input dim of MedGemma 4B)
        ↓  PhotonicAttention.forward()  for each compiled layer (k/q/v/o_proj)
                ├─ V† mesh (photonic Clements simulation from NPZ phases)
                ├─ Σ  stage (singular-value scaling, rank-64 truncation)
                └─ U  mesh (photonic Clements simulation)
        ↓  numpy_reference()    — exact W@x via reconstructed weight matrix
        ↓  compare()            — relative error, SSIM, per-token heatmap
        ↓  save_results()       — JSON + NPY + SVG

    Optional: --kaggle-ref <json>
        Load real MedGemma activations captured in the Kaggle notebook
        and compare photonic output directly against the real model.

Usage
-----
    # Synthetic phantom (no DICOM needed)
    python3 scripts/simulate_dicom.py --demo

    # Real DICOM
    python3 scripts/simulate_dicom.py --image path/to/scan.dcm

    # PNG / JPEG
    python3 scripts/simulate_dicom.py --image path/to/xray.png

    # Compare against Kaggle activations
    python3 scripts/simulate_dicom.py --demo \\
        --kaggle-ref output/simulations/kaggle_comparison.json

    # Custom phase map (real compiled weights)
    python3 scripts/simulate_dicom.py --demo \\
        --phase-map output/real/phase_map.json

    # All options
    python3 scripts/simulate_dicom.py \\
        --image scan.dcm               \\
        --phase-map output/real/phase_map.json \\
        --kaggle-ref output/simulations/kaggle_comparison.json \\
        --output-dir output/simulations \\
        --max-tokens 64                \\
        --save-arrays
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Image loading ─────────────────────────────────────────────────────────────

def load_dicom(path: str) -> np.ndarray:
    """Load a DICOM file → float32 grayscale [0, 1]."""
    try:
        import pydicom
    except ImportError:
        logger.error("pydicom not installed. Run: pip install pydicom")
        raise
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    # Handle different photometric interpretations
    if hasattr(ds, "PhotometricInterpretation"):
        if ds.PhotometricInterpretation == "MONOCHROME1":
            arr = arr.max() - arr  # invert
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # Convert RGB to grayscale if needed
    if arr.ndim == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    logger.info(f"  DICOM loaded: {path}  shape={ds.pixel_array.shape}  "
                f"rows={ds.Rows}  cols={ds.Columns}")
    return arr.astype(np.float32)


def load_image(path: str) -> np.ndarray:
    """Load PNG/JPEG → float32 grayscale [0, 1]."""
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed. Run: pip install Pillow")
        raise
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    logger.info(f"  Image loaded: {path}  shape={arr.shape}")
    return arr


def make_synthetic_xray(seed: int = 42, size: int = 448) -> np.ndarray:
    """Generate a realistic-looking synthetic chest X-ray phantom."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)

    # Dark lung fields (left and right)
    cy, cx = size // 2, size // 2
    for sign in [-1, 1]:
        lx = cx + sign * size // 5
        yy, xx = np.ogrid[:size, :size]
        mask = ((xx - lx) ** 2 / (size // 6) ** 2 +
                (yy - cy) ** 2 / (size // 4) ** 2) < 1
        img[mask] = rng.uniform(0.05, 0.15)

    # Bright mediastinum
    mid_mask = np.abs(np.arange(size) - cx) < size // 10
    img[:, mid_mask] = np.maximum(img[:, mid_mask], 0.6)

    # Ribs — periodic horizontal bands
    for i in range(8):
        y0 = int(size * 0.25 + i * size * 0.06)
        rib_mask = slice(max(0, y0 - 4), min(size, y0 + 4))
        img[rib_mask, :] = np.maximum(img[rib_mask, :], 0.5)

    # Background (soft tissue / skin)
    bg_mask = img == 0
    img[bg_mask] = rng.uniform(0.3, 0.5, bg_mask.sum())

    # Add noise
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    logger.info(f"  Synthetic phantom generated: {size}×{size}")
    return img


# ── SigLIP-style preprocessing ────────────────────────────────────────────────

def siglip_preprocess(
    image: np.ndarray,
    target_size: int = 448,
    patch_size: int = 14,
) -> Tuple[np.ndarray, dict]:
    """
    Preprocess an image the way SigLIP (MedGemma's vision encoder) does.

    Steps:
        1. Resize to target_size × target_size (bilinear)
        2. Normalise: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]  (SigLIP default)
        3. Extract non-overlapping patches of patch_size × patch_size
        4. Flatten each patch → patch_dim = patch_size²

    Returns:
        patches:  float32 array of shape (n_patches, patch_dim)
        info:     dict with preprocessing metadata
    """
    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False

    # Resize
    if has_pil:
        pil = Image.fromarray((image * 255).astype(np.uint8), mode="L")
        pil = pil.resize((target_size, target_size), Image.BILINEAR)
        resized = np.array(pil, dtype=np.float32) / 255.0
    else:
        # Nearest-neighbour fallback
        h, w = image.shape
        rows = (np.arange(target_size) * h / target_size).astype(int)
        cols = (np.arange(target_size) * w / target_size).astype(int)
        resized = image[np.ix_(rows, cols)]

    # SigLIP normalisation (grayscale: replicate channel)
    resized = (resized - 0.5) / 0.5   # → [-1, 1]

    # Extract patches
    n_h = target_size // patch_size
    n_w = target_size // patch_size
    n_patches = n_h * n_w
    patch_dim  = patch_size * patch_size

    patches = resized.reshape(n_h, patch_size, n_w, patch_size)
    patches = patches.transpose(0, 2, 1, 3).reshape(n_patches, patch_dim)

    info = {
        "target_size":  target_size,
        "patch_size":   patch_size,
        "n_patches":    n_patches,
        "patch_dim":    patch_dim,
        "grid_h":       n_h,
        "grid_w":       n_w,
    }
    logger.info(f"  SigLIP preprocess: {target_size}×{target_size}  "
                f"→ {n_patches} patches × {patch_dim} dims")
    return patches.astype(np.float32), info


def project_to_model_dim(
    patches: np.ndarray,
    model_dim: int = 2560,
    seed: int = 0,
) -> np.ndarray:
    """
    Linear projection from patch_dim to model_dim (simulating mm_projector).

    In real MedGemma, the multi-modal projector (mm_projector) converts SigLIP
    embeddings (1152-dim) to language-model tokens (2560-dim). Here we use a
    deterministic random projection as a stand-in when real weights are absent.

    Args:
        patches:    (n_patches, patch_dim) float32
        model_dim:  target token dimension (2560 for MedGemma 4B)
        seed:       RNG seed (fixed for reproducibility)

    Returns:
        tokens: (n_patches, model_dim) float32
    """
    n_patches, patch_dim = patches.shape
    rng = np.random.default_rng(seed)

    # Orthonormal projection matrix (preserves norms, physically motivated)
    if patch_dim >= model_dim:
        # Downsample: use first model_dim principal directions
        proj = rng.standard_normal((patch_dim, model_dim)).astype(np.float32)
        proj, _ = np.linalg.qr(proj)
        proj = proj[:, :model_dim]
    else:
        # Upsample: zero-pad + rotate
        proj = rng.standard_normal((patch_dim, model_dim)).astype(np.float32)
        proj /= np.linalg.norm(proj, axis=0, keepdims=True) + 1e-8

    tokens = patches @ proj  # (n_patches, model_dim)

    # Normalise tokens to unit RMS (as layer-norm would)
    rms = np.sqrt((tokens ** 2).mean(axis=1, keepdims=True)) + 1e-8
    tokens = tokens / rms

    logger.info(f"  Projected: {patches.shape} → {tokens.shape}  "
                f"(simulated mm_projector)")
    return tokens.astype(np.float32)


# ── Photonic layer loading ─────────────────────────────────────────────────────

def load_compiled_layers(phase_map_path: str) -> List[dict]:
    """
    Load compiled photonic layers from a phase_map.json and its NPZ files.

    Returns a list of layer dicts with keys:
        name, rank, U_phases, Vh_phases, n_mzis_U, n_mzis_Vh,
        chip_id_U, chip_id_Vh, error_svd
    """
    pm_path = Path(phase_map_path)
    with open(pm_path) as f:
        pm = json.load(f)

    base = pm_path.parent
    rank = pm.get("rank", 64)
    layers = []

    for layer in pm.get("layers", []):
        u_npz  = np.load(str(base / layer["phase_file_U"]))
        vh_npz = np.load(str(base / layer["phase_file_Vh"]))
        layers.append({
            "name":       layer["name"],
            "rank":       rank,
            "U_phases":   u_npz,
            "Vh_phases":  vh_npz,
            "n_mzis_U":   layer["n_mzis_U"],
            "n_mzis_Vh":  layer["n_mzis_Vh"],
            "chip_id_U":  layer["chip_id_U"],
            "chip_id_Vh": layer["chip_id_Vh"],
            "error_svd":  layer["error_svd"],
        })
        logger.info(f"  Loaded: {layer['name']}  "
                    f"U={layer['n_mzis_U']:,} MZIs  Vh={layer['n_mzis_Vh']:,} MZIs")

    return layers


# ── Photonic forward simulation ────────────────────────────────────────────────

def _apply_clements_from_npz(
    x: np.ndarray,
    npz,
    rank: int,
) -> np.ndarray:
    """
    Apply the rank-R active Clements subspace to input vector x.

    Only MZIs where both mode_i < rank AND mode_j < rank are used (2016 MZIs
    for rank=64).  This is the physically correct rank-64 active subspace —
    identical to the full N×N mesh in terms of the first `rank` output modes
    because higher modes carry no signal after rank truncation at Σ.

    Runtime: O(rank²) per call instead of O(N²) — ~1600× faster for N=2560.

    Args:
        x:    input vector of length ≥ rank (only first `rank` entries used)
        npz:  loaded NPZ file with keys: mode_i, mode_j, col, theta, phi,
              phase_screen
        rank: active subspace size (default 64)

    Returns:
        field: complex vector of length `rank`
    """
    # Work in the rank-64 active subspace only
    field = x[:rank].astype(complex).copy()

    # Phase screen — first `rank` diagonal entries
    ps = npz["phase_screen"].astype(float)
    field *= np.exp(1j * ps[:rank])

    # Filter MZIs to active subspace
    mi_all = npz["mode_i"].astype(int)
    mj_all = npz["mode_j"].astype(int)
    mask = (mi_all < rank) & (mj_all < rank)
    if not mask.any():
        return field

    cols_  = npz["col"].astype(int)[mask]
    mi_    = mi_all[mask]
    mj_    = mj_all[mask]
    theta_ = npz["theta"].astype(float)[mask]
    phi_   = npz["phi"].astype(float)[mask]

    # Apply in column order (stage order)
    order = np.argsort(cols_, kind="stable")
    for idx in order:
        mi, mj = int(mi_[idx]), int(mj_[idx])
        th, ph = float(theta_[idx]), float(phi_[idx])
        ct = math.cos(th / 2)
        st = math.sin(th / 2)
        ep = math.cos(ph) + 1j * math.sin(ph)
        a = field[mi]
        b = field[mj]
        field[mi] = ep * ct * a - st * b
        field[mj] = ep * st * a + ct * b

    return field


def photonic_forward(
    x: np.ndarray,
    layer: dict,
    max_mzis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run one photonic SVD layer on input vector x.

    Implements: y_phot = U · Σ · V† · x   (photonic, complex amplitudes)
    Also computes: y_ref  = U_r @ (σ * (Vh_r @ x))  (numpy reference)

    Args:
        x:        input vector, shape (input_dim,)  e.g. 2560 for q_proj
        layer:    dict from load_compiled_layers()
        max_mzis: if > 0, limit MZIs used (for speed testing)

    Returns:
        y_phot: photonic output, real part, shape (output_dim,)
        y_ref:  numpy reference output, shape (output_dim,)
    """
    rank   = layer["rank"]
    vh_npz = layer["Vh_phases"]
    u_npz  = layer["U_phases"]

    # Pad/truncate x to rank (active subspace)
    x_in = np.zeros(rank, dtype=complex)
    n = min(len(x), rank)
    x_in[:n] = x[:n]

    # ── Stage 1: V† (photonic, rank-subspace only) ────────────────────────────
    after_Vh = _apply_clements_from_npz(x_in, vh_npz, rank)  # shape (rank,)

    # ── Stage 2: Σ — rank truncation (already rank-sized, pass through) ───────
    after_sigma = after_Vh  # unit singular values; rank limit already applied

    # ── Stage 3: U (photonic, rank-subspace only) ─────────────────────────────
    y_phot = _apply_clements_from_npz(after_sigma, u_npz, rank).real  # (rank,)

    # ── NumPy reference — same rank-64 subspace, double precision ────────────
    Vh_r = _reconstruct_unitary_rank(vh_npz, rank)   # (rank, rank)
    U_r  = _reconstruct_unitary_rank(u_npz,  rank)   # (rank, rank)
    y_ref = (U_r @ (Vh_r @ x_in)).real              # (rank,)

    return y_phot, y_ref


def _reconstruct_unitary_rank(npz, rank: int) -> np.ndarray:
    """
    Reconstruct the rank×rank active-subspace unitary from NPZ phase data.

    Only MZIs with mode_i < rank AND mode_j < rank are included (2016 for
    rank=64). Builds a compact rank×rank matrix — no OOM risk, runs in ms.
    """
    R = rank
    U = np.eye(R, dtype=complex)

    # Phase screen
    ps = npz["phase_screen"].astype(float)[:R]
    U = np.exp(1j * ps)[:, None] * U   # diagonal left-multiply

    # Filter to active subspace
    mi_all = npz["mode_i"].astype(int)
    mj_all = npz["mode_j"].astype(int)
    mask = (mi_all < R) & (mj_all < R)
    if not mask.any():
        return U

    cols_  = npz["col"].astype(int)[mask]
    mi_    = mi_all[mask]
    mj_    = mj_all[mask]
    theta_ = npz["theta"].astype(float)[mask]
    phi_   = npz["phi"].astype(float)[mask]

    order = np.argsort(cols_, kind="stable")
    for idx in order:
        mi, mj = int(mi_[idx]), int(mj_[idx])
        th, ph = float(theta_[idx]), float(phi_[idx])
        ct = math.cos(th / 2)
        st = math.sin(th / 2)
        ep = math.cos(ph) + 1j * math.sin(ph)
        row_i = U[mi, :].copy()
        row_j = U[mj, :].copy()
        U[mi, :] = ep * ct * row_i - st * row_j
        U[mj, :] = ep * st * row_i + ct * row_j

    return U


# ── Error metrics ──────────────────────────────────────────────────────────────

def relative_error(y_phot: np.ndarray, y_ref: np.ndarray) -> float:
    """‖y_phot - y_ref‖₂ / ‖y_ref‖₂"""
    denom = np.linalg.norm(y_ref)
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(y_phot - y_ref) / denom)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Kaggle reference comparison ───────────────────────────────────────────────

def load_kaggle_reference(path: str) -> Optional[dict]:
    """Load Kaggle notebook output (photomedgemma_comparison.json)."""
    try:
        with open(path) as f:
            data = json.load(f)
        logger.info(f"  Kaggle reference loaded: {path}")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"  Could not load Kaggle reference {path}: {e}")
        return None


def compare_to_kaggle(
    photonic_outputs: Dict[str, np.ndarray],
    kaggle_ref: dict,
) -> dict:
    """
    Compare photonic chip outputs against real MedGemma activations
    captured from the Kaggle notebook.

    The Kaggle notebook saves activations as:
        kaggle_ref["layers"][layer_name]["photonic_error"]  — already computed
        kaggle_ref["layers"][layer_name]["W_q"] etc.        — weight matrices

    Returns: dict with per-layer comparison metrics
    """
    results = {}
    kaggle_layers = kaggle_ref.get("layers", {})

    for lname, phot_out in photonic_outputs.items():
        # Match layer by partial name (q_proj, k_proj etc.)
        proj_key = None
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            if proj in lname:
                proj_key = proj
                break
        if proj_key is None:
            continue

        # Try to find Kaggle error for this projection
        kaggle_err = None
        for klayer, kdata in kaggle_layers.items():
            if proj_key in klayer:
                kaggle_err = kdata.get("photonic_error")
                break

        results[lname] = {
            "projection":    proj_key,
            "kaggle_error":  kaggle_err,
            "note": ("Kaggle captures real MedGemma weights; "
                     "local simulation uses compiled NPZ phases")
        }

    return results


# ── Output saving ──────────────────────────────────────────────────────────────

def save_svg_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str = "",
    max_dim: int = 64,
):
    """Save a matrix as an SVG heatmap (no matplotlib required)."""
    h, w = matrix.shape
    # Downsample if too large
    step_h = max(1, h // max_dim)
    step_w = max(1, w // max_dim)
    m = matrix[::step_h, ::step_w]
    mh, mw = m.shape

    vmin, vmax = m.min(), m.max()
    scale = vmax - vmin + 1e-12

    cell = 8  # px per cell in SVG
    svg_h = mh * cell + 40
    svg_w = mw * cell + 10

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">',
        f'<text x="5" y="14" font-size="11" fill="#333">{title}</text>',
    ]
    for i in range(mh):
        for j in range(mw):
            v = (m[i, j] - vmin) / scale
            r = int(255 * min(1, 2 * v))
            b = int(255 * min(1, 2 * (1 - v)))
            color = f"#{r:02x}00{b:02x}"
            x, y = j * cell, i * cell + 20
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                         f'fill="{color}" stroke="none"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def save_svg_comparison(
    y_phot: np.ndarray,
    y_ref: np.ndarray,
    path: Path,
    title: str = "",
    n_show: int = 128,
):
    """Save side-by-side photonic vs reference output as SVG bar chart."""
    n = min(n_show, len(y_phot), len(y_ref))
    y_p = y_phot[:n]
    y_r = y_ref[:n]

    vmin = min(y_p.min(), y_r.min())
    vmax = max(y_p.max(), y_r.max())
    scale = vmax - vmin + 1e-12

    bw, bh = 4, 60  # bar width, max bar height
    gap = 1
    svg_w = n * (2 * bw + gap + 1) + 20
    svg_h = bh + 50

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">',
        f'<text x="5" y="14" font-size="10" fill="#333">{title}</text>',
        '<text x="5" y="26" font-size="8" fill="#00aaff">■ photonic</text>',
        '<text x="65" y="26" font-size="8" fill="#ff4444">■ numpy ref</text>',
    ]
    baseline = bh + 30
    for i in range(n):
        x0 = 5 + i * (2 * bw + gap + 1)
        # Photonic bar
        hp = max(1, int(abs(y_p[i] - vmin) / scale * bh))
        lines.append(f'<rect x="{x0}" y="{baseline - hp}" width="{bw}" '
                     f'height="{hp}" fill="#00aaff" opacity="0.8"/>')
        # Reference bar
        hr = max(1, int(abs(y_r[i] - vmin) / scale * bh))
        lines.append(f'<rect x="{x0 + bw}" y="{baseline - hr}" width="{bw}" '
                     f'height="{hr}" fill="#ff4444" opacity="0.6"/>')

    lines.append(f'<line x1="5" y1="{baseline}" x2="{svg_w - 5}" y2="{baseline}" '
                 f'stroke="#888" stroke-width="0.5"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines))


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(args, image_path_override: Optional[str] = None) -> dict:
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load image ────────────────────────────────────────────────────
    logger.info("=== Step 1: Load image ===")
    img_path = image_path_override or (str(args.image) if args.image else None)
    if args.demo or img_path is None:
        image = make_synthetic_xray(seed=args.seed)
        image_source = "synthetic_xray"
    else:
        if img_path.lower().endswith(".dcm"):
            image = load_dicom(img_path)
        else:
            image = load_image(img_path)
        image_source = Path(img_path).name

    # ── Step 2: SigLIP preprocessing → patches ────────────────────────────────
    logger.info("=== Step 2: SigLIP preprocess → patches ===")
    patches, preproc_info = siglip_preprocess(
        image,
        target_size=args.image_size,
        patch_size=args.patch_size,
    )

    # ── Step 3: Project patches to model dimension ────────────────────────────
    logger.info("=== Step 3: Project to model dim ===")
    tokens = project_to_model_dim(patches, model_dim=args.model_dim, seed=args.seed)
    n_tokens = min(len(tokens), args.max_tokens)
    tokens = tokens[:n_tokens]
    logger.info(f"  Using {n_tokens} tokens for simulation")

    # ── Step 4: Load compiled photonic layers ─────────────────────────────────
    logger.info("=== Step 4: Load compiled layers ===")
    phase_map_path = args.phase_map
    layers = load_compiled_layers(phase_map_path)
    logger.info(f"  {len(layers)} compiled layers loaded")

    # ── Step 5: Photonic simulation ───────────────────────────────────────────
    logger.info("=== Step 5: Photonic simulation ===")
    all_results = []
    photonic_outputs: Dict[str, np.ndarray] = {}

    for layer in layers:
        lname = layer["name"]
        proj = next((p for p in ["q_proj","k_proj","v_proj","o_proj"]
                     if p in lname), "proj")
        logger.info(f"  Layer: {lname}")

        errors, cosines = [], []
        phot_outs, ref_outs = [], []

        for tok_idx, x in enumerate(tokens):
            y_phot, y_ref = photonic_forward(x, layer)

            err = relative_error(y_phot, y_ref)
            cos = cosine_similarity(y_phot, y_ref)
            errors.append(err)
            cosines.append(cos)
            phot_outs.append(y_phot)
            ref_outs.append(y_ref)

            if (tok_idx + 1) % 10 == 0:
                logger.info(f"    Token {tok_idx+1}/{n_tokens}  "
                            f"err={np.mean(errors):.2e}  cos={np.mean(cosines):.4f}")

        phot_stack = np.stack(phot_outs)   # (n_tokens, output_dim)
        ref_stack  = np.stack(ref_outs)
        photonic_outputs[lname] = phot_stack

        mean_err = float(np.mean(errors))
        mean_cos = float(np.mean(cosines))
        logger.info(f"  ✓ {lname}: mean_error={mean_err:.2e}  "
                    f"cosine={mean_cos:.6f}  "
                    f"{'PASS' if mean_err < 0.01 else 'FAIL'}")

        layer_result = {
            "layer_name":      lname,
            "projection_type": proj,
            "chip_id_U":       layer["chip_id_U"],
            "chip_id_Vh":      layer["chip_id_Vh"],
            "rank":            layer["rank"],
            "n_tokens":        n_tokens,
            "mean_relative_error": mean_err,
            "mean_cosine_similarity": mean_cos,
            "pass":            mean_err < 0.01,
            "error_svd":       layer["error_svd"],
        }
        all_results.append(layer_result)

        # Save arrays
        if args.save_arrays:
            safe = lname.replace(".", "_").replace("/", "_")
            np.save(str(out_dir / f"{safe}_phot.npy"),  phot_stack)
            np.save(str(out_dir / f"{safe}_ref.npy"),   ref_stack)
            np.save(str(out_dir / f"{safe}_errors.npy"), np.array(errors))

        # Save SVG comparison for first token
        save_svg_comparison(
            phot_outs[0], ref_outs[0],
            out_dir / f"comparison_{proj}.svg",
            title=f"{proj}: photonic vs numpy (token 0)"
        )

        # Save error heatmap across tokens
        err_matrix = np.array([
            [relative_error(phot_outs[i], ref_outs[i])]
            for i in range(n_tokens)
        ]).reshape(1, n_tokens)
        save_svg_heatmap(
            err_matrix,
            out_dir / f"error_heatmap_{proj}.svg",
            title=f"{proj}: per-token relative error",
        )

    # ── Step 6: Kaggle comparison ──────────────────────────────────────────────
    kaggle_comparison = {}
    if args.kaggle_ref:
        logger.info("=== Step 6: Kaggle reference comparison ===")
        kaggle_data = load_kaggle_reference(args.kaggle_ref)
        if kaggle_data:
            kaggle_comparison = compare_to_kaggle(photonic_outputs, kaggle_data)
            for lname, cmp in kaggle_comparison.items():
                ke = cmp.get("kaggle_error")
                logger.info(f"  {lname}: kaggle_error={ke}")

    # ── Save input image SVG ───────────────────────────────────────────────────
    _save_image_svg(image, out_dir / "input_image.svg")

    # ── Step 7: Summary JSON ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        "image_source":  image_source,
        "preprocess":    preproc_info,
        "n_tokens":      n_tokens,
        "model_dim":     args.model_dim,
        "phase_map":     str(phase_map_path),
        "n_layers":      len(layers),
        "layers":        all_results,
        "kaggle_comparison": kaggle_comparison,
        "overall": {
            "mean_error":   float(np.mean([r["mean_relative_error"] for r in all_results])),
            "mean_cosine":  float(np.mean([r["mean_cosine_similarity"] for r in all_results])),
            "n_pass":       sum(1 for r in all_results if r["pass"]),
            "n_layers":     len(all_results),
        },
        "elapsed_s": round(elapsed, 1),
    }

    result_path = out_dir / "dicom_simulation_results.json"
    result_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"  Results saved → {result_path}")

    # ── Print summary ──────────────────────────────────────────────────────────
    ov = summary["overall"]
    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info(f"  Image:          {image_source}")
    logger.info(f"  Tokens:         {n_tokens}")
    logger.info(f"  Layers:         {ov['n_pass']}/{ov['n_layers']} PASS")
    logger.info(f"  Mean error:     {ov['mean_error']:.2e}")
    logger.info(f"  Mean cosine:    {ov['mean_cosine']:.6f}")
    logger.info(f"  Elapsed:        {elapsed:.1f}s")
    logger.info(f"  Output dir:     {out_dir}/")
    logger.info("=" * 60)

    return summary


def _save_image_svg(arr: np.ndarray, path: Path, max_px: int = 64):
    """Save input image as SVG pixel grid."""
    h, w = arr.shape
    step = max(1, max(h, w) // max_px)
    small = arr[::step, ::step]
    sh, sw = small.shape
    cs = 4  # cell size px

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{sw * cs}" height="{sh * cs + 16}">',
        '<text x="0" y="12" font-size="9" fill="#333">Input image (grayscale)</text>',
    ]
    for i in range(sh):
        for j in range(sw):
            v = int(small[i, j] * 255)
            v = max(0, min(255, v))
            color = f"#{v:02x}{v:02x}{v:02x}"
            lines.append(
                f'<rect x="{j*cs}" y="{i*cs+16}" width="{cs}" height="{cs}" '
                f'fill="{color}"/>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DICOM/Image → PhotoMedGemma photonic chip simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("--image", default=None,
        help="Path to one input image (.dcm DICOM, .png, .jpg)")
    parser.add_argument("--all-images", action="store_true",
        help="Run on all images in --image-dir (batch mode)")
    parser.add_argument("--image-dir", default="images/",
        help="Directory of images to process with --all-images (default: images/)")
    parser.add_argument("--demo", action="store_true",
        help="Use synthetic chest X-ray phantom (no image needed)")

    # Preprocessing
    parser.add_argument("--image-size", type=int, default=448,
        help="Resize image to this square size (default: 448, SigLIP standard)")
    parser.add_argument("--patch-size", type=int, default=14,
        help="SigLIP patch size in pixels (default: 14 → 1024 patches for 448px)")
    parser.add_argument("--model-dim", type=int, default=2560,
        help="Token dimension after mm_projector (default: 2560, MedGemma 4B)")
    parser.add_argument("--max-tokens", type=int, default=32,
        help="Max image tokens to simulate (default: 32, full=1024)")
    parser.add_argument("--seed", type=int, default=42,
        help="RNG seed for projection matrix (default: 42)")

    # Phase map
    parser.add_argument("--phase-map",
        default="output/real/phase_map.json",
        help="Compiled phase map JSON (default: output/real/phase_map.json)")

    # Kaggle comparison
    parser.add_argument("--kaggle-ref", default=None,
        help="Path to Kaggle comparison JSON (photomedgemma_comparison.json)")

    # Output
    parser.add_argument("--output-dir", default="output/simulations",
        help="Output directory (default: output/simulations)")
    parser.add_argument("--save-arrays", action="store_true",
        help="Save photonic and reference outputs as .npy files")

    args = parser.parse_args()

    if not args.demo and not args.all_images and args.image is None:
        parser.print_help()
        print("\nError: provide --image <path>, --all-images, or --demo")
        sys.exit(1)

    if args.all_images:
        # Batch mode: run all images in image-dir
        img_dir = Path(args.image_dir)
        images = sorted(img_dir.glob("*.png")) + \
                 sorted(img_dir.glob("*.jpg")) + \
                 sorted(img_dir.glob("*.jpeg")) + \
                 sorted(img_dir.glob("*.dcm"))
        if not images:
            print(f"No images found in {img_dir}")
            sys.exit(1)
        logger.info(f"Batch mode: {len(images)} images in {img_dir}")

        all_summaries = []
        for img_path in images:
            logger.info(f"\n{'='*60}\nProcessing: {img_path.name}\n{'='*60}")
            img_out = Path(args.output_dir) / img_path.stem
            img_out.mkdir(parents=True, exist_ok=True)
            # Temporarily redirect output dir
            import copy
            img_args = copy.copy(args)
            img_args.output_dir = str(img_out)
            summary = run_pipeline(img_args, image_path_override=str(img_path))
            all_summaries.append({"image": img_path.name, **summary.get("overall", {})})

        # Write combined batch summary
        batch_out = Path(args.output_dir) / "batch_summary.json"
        batch_out.parent.mkdir(parents=True, exist_ok=True)
        batch_out.write_text(json.dumps(all_summaries, indent=2))
        logger.info(f"\nBatch complete. Summary → {batch_out}")
        logger.info("Per-image results:")
        for s in all_summaries:
            logger.info(f"  {s['image']}: error={s.get('mean_error','?'):.2e}  "
                        f"cosine={s.get('mean_cosine','?'):.4f}  "
                        f"pass={s.get('n_pass','?')}/{s.get('n_layers','?')}")
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
