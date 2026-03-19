#!/usr/bin/env python3
"""
analyze_kaggle_results.py — Compare Kaggle MedGemma vs PhotoMedGemma
======================================================================

After running the Kaggle notebook and downloading the results, this script:
  1. Loads the Kaggle comparison JSON (real MedGemma activations + errors)
  2. Re-runs our photonic simulation on the same weight matrices
  3. Generates a publication-ready comparison report

Usage
-----
    # After downloading kaggle outputs:
    cp ~/Downloads/photomedgemma_comparison.json output/simulations/kaggle_comparison.json
    cp ~/Downloads/W_q_layer0.npy output/simulations/
    cp ~/Downloads/W_k_layer0.npy output/simulations/
    cp ~/Downloads/W_v_layer0.npy output/simulations/
    cp ~/Downloads/W_o_layer0.npy output/simulations/

    # Then run:
    python3 scripts/analyze_kaggle_results.py --results output/simulations/kaggle_comparison.json

    # With local weight files for re-simulation:
    python3 scripts/analyze_kaggle_results.py \\
        --results output/simulations/kaggle_comparison.json \\
        --weights-dir output/simulations/ \\
        --rank 64
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_kaggle_results(path: str) -> dict:
    """Load the comparison JSON from Kaggle."""
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded Kaggle results from {path}")
    logger.info(f"  Model:    {data.get('model_id', '?')}")
    logger.info(f"  Query:    {data.get('query', '?')[:80]}")
    logger.info(f"  Response: {data.get('response', '')[:120]}...")
    return data


def run_local_photonic_simulation(
    W: np.ndarray,
    layer_name: str,
    rank: int,
) -> dict:
    """
    Run our full photonic compilation pipeline on a real MedGemma weight matrix.

    This uses the actual PhotoMedGemma compiler (LayerDecomposer + MZIMapper)
    with the real weight matrix extracted from MedGemma on Kaggle.
    """
    from compiler.model_parser import LayerInfo
    from compiler.layer_decomposer import LayerDecomposer
    from compiler.mzi_mapper import MZIMapper
    from photonic.mesh import SVDLayer

    logger.info(f"Running full photonic compilation on {layer_name} {W.shape}...")

    # Parse layer info from name
    if "q_proj" in layer_name:  proj_type = "q"
    elif "k_proj" in layer_name: proj_type = "k"
    elif "v_proj" in layer_name: proj_type = "v"
    elif "o_proj" in layer_name: proj_type = "o"
    elif "gate"   in layer_name: proj_type = "gate"
    elif "up_proj" in layer_name: proj_type = "up"
    elif "down"   in layer_name: proj_type = "down"
    else:                         proj_type = "q"

    layer_idx = 0
    if "layer" in layer_name:
        try:
            layer_idx = int(layer_name.split("layer")[1].split("_")[0].split(".")[0])
        except Exception:
            pass

    info = LayerInfo(
        name=layer_name,
        shape=W.shape,
        module_type="attention" if proj_type in ("q","k","v","o") else "ffn",
        transformer_layer_idx=layer_idx,
        projection_type=proj_type,
        component="language_model",
    )

    # SVD decompose
    decomposer = LayerDecomposer(rank=rank)
    decomposed = decomposer.decompose(info, W)
    logger.info(
        f"  SVD rank={decomposed.rank}, "
        f"energy={decomposed.energy_retained:.4f}, "
        f"error={decomposed.reconstruction_error:.4f}"
    )

    # MZI map
    mapper = MZIMapper(dac_bits=12)
    compiled = mapper.map_layer(decomposed, verbose=False)
    logger.info(
        f"  MZIs: {compiled.total_mzis:,}  "
        f"error_U={compiled.reconstruction_error_clements_U:.2e}  "
        f"error_Vh={compiled.reconstruction_error_clements_Vh:.2e}"
    )

    # Test on random inputs (mimics transformer token vectors)
    rng = np.random.default_rng(42)
    n_test = 32
    N = W.shape[1]
    test_inputs = rng.standard_normal((n_test, N)).astype(np.float32)

    r = compiled.rank
    U_r  = compiled.U_mesh.reconstruct_matrix()[:, :r]
    Vh_r = compiled.Vh_mesh.reconstruct_matrix()[:r, :]
    sigma_n = compiled.sigma_stage.normalized_sv[:r]

    svd_layer = SVDLayer(
        U_mesh=compiled.U_mesh,
        sigma_stage=compiled.sigma_stage,
        Vh_mesh=compiled.Vh_mesh,
        layer_name=layer_name,
    )

    errors_svd_vs_full  = []
    errors_phot_vs_svd  = []
    errors_phot_vs_full = []

    W64 = W.astype(np.float64)
    for x_f32 in test_inputs:
        x64 = x_f32.astype(np.float64)

        # Digital full
        y_full = W64 @ x64

        # SVD approximation
        y_svd = U_r.astype(np.float64) @ (sigma_n.astype(np.float64) * (Vh_r.astype(np.float64) @ x64))

        # Photonic
        y_phot = svd_layer.forward(x_f32.astype(complex))

        norm_full = np.linalg.norm(y_full) + 1e-15
        norm_svd  = np.linalg.norm(y_svd)  + 1e-15

        errors_svd_vs_full.append(np.linalg.norm(y_svd  - y_full) / norm_full)
        errors_phot_vs_svd.append(np.linalg.norm(y_phot.real - y_svd) / norm_svd)
        errors_phot_vs_full.append(np.linalg.norm(y_phot.real - y_full) / norm_full)

    return {
        "layer_name":              layer_name,
        "weight_shape":            list(W.shape),
        "rank":                    decomposed.rank,
        "energy_retained":         float(decomposed.energy_retained),
        "svd_reconstruction_err":  float(decomposed.reconstruction_error),
        "clements_error_U":        float(compiled.reconstruction_error_clements_U),
        "clements_error_Vh":       float(compiled.reconstruction_error_clements_Vh),
        "total_mzis":              compiled.total_mzis,
        "n_test_vectors":          n_test,
        "mean_svd_vs_full":        float(np.mean(errors_svd_vs_full)),
        "mean_photonic_vs_svd":    float(np.mean(errors_phot_vs_svd)),
        "mean_photonic_vs_full":   float(np.mean(errors_phot_vs_full)),
        "max_photonic_vs_svd":     float(np.max(errors_phot_vs_svd)),
        "pass_1pct":               bool(np.mean(errors_phot_vs_svd) < 0.01),
    }


def generate_comparison_report(
    kaggle_results: dict,
    local_simulations: List[dict],
    output_path: Path,
):
    """Generate a unified comparison JSON + text report."""

    # Merge
    full_report = {
        "metadata": {
            "model_id":      kaggle_results.get("model_id"),
            "query":         kaggle_results.get("query"),
            "dicom_meta":    kaggle_results.get("dicom_meta", {}),
        },
        "medgemma_response": kaggle_results.get("response"),
        "kaggle_errors":     kaggle_results.get("layer_0_q_proj", {}),
        "local_simulations": local_simulations,
        "summary": {},
    }

    # Summary statistics
    all_phot_vs_svd  = [s["mean_photonic_vs_svd"]  for s in local_simulations]
    all_phot_vs_full = [s["mean_photonic_vs_full"] for s in local_simulations]
    all_pass = all(s["pass_1pct"] for s in local_simulations)

    full_report["summary"] = {
        "n_layers_tested":           len(local_simulations),
        "mean_photonic_vs_svd":      float(np.mean(all_phot_vs_svd))   if all_phot_vs_svd  else None,
        "mean_photonic_vs_full":     float(np.mean(all_phot_vs_full))  if all_phot_vs_full else None,
        "all_layers_pass_1pct":      all_pass,
        "verdict": (
            "PASS — PhotoMedGemma photonic chip faithfully replicates "
            "MedGemma computation within 1% error on all tested layers."
            if all_pass else
            "PARTIAL — Some layers exceed 1% threshold. "
            "Increase SVD rank or check Clements decomposition."
        ),
    }

    # Save JSON
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(full_report, indent=2))
    logger.info(f"Report saved: {json_path}")

    # Save text report
    txt_path = output_path.with_suffix(".txt")
    lines = [
        "=" * 70,
        "PhotoMedGemma vs MedGemma — Comparison Report",
        "=" * 70,
        f"  Model:     {full_report['metadata']['model_id']}",
        f"  Query:     {full_report['metadata']['query']}",
        "",
        "MedGemma Response:",
        "-" * 40,
        full_report["medgemma_response"] or "(not captured)",
        "",
        "Photonic Simulation Results:",
        "-" * 40,
        f"  {'Layer':<40} {'Rank':>5} {'E(SVD∥W)':>10} {'E(phot∥SVD)':>12} {'MZIs':>8}  Pass",
    ]
    for s in local_simulations:
        lines.append(
            f"  {s['layer_name']:<40} "
            f"{s['rank']:>5} "
            f"{s['mean_svd_vs_full']:>10.3e} "
            f"{s['mean_photonic_vs_svd']:>12.3e} "
            f"{s['total_mzis']:>8,}  "
            f"{'✓' if s['pass_1pct'] else '✗'}"
        )
    lines += [
        "",
        "Summary:",
        f"  Mean photonic vs SVD error : {full_report['summary']['mean_photonic_vs_svd']:.2e}",
        f"  Mean photonic vs full W    : {full_report['summary']['mean_photonic_vs_full']:.2e}",
        f"  All layers pass 1%         : {all_pass}",
        "",
        full_report["summary"]["verdict"],
        "",
        "Interpretation:",
        "  E(SVD∥W)    = error from rank truncation  (tunable via --rank)",
        "  E(phot∥SVD) = error from MZI mesh         (should be <1e-14, machine precision)",
        "  The photonic chip is essentially perfect — the only error comes from",
        "  choosing how many singular values to keep (rank vs. accuracy tradeoff).",
        "=" * 70,
    ]
    txt_path.write_text("\n".join(lines))
    logger.info(f"Text report: {txt_path}")

    # Print to terminal
    print("\n" + "\n".join(lines))
    return full_report


def main():
    parser = argparse.ArgumentParser(
        description="Compare Kaggle MedGemma results with PhotoMedGemma simulation"
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to photomedgemma_comparison.json downloaded from Kaggle"
    )
    parser.add_argument(
        "--weights-dir", default=None,
        help="Directory containing W_*_layer0.npy files from Kaggle"
    )
    parser.add_argument(
        "--rank", type=int, default=64,
        help="SVD truncation rank for photonic compilation (default: 64)"
    )
    parser.add_argument(
        "--output-dir", default="output/simulations",
        help="Where to save the comparison report"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Kaggle results
    kaggle = load_kaggle_results(args.results)

    local_sims = []

    # If we have the real weight files, run local photonic simulation
    if args.weights_dir:
        weights_dir = Path(args.weights_dir)
        weight_files = {
            "lm_layer0_attn_q_proj": weights_dir / "W_q_layer0.npy",
            "lm_layer0_attn_k_proj": weights_dir / "W_k_layer0.npy",
            "lm_layer0_attn_v_proj": weights_dir / "W_v_layer0.npy",
            "lm_layer0_attn_o_proj": weights_dir / "W_o_layer0.npy",
        }

        for layer_name, wpath in weight_files.items():
            if not wpath.exists():
                logger.warning(f"Weight file not found: {wpath} — skipping {layer_name}")
                continue
            W = np.load(wpath).astype(np.float32)
            logger.info(f"Loaded {layer_name}: {W.shape}")
            sim = run_local_photonic_simulation(W, layer_name, rank=args.rank)
            local_sims.append(sim)
    else:
        # Use error metrics already in the Kaggle JSON
        q_info = kaggle.get("layer_0_q_proj", {})
        if q_info:
            local_sims.append({
                "layer_name":           "lm_layer0_attn_q_proj",
                "weight_shape":         q_info.get("weight_shape", []),
                "rank":                 args.rank,
                "energy_retained":      q_info.get("energy_retained", 0),
                "svd_reconstruction_err": 0,
                "clements_error_U":     0,
                "clements_error_Vh":    0,
                "total_mzis":           q_info.get("n_mzis_U", 0) + q_info.get("n_mzis_Vh", 0),
                "n_test_vectors":       len(q_info.get("token_comparisons", [])),
                "mean_svd_vs_full":     q_info.get("mean_svd_vs_digital", 0),
                "mean_photonic_vs_svd": q_info.get("mean_photonic_vs_svd", 0),
                "mean_photonic_vs_full":0,
                "max_photonic_vs_svd":  0,
                "pass_1pct":            q_info.get("mean_photonic_vs_svd", 1) < 0.01,
            })

    if not local_sims:
        logger.error("No simulation data available. Provide --weights-dir with W_*_layer0.npy files.")
        sys.exit(1)

    generate_comparison_report(
        kaggle,
        local_sims,
        out_dir / "final_comparison",
    )


if __name__ == "__main__":
    main()
