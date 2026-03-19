#!/usr/bin/env python3
"""
Analyze MedGemma Model for Photonic Compilation
================================================

This script analyzes the MedGemma model architecture and produces:
1. Resource estimates (MZI counts, chip counts, power)
2. SVD rank sensitivity analysis (accuracy vs. compression)
3. Layer-by-layer compilation cost breakdown

Run without downloading the full model using --synthetic to use
a synthetic model with matching dimensions.

Usage:
    python scripts/analyze_model.py
    python scripts/analyze_model.py --rank 64 --output report.json
    python scripts/analyze_model.py --synthetic --rank 32
    python scripts/analyze_model.py --model /local/path/to/medgemma
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compiler.model_parser import ModelParser, estimate_compilation_resources
from utils.error_analysis import ErrorAnalyzer
from utils.power_model import PowerModel, GPUInferencePower
from utils.svd_utils import rank_for_energy, energy_at_rank


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Synthetic model for offline analysis (no HuggingFace needed) ─────────────

MEDGEMMA_4B_LAYERS = [
    # (name, shape, projection_type, component)
    # Language model — 46 transformer layers
    *[
        item
        for i in range(46)
        for item in [
            (f"model.layers.{i}.self_attn.q_proj.weight", (2048, 2048), "q", "language_model"),
            (f"model.layers.{i}.self_attn.k_proj.weight", (1024, 2048), "k", "language_model"),
            (f"model.layers.{i}.self_attn.v_proj.weight", (1024, 2048), "v", "language_model"),
            (f"model.layers.{i}.self_attn.o_proj.weight", (2048, 2048), "o", "language_model"),
            (f"model.layers.{i}.mlp.gate_proj.weight",    (16384, 2048), "gate", "language_model"),
            (f"model.layers.{i}.mlp.up_proj.weight",      (16384, 2048), "up",   "language_model"),
            (f"model.layers.{i}.mlp.down_proj.weight",    (2048, 16384), "down", "language_model"),
        ]
    ],
    # Vision encoder — 27 SigLIP layers
    *[
        item
        for i in range(27)
        for item in [
            (f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight", (1152, 1152), "q", "vision_encoder"),
            (f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight", (1152, 1152), "k", "vision_encoder"),
            (f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight", (1152, 1152), "v", "vision_encoder"),
            (f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight", (1152, 1152), "o", "vision_encoder"),
            (f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight", (4304, 1152), "fc1", "vision_encoder"),
            (f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight", (1152, 4304), "fc2", "vision_encoder"),
        ]
    ],
]


def analyze_synthetic(rank: int = 64, dac_bits: int = 12) -> dict:
    """
    Analyze resource requirements using synthetic model dimensions.
    No model download required.

    Args:
        rank: SVD truncation rank
        dac_bits: DAC resolution

    Returns:
        Analysis results dictionary
    """
    logger.info("Running synthetic analysis (no model download required)...")

    total_mzis = 0
    total_params = 0
    layer_breakdown = {}
    rank_sensitivity = {}

    for name, shape, proj_type, component in MEDGEMMA_4B_LAYERS:
        m, n = shape
        r = min(rank, m, n)

        # MZIs for Clements decomposition
        mzis_U = m * (m - 1) // 2    # Full m×m unitary
        mzis_Vh = n * (n - 1) // 2   # Full n×n unitary
        # Rectangular Clements (rank-r): approximately m×r + n×r
        mzis_U_rect = m * r
        mzis_Vh_rect = n * r
        mzis_full = mzis_U_rect + mzis_Vh_rect

        n_params = m * n
        total_mzis += mzis_full
        total_params += n_params

        key = f"{component}.{proj_type}"
        if key not in layer_breakdown:
            layer_breakdown[key] = {
                "count": 0,
                "shape": shape,
                "mzis_per_instance": mzis_full,
                "total_mzis": 0,
                "total_params": 0,
            }
        layer_breakdown[key]["count"] += 1
        layer_breakdown[key]["total_mzis"] += mzis_full
        layer_breakdown[key]["total_params"] += n_params

    # Resource estimates
    mzis_per_chip = 4096
    n_chips = (total_mzis + mzis_per_chip - 1) // mzis_per_chip

    power_model = PowerModel(
        n_mzis=total_mzis,
        n_chips=n_chips,
        inference_tokens_per_sec=100.0,
    )
    system_power = power_model.estimate()
    gpu_comparison = power_model.compare_to_gpu()

    # Error analysis
    error_analyzer = ErrorAnalyzer(N=2048, rank=rank, dac_bits=dac_bits)
    budget = error_analyzer.full_error_budget()

    # Rank sensitivity (for 2048×2048 Q projection as representative case)
    logger.info("Computing rank sensitivity analysis for Q projection (2048×2048)...")
    test_ranks = [4, 8, 16, 32, 64, 128, 256, 512]
    for r in test_ranks:
        mzis = 2048 * r + 2048 * r
        energy_approx = 1.0 - np.exp(-r / 50.0)  # rough estimate without real weights
        rank_sensitivity[r] = {
            "mzis_for_q_proj": mzis,
            "energy_estimated": float(min(energy_approx, 0.999)),
            "error_floor_dac": ErrorAnalyzer(N=2048, rank=r, dac_bits=dac_bits)
                               .phase_quantization_error_bound(),
        }

    return {
        "model": "google/medgemma-4b-it (synthetic analysis)",
        "rank": rank,
        "dac_bits": dac_bits,
        "total_parameters": total_params,
        "total_parameters_B": total_params / 1e9,
        "total_mzis": total_mzis,
        "total_mzis_B": total_mzis / 1e9,
        "chips_required": n_chips,
        "chip_footprint_cm2": n_chips * 1.0,  # 1cm² per chip
        "power": {
            "total_inference_W": system_power.total_inference_power_W,
            "laser_power_mW": system_power.laser_power_mW,
            "fpga_power_mW": system_power.fpga_power_mW,
            "vs_gpu_A100_power_reduction": gpu_comparison["power_reduction_factor"],
            "energy_per_token_mJ": power_model.energy_per_token(system_power),
            "gpu_energy_per_token_mJ": gpu_comparison["gpu_energy_mJ_per_token"],
        },
        "error_budget": {
            "svd_truncation": budget.svd_error,
            "phase_quantization": budget.phase_quantization_error,
            "fabrication": budget.fabrication_total,
            "total_rms": budget.total_error_estimate,
        },
        "layer_breakdown": layer_breakdown,
        "rank_sensitivity_q_proj": rank_sensitivity,
    }


def print_summary(results: dict):
    """Print a human-readable analysis summary."""
    print("\n" + "=" * 60)
    print("  PhotoMedGemma Compilation Analysis")
    print("=" * 60)
    print(f"\nModel: {results['model']}")
    print(f"SVD Rank: {results['rank']}")
    print(f"DAC Resolution: {results['dac_bits']} bits")

    print(f"\n{'─'*60}")
    print("WEIGHT PARAMETERS")
    print(f"{'─'*60}")
    print(f"  Total: {results['total_parameters']:,} ({results['total_parameters_B']:.2f}B)")

    print(f"\n{'─'*60}")
    print("PHOTONIC RESOURCES (rank={})".format(results['rank']))
    print(f"{'─'*60}")
    print(f"  Total MZIs:        {results['total_mzis']:,}")
    print(f"  Chips required:    {results['chips_required']:,} (10mm×10mm each)")
    print(f"  System footprint:  {results['chip_footprint_cm2']:.0f} cm²")

    print(f"\n{'─'*60}")
    print("POWER & ENERGY")
    print(f"{'─'*60}")
    p = results['power']
    print(f"  System power:              {p['total_inference_W']:.1f} W")
    print(f"  GPU A100 power:            ~200 W")
    print(f"  Power reduction:           {p['vs_gpu_A100_power_reduction']:.0f}×")
    print(f"  Energy/token (photonic):   {p['energy_per_token_mJ']:.1f} mJ")
    print(f"  Energy/token (A100 GPU):   {p['gpu_energy_per_token_mJ']:.0f} mJ")
    print(f"  Energy reduction:          {p['gpu_energy_per_token_mJ'] / p['energy_per_token_mJ']:.0f}×")

    print(f"\n{'─'*60}")
    print("ERROR BUDGET (rank={}, {}-bit DAC)".format(results['rank'], results['dac_bits']))
    print(f"{'─'*60}")
    eb = results['error_budget']
    print(f"  SVD truncation:     {eb['svd_truncation']:.4f}")
    print(f"  Phase quantization: {eb['phase_quantization']:.4f}")
    print(f"  Fabrication:        {eb['fabrication']:.4f}")
    print(f"  TOTAL (RMS):        {eb['total_rms']:.4f}")

    print(f"\n{'─'*60}")
    print("RANK SENSITIVITY (Q projection, 2048×2048)")
    print(f"{'─'*60}")
    print(f"  {'Rank':>6} | {'MZIs':>12} | {'DAC error':>10}")
    print(f"  {'─'*6}-+-{'─'*12}-+-{'─'*10}")
    for r, info in results['rank_sensitivity_q_proj'].items():
        print(
            f"  {r:>6} | {info['mzis_for_q_proj']:>12,} | {info['error_floor_dac']:>10.4f}"
        )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MedGemma model for photonic compilation"
    )
    parser.add_argument(
        "--model", default="google/medgemma-4b-it",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--rank", type=int, default=64,
        help="SVD truncation rank (default: 64)"
    )
    parser.add_argument(
        "--dac-bits", type=int, default=12,
        help="DAC resolution in bits (default: 12)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic analysis (no model download)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file"
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.synthetic:
        results = analyze_synthetic(rank=args.rank, dac_bits=args.dac_bits)
    else:
        # Try to load the actual model
        try:
            parser_obj = ModelParser(model_id=args.model)
            logger.info(f"Loading model {args.model}...")
            parser_obj.load()
            print(parser_obj.summary())

            resources = estimate_compilation_resources(
                parser_obj, rank=args.rank
            )
            # Convert to same format as synthetic for display
            results = {
                "model": args.model,
                "rank": args.rank,
                "dac_bits": args.dac_bits,
                **resources,
            }
        except Exception as e:
            logger.warning(
                f"Could not load model ({e}). "
                f"Falling back to synthetic analysis."
            )
            results = analyze_synthetic(rank=args.rank, dac_bits=args.dac_bits)

    print_summary(results)

    if args.output:
        # Make results JSON-serializable
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        logger.info(f"Results saved to {args.output}")

    elapsed = time.time() - t_start
    logger.info(f"Analysis complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
