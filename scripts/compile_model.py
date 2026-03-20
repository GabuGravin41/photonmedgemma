#!/usr/bin/env python3
"""
Compile MedGemma to Photonic Chip Configuration
================================================

Main compilation script. Runs the full pipeline:
    1. Load MedGemma weights
    2. SVD decompose each weight matrix
    3. Clements decompose U and V† factors
    4. Encode phases to DAC codes
    5. Generate photonic netlist
    6. Save phase configuration for chip programming

This is the PRIMARY entry point for chip compilation.

Usage:
    # Full model compilation (requires model download ~8GB)
    python scripts/compile_model.py --output-dir output/

    # Single layer (fast, good for testing)
    python scripts/compile_model.py --layer 0 --output-dir output/

    # Demo with small synthetic matrices
    python scripts/compile_model.py --demo --output-dir output/

    # Custom rank
    python scripts/compile_model.py --rank 32 --demo --output-dir output/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compiler.model_parser import ModelParser, LayerInfo
from compiler.layer_decomposer import LayerDecomposer
from compiler.mzi_mapper import MZIMapper
from compiler.phase_encoder import PhaseEncoder
from compiler.netlist_generator import NetlistGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compile_demo(rank: int = 8, output_dir: str = "output") -> dict:
    """
    Demo compilation using small synthetic matrices.

    Creates synthetic weight matrices matching MedGemma's layer structure
    but at reduced scale (small N for fast Clements decomposition).

    Args:
        rank: SVD truncation rank
        output_dir: Output directory

    Returns:
        Compilation statistics
    """
    logger.info("=" * 60)
    logger.info("PhotoMedGemma Demo Compilation")
    logger.info("=" * 60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Demo dimensions: 1 transformer layer, reduced size for speed
    DEMO_N = 64   # instead of 2048 — 64×64 matrix for fast demo
    DEMO_RANK = min(rank, DEMO_N - 1)

    logger.info(f"Demo parameters: N={DEMO_N}, rank={DEMO_RANK}")
    logger.info(f"Full model has N=2048, which would take ~30 min/layer")

    # Create synthetic weight matrices (realistic shape, small N)
    rng = np.random.default_rng(42)

    # Use low-rank + noise structure to simulate realistic transformer weights
    def make_synthetic_weight(m, n, eff_rank=10):
        """Create a synthetic weight matrix with controlled rank structure."""
        U = rng.standard_normal((m, eff_rank))
        V = rng.standard_normal((eff_rank, n))
        S = np.exp(-np.arange(eff_rank) / 3.0)  # exponentially decaying singular values
        W = (U * S) @ V
        W += 0.01 * rng.standard_normal((m, n))  # small noise
        return W.astype(np.float32)

    demo_layers = [
        (
            LayerInfo(
                name=f"model.layers.0.self_attn.{proj}.weight",
                shape=(DEMO_N, DEMO_N),
                module_type="attention",
                transformer_layer_idx=0,
                projection_type=proj,
                component="language_model",
            ),
            make_synthetic_weight(DEMO_N, DEMO_N),
        )
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]

    # Also add one FFN layer (smaller inner dim for demo)
    DEMO_FFN = DEMO_N * 4  # 256 instead of 16384
    for proj in ["gate_proj", "up_proj"]:
        demo_layers.append((
            LayerInfo(
                name=f"model.layers.0.mlp.{proj}.weight",
                shape=(DEMO_FFN, DEMO_N),
                module_type="ffn",
                transformer_layer_idx=0,
                projection_type=proj.replace("_proj", ""),
                component="language_model",
            ),
            make_synthetic_weight(DEMO_FFN, DEMO_N),
        ))
    demo_layers.append((
        LayerInfo(
            name="model.layers.0.mlp.down_proj.weight",
            shape=(DEMO_N, DEMO_FFN),
            module_type="ffn",
            transformer_layer_idx=0,
            projection_type="down",
            component="language_model",
        ),
        make_synthetic_weight(DEMO_N, DEMO_FFN),
    ))

    # ── Stage 2: SVD Decomposition ────────────────────────────────────────────
    logger.info("\n[Stage 2/5] SVD Decomposition...")
    decomposer = LayerDecomposer(rank=DEMO_RANK, min_rank=DEMO_RANK)

    decomposed_layers = []
    for info, weight in demo_layers:
        t0 = time.perf_counter()
        decomposed = decomposer.decompose(info, weight)
        t1 = time.perf_counter()
        logger.info(
            f"  ✓ {info.name} ({weight.shape[0]}×{weight.shape[1]}): "
            f"rank={decomposed.rank}, "
            f"energy={decomposed.energy_retained:.4f}, "
            f"error={decomposed.reconstruction_error:.4f}, "
            f"time={t1-t0:.2f}s"
        )
        decomposed_layers.append(decomposed)

    # ── Stage 3: MZI Mapping (Clements Decomposition) ─────────────────────────
    logger.info(f"\n[Stage 3/5] Clements MZI Decomposition (N={DEMO_N})...")
    logger.info(f"  Each {DEMO_N}×{DEMO_N} matrix → {DEMO_N*(DEMO_N-1)//2:,} MZIs")

    mapper = MZIMapper(mzis_per_chip=4096, dac_bits=12)
    compiled_layers = []

    for decomposed in decomposed_layers:
        logger.info(f"\n  Compiling {decomposed.layer_info.name}...")
        compiled = mapper.map_layer(decomposed)
        compiled_layers.append(compiled)
        logger.info(
            f"  ✓ Compiled: {compiled.total_mzis:,} MZIs, "
            f"chips U={compiled.chip_id_U}, Vh={compiled.chip_id_Vh}"
        )

    logger.info(f"\n  Total MZIs: {mapper.total_mzis():,}")
    logger.info(f"  Total chips: {mapper.total_chips()}")

    # ── Stage 4: Phase Encoding ───────────────────────────────────────────────
    logger.info("\n[Stage 4/5] Phase Encoding...")
    encoder = PhaseEncoder(dac_bits=12)
    phase_map = encoder.encode(
        compiled_layers,
        model_id="medgemma-4b-demo",
        rank=DEMO_RANK,
    )

    phase_json_path = output_path / "phase_map_demo.json"
    phase_map.save_json(str(phase_json_path))

    phase_bin_path = output_path / "phase_map_demo.phcfg"
    phase_map.save_binary(str(phase_bin_path))

    # Print sample SPI bitstream for first chip
    spi_bytes = phase_map.generate_spi_bitstream(chip_id=0)
    logger.info(
        f"  Generated SPI bitstream for chip 0: {len(spi_bytes):,} bytes "
        f"({len(spi_bytes)//3:,} phase register writes)"
    )

    # ── Stage 5: Netlist Generation ───────────────────────────────────────────
    logger.info("\n[Stage 5/5] Netlist Generation...")
    netlist_dir = output_path / "netlists"
    generator = NetlistGenerator(output_dir=str(netlist_dir))

    netlist = generator.generate_full_model_netlist(
        compiled_layers,
        output_file="medgemma_demo.pntl",
    )
    manifest = generator.generate_json_manifest(
        compiled_layers,
        phase_map=phase_map,
        output_file="manifest_demo.json",
    )

    n_lines = len(netlist.splitlines())
    logger.info(f"  Generated netlist: {n_lines:,} lines")

    # ── Verification: Software Simulation ────────────────────────────────────
    logger.info("\n[Verification] Simulating photonic inference for first layer...")
    first_compiled = compiled_layers[0]
    test_x = np.random.randn(DEMO_N).astype(np.float32)

    # Ground truth: SVD reconstruction using normalized singular values
    # (SigmaStage normalizes by max_sv to fit in [0,1] range for optical attenuators)
    r = first_compiled.rank
    U = first_compiled.U_mesh.reconstruct_matrix()[:, :r]
    Vh = first_compiled.Vh_mesh.reconstruct_matrix()[:r, :]
    sigma_normalized = first_compiled.sigma_stage.normalized_sv  # sigma / max_sv

    y_svd = U @ (sigma_normalized[:r] * (Vh @ test_x))

    # Photonic simulation (uses normalized sigma internally)
    from photonic.mesh import SVDLayer
    svd_layer = SVDLayer(
        U_mesh=first_compiled.U_mesh,
        sigma_stage=first_compiled.sigma_stage,
        Vh_mesh=first_compiled.Vh_mesh,
        layer_name=first_compiled.layer_name,
    )
    y_photonic = svd_layer.forward(test_x.astype(complex))

    sim_error = np.linalg.norm(y_svd.real - y_photonic.real) / (np.linalg.norm(y_svd.real) + 1e-15)
    logger.info(f"  Photonic simulation error vs. SVD: {sim_error:.6f}")

    if sim_error < 0.01:
        logger.info("  ✓ PASS: Photonic simulation matches SVD within 1%")
    else:
        logger.warning(f"  ⚠ HIGH error: {sim_error:.4f} — check Clements decomposition")

    # ── Final Summary ─────────────────────────────────────────────────────────
    stats = {
        "demo_N": DEMO_N,
        "demo_rank": DEMO_RANK,
        "n_layers": len(compiled_layers),
        "total_mzis": mapper.total_mzis(),
        "total_chips": mapper.total_chips(),
        "phase_entries": len(phase_map.entries),
        "netlist_lines": n_lines,
        "simulation_error": float(sim_error),
        "output_files": {
            "phase_map_json": str(phase_json_path),
            "phase_map_binary": str(phase_bin_path),
            "netlist": str(netlist_dir / "medgemma_demo.pntl"),
            "manifest": str(netlist_dir / "manifest_demo.json"),
        },
    }

    logger.info("\n" + "=" * 60)
    logger.info("Demo Compilation Complete!")
    logger.info("=" * 60)
    logger.info(f"  Layers compiled: {stats['n_layers']}")
    logger.info(f"  Total MZIs:      {stats['total_mzis']:,}")
    logger.info(f"  Total chips:     {stats['total_chips']}")
    logger.info(f"  Phase entries:   {stats['phase_entries']:,}")
    logger.info(f"  Sim error:       {stats['simulation_error']:.6f}")
    logger.info(f"\nOutput files:")
    for k, v in stats['output_files'].items():
        logger.info(f"  {k}: {v}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Compile MedGemma to photonic chip configuration"
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
        "--layer", type=int, default=None,
        help="Compile only this transformer layer index (None = all)"
    )
    parser.add_argument(
        "--component",
        choices=["language_model", "vision_encoder", "both"],
        default="language_model",
        help="Which model component to compile",
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Output directory for compiled chip config"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with small synthetic matrices (fast, no download)"
    )
    parser.add_argument(
        "--dac-bits", type=int, default=12,
        help="DAC resolution in bits"
    )
    parser.add_argument(
        "--n-workers", type=int, default=1,
        help="Number of parallel workers for decomposition"
    )
    parser.add_argument(
        "--max-mesh-size", type=int, default=4096,
        help=(
            "Skip layers where max(m,n) exceeds this (default: 4096). "
            "Clements is O(N^3): N=2048 ~1min, N=4096 ~10min, N=10240 ~2h. "
            "Use 0 to disable limit (compile everything, may take many hours)."
        )
    )
    args = parser.parse_args()

    t_start = time.time()

    if args.demo:
        stats = compile_demo(rank=args.rank, output_dir=args.output_dir)
    else:
        # Full model compilation
        logger.info(f"Starting compilation of {args.model}")
        logger.warning(
            "Full model compilation requires:\n"
            "  - ~8GB disk space (model download)\n"
            "  - ~16GB RAM\n"
            "  - Several hours compute time\n"
            "Use --demo for a fast demonstration."
        )

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load model
        model_parser = ModelParser(model_id=args.model)
        model_parser.load()

        layer_range = None
        if args.layer is not None:
            layer_range = (args.layer, args.layer + 1)

        # Collect layers
        layers = list(model_parser.iter_linear_layers(
            include_language_model=(args.component in ["language_model", "both"]),
            include_vision_encoder=(args.component in ["vision_encoder", "both"]),
            layer_range=layer_range,
        ))

        logger.info(f"Found {len(layers)} compilable layers")

        # Decompose
        decomposer = LayerDecomposer(rank=args.rank)
        mapper = MZIMapper(dac_bits=args.dac_bits, max_mesh_size=args.max_mesh_size)
        encoder = PhaseEncoder(dac_bits=args.dac_bits)
        generator = NetlistGenerator(output_dir=str(output_path / "netlists"))

        compiled_layers = []
        for info, weight in layers:
            decomposed = decomposer.decompose(info, weight)
            compiled = mapper.map_layer(decomposed)
            if compiled is None:
                continue   # skipped due to max_mesh_size
            compiled_layers.append(compiled)

        # Encode and generate
        phase_map = encoder.encode(compiled_layers, model_id=args.model, rank=args.rank)
        phase_map.save_json(str(output_path / "phase_map.json"))
        phase_map.save_binary(str(output_path / "phase_map.phcfg"))

        generator.generate_full_model_netlist(compiled_layers)
        generator.generate_json_manifest(compiled_layers, phase_map)

        logger.info(mapper.resource_report())
        stats = {
            "n_layers": len(compiled_layers),
            "total_mzis": mapper.total_mzis(),
            "total_chips": mapper.total_chips(),
        }

    elapsed = time.time() - t_start
    logger.info(f"\nTotal compilation time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return stats


if __name__ == "__main__":
    main()
