#!/usr/bin/env python3
"""
generate_chiplet_mask.py — PhotoMedGemma Full Chiplet Mask Generator
=====================================================================

Orchestrates the full pipeline:
  1. Load phase map (or run demo compilation)
  2. Partition chips into chiplets via chiplet_partition.py
  3. Reuse existing per-chip GDS files (from output/gds/)
  4. Generate per-chiplet GDS (groups of chips on a chiplet)
  5. Generate assembly GDS (all chiplets on the 2.5D interposer)
  6. Write KLayout layer properties (.lyp)

Output
------
  output/chiplet_mask/
    chiplet_manifest.json
    chiplets/
      chiplet_000_attention_L0.gds
      chiplet_001_ffn_L0.gds
    assembly/
      assembly.gds          ← OPEN THIS IN KLAYOUT
      photomedgemma.lyp     ← LOAD THIS AS LAYER PROPERTIES

Usage
-----
    # Demo (rank-8, 10 chips — uses existing output/gds/ chips)
    python3 scripts/generate_chiplet_mask.py \\
        --phase-map output/phase_map_demo.json \\
        --chip-gds-dir output/gds \\
        --output-dir output/chiplet_mask

    # Real model (layer 0 only)
    python3 scripts/generate_chiplet_mask.py \\
        --phase-map output/real/phase_map.json \\
        --chip-gds-dir output/real/gds \\
        --output-dir output/chiplet_mask_real
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS = ROOT / "scripts"


# ── Phase map loader ────────────────────────────────────────────────────────

def load_phase_map(phase_map_path: Path) -> dict:
    """Load and return the phase map JSON."""
    with open(phase_map_path) as f:
        return json.load(f)


# ── Chiplet partitioning ────────────────────────────────────────────────────

def run_chiplet_partition(
    phase_map_path: Path,
    output_path: Path,
    strategy: str = "per_transformer_layer",
    chip_gap_um: float = 500.0,
    dicing_lane_um: float = 75.0,
) -> dict:
    """
    Run chiplet_partition.py and return the manifest dict.
    """
    from chiplet_partition import (
        load_phase_map_summary,
        partition_by_function,
        assign_interposer_positions,
        write_chiplet_manifest,
    )

    summary  = load_phase_map_summary(phase_map_path)
    chiplets = partition_by_function(summary["chip_entries"], strategy=strategy)
    chiplets = assign_interposer_positions(
        chiplets,
        summary["chip_entries"],
        chip_gap_um=chip_gap_um,
        dicing_lane_um=dicing_lane_um,
    )
    manifest = write_chiplet_manifest(
        chiplets,
        phase_map_path,
        output_path,
        model_id=summary["model_id"],
        rank=summary["rank"],
        strategy=strategy,
        chip_gap_um=chip_gap_um,
    )
    return manifest


# ── Chip GDS path collection ────────────────────────────────────────────────

def collect_chip_gds(
    chip_ids: List[int],
    chip_gds_dir: Path,
    phase_map_path: Optional[Path] = None,
    dac_bits: int = 12,
    rank: int = 63,
    force_regen: bool = False,
) -> Dict[int, Path]:
    """
    Collect existing per-chip GDS paths. If a chip GDS is missing and
    phase_map_path is given, regenerate it via export_gds.py.

    Returns dict mapping chip_id → Path.
    """
    chip_gds_paths: Dict[int, Path] = {}

    for cid in chip_ids:
        gds_path = chip_gds_dir / f"chip_{cid:03d}.gds"
        if gds_path.exists() and not force_regen:
            chip_gds_paths[cid] = gds_path
        elif phase_map_path and phase_map_path.exists():
            logger.info(f"  Chip {cid} GDS not found — regenerating from phase map...")
            chip_gds_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run([
                sys.executable,
                str(SCRIPTS / "export_gds.py"),
                "--phase-map", str(phase_map_path),
                "--chip-id",   str(cid),
                "--output-dir", str(chip_gds_dir),
                "--dac-bits",  str(dac_bits),
                "--rank",      str(rank),
            ], capture_output=True, text=True)
            if result.returncode == 0 and gds_path.exists():
                chip_gds_paths[cid] = gds_path
                logger.info(f"  Chip {cid} regenerated: {gds_path}")
            else:
                logger.warning(f"  Chip {cid} regen failed: {result.stderr[:200]}")
        else:
            logger.warning(f"  Chip {cid} GDS missing: {gds_path}")

    found = len(chip_gds_paths)
    total = len(chip_ids)
    logger.info(f"Chip GDS files: {found}/{total} found in {chip_gds_dir}")
    return chip_gds_paths


# ── Chiplet GDS generation ─────────────────────────────────────────────────

def generate_all_chiplets(
    manifest: dict,
    chip_gds_paths: Dict[int, Path],
    chiplets_dir: Path,
    force_regen: bool = False,
) -> Dict[int, Path]:
    """
    Generate per-chiplet GDS files for all chiplets in the manifest.

    Returns dict mapping chiplet_id → Path.
    """
    from export_gds import generate_chiplet_gds

    chiplet_gds_paths: Dict[int, Path] = {}
    chip_size_um = manifest.get("chip_size_um", 10_000.0)

    for chiplet in manifest["chiplets"]:
        cid    = chiplet["chiplet_id"]
        ctype  = chiplet["chiplet_type"]
        lidx   = chiplet["layer_idx"]
        out_path = chiplets_dir / f"chiplet_{cid:03d}_{ctype}_L{lidx}.gds"

        if out_path.exists() and not force_regen:
            logger.info(f"  Chiplet {cid} already exists: {out_path}")
            chiplet_gds_paths[cid] = out_path
            continue

        # Collect chip GDS paths for this chiplet
        local_chip_paths = {
            cl["chip_id"]: chip_gds_paths.get(cl["chip_id"])
            for cl in chiplet.get("chip_layout", [])
        }
        # Fill in any chip_ids not in chip_layout
        for chip_id in chiplet["chip_ids"]:
            if chip_id not in local_chip_paths:
                local_chip_paths[chip_id] = chip_gds_paths.get(chip_id)

        stats = generate_chiplet_gds(
            chiplet_id=cid,
            chiplet_type=ctype,
            layer_idx=lidx,
            chip_ids=chiplet["chip_ids"],
            chip_layout=chiplet.get("chip_layout", []),
            width_um=chiplet["width_um"],
            height_um=chiplet["height_um"],
            chip_gds_paths=local_chip_paths,
            output_path=out_path,
            chip_size_um=chip_size_um,
        )
        if stats:
            chiplet_gds_paths[cid] = out_path

    logger.info(f"Chiplet GDS: {len(chiplet_gds_paths)}/{len(manifest['chiplets'])} generated")
    return chiplet_gds_paths


# ── Assembly GDS + KLayout export ─────────────────────────────────────────

def generate_assembly(
    manifest: dict,
    chiplet_gds_paths: Dict[int, Path],
    assembly_dir: Path,
    model_id: str = "PhotoMedGemma",
    representative_only: bool = True,
) -> dict:
    """
    Generate the final assembly GDS and KLayout properties file.

    Returns dict with assembly stats and output paths.
    """
    from export_gds import generate_assembly_gds, write_klayout_props

    assembly_dir.mkdir(parents=True, exist_ok=True)
    assembly_gds_path = assembly_dir / "assembly.gds"

    logger.info("Generating assembly GDS...")
    stats = generate_assembly_gds(
        chiplets=manifest["chiplets"],
        chiplet_gds_paths=chiplet_gds_paths,
        output_path=assembly_gds_path,
        interposer_width_um=manifest["interposer_width_um"],
        interposer_height_um=manifest["interposer_height_um"],
        model_id=model_id,
        representative_only=representative_only,
    )

    # KLayout layer properties
    lyp_path = write_klayout_props(assembly_dir)

    # Write assembly manifest
    assembly_manifest = {
        "model_id": model_id,
        "assembly_gds": str(assembly_gds_path),
        "klayout_lyp":  str(lyp_path),
        **stats,
        "chiplet_gds_files": {
            str(cid): str(p) for cid, p in chiplet_gds_paths.items()
        },
    }
    manifest_path = assembly_dir / "assembly_manifest.json"
    manifest_path.write_text(json.dumps(assembly_manifest, indent=2))
    logger.info(f"Assembly manifest: {manifest_path}")

    return {**assembly_manifest, "lyp_path": str(lyp_path)}


# ── Full pipeline ──────────────────────────────────────────────────────────

def run_full_pipeline(
    phase_map_path: Path,
    chip_gds_dir: Path,
    output_dir: Path,
    strategy: str = "per_transformer_layer",
    chip_gap_um: float = 500.0,
    dicing_lane_um: float = 75.0,
    representative_only: bool = True,
    force_regen: bool = False,
) -> dict:
    """
    Full pipeline: partition → collect chips → generate chiplets → assemble.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load phase map ───────────────────────────────────────────────
    logger.info(f"Loading phase map: {phase_map_path}")
    pm = load_phase_map(phase_map_path)
    model_id = pm.get("model_id", "PhotoMedGemma")
    rank     = pm.get("rank", 63)

    # ── Step 2: Chiplet partitioning ─────────────────────────────────────────
    logger.info("Step 1/4: Partitioning chips into chiplets...")
    manifest_path = output_dir / "chiplet_manifest.json"
    manifest = run_chiplet_partition(
        phase_map_path,
        manifest_path,
        strategy=strategy,
        chip_gap_um=chip_gap_um,
        dicing_lane_um=dicing_lane_um,
    )

    # ── Step 3: Collect chip GDS paths ───────────────────────────────────────
    logger.info("Step 2/4: Collecting per-chip GDS files...")
    all_chip_ids = sorted(set(
        cid
        for chiplet in manifest["chiplets"]
        for cid in chiplet["chip_ids"]
    ))
    chip_gds_paths = collect_chip_gds(
        all_chip_ids,
        chip_gds_dir,
        phase_map_path=phase_map_path,
        dac_bits=pm.get("dac_bits", 12),
        rank=rank,
        force_regen=force_regen,
    )

    # ── Step 4: Generate chiplet GDS ─────────────────────────────────────────
    logger.info("Step 3/4: Generating chiplet GDS files...")
    chiplets_dir = output_dir / "chiplets"
    chiplets_dir.mkdir(parents=True, exist_ok=True)
    chiplet_gds_paths = generate_all_chiplets(
        manifest,
        chip_gds_paths,
        chiplets_dir,
        force_regen=force_regen,
    )

    # ── Step 5: Generate assembly GDS + KLayout .lyp ─────────────────────────
    logger.info("Step 4/4: Generating assembly GDS + KLayout properties...")
    assembly_dir = output_dir / "assembly"
    assembly_result = generate_assembly(
        manifest,
        chiplet_gds_paths,
        assembly_dir,
        model_id=model_id,
        representative_only=representative_only,
    )

    return {
        "manifest_path":       str(manifest_path),
        "n_chiplets":          manifest["n_chiplets"],
        "n_chips":             len(all_chip_ids),
        "n_chip_gds_found":    len(chip_gds_paths),
        "n_chiplet_gds":       len(chiplet_gds_paths),
        "assembly_gds":        assembly_result["assembly_gds"],
        "klayout_lyp":         assembly_result["lyp_path"],
        "interposer_width_mm": manifest["interposer_width_um"] / 1000,
        "interposer_height_mm":manifest["interposer_height_um"] / 1000,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate PhotoMedGemma chiplet assembly mask for KLayout"
    )
    parser.add_argument(
        "--phase-map",
        default="output/phase_map_demo.json",
        help="Compiled phase map JSON (default: output/phase_map_demo.json)"
    )
    parser.add_argument(
        "--chip-gds-dir",
        default="output/gds",
        help="Directory containing per-chip GDS files (default: output/gds)"
    )
    parser.add_argument(
        "--output-dir",
        default="output/chiplet_mask",
        help="Output directory (default: output/chiplet_mask)"
    )
    parser.add_argument(
        "--strategy",
        default="per_transformer_layer",
        choices=["per_transformer_layer", "flat"],
        help="Chiplet partitioning strategy"
    )
    parser.add_argument(
        "--chip-gap", type=float, default=500.0,
        help="Gap between chips within a chiplet (μm, default 500)"
    )
    parser.add_argument(
        "--dicing-lane", type=float, default=75.0,
        help="Dicing lane width between chiplets (μm, default 75)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Render all chiplets fully (slow for large models; default: representative only)"
    )
    parser.add_argument(
        "--force-regen", action="store_true",
        help="Force regeneration of existing GDS files"
    )
    args = parser.parse_args()

    # Add scripts dir to path so we can import chiplet_partition and export_gds
    sys.path.insert(0, str(SCRIPTS))

    phase_map_path = Path(args.phase_map)
    chip_gds_dir   = Path(args.chip_gds_dir)
    output_dir     = Path(args.output_dir)

    if not phase_map_path.exists():
        logger.error(f"Phase map not found: {phase_map_path}")
        logger.error("Run: python3 scripts/compile_model.py --demo --rank 8 --output-dir output/")
        sys.exit(1)

    result = run_full_pipeline(
        phase_map_path,
        chip_gds_dir,
        output_dir,
        strategy=args.strategy,
        chip_gap_um=args.chip_gap,
        dicing_lane_um=args.dicing_lane,
        representative_only=not args.full,
        force_regen=args.force_regen,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("PhotoMedGemma Chiplet Mask — Generation Complete")
    print("=" * 72)
    print(f"  Chiplets:        {result['n_chiplets']}")
    print(f"  Chips total:     {result['n_chips']}  ({result['n_chip_gds_found']} GDS found)")
    print(f"  Chiplet GDS:     {result['n_chiplet_gds']} files")
    print(f"  Interposer:      {result['interposer_width_mm']:.1f}mm × "
          f"{result['interposer_height_mm']:.1f}mm")
    print()
    print("  Open in KLayout:")
    print(f"    klayout {result['assembly_gds']} \\")
    print(f"            -l {result['klayout_lyp']}")
    print()
    print("  KLayout zoom guide:")
    print("    Zoom out  → Interposer + chiplet outlines (layers 10, 11, 12)")
    print("    Zoom mid  → Chips + fiber ribbons + bond pads (layers 5, 13, 14)")
    print("    Zoom in   → MZI mesh — Si waveguide rails (layer 1)")
    print("    Zoom max  → Individual MZIs: TiN heaters (layers 2, 3) + contacts (4)")
    print("=" * 72)


if __name__ == "__main__":
    main()
