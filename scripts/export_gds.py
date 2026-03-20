#!/usr/bin/env python3
"""
export_gds.py — Photonic Chip GDS-II Mask Generator
=====================================================

Generates KLayout-viewable GDS-II chip masks for PhotoMedGemma.

Each chip is rendered with physically accurate MZI geometry on a 220nm SOI
platform at 1310nm. The resulting .gds files can be opened directly in KLayout
for inspection and design rule checking.

GDS Layer Map
-------------
    Layer 1 (Si)    : Silicon waveguide core
    Layer 2 (Heater): TiN phase-shift heaters (θ arm)
    Layer 3 (Heater): TiN phase-shift heaters (φ arm)
    Layer 4 (Metal) : Metal routing / heater contacts
    Layer 5 (Oxide) : Chip boundary / floorplan box
    Layer 6 (Label) : Text labels

MZI Physical Geometry (220nm SOI, 1310nm)
------------------------------------------
    Waveguide width   : 450 nm
    DC gap            : 200 nm
    DC length         : 10 μm
    Phase shifter len : 100 μm (θ arm) + 40 μm (φ arm)
    Total MZI length  : ~300 μm  (matches compile_model.py constant)
    Mode pitch        : 127 μm   (fiber array standard)

Usage
-----
    # All chips from a phase map
    python3 scripts/export_gds.py --phase-map output/phase_map_demo.json

    # Single chip
    python3 scripts/export_gds.py --phase-map output/phase_map_demo.json --chip-id 0

    # Output directory
    python3 scripts/export_gds.py --phase-map output/phase_map_demo.json --output-dir output/gds
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from compiler.phase_encoder import PhaseMap, MZIPhaseEntry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Physical constants (all in μm) ────────────────────────────────────────────

WG_WIDTH_UM       = 0.450    # Si waveguide width
WG_SPACING_UM     = 2.0      # intra-MZI waveguide separation at coupler gap
MODE_PITCH_UM     = 127.0    # vertical spacing between adjacent mesh modes
DC_LENGTH_UM      = 10.0     # directional coupler length
DC_GAP_UM         = 0.200    # coupler gap (200 nm)
THETA_LEN_UM      = 100.0    # θ heater length
PHI_LEN_UM        = 40.0     # φ heater length
HEATER_WIDTH_UM   = 2.0      # TiN heater width
TOTAL_MZI_LEN_UM  = 300.0    # total MZI cell length (matches compiler constant)
STAGE_GAP_UM      = 20.0     # horizontal gap between stages
CHIP_MARGIN_UM    = 500.0    # chip border margin

# GDS layers
L_SI      = 1   # silicon core
L_THETA   = 2   # TiN heater — θ phase
L_PHI     = 3   # TiN heater — φ phase
L_METAL   = 4   # metal contacts
L_BOUND   = 5   # chip boundary
L_LABEL   = 6   # text labels


# ── GDS primitives ─────────────────────────────────────────────────────────────

def _rect(lib_cell, x0: float, y0: float, x1: float, y1: float, layer: int):
    """Add a rectangle to a gdspy cell."""
    import gdspy
    lib_cell.add(gdspy.Rectangle((x0, y0), (x1, y1), layer=layer))


def _label(lib_cell, text: str, x: float, y: float, size: float = 5.0):
    """Add a text label."""
    import gdspy
    lib_cell.add(gdspy.Label(text, (x, y), "nw", magnification=size, texttype=L_LABEL))


def _waveguide(lib_cell, x0: float, y0: float, x1: float, y1: float):
    """Add a waveguide segment (thin Si rectangle)."""
    hw = WG_WIDTH_UM / 2
    _rect(lib_cell, x0, y0 - hw, x1, y1 + hw, L_SI)


def _mzi_cell(lib, chip_id: int, mzi_row: int, mzi_col: int,
               theta: float, phi: float) -> str:
    """
    Create (or retrieve) a uniquely placed MZI instance.

    Returns the cell name.  Each MZI is placed at its absolute position
    on the chip so the top-level cell simply references each MZI cell.

    The MZI structure from left to right:
        [φ heater (40μm)] → [DC1 (10μm)] → [θ heater (100μm)] → [DC2 (10μm)]
    Total = φ_len + DC + θ_len + DC + routing = 300 μm
    """
    import gdspy

    cell_name = f"chip{chip_id}_mzi_r{mzi_row}_c{mzi_col}"
    if cell_name in lib.cells:
        return cell_name

    cell = lib.new_cell(cell_name)

    # Absolute chip position for this MZI
    stage_pitch = TOTAL_MZI_LEN_UM + STAGE_GAP_UM
    x_origin = CHIP_MARGIN_UM + mzi_col * stage_pitch
    # Row index = top mode of the pair; y increases upward in GDS
    y_top = CHIP_MARGIN_UM + mzi_row * MODE_PITCH_UM + MODE_PITCH_UM  # upper wg
    y_bot = CHIP_MARGIN_UM + mzi_row * MODE_PITCH_UM                  # lower wg

    # ── φ heater (left, on upper waveguide before DC1) ───────────────────────
    phi_x0 = x_origin
    phi_x1 = phi_x0 + PHI_LEN_UM
    _rect(cell,
          phi_x0, y_top - HEATER_WIDTH_UM / 2,
          phi_x1, y_top + HEATER_WIDTH_UM / 2,
          L_PHI)
    # φ background waveguides
    _waveguide(cell, phi_x0, y_top, phi_x1, y_top)
    _waveguide(cell, phi_x0, y_bot, phi_x1, y_bot)

    # ── DC1 (directional coupler) ─────────────────────────────────────────────
    dc1_x0 = phi_x1
    dc1_x1 = dc1_x0 + DC_LENGTH_UM
    # Si coupling region — two waveguides come close together
    dc_y_top = y_bot + WG_SPACING_UM / 2 + DC_GAP_UM / 2 + WG_WIDTH_UM / 2
    dc_y_bot = y_bot + WG_SPACING_UM / 2 - DC_GAP_UM / 2 - WG_WIDTH_UM / 2
    _rect(cell, dc1_x0, dc_y_bot, dc1_x1, dc_y_top, L_SI)

    # ── θ heater (main phase, on upper waveguide between DCs) ─────────────────
    theta_x0 = dc1_x1
    theta_x1 = theta_x0 + THETA_LEN_UM
    _rect(cell,
          theta_x0, y_top - HEATER_WIDTH_UM / 2,
          theta_x1, y_top + HEATER_WIDTH_UM / 2,
          L_THETA)
    _waveguide(cell, theta_x0, y_top, theta_x1, y_top)
    _waveguide(cell, theta_x0, y_bot, theta_x1, y_bot)

    # ── DC2 ───────────────────────────────────────────────────────────────────
    dc2_x0 = theta_x1
    dc2_x1 = dc2_x0 + DC_LENGTH_UM
    _rect(cell, dc2_x0, dc_y_bot, dc2_x1, dc_y_top, L_SI)

    # ── Routing waveguides (remaining length to fill TOTAL_MZI_LEN_UM) ────────
    route_x0 = dc2_x1
    route_x1 = x_origin + TOTAL_MZI_LEN_UM
    if route_x1 > route_x0:
        _waveguide(cell, route_x0, y_top, route_x1, y_top)
        _waveguide(cell, route_x0, y_bot, route_x1, y_bot)

    # ── Metal contacts on heaters ─────────────────────────────────────────────
    contact_w = 4.0
    contact_h = 4.0
    # θ contacts
    _rect(cell,
          theta_x0, y_top + HEATER_WIDTH_UM / 2,
          theta_x0 + contact_w, y_top + HEATER_WIDTH_UM / 2 + contact_h,
          L_METAL)
    _rect(cell,
          theta_x1 - contact_w, y_top + HEATER_WIDTH_UM / 2,
          theta_x1, y_top + HEATER_WIDTH_UM / 2 + contact_h,
          L_METAL)
    # φ contacts
    _rect(cell,
          phi_x0, y_top + HEATER_WIDTH_UM / 2,
          phi_x0 + contact_w, y_top + HEATER_WIDTH_UM / 2 + contact_h,
          L_METAL)
    _rect(cell,
          phi_x1 - contact_w, y_top + HEATER_WIDTH_UM / 2,
          phi_x1, y_top + HEATER_WIDTH_UM / 2 + contact_h,
          L_METAL)

    return cell_name


# ── Per-chip GDS generator ──────────────────────────────────────────────────────

def generate_chip_gds(
    entries: List[MZIPhaseEntry],
    chip_id: int,
    output_path: Path,
    dac_bits: int = 12,
) -> dict:
    """
    Generate a GDS-II file for one photonic chip.

    Args:
        entries:     All MZIPhaseEntry records for this chip.
        chip_id:     Chip index.
        output_path: Where to write the .gds file.
        dac_bits:    DAC resolution (for phase reconstruction).

    Returns:
        dict with chip statistics (n_mzis, width_um, height_um, path).
    """
    try:
        import gdspy
    except ImportError:
        logger.error("gdspy not installed. Run: pip install gdspy")
        return {}

    # ── Separate MZI entries from phase-screen entries ─────────────────────────
    mzi_entries = [e for e in entries if e.mzi_row >= 0]
    if not mzi_entries:
        logger.warning(f"Chip {chip_id}: no MZI entries — skipping.")
        return {}

    max_code = (1 << dac_bits) - 1

    # ── Compute chip dimensions ───────────────────────────────────────────────
    n_modes = max(e.mzi_row for e in mzi_entries) + 2  # +2: row is top mode of pair
    n_cols  = max(e.mzi_col for e in mzi_entries) + 1
    stage_pitch = TOTAL_MZI_LEN_UM + STAGE_GAP_UM
    chip_w = 2 * CHIP_MARGIN_UM + n_cols * stage_pitch
    chip_h = 2 * CHIP_MARGIN_UM + n_modes * MODE_PITCH_UM

    logger.info(
        f"Chip {chip_id}: {n_modes} modes × {n_cols} cols → "
        f"{chip_w/1000:.1f}mm × {chip_h/1000:.1f}mm ({len(mzi_entries):,} MZIs)"
    )

    # ── Build GDS library ──────────────────────────────────────────────────────
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)  # units in μm

    top = lib.new_cell(f"CHIP_{chip_id:03d}")

    # Chip boundary
    _rect(top, 0, 0, chip_w, chip_h, L_BOUND)
    _label(top, f"PhotoMedGemma Chip {chip_id}", 10, chip_h - 20, size=20)
    _label(top, f"{n_modes} modes  {n_cols} stages  {len(mzi_entries):,} MZIs",
           10, chip_h - 45, size=10)

    # ── Place each MZI ─────────────────────────────────────────────────────────
    n_placed = 0
    for entry in mzi_entries:
        theta = entry.theta_dac / max_code * (2 * math.pi)
        phi   = entry.phi_dac   / max_code * (2 * math.pi)
        cell_name = _mzi_cell(lib, chip_id, entry.mzi_row, entry.mzi_col, theta, phi)
        # Reference the cell (it's already at absolute coordinates)
        ref = gdspy.CellReference(lib.cells[cell_name], (0, 0))
        top.add(ref)
        n_placed += 1

    # ── Phase screen (left-edge circles represented as rectangles) ────────────
    screen_entries = [e for e in entries if e.mzi_row < 0]
    for e in screen_entries:
        mode_idx = e.mzi_col
        y_c = CHIP_MARGIN_UM + mode_idx * MODE_PITCH_UM + MODE_PITCH_UM / 2
        x_c = CHIP_MARGIN_UM / 2  # left margin
        phase = e.theta_dac / max_code * (2 * math.pi)
        # Draw as a small square color-coded by phase value
        s = 20.0  # 20μm box
        _rect(top, x_c - s/2, y_c - s/2, x_c + s/2, y_c + s/2, L_PHI)
        _label(top, f"{math.degrees(phase):.0f}deg", x_c + s/2 + 2, y_c, size=4)

    # ── Write GDS ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(output_path))
    logger.info(f"  → GDS written: {output_path}  ({output_path.stat().st_size//1024} KB)")

    return {
        "chip_id":   chip_id,
        "n_mzis":    n_placed,
        "n_modes":   n_modes,
        "n_cols":    n_cols,
        "width_mm":  chip_w / 1000,
        "height_mm": chip_h / 1000,
        "gds_path":  str(output_path),
    }


# ── KLayout layer properties file ─────────────────────────────────────────────

KLAYOUT_LYPROPS = """\
<?xml version="1.0" encoding="utf-8"?>
<layer-properties>
 <properties>
  <frame-color>#00aaff</frame-color>
  <fill-color>#00aaff</fill-color>
  <name>Si waveguide</name>
  <source>1/0@1</source>
 </properties>
 <properties>
  <frame-color>#ff4444</frame-color>
  <fill-color>#ff4444</fill-color>
  <name>TiN heater theta</name>
  <source>2/0@1</source>
 </properties>
 <properties>
  <frame-color>#ff9900</frame-color>
  <fill-color>#ff9900</fill-color>
  <name>TiN heater phi</name>
  <source>3/0@1</source>
 </properties>
 <properties>
  <frame-color>#ffdd00</frame-color>
  <fill-color>#ffdd00</fill-color>
  <name>Metal contacts</name>
  <source>4/0@1</source>
 </properties>
 <properties>
  <frame-color>#aaaaaa</frame-color>
  <fill-color>#00000000</fill-color>
  <name>Chip boundary</name>
  <source>5/0@1</source>
 </properties>
 <properties>
  <frame-color>#ffffff</frame-color>
  <fill-color>#00000000</fill-color>
  <name>Labels</name>
  <source>6/0@1</source>
 </properties>
</layer-properties>
"""


def write_klayout_props(output_dir: Path):
    """Write a KLayout layer properties file for nice rendering."""
    props_path = output_dir / "photomedgemma.lyp"
    props_path.write_text(KLAYOUT_LYPROPS)
    logger.info(f"KLayout layer properties: {props_path}")
    return props_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export PhotoMedGemma chip GDS-II masks for KLayout"
    )
    parser.add_argument(
        "--phase-map", required=True,
        help="Compiled phase map JSON (output of compile_model.py)"
    )
    parser.add_argument(
        "--output-dir", default="output/gds",
        help="Output directory for .gds files (default: output/gds)"
    )
    parser.add_argument(
        "--chip-id", type=int, default=None,
        help="Only export this chip index (default: all chips)"
    )
    parser.add_argument(
        "--dac-bits", type=int, default=12,
        help="DAC resolution (must match compilation)"
    )
    parser.add_argument(
        "--max-mzis", type=int, default=4096,
        help=(
            "Max MZIs to render per chip GDS (default: 4096). "
            "A full 2560×2560 mesh has 3.27M MZIs — KLayout cannot display that. "
            "Use 0 for no limit (produces very large files)."
        )
    )
    args = parser.parse_args()

    phase_map_path = Path(args.phase_map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load phase map — support both old (entries) and new (layers+npz) format
    logger.info(f"Loading phase map: {phase_map_path}")
    with open(phase_map_path) as f:
        pm_data = json.load(f)

    by_chip: Dict[int, List[MZIPhaseEntry]] = {}
    model_id = pm_data.get("model_id", "unknown")

    if "entries" in pm_data:
        # Old format: inline MZI entries
        phase_map = PhaseMap.load_json(args.phase_map)
        for e in phase_map.entries:
            by_chip.setdefault(e.chip_id, []).append(e)
        logger.info(
            f"  {len(phase_map.entries):,} entries (old format), "
            f"{len(by_chip)} chips"
        )

    else:
        # New format: phases stored in per-layer NPZ files
        base_dir = phase_map_path.parent
        total_entries = 0
        for layer in pm_data.get("layers", []):
            for mesh_key, chip_id_key in [("phase_file_U", "chip_id_U"),
                                           ("phase_file_Vh", "chip_id_Vh")]:
                npz_rel = layer.get(mesh_key)
                chip_id = layer.get(chip_id_key)
                if npz_rel is None or chip_id is None:
                    continue
                npz_path = base_dir / npz_rel
                if not npz_path.exists():
                    logger.warning(f"  NPZ not found: {npz_path} — skipping")
                    continue
                data = np.load(str(npz_path))
                rows      = data["row"].astype(int)
                cols      = data["col"].astype(int)
                th_dac    = data["theta_dac"].astype(int)
                phi_dac   = data["phi_dac"].astype(int)
                ps        = data["phase_screen_dac"].astype(int)  # phase screen
                n = len(rows)
                if args.max_mzis > 0:
                    n = min(n, args.max_mzis)
                mode_i = data["mode_i"].astype(int)
                mode_j = data["mode_j"].astype(int)
                theta_rad = data["theta"].astype(float)
                phi_rad   = data["phi"].astype(float)
                mtype = "U" if mesh_key == "phase_file_U" else "Vh"
                lname = layer.get("name", "unknown")
                for i in range(n):
                    by_chip.setdefault(chip_id, []).append(
                        MZIPhaseEntry(
                            chip_id=chip_id,
                            mzi_row=int(rows[i]),
                            mzi_col=int(cols[i]),
                            mode_i=int(mode_i[i]),
                            mode_j=int(mode_j[i]),
                            theta_rad=float(theta_rad[i]),
                            phi_rad=float(phi_rad[i]),
                            theta_dac=int(th_dac[i]),
                            phi_dac=int(phi_dac[i]),
                            layer_name=lname,
                            matrix_type=mtype,
                        )
                    )
                total_entries += n
                # Phase screen entries (mzi_row = -1)
                for idx, phase in enumerate(ps):
                    by_chip.setdefault(chip_id, []).append(
                        MZIPhaseEntry(
                            chip_id=chip_id,
                            mzi_row=-1,
                            mzi_col=idx,
                            mode_i=idx,
                            mode_j=idx,
                            theta_rad=float(phase) * 2 * np.pi / ((1 << args.dac_bits) - 1),
                            phi_rad=0.0,
                            theta_dac=int(phase),
                            phi_dac=0,
                            layer_name=lname,
                            matrix_type=mtype,
                        )
                    )
        logger.info(
            f"  {total_entries:,} MZI entries loaded from NPZ (new format, "
            f"capped at {args.max_mzis}/chip), {len(by_chip)} chips"
        )

    chip_ids = sorted(by_chip.keys())
    if args.chip_id is not None:
        if args.chip_id not in by_chip:
            logger.error(f"Chip {args.chip_id} not in phase map (available: {chip_ids})")
            sys.exit(1)
        chip_ids = [args.chip_id]

    # ── Generate GDS for each chip ─────────────────────────────────────────────
    manifest = []
    for chip_id in chip_ids:
        gds_path = out_dir / f"chip_{chip_id:03d}.gds"
        stats = generate_chip_gds(
            by_chip[chip_id], chip_id, gds_path,
            dac_bits=args.dac_bits,
        )
        if stats:
            manifest.append(stats)

    # ── Write KLayout layer properties ────────────────────────────────────────
    write_klayout_props(out_dir)

    # ── Write manifest ─────────────────────────────────────────────────────────
    manifest_path = out_dir / "gds_manifest.json"
    manifest_data = {
        "model_id":  model_id,
        "dac_bits":  args.dac_bits,
        "n_chips":   len(manifest),
        "total_mzis": sum(c["n_mzis"] for c in manifest),
        "chips":     manifest,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GDS Export Complete")
    print("=" * 60)
    print(f"  Chips exported : {len(manifest)}")
    print(f"  Total MZIs     : {manifest_data['total_mzis']:,}")
    print(f"\nTo view in KLayout:")
    print(f"  klayout {out_dir}/chip_000.gds -l {out_dir}/photomedgemma.lyp")
    print(f"\nOutput directory: {out_dir}/")
    for c in manifest:
        print(
            f"  chip_{c['chip_id']:03d}.gds  "
            f"{c['n_modes']} modes  "
            f"{c['width_mm']:.1f}mm × {c['height_mm']:.1f}mm  "
            f"({c['n_mzis']:,} MZIs)"
        )


if __name__ == "__main__":
    main()
