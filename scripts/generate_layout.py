"""
generate_layout.py — Photonic Chip Layout Generator
=====================================================

Generates GDS-II layout files for the PhotoMedGemma photonic chip.

Takes a compiled phase map (.json or .phcfg) and produces:
  1. Per-chip GDS-II layout (requires gdsfactory)
  2. SVG overview diagram (no dependencies, always runs)
  3. JSON layout manifest with physical dimensions and MZI placement

Physical design targets (220nm SOI, 1310nm):
  - MZI pitch:        127 μm  (standard fiber array pitch)
  - MZI length:       300 μm  (includes 2× phase shifters + 2× DCs)
  - Waveguide width:  450 nm
  - Heater:           TiN, 15mW/π, 100 μm long

Usage
-----
    # Generate layout from a compiled phase map
    python3 scripts/generate_layout.py --phase-map output/phase_map_demo.json

    # Target a specific chip
    python3 scripts/generate_layout.py --phase-map output/phase_map_demo.json --chip-id 0

    # SVG only (no gdsfactory required)
    python3 scripts/generate_layout.py --phase-map output/phase_map_demo.json --svg-only
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
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


# ── Physical constants ─────────────────────────────────────────────────────────

# All dimensions in micrometers (μm)
MZI_PITCH_UM     = 127.0   # vertical spacing between adjacent waveguide modes (fiber array pitch)
MZI_LENGTH_UM    = 300.0   # horizontal length of one MZI element (θ + φ + 2×DC)
STAGE_GAP_UM     = 20.0    # horizontal gap between MZI stages
CHIP_MARGIN_UM   = 500.0   # border margin around the active area
HEATER_WIDTH_UM  = 2.0     # TiN heater stripe width
HEATER_LENGTH_UM = 100.0   # TiN heater length
WG_WIDTH_NM      = 450     # waveguide width in nanometers


def mzi_position(row: int, col: int, stage_pitch_x: float = None) -> Tuple[float, float]:
    """
    Compute the (x, y) center position of an MZI at (row, col).

    Args:
        row: MZI row (0 = top mode pair)
        col: MZI column / stage index
        stage_pitch_x: Horizontal stage pitch. Defaults to MZI_LENGTH_UM + STAGE_GAP_UM.

    Returns:
        (x_um, y_um) center position
    """
    if stage_pitch_x is None:
        stage_pitch_x = MZI_LENGTH_UM + STAGE_GAP_UM
    x = CHIP_MARGIN_UM + col * stage_pitch_x
    y = CHIP_MARGIN_UM + row * MZI_PITCH_UM
    return x, y


def chip_dimensions(n_modes: int, n_cols: int) -> Tuple[float, float]:
    """
    Compute chip bounding box size.

    The Reck triangular mesh has n_modes-1 physical columns (0 to N-2),
    each of different height. The bounding box spans all columns and all
    mode rows.

    Args:
        n_modes: Number of optical modes (N); physical cols = n_modes - 1
        n_cols:  Number of distinct MZI columns (= n_modes - 1 for full mesh)

    Returns:
        (width_um, height_um)
    """
    stage_pitch_x = MZI_LENGTH_UM + STAGE_GAP_UM
    width  = 2 * CHIP_MARGIN_UM + n_cols * stage_pitch_x
    height = 2 * CHIP_MARGIN_UM + (n_modes - 1) * MZI_PITCH_UM
    return width, height


# ── SVG Layout Generation ──────────────────────────────────────────────────────

def phase_to_color(dac_code: int, max_code: int = 4095) -> str:
    """Map a DAC code to an SVG color (blue=0, red=max)."""
    t = dac_code / max_code
    r = int(255 * t)
    b = int(255 * (1 - t))
    return f"rgb({r},64,{b})"


def generate_svg(
    entries: List[MZIPhaseEntry],
    chip_id: int,
    output_path: Path,
    dac_bits: int = 12,
) -> None:
    """
    Generate an SVG visualisation of the MZI mesh for one chip.

    Each MZI is drawn as a coloured rectangle; colour encodes θ (hue = 0→blue, 2π→red).
    Phase-screen entries (mzi_row == -1) are shown as circles at the left edge.

    Args:
        entries:     Phase map entries for this chip
        chip_id:     Chip identifier
        output_path: Where to write the SVG file
        dac_bits:    DAC resolution
    """
    max_code = (1 << dac_bits) - 1

    # Filter to normal MZI entries (exclude phase screen)
    mzi_entries    = [e for e in entries if e.mzi_row >= 0]
    screen_entries = [e for e in entries if e.mzi_row == -1]

    if not mzi_entries:
        logger.warning(f"No MZI entries for chip {chip_id}; nothing to draw.")
        return

    # Determine grid bounds
    # mzi_row = mode_i (0 to N-2), mzi_col = physical column (0 to N-2)
    max_row   = max(e.mzi_row for e in mzi_entries)
    max_col   = max(e.mzi_col for e in mzi_entries)
    n_modes   = max_row + 2           # N = max_mode_i + 2 (modes are 0..N-1)
    n_cols    = max_col + 1           # physical mesh columns (0 to N-2)

    # SVG scale: pixels per micron
    PX_PER_UM = 0.8
    stage_pitch_x = MZI_LENGTH_UM + STAGE_GAP_UM

    chip_w, chip_h = chip_dimensions(n_modes, n_cols)
    svg_w = chip_w * PX_PER_UM
    svg_h = chip_h * PX_PER_UM

    mzi_w_px = MZI_LENGTH_UM * PX_PER_UM
    mzi_h_px = 0.6 * MZI_PITCH_UM * PX_PER_UM

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" height="{svg_h:.0f}">',
        '<rect width="100%" height="100%" fill="#0a0a1a"/>',
        f'<text x="20" y="24" font-family="monospace" font-size="14" fill="#ccc">',
        f'  PhotoMedGemma — Chip {chip_id} MZI Mesh ({n_modes}×{n_modes}, {len(mzi_entries)} MZIs)',
        f'</text>',
    ]

    # Draw waveguide rails (horizontal lines, one per mode)
    for mode in range(n_modes):
        y = (CHIP_MARGIN_UM + mode * MZI_PITCH_UM) * PX_PER_UM
        lines.append(
            f'<line x1="{CHIP_MARGIN_UM * PX_PER_UM * 0.5:.1f}" y1="{y:.1f}" '
            f'x2="{(chip_w - CHIP_MARGIN_UM * 0.5) * PX_PER_UM:.1f}" y2="{y:.1f}" '
            f'stroke="#334" stroke-width="1"/>'
        )

    # Draw MZIs
    for e in mzi_entries:
        x_c, y_c = mzi_position(e.mzi_row, e.mzi_col, stage_pitch_x)
        x_px = x_c * PX_PER_UM - mzi_w_px / 2
        y_px = y_c * PX_PER_UM - mzi_h_px / 2

        theta_color = phase_to_color(e.theta_dac, max_code)
        phi_color   = phase_to_color(e.phi_dac, max_code)

        # θ half (left)
        lines.append(
            f'<rect x="{x_px:.1f}" y="{y_px:.1f}" '
            f'width="{mzi_w_px/2:.1f}" height="{mzi_h_px:.1f}" '
            f'fill="{theta_color}" rx="2" opacity="0.9" '
            f'><title>MZI({e.mzi_row},{e.mzi_col}) θ={e.theta_rad:.3f}rad '
            f'[{e.layer_name}]</title></rect>'
        )
        # φ half (right)
        lines.append(
            f'<rect x="{x_px + mzi_w_px/2:.1f}" y="{y_px:.1f}" '
            f'width="{mzi_w_px/2:.1f}" height="{mzi_h_px:.1f}" '
            f'fill="{phi_color}" rx="2" opacity="0.9" '
            f'><title>MZI({e.mzi_row},{e.mzi_col}) φ={e.phi_rad:.3f}rad</title></rect>'
        )

    # Draw phase screen circles (left edge)
    for e in screen_entries:
        cx = (CHIP_MARGIN_UM * 0.7) * PX_PER_UM
        cy = (CHIP_MARGIN_UM + e.mzi_col * MZI_PITCH_UM) * PX_PER_UM
        r_px = 0.3 * MZI_PITCH_UM * PX_PER_UM
        color = phase_to_color(e.theta_dac, max_code)
        lines.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_px:.1f}" '
            f'fill="{color}" opacity="0.85" '
            f'><title>Phase screen mode {e.mzi_col} φ={e.theta_rad:.3f}rad</title></circle>'
        )

    # Color scale legend
    legend_x = svg_w - 220
    legend_y = 40
    lines.append(
        f'<text x="{legend_x:.0f}" y="{legend_y:.0f}" '
        f'font-family="monospace" font-size="11" fill="#aaa">θ (left) / φ (right) :</text>'
    )
    n_steps = 20
    for i in range(n_steps):
        t = i / (n_steps - 1)
        dac_v = int(t * max_code)
        color = phase_to_color(dac_v, max_code)
        lx = legend_x + i * (150 / n_steps)
        lines.append(
            f'<rect x="{lx:.1f}" y="{legend_y + 6:.0f}" '
            f'width="{150/n_steps + 0.5:.1f}" height="10" fill="{color}"/>'
        )
    lines.append(
        f'<text x="{legend_x:.0f}" y="{legend_y + 26:.0f}" '
        f'font-family="monospace" font-size="10" fill="#888">0</text>'
    )
    lines.append(
        f'<text x="{legend_x + 136:.0f}" y="{legend_y + 26:.0f}" '
        f'font-family="monospace" font-size="10" fill="#888">2π</text>'
    )

    lines.append('</svg>')

    output_path.write_text('\n'.join(lines))
    logger.info(f"SVG layout written to {output_path} "
                f"({svg_w:.0f}×{svg_h:.0f} px, {len(mzi_entries)} MZIs)")


# ── GDS Layout Generation ──────────────────────────────────────────────────────

def _make_mzi_cell(gf, dac_bits: int = 12):
    """
    Create a parametric MZI cell in gdsfactory.

    Returns a gf.Component function that takes (theta_dac, phi_dac) and
    places the appropriate heater polygons for the phase settings.
    """
    import gdsfactory as gf

    @gf.cell
    def mzi_unit(
        theta_dac: int = 0,
        phi_dac: int = 0,
        wg_width: float = WG_WIDTH_NM / 1000,    # convert nm → μm
        length: float = MZI_LENGTH_UM,
    ) -> gf.Component:
        """Single MZI unit cell with heaters."""
        c = gf.Component()

        # Directional coupler (50:50 beamsplitter) — simple rectangle placeholder
        dc_len = 20.0
        gap    = 0.2  # μm gap between waveguides in DC

        # Top waveguide
        c.add_polygon(
            [(0, 0), (length, 0), (length, wg_width), (0, wg_width)],
            layer=(1, 0)  # CORE layer
        )
        # Bottom waveguide
        y_bot = -(gap + wg_width)
        c.add_polygon(
            [(0, y_bot), (length, y_bot),
             (length, y_bot + wg_width), (0, y_bot + wg_width)],
            layer=(1, 0)
        )

        # θ heater (top waveguide, left half)
        # Heater length scales with phase: l_heater ∝ theta_dac / max_code
        max_code = (1 << dac_bits) - 1
        theta_frac = theta_dac / max_code
        phi_frac   = phi_dac   / max_code

        theta_heater_len = HEATER_LENGTH_UM * theta_frac
        if theta_heater_len > 0.5:
            c.add_polygon(
                [
                    (length * 0.1, wg_width),
                    (length * 0.1 + theta_heater_len, wg_width),
                    (length * 0.1 + theta_heater_len, wg_width + HEATER_WIDTH_UM),
                    (length * 0.1, wg_width + HEATER_WIDTH_UM),
                ],
                layer=(2, 0)  # HEATER layer
            )

        # φ heater (top waveguide, right half)
        phi_heater_len = HEATER_LENGTH_UM * phi_frac
        if phi_heater_len > 0.5:
            c.add_polygon(
                [
                    (length * 0.55, wg_width),
                    (length * 0.55 + phi_heater_len, wg_width),
                    (length * 0.55 + phi_heater_len, wg_width + HEATER_WIDTH_UM),
                    (length * 0.55, wg_width + HEATER_WIDTH_UM),
                ],
                layer=(2, 0)
            )

        return c

    return mzi_unit


def generate_gds(
    entries: List[MZIPhaseEntry],
    chip_id: int,
    output_path: Path,
    dac_bits: int = 12,
) -> None:
    """
    Generate a GDS-II layout file for one chip using gdsfactory.

    Args:
        entries:     Phase map entries for this chip
        chip_id:     Chip identifier
        output_path: Where to write the .gds file
        dac_bits:    DAC resolution
    """
    try:
        import gdsfactory as gf
    except ImportError:
        logger.warning(
            "gdsfactory not installed. Skipping GDS generation. "
            "Install with: pip install gdsfactory"
        )
        return

    logger.info(f"Generating GDS layout for chip {chip_id}...")

    mzi_entries = [e for e in entries if e.mzi_row >= 0]
    if not mzi_entries:
        logger.warning(f"No MZI entries for chip {chip_id}; skipping GDS.")
        return

    mzi_cell = _make_mzi_cell(gf, dac_bits)

    # Top-level chip component
    chip = gf.Component(f"photomedgemma_chip_{chip_id}")

    # Add chip boundary
    max_row = max(e.mzi_row for e in mzi_entries)
    max_col = max(e.mzi_col for e in mzi_entries)
    n_modes = max_row + 2
    n_cols  = max_col + 1
    w, h = chip_dimensions(n_modes, n_cols)

    chip.add_polygon(
        [(0, 0), (w, 0), (w, h), (0, h)],
        layer=(99, 0)  # FLOORPLAN layer
    )

    # Place MZI cells
    stage_pitch_x = MZI_LENGTH_UM + STAGE_GAP_UM
    n_placed = 0
    for e in mzi_entries:
        x, y = mzi_position(e.mzi_row, e.mzi_col, stage_pitch_x)
        cell = mzi_cell(theta_dac=e.theta_dac, phi_dac=e.phi_dac)
        ref = chip << cell
        ref.move((x, y))
        n_placed += 1

    chip.write_gds(str(output_path))
    logger.info(f"GDS layout written to {output_path} ({n_placed} MZIs placed)")


# ── Layout Manifest ─────────────────────────────────────────────────────────────

def generate_manifest(
    phase_map: PhaseMap,
    output_path: Path,
) -> dict:
    """
    Generate a JSON layout manifest with physical placement data.

    Args:
        phase_map:   Loaded PhaseMap
        output_path: Where to write the manifest JSON

    Returns:
        Manifest dictionary
    """
    chip_ids = sorted(set(e.chip_id for e in phase_map.entries if e.mzi_row >= 0))
    stage_pitch_x = MZI_LENGTH_UM + STAGE_GAP_UM

    chips_info = []
    for chip_id in chip_ids:
        entries = phase_map.entries_for_chip(chip_id)
        mzi_entries = [e for e in entries if e.mzi_row >= 0]
        if not mzi_entries:
            continue

        max_row  = max(e.mzi_row for e in mzi_entries)
        max_col  = max(e.mzi_col for e in mzi_entries)
        n_modes  = max_row + 2   # N modes (mode_i goes 0..N-2, so N = max+2)
        n_cols   = max_col + 1   # physical mesh columns (= N-1 for full mesh)
        w, h = chip_dimensions(n_modes, n_cols)

        # Unique layers on this chip
        layers = sorted(set(e.layer_name for e in mzi_entries))

        chips_info.append({
            "chip_id":         chip_id,
            "n_mzis":          len(mzi_entries),
            "n_modes":         n_modes,
            "n_cols":          n_cols,
            "chip_width_um":   round(w, 2),
            "chip_height_um":  round(h, 2),
            "chip_area_mm2":   round(w * h / 1e6, 4),
            "layers":          layers,
            "matrix_types":    sorted(set(e.matrix_type for e in mzi_entries)),
            "mzi_pitch_um":    MZI_PITCH_UM,
            "mzi_length_um":   MZI_LENGTH_UM,
        })

    total_area_mm2 = sum(c["chip_area_mm2"] for c in chips_info)

    manifest = {
        "model_id":        phase_map.model_id,
        "rank":            phase_map.rank,
        "dac_bits":        phase_map.dac_bits,
        "n_chips":         len(chips_info),
        "total_mzis":      phase_map.n_mzis,
        "total_area_mm2":  round(total_area_mm2, 4),
        "platform": {
            "process":       "220nm SOI",
            "wavelength_nm": 1310,
            "wg_width_nm":   WG_WIDTH_NM,
            "mzi_pitch_um":  MZI_PITCH_UM,
            "mzi_length_um": MZI_LENGTH_UM,
            "heater":        "TiN thermal phase shifter",
            "pi_power_mW":   15.0,
        },
        "chips": chips_info,
    }

    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info(
        f"Layout manifest written to {output_path} "
        f"({len(chips_info)} chips, {total_area_mm2:.3f} mm² total)"
    )
    return manifest


# ── Chip Summary ────────────────────────────────────────────────────────────────

def print_chip_summary(manifest: dict) -> None:
    """Print a human-readable summary of the chip layout."""
    print()
    print("=" * 62)
    print(f"  PhotoMedGemma Chip Layout Summary")
    print("=" * 62)
    print(f"  Model:         {manifest['model_id']}")
    print(f"  SVD rank:      {manifest['rank']}")
    print(f"  DAC resolution:{manifest['dac_bits']} bits")
    print(f"  Total MZIs:    {manifest['total_mzis']:,}")
    print(f"  Total chips:   {manifest['n_chips']}")
    print(f"  Total area:    {manifest['total_area_mm2']:.3f} mm²")
    print()
    print(f"  Platform: {manifest['platform']['process']}, "
          f"λ={manifest['platform']['wavelength_nm']}nm")
    print(f"  MZI pitch: {manifest['platform']['mzi_pitch_um']} μm, "
          f"length: {manifest['platform']['mzi_length_um']} μm")
    print()
    print(f"  {'ChipID':>6}  {'MZIs':>8}  {'Modes':>6}  {'Cols':>6}  "
          f"{'W (mm)':>7}  {'H (mm)':>7}  {'Area (mm²)':>10}")
    print(f"  {'──────':>6}  {'────':>8}  {'─────':>6}  {'────':>6}  "
          f"{'──────':>7}  {'──────':>7}  {'──────────':>10}")
    for chip in manifest['chips']:
        print(
            f"  {chip['chip_id']:>6}  {chip['n_mzis']:>8,}  "
            f"{chip['n_modes']:>6}  {chip['n_cols']:>6}  "
            f"{chip['chip_width_um']/1000:>7.1f}  {chip['chip_height_um']/1000:>7.1f}  "
            f"{chip['chip_area_mm2']:>10.4f}"
        )
    print("=" * 62)
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate photonic chip layout from compiled phase map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase-map",
        default="output/phase_map_demo.json",
        help="Path to compiled phase map JSON (default: output/phase_map_demo.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/layout",
        help="Output directory for layout files (default: output/layout/)",
    )
    parser.add_argument(
        "--chip-id",
        type=int,
        default=None,
        help="Generate layout for specific chip ID only (default: all chips)",
    )
    parser.add_argument(
        "--svg-only",
        action="store_true",
        help="Generate SVG diagrams only (skip GDS, no gdsfactory required)",
    )
    parser.add_argument(
        "--max-chips",
        type=int,
        default=4,
        help="Maximum number of chips to render (default: 4, to limit file size)",
    )
    args = parser.parse_args()

    # ── Load phase map ─────────────────────────────────────────────────────────
    phase_map_path = ROOT / args.phase_map
    if not phase_map_path.exists():
        # Try relative to CWD
        phase_map_path = Path(args.phase_map)
    if not phase_map_path.exists():
        logger.error(f"Phase map not found: {args.phase_map}")
        logger.error("Run compile_model.py first to generate a phase map.")
        sys.exit(1)

    logger.info(f"Loading phase map from {phase_map_path}...")
    phase_map = PhaseMap.load_json(str(phase_map_path))
    logger.info(
        f"Loaded: {phase_map.n_mzis:,} phase entries across "
        f"{phase_map.n_chips} chips, rank={phase_map.rank}"
    )

    # ── Set up output directory ────────────────────────────────────────────────
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate layout manifest ───────────────────────────────────────────────
    manifest_path = out_dir / "layout_manifest.json"
    manifest = generate_manifest(phase_map, manifest_path)
    print_chip_summary(manifest)

    # ── Determine which chips to render ───────────────────────────────────────
    all_chip_ids = sorted(set(e.chip_id for e in phase_map.entries if e.mzi_row >= 0))
    if args.chip_id is not None:
        chip_ids_to_render = [args.chip_id]
    else:
        chip_ids_to_render = all_chip_ids[:args.max_chips]
        if len(all_chip_ids) > args.max_chips:
            logger.info(
                f"Rendering first {args.max_chips} of {len(all_chip_ids)} chips "
                f"(use --max-chips N to render more)"
            )

    # ── Generate per-chip layouts ──────────────────────────────────────────────
    svg_paths = []
    gds_paths = []

    for chip_id in chip_ids_to_render:
        entries = phase_map.entries_for_chip(chip_id)
        if not entries:
            logger.warning(f"Chip {chip_id}: no entries found, skipping.")
            continue

        # SVG (always)
        svg_path = out_dir / f"chip_{chip_id:03d}.svg"
        generate_svg(entries, chip_id, svg_path, dac_bits=phase_map.dac_bits)
        svg_paths.append(svg_path)

        # GDS (if not --svg-only)
        if not args.svg_only:
            gds_path = out_dir / f"chip_{chip_id:03d}.gds"
            generate_gds(entries, chip_id, gds_path, dac_bits=phase_map.dac_bits)
            gds_paths.append(gds_path)

    # ── Generate ASCII overview ────────────────────────────────────────────────
    _print_ascii_layout(phase_map, all_chip_ids[:min(4, len(all_chip_ids))])

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("Output files:")
    print(f"  Manifest:   {manifest_path}")
    for p in svg_paths:
        print(f"  SVG:        {p}")
    for p in gds_paths:
        print(f"  GDS:        {p}")
    print()


def _print_ascii_layout(phase_map: PhaseMap, chip_ids: List[int]) -> None:
    """Print a small ASCII art view of the MZI mesh for quick inspection."""
    print("  MZI Phase Map — ASCII Preview")
    print("  (θ: . low   o mid   @ high   | = phase screen)")
    print()

    for chip_id in chip_ids:
        entries = phase_map.entries_for_chip(chip_id)
        mzi_entries = [e for e in entries if e.mzi_row >= 0]
        if not mzi_entries:
            continue

        max_row = max(e.mzi_row for e in mzi_entries)
        max_col = max(e.mzi_col for e in mzi_entries)

        # Build grid (cap display at 32×32)
        disp_rows = min(max_row + 1, 16)
        disp_cols = min(max_col + 1, 32)

        grid = [['  '] * disp_cols for _ in range(disp_rows)]
        max_code = (1 << phase_map.dac_bits) - 1

        for e in mzi_entries:
            if e.mzi_row < disp_rows and e.mzi_col < disp_cols:
                t = e.theta_dac / max_code
                ch = '.' if t < 0.33 else ('o' if t < 0.66 else '@')
                grid[e.mzi_row][e.mzi_col] = f"{ch} "

        print(f"  Chip {chip_id}  ({max_row+1} rows, {max_col+1} stages):")
        for row_idx, row in enumerate(grid):
            print(f"  {''.join(row)}")
        if max_row >= disp_rows or max_col >= disp_cols:
            print(f"  ... (showing {disp_rows}×{disp_cols} of "
                  f"{max_row+1}×{max_col+1})")
        print()


if __name__ == "__main__":
    main()
