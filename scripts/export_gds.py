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
MODE_PITCH_UM     = 127.0    # vertical spacing between adjacent mesh modes (fiber pitch)
DC_LENGTH_UM      = 10.0     # directional coupler length
DC_GAP_UM         = 0.200    # coupler gap (200 nm)
THETA_LEN_UM      = 100.0    # θ heater length
PHI_LEN_UM        = 40.0     # φ heater length
HEATER_WIDTH_UM   = 2.0      # TiN heater width
TOTAL_MZI_LEN_UM  = 300.0    # reference MZI cell length (scaled to fit chip)
STAGE_GAP_UM      = 20.0     # horizontal gap between stages
CHIP_MARGIN_UM    = 500.0    # chip border margin
CHIP_SIZE_UM      = 10_000.0 # target die size: 10mm × 10mm
GC_WIDTH_UM       = 15.0     # grating coupler pad width (schematic)
GC_LENGTH_UM      = 30.0     # grating coupler pad length
HEATER_ROUTING_H  = 300.0    # height of heater metal routing strip at chip bottom

# GDS layers — per-chip
L_SI      = 1   # silicon core
L_THETA   = 2   # TiN heater — θ phase
L_PHI     = 3   # TiN heater — φ phase
L_METAL   = 4   # metal contacts
L_BOUND   = 5   # chip boundary
L_LABEL   = 6   # text labels

# GDS layers — assembly (chiplet + interposer)
L_CHIPLET_BOUND  = 10  # chiplet group outline
L_INTERPOSER     = 11  # silicon interposer substrate
L_DICING_LANE    = 12  # dicing lanes between chiplets
L_FIBER_RIBBON   = 13  # inter-chip fiber ribbon routing
L_BOND_PAD_ARRAY = 14  # C4 micro-bump bond pad arrays
L_INTERPOSER_WG  = 15  # interposer waveguide routing
L_ASSEMBLY_LABEL = 16  # chiplet-level text annotations
L_STACK_CROSS    = 20  # 3D stack cross-section geometry
L_STACK_LABEL    = 21  # 3D stack cross-section labels


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


def _grating_coupler_column(cell, x_center: float, y_base: float,
                             n_modes: int, pitch: float, side: str):
    """Draw a column of grating coupler pads (schematic representation)."""
    import gdspy
    for i in range(n_modes):
        y = y_base + i * pitch
        # Grating coupler pad (Si layer)
        x0 = x_center - GC_WIDTH_UM / 2
        x1 = x_center + GC_WIDTH_UM / 2
        y0 = y - GC_LENGTH_UM / 2
        y1 = y + GC_LENGTH_UM / 2
        cell.add(gdspy.Rectangle((x0, y0), (x1, y1), layer=L_SI))
        # Routing waveguide connecting coupler to mesh edge
        wg_x0, wg_x1 = (x1, x1 + 30) if side == "left" else (x0 - 30, x0)
        hw = WG_WIDTH_UM / 2
        cell.add(gdspy.Rectangle((wg_x0, y - hw), (wg_x1, y + hw), layer=L_SI))
    label = "INPUT (64ch)" if side == "left" else "OUTPUT (64ch)"
    _label(cell, label, x_center - 20, y_base + n_modes * pitch + 10, size=8)


def _heater_routing_strip(cell, x0: float, y0: float, width: float):
    """Draw the metal heater routing strip at the bottom of the chip."""
    # Main metal routing zone
    _rect(cell, x0, y0, x0 + width, y0 + HEATER_ROUTING_H, L_METAL)
    # Bond pad row inside strip
    pad_w, pad_h = 60.0, 60.0
    pad_pitch = width / 34.0  # ~32 bond pads across chip width
    for i in range(33):
        px = x0 + pad_pitch * i + pad_pitch / 2 - pad_w / 2
        py = y0 + (HEATER_ROUTING_H - pad_h) / 2
        _rect(cell, px, py, px + pad_w, py + pad_h, L_METAL)
    _label(cell, "Heater routing + bond pads (1024 channels -> DAC)", x0 + 10, y0 + 10, size=6)


# ── Per-chip GDS generator ──────────────────────────────────────────────────────

def generate_chip_gds(
    entries: List[MZIPhaseEntry],
    chip_id: int,
    output_path: Path,
    dac_bits: int = 12,
    rank: int = 64,
) -> dict:
    """
    Generate a GDS-II file for one photonic chip.

    Layout follows chip_architecture.md: 10mm × 10mm die, 220nm SOI.
    Only the rank-64 active subspace is drawn (mode_i < rank, mode_j < rank),
    giving 64 waveguide rails at 127μm pitch (fiber-array standard).
    Stages are scaled to fit the active area width (8mm).

    Args:
        entries:     All MZIPhaseEntry records for this chip.
        chip_id:     Chip index.
        output_path: Where to write the .gds file.
        dac_bits:    DAC resolution.
        rank:        Number of active optical modes (default 64).

    Returns:
        dict with chip statistics.
    """
    try:
        import gdspy
    except ImportError:
        logger.error("gdspy not installed. Run: pip install gdspy")
        return {}

    max_code = (1 << dac_bits) - 1

    # ── Filter to rank-64 active subspace ──────────────────────────────────────
    # Only MZIs where both coupled modes are within the rank-R active subspace.
    # This gives the genuine N×N → rank×rank active Clements mesh.
    all_mzi  = [e for e in entries if e.mzi_row >= 0]
    active   = [e for e in all_mzi if e.mode_i < rank and e.mode_j < rank]

    if not active:
        # Fallback: remap all entries into rank-sized display space
        active = all_mzi
        logger.warning(f"Chip {chip_id}: no MZIs in rank-{rank} subspace — showing full mesh remapped.")

    # ── Chip geometry — fixed 10mm × 10mm die ─────────────────────────────────
    # Active mesh area: 8mm wide × (rank modes × 127μm) tall, centred in die.
    n_modes      = rank                             # 64 active mode rails
    mode_pitch   = MODE_PITCH_UM                    # 127 μm (fiber-array standard)
    mesh_h       = n_modes * mode_pitch             # 64 × 127 = 8128 μm ≈ 8.1 mm

    unique_cols  = sorted(set(e.mzi_col for e in active))
    n_stages     = len(unique_cols)
    col_index    = {c: i for i, c in enumerate(unique_cols)}  # compress column indices

    # Scale stage pitch so all stages fit within 8 mm active width
    available_w  = CHIP_SIZE_UM - 2 * CHIP_MARGIN_UM - 2 * (GC_LENGTH_UM + 40)
    stage_pitch  = min(TOTAL_MZI_LEN_UM + STAGE_GAP_UM,
                       available_w / max(n_stages, 1))
    mzi_len      = stage_pitch * TOTAL_MZI_LEN_UM / (TOTAL_MZI_LEN_UM + STAGE_GAP_UM)
    mesh_w       = n_stages * stage_pitch

    # Die boundary always 10mm × 10mm
    chip_w  = CHIP_SIZE_UM
    chip_h  = CHIP_SIZE_UM

    # Offsets: centre the mesh vertically, leave left/right margins for GCs
    gc_zone     = GC_LENGTH_UM + 40           # grating coupler zone width
    mesh_x0     = CHIP_MARGIN_UM + gc_zone    # mesh left edge
    mesh_y0     = (chip_h - HEATER_ROUTING_H - mesh_h) / 2 + HEATER_ROUTING_H

    logger.info(
        f"Chip {chip_id}: {n_modes} modes × {n_stages} stages  "
        f"stage_pitch={stage_pitch:.0f}μm  "
        f"mesh={mesh_w/1000:.1f}mm × {mesh_h/1000:.1f}mm  "
        f"({len(active):,} MZIs in rank-{rank} subspace)"
    )

    # ── Build GDS library ──────────────────────────────────────────────────────
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    top = lib.new_cell(f"CHIP_{chip_id:03d}")

    # ── Die boundary ───────────────────────────────────────────────────────────
    _rect(top, 0, 0, chip_w, chip_h, L_BOUND)
    _label(top, f"PhotoMedGemma Chip {chip_id:02d}  |  google/medgemma-4b-it",
           CHIP_MARGIN_UM, chip_h - CHIP_MARGIN_UM + 30, size=15)
    _label(top, f"{n_modes} modes  {n_stages} stages  {len(active):,} MZIs  "
                f"rank={rank}  220nm SOI  lam=1310nm",
           CHIP_MARGIN_UM, chip_h - CHIP_MARGIN_UM + 10, size=7)

    # ── Heater routing strip at bottom ─────────────────────────────────────────
    _heater_routing_strip(top, CHIP_MARGIN_UM, CHIP_MARGIN_UM,
                          chip_w - 2 * CHIP_MARGIN_UM)

    # ── Grating coupler arrays — left (input) and right (output) ──────────────
    gc_x_left  = CHIP_MARGIN_UM + GC_WIDTH_UM / 2
    gc_x_right = chip_w - CHIP_MARGIN_UM - GC_WIDTH_UM / 2
    _grating_coupler_column(top, gc_x_left,  mesh_y0, n_modes, mode_pitch, "left")
    _grating_coupler_column(top, gc_x_right, mesh_y0, n_modes, mode_pitch, "right")

    # ── Through-waveguides: horizontal rails for each mode ─────────────────────
    rail_x0 = gc_x_left  + GC_WIDTH_UM / 2 + 30
    rail_x1 = gc_x_right - GC_WIDTH_UM / 2 - 30
    for i in range(n_modes):
        y = mesh_y0 + i * mode_pitch
        _waveguide(top, rail_x0, y, rail_x1, y)

    # ── Place MZI cells — using compressed column index & mode_i as row ────────
    n_placed = 0
    for entry in active:
        theta = entry.theta_dac / max_code * (2 * math.pi)
        phi   = entry.phi_dac   / max_code * (2 * math.pi)

        display_col = col_index[entry.mzi_col]
        display_row = min(entry.mode_i, entry.mode_j)   # top rail of the pair

        # Absolute position in μm
        x_origin = mesh_x0 + display_col * stage_pitch
        y_top    = mesh_y0 + (display_row + 1) * mode_pitch
        y_bot    = mesh_y0 + display_row       * mode_pitch

        cell_name = f"chip{chip_id}_mzi_r{display_row}_c{display_col}"
        if cell_name not in lib.cells:
            c = lib.new_cell(cell_name)
            hw = WG_WIDTH_UM / 2

            # φ heater
            phi_x0 = x_origin
            phi_x1 = phi_x0 + mzi_len * PHI_LEN_UM / TOTAL_MZI_LEN_UM
            _rect(c, phi_x0, y_top - HEATER_WIDTH_UM/2,
                     phi_x1, y_top + HEATER_WIDTH_UM/2, L_PHI)
            _waveguide(c, phi_x0, y_top, phi_x1, y_top)
            _waveguide(c, phi_x0, y_bot, phi_x1, y_bot)

            # DC1
            dc1_x0 = phi_x1
            dc1_x1 = dc1_x0 + mzi_len * DC_LENGTH_UM / TOTAL_MZI_LEN_UM
            dc_ymid = (y_top + y_bot) / 2
            _rect(c, dc1_x0, dc_ymid - WG_SPACING_UM/2 - WG_WIDTH_UM,
                     dc1_x1, dc_ymid + WG_SPACING_UM/2 + WG_WIDTH_UM, L_SI)

            # θ heater
            theta_x0 = dc1_x1
            theta_x1 = theta_x0 + mzi_len * THETA_LEN_UM / TOTAL_MZI_LEN_UM
            _rect(c, theta_x0, y_top - HEATER_WIDTH_UM/2,
                     theta_x1, y_top + HEATER_WIDTH_UM/2, L_THETA)
            _waveguide(c, theta_x0, y_top, theta_x1, y_top)
            _waveguide(c, theta_x0, y_bot, theta_x1, y_bot)

            # DC2
            dc2_x0 = theta_x1
            dc2_x1 = min(dc2_x0 + mzi_len * DC_LENGTH_UM / TOTAL_MZI_LEN_UM,
                         x_origin + mzi_len)
            _rect(c, dc2_x0, dc_ymid - WG_SPACING_UM/2 - WG_WIDTH_UM,
                     dc2_x1, dc_ymid + WG_SPACING_UM/2 + WG_WIDTH_UM, L_SI)

            # Metal contacts on θ heater
            cw, ch = 4.0, 4.0
            _rect(c, theta_x0,      y_top + HEATER_WIDTH_UM/2,
                     theta_x0 + cw, y_top + HEATER_WIDTH_UM/2 + ch, L_METAL)
            _rect(c, theta_x1 - cw, y_top + HEATER_WIDTH_UM/2,
                     theta_x1,      y_top + HEATER_WIDTH_UM/2 + ch, L_METAL)

        import gdspy
        top.add(gdspy.CellReference(lib.cells[cell_name], (0, 0)))
        n_placed += 1

    # ── Phase screen elements on left edge (diagonal phases) ───────────────────
    screen_entries = [e for e in entries if e.mzi_row < 0 and e.mzi_col < rank]
    for e in screen_entries:
        mode_idx = e.mzi_col
        y_c  = mesh_y0 + mode_idx * mode_pitch + mode_pitch / 2
        x_c  = mesh_x0 - 30.0
        phase = e.theta_dac / max_code * (2 * math.pi)
        s = 15.0
        _rect(top, x_c - s/2, y_c - s/2, x_c + s/2, y_c + s/2, L_PHI)

    # ── Write GDS ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(output_path))
    logger.info(f"  → GDS written: {output_path}  ({output_path.stat().st_size//1024} KB)")

    return {
        "chip_id":    chip_id,
        "n_mzis":     n_placed,
        "n_modes":    n_modes,
        "n_stages":   n_stages,
        "rank":       rank,
        "width_mm":   chip_w / 1000,
        "height_mm":  chip_h / 1000,
        "gds_path":   str(output_path),
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
 <properties>
  <frame-color>#22cc44</frame-color>
  <fill-color>#00000000</fill-color>
  <name>Chiplet boundary</name>
  <source>10/0@1</source>
 </properties>
 <properties>
  <frame-color>#8844ff</frame-color>
  <fill-color>#8844ff</fill-color>
  <fill-brightness>-180</fill-brightness>
  <name>Interposer substrate</name>
  <source>11/0@1</source>
 </properties>
 <properties>
  <frame-color>#ff2222</frame-color>
  <fill-color>#ff2222</fill-color>
  <fill-brightness>-120</fill-brightness>
  <name>Dicing lanes</name>
  <source>12/0@1</source>
 </properties>
 <properties>
  <frame-color>#00ffff</frame-color>
  <fill-color>#00ffff</fill-color>
  <fill-brightness>-80</fill-brightness>
  <name>Fiber ribbon routing</name>
  <source>13/0@1</source>
 </properties>
 <properties>
  <frame-color>#ffcc00</frame-color>
  <fill-color>#ffcc00</fill-color>
  <name>Bond pad arrays (C4)</name>
  <source>14/0@1</source>
 </properties>
 <properties>
  <frame-color>#00aacc</frame-color>
  <fill-color>#00aacc</fill-color>
  <name>Interposer waveguides</name>
  <source>15/0@1</source>
 </properties>
 <properties>
  <frame-color>#cccccc</frame-color>
  <fill-color>#00000000</fill-color>
  <name>Assembly labels</name>
  <source>16/0@1</source>
 </properties>
 <properties>
  <frame-color>#888888</frame-color>
  <fill-color>#888888</fill-color>
  <fill-brightness>-100</fill-brightness>
  <name>Stack cross-section</name>
  <source>20/0@1</source>
 </properties>
 <properties>
  <frame-color>#ffffff</frame-color>
  <fill-color>#00000000</fill-color>
  <name>Stack labels</name>
  <source>21/0@1</source>
 </properties>
</layer-properties>
"""


def write_klayout_props(output_dir: Path, filename: str = "photomedgemma.lyp") -> Path:
    """Write a KLayout layer properties file for nice rendering of the full assembly."""
    props_path = output_dir / filename
    props_path.write_text(KLAYOUT_LYPROPS)
    logger.info(f"KLayout layer properties: {props_path}")
    return props_path


# ── Chiplet GDS generator ──────────────────────────────────────────────────────

def generate_chiplet_gds(
    chiplet_id: int,
    chiplet_type: str,
    layer_idx: int,
    chip_ids: List[int],
    chip_layout: List[dict],          # list of {chip_id, local_x_um, local_y_um, proj_key, matrix_type}
    width_um: float,
    height_um: float,
    chip_gds_paths: Dict[int, Path],
    output_path: Path,
    chip_size_um: float = 10_000.0,
    n_modes: int = 64,
    mode_pitch: float = 127.0,
    draw_fiber_routing: bool = True,
    draw_bond_pads: bool = True,
    dicing_lane_um: float = 75.0,
    chiplet_margin_um: float = 200.0,
) -> dict:
    """
    Generate a GDS for a single chiplet — a group of chips arranged in a grid
    with fiber ribbon routing, bond pads, and dicing lanes.

    Uses a hierarchical cell structure:
      CHIPLET_{id}_{type}_L{layer}
        ├── Imported chip cells (prefixed C{chip_id}_)
        ├── L_CHIPLET_BOUND outline
        ├── L_FIBER_RIBBON inter-chip routing lines
        ├── L_BOND_PAD_ARRAY C4 bond pad rows
        └── L_DICING_LANE strips

    Args:
        chiplet_id:      Chiplet index.
        chiplet_type:    "attention", "ffn", or "other".
        layer_idx:       Transformer layer index.
        chip_ids:        Ordered list of chip IDs in this chiplet.
        chip_layout:     Per-chip local positions and projection info.
        width_um:        Full chiplet width (including margins).
        height_um:       Full chiplet height (including margins).
        chip_gds_paths:  Dict mapping chip_id → Path of individual chip GDS.
        output_path:     Where to write chiplet GDS.
        chip_size_um:    Per-chip die size (square, μm).
        n_modes:         Number of optical modes (fiber channels per chip edge).
        mode_pitch:      Mode pitch for fiber ribbon (μm).
        draw_fiber_routing: Draw inter-chip fiber ribbon lines.
        draw_bond_pads:  Draw C4 bond pad arrays.
        dicing_lane_um:  Width of dicing lanes.
        chiplet_margin_um: Margin inside chiplet boundary.

    Returns:
        dict with chiplet stats.
    """
    try:
        import gdspy
    except ImportError:
        logger.error("gdspy not installed. Run: pip install gdspy")
        return {}

    cell_name = f"CHIPLET_{chiplet_id:03d}_{chiplet_type.upper()}_L{layer_idx}"
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    top = lib.new_cell(cell_name)

    # Build a map of local positions
    layout_by_id: Dict[int, dict] = {cl["chip_id"]: cl for cl in chip_layout}

    # ── Import each chip's GDS and place it ──────────────────────────────────
    for cid in chip_ids:
        gds_path = chip_gds_paths.get(cid)
        if gds_path is None or not Path(gds_path).exists():
            logger.warning(f"  Chip {cid} GDS not found: {gds_path} — drawing outline only")
            # Draw placeholder outline
            cl = layout_by_id.get(cid, {"local_x_um": 0, "local_y_um": 0})
            ox = chiplet_margin_um + cl["local_x_um"]
            oy = chiplet_margin_um + cl["local_y_um"]
            _rect(top, ox, oy, ox + chip_size_um, oy + chip_size_um, L_BOUND)
            continue

        # Read chip library and import cells with unique prefix
        chip_lib = gdspy.GdsLibrary()
        chip_lib.read_gds(str(gds_path))

        # Find the top cell (e.g. "CHIP_000")
        chip_top_cells = chip_lib.top_level()
        if not chip_top_cells:
            logger.warning(f"  Chip {cid}: no top-level cell found in {gds_path}")
            continue
        chip_top_cell = chip_top_cells[0]

        # Import all cells from chip library, renaming to avoid collisions
        prefix = f"C{cid:03d}_"
        imported_names: Dict[str, str] = {}
        for orig_name, cell in chip_lib.cells.items():
            new_name = prefix + orig_name
            if new_name not in lib.cells:
                new_cell = gdspy.Cell(new_name)
                # Copy polygons
                for poly in cell.get_polygons(by_spec=True).items():
                    spec, polys = poly
                    layer, datatype = spec
                    for pts in polys:
                        new_cell.add(gdspy.Polygon(pts, layer=layer, datatype=datatype))
                # Copy labels (anchor stored as int in gdspy internals — use "nw")
                for lbl in cell.labels:
                    anchor = lbl.anchor if isinstance(lbl.anchor, str) else "nw"
                    new_cell.add(gdspy.Label(
                        lbl.text, lbl.position, anchor,
                        lbl.rotation, lbl.magnification, lbl.x_reflection,
                        layer=lbl.layer, texttype=lbl.texttype
                    ))
                lib.cells[new_name] = new_cell
            imported_names[orig_name] = new_name

        # Place chip top cell at its local position within the chiplet
        cl = layout_by_id.get(cid, {"local_x_um": 0, "local_y_um": 0})
        ox = chiplet_margin_um + cl["local_x_um"]
        oy = chiplet_margin_um + cl["local_y_um"]

        renamed_top = imported_names.get(chip_top_cell.name)
        if renamed_top and renamed_top in lib.cells:
            top.add(gdspy.CellReference(lib.cells[renamed_top], (ox, oy)))

        # Draw dicing lane between chips (above and right edges)
        _rect(top, ox + chip_size_um, oy,
              ox + chip_size_um + dicing_lane_um, oy + chip_size_um,
              L_DICING_LANE)

    # ── Chiplet boundary ───────────────────────────────────────────────────────
    _rect(top, 0, 0, width_um, height_um, L_CHIPLET_BOUND)
    _label(top,
           f"Chiplet {chiplet_id}: {chiplet_type.upper()}  Layer {layer_idx}  "
           f"({len(chip_ids)} chips)",
           10, height_um - 20, size=20)

    # ── Fiber ribbon routing between chips ────────────────────────────────────
    if draw_fiber_routing and len(chip_layout) > 1:
        ribbon_w = 20.0   # schematic ribbon width (μm)
        # Sort chip layout by x position to find adjacent pairs
        sorted_layout = sorted(chip_layout, key=lambda cl: cl["local_x_um"])
        for i in range(len(sorted_layout) - 1):
            left  = sorted_layout[i]
            right = sorted_layout[i + 1]
            lx = chiplet_margin_um + left["local_x_um"] + chip_size_um
            rx = chiplet_margin_um + right["local_x_um"]
            # Only draw ribbon if chips are horizontally adjacent
            if rx > lx and (rx - lx) < chip_size_um * 1.5:
                ly = chiplet_margin_um + left["local_y_um"]
                # Draw 4 ribbon bands (representing 64-channel fiber ribbon groups)
                for band in range(4):
                    y_frac = (band + 0.5) / 4
                    yc = ly + y_frac * chip_size_um
                    _rect(top, lx, yc - ribbon_w / 2, rx, yc + ribbon_w / 2,
                          L_FIBER_RIBBON)

    # ── Inter-chiplet waveguide stubs (optical output connectors) ─────────────
    # Short waveguide stubs at left and right edges of the chiplet
    stub_len = 80.0
    stub_pitch = mode_pitch * 4  # sparse representation
    n_stubs = min(4, n_modes // 16)
    for i in range(n_stubs):
        y = chiplet_margin_um + (i + 1) * chip_size_um / (n_stubs + 1)
        # Left input stub
        _rect(top, 0, y - WG_WIDTH_UM / 2, stub_len, y + WG_WIDTH_UM / 2,
              L_INTERPOSER_WG)
        # Right output stub
        _rect(top, width_um - stub_len, y - WG_WIDTH_UM / 2,
              width_um, y + WG_WIDTH_UM / 2, L_INTERPOSER_WG)

    # ── Bond pad arrays (C4 micro-bumps at top edge of each chip) ─────────────
    if draw_bond_pads:
        pad_w, pad_h = 60.0, 60.0
        pad_pitch = 120.0
        n_pads = min(16, int(chip_size_um / pad_pitch))
        for cl in chip_layout:
            ox = chiplet_margin_um + cl["local_x_um"]
            oy = chiplet_margin_um + cl["local_y_um"]
            # Top edge bond pads
            pad_y0 = oy + chip_size_um - pad_h - 10
            for p in range(n_pads):
                px = ox + (chip_size_um - n_pads * pad_pitch) / 2 + p * pad_pitch
                _rect(top, px, pad_y0, px + pad_w, pad_y0 + pad_h, L_BOND_PAD_ARRAY)

    # ── Write GDS ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(output_path))
    size_kb = output_path.stat().st_size // 1024
    logger.info(f"  Chiplet {chiplet_id} GDS: {output_path}  ({size_kb} KB)")

    return {
        "chiplet_id":   chiplet_id,
        "chiplet_type": chiplet_type,
        "layer_idx":    layer_idx,
        "n_chips":      len(chip_ids),
        "chip_ids":     chip_ids,
        "width_mm":     width_um / 1000,
        "height_mm":    height_um / 1000,
        "gds_path":     str(output_path),
        "cell_name":    cell_name,
    }


# ── Assembly GDS generator ─────────────────────────────────────────────────────

def generate_assembly_gds(
    chiplets: List[dict],              # chiplet dicts from manifest
    chiplet_gds_paths: Dict[int, Path],
    output_path: Path,
    interposer_width_um: float,
    interposer_height_um: float,
    model_id: str = "PhotoMedGemma",
    representative_only: bool = True,
    interposer_margin_um: float = 2_000.0,
) -> dict:
    """
    Generate the full assembly GDS showing all chiplets on a 2.5D silicon interposer.

    Cell hierarchy (visible as tree in KLayout):
      ASSEMBLY_TOP
        ├── INTERPOSER_SUBSTRATE (L_INTERPOSER, full interposer box)
        ├── CHIPLET_000_ATTENTION_L0 → CellReference (full geometry)
        ├── CHIPLET_001_FFN_L0       → CellReference (full geometry)
        ├── [placeholder rectangles for remaining layers on L_CHIPLET_BOUND]
        └── ASSEMBLY_LABELS

    representative_only=True:
      Only chiplets[0] and chiplets[1] are drawn with full GDS geometry.
      Any additional chiplets (from more transformer layers) are shown as
      grey placeholder outlines with text labels — keeps the file renderable.

    Args:
        chiplets:            List of chiplet dicts from chiplet_manifest.json.
        chiplet_gds_paths:   Dict mapping chiplet_id → Path to chiplet GDS.
        output_path:         Where to write assembly.gds.
        interposer_width_um: Total interposer width from manifest.
        interposer_height_um:Total interposer height from manifest.
        model_id:            Label string.
        representative_only: Only render first layer pair fully.
        interposer_margin_um:Margin around chiplets on the interposer.

    Returns:
        dict with assembly stats.
    """
    try:
        import gdspy
    except ImportError:
        logger.error("gdspy not installed. Run: pip install gdspy")
        return {}

    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    top = lib.new_cell("ASSEMBLY_TOP")

    # Total interposer size with margins
    ip_w = interposer_width_um  + 2 * interposer_margin_um
    ip_h = interposer_height_um + 2 * interposer_margin_um
    ip_x0 = -interposer_margin_um
    ip_y0 = -interposer_margin_um

    # ── Interposer substrate ───────────────────────────────────────────────────
    ip_cell = lib.new_cell("INTERPOSER_SUBSTRATE")
    _rect(ip_cell, ip_x0, ip_y0, ip_x0 + ip_w, ip_y0 + ip_h, L_INTERPOSER)
    _label(ip_cell,
           f"PhotoMedGemma 2.5D Silicon Interposer  |  {model_id}",
           ip_x0 + 200, ip_y0 + ip_h - 400, size=100)
    _label(ip_cell,
           f"220nm SOI  |  1310nm  |  MZI mesh photonic compute",
           ip_x0 + 200, ip_y0 + ip_h - 700, size=60)
    _label(ip_cell,
           f"Interposer: {ip_w/1000:.1f}mm x {ip_h/1000:.1f}mm",
           ip_x0 + 200, ip_y0 + 200, size=60)
    top.add(gdspy.CellReference(ip_cell, (0, 0)))

    n_full_rendered = 0
    n_placeholder = 0

    for i, chiplet in enumerate(chiplets):
        cid    = chiplet["chiplet_id"]
        ctype  = chiplet["chiplet_type"]
        lidx   = chiplet["layer_idx"]
        ox     = chiplet["origin_x_um"]
        oy     = chiplet["origin_y_um"]
        cw     = chiplet["width_um"]
        ch     = chiplet["height_um"]

        render_full = (not representative_only) or (i < 2)

        if render_full and cid in chiplet_gds_paths:
            gds_path = chiplet_gds_paths[cid]
            if Path(gds_path).exists():
                # Import chiplet GDS cells
                clib = gdspy.GdsLibrary()
                clib.read_gds(str(gds_path))
                prefix = f"CL{cid:03d}_"
                imported_top = None
                for orig_name, cell in clib.cells.items():
                    new_name = prefix + orig_name
                    if new_name not in lib.cells:
                        new_cell = gdspy.Cell(new_name)
                        for spec, polys in cell.get_polygons(by_spec=True).items():
                            layer, datatype = spec
                            for pts in polys:
                                new_cell.add(gdspy.Polygon(pts, layer=layer, datatype=datatype))
                        for lbl in cell.labels:
                            anchor = lbl.anchor if isinstance(lbl.anchor, str) else "nw"
                            new_cell.add(gdspy.Label(
                                lbl.text, lbl.position, anchor,
                                lbl.rotation, lbl.magnification, lbl.x_reflection,
                                layer=lbl.layer, texttype=lbl.texttype
                            ))
                        lib.cells[new_name] = new_cell
                    if not imported_top:
                        imported_top = new_name

                # Find the top cell of the chiplet
                ctop_cells = clib.top_level()
                if ctop_cells:
                    ctop_name = prefix + ctop_cells[0].name
                    if ctop_name in lib.cells:
                        top.add(gdspy.CellReference(lib.cells[ctop_name], (ox, oy)))
                        n_full_rendered += 1
                        continue

        # Placeholder: draw outline + label
        ph_cell_name = f"PLACEHOLDER_CL{cid:03d}"
        if ph_cell_name not in lib.cells:
            ph_cell = lib.new_cell(ph_cell_name)
            _rect(ph_cell, 0, 0, cw, ch, L_CHIPLET_BOUND)
            _label(ph_cell,
                   f"Layer {lidx}: {ctype.upper()} chiplet  "
                   f"(chips {chiplet['chip_ids']})",
                   50, ch / 2, size=max(30, int(cw / 500)))
        top.add(gdspy.CellReference(lib.cells[ph_cell_name], (ox, oy)))
        n_placeholder += 1

    # ── Inter-chiplet optical routing on interposer ────────────────────────────
    # Draw short waveguide lines connecting chiplet output stubs to adjacent chiplets
    interposer_wg_cell = lib.new_cell("INTERPOSER_WG_ROUTING")
    for i in range(len(chiplets) - 1):
        ca = chiplets[i]
        cb = chiplets[i + 1]
        # Horizontal connection between right edge of A and left edge of B
        ax_right = ca["origin_x_um"] + ca["width_um"]
        bx_left  = cb["origin_x_um"]
        if abs(ca["origin_y_um"] - cb["origin_y_um"]) < ca["height_um"] * 0.5:
            # Same row — connect horizontally
            ymid = ca["origin_y_um"] + ca["height_um"] / 2
            _rect(interposer_wg_cell,
                  ax_right, ymid - WG_WIDTH_UM * 4,
                  bx_left,  ymid + WG_WIDTH_UM * 4,
                  L_INTERPOSER_WG)
    top.add(gdspy.CellReference(interposer_wg_cell, (0, 0)))

    # ── Write GDS ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Assembly GDS: {output_path}  ({size_mb:.1f} MB)  "
        f"[{n_full_rendered} full + {n_placeholder} placeholder chiplets]"
    )
    logger.info(
        f"  Interposer: {ip_w/1000:.1f}mm x {ip_h/1000:.1f}mm"
    )

    return {
        "n_chiplets_total":    len(chiplets),
        "n_full_rendered":     n_full_rendered,
        "n_placeholder":       n_placeholder,
        "interposer_width_mm": ip_w / 1000,
        "interposer_height_mm":ip_h / 1000,
        "interposer_area_mm2": ip_w * ip_h / 1e6,
        "gds_path":            str(output_path),
        "size_mb":             size_mb,
    }


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
        "--max-mzis", type=int, default=0,
        help=(
            "Max MZIs to load per chip from NPZ (default: 0 = no limit). "
            "The rank-64 filter reduces this to ~2016 MZIs per chip regardless."
        )
    )
    parser.add_argument(
        "--rank", type=int, default=None,
        help="Number of active optical modes / SVD rank (default: read from phase-map JSON)"
    )
    args = parser.parse_args()

    phase_map_path = Path(args.phase_map)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load phase map — support both old (entries) and new (layers+npz) format
    logger.info(f"Loading phase map: {phase_map_path}")
    with open(phase_map_path) as f:
        pm_data = json.load(f)

    rank = args.rank if args.rank is not None else pm_data.get("rank", 64)

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
            rank=rank,
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
