#!/usr/bin/env python3
"""
chiplet_partition.py — PhotoMedGemma Chiplet Partitioner
=========================================================

Groups photonic chips into functional chiplets and assigns 2D positions
on a 2.5D silicon interposer for the full assembly mask.

Chiplet strategy (default "per_transformer_layer"):
  - Attention chiplet: Q, K, V, O projection chips — 2×2 grid
  - FFN chiplet:       gate, up, down projection chips — 3×2 grid
  One attention + one FFN chiplet per transformer layer.

Usage
-----
    python3 scripts/chiplet_partition.py \\
        --phase-map output/phase_map_demo.json \\
        --output   output/chiplet_mask/chiplet_manifest.json
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Projection → module type
ATTENTION_PROJS = {"q_proj", "k_proj", "v_proj", "o_proj"}
FFN_PROJS       = {"gate_proj", "up_proj", "down_proj",
                   "gate",      "up",      "down"}

# Attention chiplet grid order (row-major, left-to-right, top-to-bottom)
ATTN_PROJ_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj"]
# FFN chiplet: each FFN projection may span 2 chips (Vh and U meshes)
# Arranged in 3 cols × 2 rows: top row = Vh meshes, bottom row = U meshes
FFN_PROJ_ORDER = ["gate_proj", "up_proj", "down_proj",
                  "gate",      "up",      "down"]


@dataclass
class ChipEntry:
    chip_id: int
    proj_key: str          # canonical projection name, e.g. "q_proj"
    matrix_type: str       # "U" or "Vh"
    layer_name: str        # original layer name from phase map
    transformer_layer: int # 0-based transformer layer index


@dataclass
class ChipLayout:
    chip_id: int
    local_x_um: float
    local_y_um: float
    proj_key: str
    matrix_type: str


@dataclass
class ChipletSpec:
    chiplet_id: int
    chiplet_type: str          # "attention" | "ffn" | "vision"
    layer_idx: int             # transformer layer index
    projections: List[str]     # e.g. ["q_proj","k_proj","v_proj","o_proj"]
    chip_ids: List[int]
    n_chips: int
    origin_x_um: float = 0.0
    origin_y_um: float = 0.0
    width_um: float = 0.0
    height_um: float = 0.0
    chip_layout: List[ChipLayout] = field(default_factory=list)


# ── Phase map loading ──────────────────────────────────────────────────────────

def _proj_key(layer_name: str) -> str:
    """Extract canonical projection name from a layer_name string."""
    for proj in [*ATTENTION_PROJS, *FFN_PROJS]:
        if proj in layer_name:
            return proj
    # Fallback: last component before .weight
    parts = layer_name.replace(".weight", "").split(".")
    return parts[-1] if parts else layer_name


def _transformer_layer(layer_name: str) -> int:
    """Extract the transformer layer index from a layer name like
    'model.layers.3.self_attn.q_proj.weight'."""
    parts = layer_name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return 0


def load_phase_map_summary(phase_map_path: Path) -> Dict:
    """
    Load the phase map JSON and return a summary dict:
    {
        model_id: str,
        rank: int,
        chip_entries: List[ChipEntry],   # one per chip_id × matrix_type
    }

    Supports both the old 'entries' list format and the new 'layers+npz' format.
    """
    with open(phase_map_path) as f:
        pm = json.load(f)

    model_id = pm.get("model_id", "unknown")
    rank     = pm.get("rank", 64)

    # Collect (chip_id, layer_name, matrix_type) → ChipEntry
    seen: Dict[tuple, ChipEntry] = {}

    if "entries" in pm:
        # Old format: inline MZI entries list
        for e in pm["entries"]:
            key = (e["chip_id"], e.get("layer_name", ""), e.get("matrix_type", "U"))
            if key not in seen:
                seen[key] = ChipEntry(
                    chip_id=e["chip_id"],
                    proj_key=_proj_key(e.get("layer_name", "")),
                    matrix_type=e.get("matrix_type", "U"),
                    layer_name=e.get("layer_name", ""),
                    transformer_layer=_transformer_layer(e.get("layer_name", "")),
                )

    else:
        # New format: layers array with NPZ references
        for layer in pm.get("layers", []):
            lname = layer.get("name", "")
            for mtype, chip_key in [("U", "chip_id_U"), ("Vh", "chip_id_Vh")]:
                chip_id = layer.get(chip_key)
                if chip_id is None:
                    continue
                key = (chip_id, lname, mtype)
                if key not in seen:
                    seen[key] = ChipEntry(
                        chip_id=chip_id,
                        proj_key=_proj_key(lname),
                        matrix_type=mtype,
                        layer_name=lname,
                        transformer_layer=_transformer_layer(lname),
                    )

    chip_entries = sorted(seen.values(), key=lambda e: (e.chip_id, e.layer_name))
    logger.info(f"Phase map '{model_id}': {len(chip_entries)} chip-projection assignments, rank={rank}")
    return {"model_id": model_id, "rank": rank, "chip_entries": chip_entries}


# ── Partitioning ──────────────────────────────────────────────────────────────

def _normalize_proj(proj_key: str) -> str:
    """Normalize projection key to its canonical form."""
    # Map short forms to long forms
    mapping = {"gate": "gate_proj", "up": "up_proj", "down": "down_proj"}
    return mapping.get(proj_key, proj_key)


def partition_by_function(
    chip_entries: List[ChipEntry],
    strategy: str = "per_transformer_layer",
) -> List[ChipletSpec]:
    """
    Group chips into chiplets by functional role.

    strategy "per_transformer_layer" (default):
      For each unique transformer layer index:
        - All chips for q/k/v/o projections → one attention chiplet
        - All chips for gate/up/down projections → one FFN chiplet

    Returns list of ChipletSpec sorted by (layer_idx, chiplet_type).
    """
    if strategy == "flat":
        # One chiplet per chip — no grouping
        return [
            ChipletSpec(
                chiplet_id=i,
                chiplet_type="flat",
                layer_idx=e.transformer_layer,
                projections=[_normalize_proj(e.proj_key)],
                chip_ids=[e.chip_id],
                n_chips=1,
            )
            for i, e in enumerate(chip_entries)
        ]

    # Group by transformer layer
    by_layer: Dict[int, List[ChipEntry]] = {}
    for e in chip_entries:
        by_layer.setdefault(e.transformer_layer, []).append(e)

    chiplets: List[ChipletSpec] = []
    chiplet_id = 0

    for layer_idx in sorted(by_layer.keys()):
        layer_entries = by_layer[layer_idx]

        # Split into attention vs FFN
        attn_entries = [e for e in layer_entries
                        if e.proj_key in ATTENTION_PROJS]
        ffn_entries  = [e for e in layer_entries
                        if e.proj_key in FFN_PROJS or
                        _normalize_proj(e.proj_key) in FFN_PROJS]
        other_entries = [e for e in layer_entries
                         if e not in attn_entries and e not in ffn_entries]

        if attn_entries:
            chip_ids   = sorted(set(e.chip_id for e in attn_entries))
            projs      = sorted(set(_normalize_proj(e.proj_key) for e in attn_entries))
            chiplets.append(ChipletSpec(
                chiplet_id=chiplet_id,
                chiplet_type="attention",
                layer_idx=layer_idx,
                projections=projs,
                chip_ids=chip_ids,
                n_chips=len(chip_ids),
            ))
            chiplet_id += 1

        if ffn_entries:
            chip_ids = sorted(set(e.chip_id for e in ffn_entries))
            projs    = sorted(set(_normalize_proj(e.proj_key) for e in ffn_entries))
            chiplets.append(ChipletSpec(
                chiplet_id=chiplet_id,
                chiplet_type="ffn",
                layer_idx=layer_idx,
                projections=projs,
                chip_ids=chip_ids,
                n_chips=len(chip_ids),
            ))
            chiplet_id += 1

        if other_entries:
            chip_ids = sorted(set(e.chip_id for e in other_entries))
            projs    = sorted(set(_normalize_proj(e.proj_key) for e in other_entries))
            chiplets.append(ChipletSpec(
                chiplet_id=chiplet_id,
                chiplet_type="other",
                layer_idx=layer_idx,
                projections=projs,
                chip_ids=chip_ids,
                n_chips=len(chip_ids),
            ))
            chiplet_id += 1

    logger.info(f"Partitioned into {len(chiplets)} chiplets "
                f"({sum(1 for c in chiplets if c.chiplet_type=='attention')} attention, "
                f"{sum(1 for c in chiplets if c.chiplet_type=='ffn')} FFN)")
    return chiplets


# ── Chip layout within chiplet ────────────────────────────────────────────────

def _build_chip_layout(
    chiplet: ChipletSpec,
    chip_entries: List[ChipEntry],
    chip_size_um: float,
    chip_gap_um: float,
) -> List[ChipLayout]:
    """
    Assign local (x, y) positions for chips within a chiplet.

    Attention chiplet (4 chips: Q, K, V, O) → 2 cols × 2 rows:
        col 0       col 1
      +----------+----------+
      | Q (top)  | K (top)  |  row 1 (top)
      +----------+----------+
      | V (bot)  | O (bot)  |  row 0 (bottom)
      +----------+----------+

    FFN chiplet (up to 6 chips: gate_Vh, gate_U, up_Vh, up_U, down_Vh, down_U):
      3 cols × 2 rows — Vh meshes in top row, U meshes in bottom row:
        col 0       col 1       col 2
      +----------+----------+----------+
      | gate_Vh  | up_Vh    | down_Vh  |  row 1 (top)
      +----------+----------+----------+
      | gate_U   | up_U     | down_U   |  row 0
      +----------+----------+----------+

    Other/flat chiplets: single row, sorted by chip_id.
    """
    pitch = chip_size_um + chip_gap_um

    # Build a map from chip_id to its entries
    id_to_entries: Dict[int, List[ChipEntry]] = {}
    for e in chip_entries:
        if e.chip_id in chiplet.chip_ids:
            id_to_entries.setdefault(e.chip_id, []).append(e)

    layout: List[ChipLayout] = []

    if chiplet.chiplet_type == "attention":
        # 2×2 grid: [q, k] top row left-to-right, [v, o] bottom row
        attn_order = ["q_proj", "k_proj", "v_proj", "o_proj"]
        chip_by_proj: Dict[str, List[int]] = {}
        for cid in chiplet.chip_ids:
            for e in id_to_entries.get(cid, []):
                pk = _normalize_proj(e.proj_key)
                chip_by_proj.setdefault(pk, []).append(cid)

        grid_pos = [
            ("q_proj", 0, 1), ("k_proj", 1, 1),  # top row
            ("v_proj", 0, 0), ("o_proj", 1, 0),  # bottom row
        ]
        placed: set = set()
        for proj, col, row in grid_pos:
            for cid in sorted(chip_by_proj.get(proj, [])):
                if cid not in placed:
                    layout.append(ChipLayout(
                        chip_id=cid,
                        local_x_um=col * pitch,
                        local_y_um=row * pitch,
                        proj_key=proj,
                        matrix_type=id_to_entries[cid][0].matrix_type if cid in id_to_entries else "?",
                    ))
                    placed.add(cid)
        # Any remaining chips (beyond the 4 standard slots)
        for cid in sorted(chiplet.chip_ids):
            if cid not in placed:
                col = len(placed) % 2
                row = len(placed) // 2
                layout.append(ChipLayout(
                    chip_id=cid,
                    local_x_um=col * pitch,
                    local_y_um=row * pitch,
                    proj_key="extra",
                    matrix_type="?",
                ))
                placed.add(cid)

    elif chiplet.chiplet_type == "ffn":
        # 3 cols × 2 rows: Vh in top row, U in bottom row
        # Columns ordered: gate, up, down
        ffn_proj_order = ["gate_proj", "up_proj", "down_proj"]

        # Group chips by projection + matrix_type
        chip_assignments: Dict[tuple, int] = {}  # (proj, mtype) → chip_id
        for cid in sorted(chiplet.chip_ids):
            for e in id_to_entries.get(cid, []):
                pk = _normalize_proj(e.proj_key)
                mtype = e.matrix_type
                key = (pk, mtype)
                if key not in chip_assignments:
                    chip_assignments[key] = cid

        placed: set = set()
        # Place in 3×2 grid
        for col_idx, proj in enumerate(ffn_proj_order):
            for row_idx, mtype in enumerate(["Vh", "U"]):
                cid = chip_assignments.get((prop, mtype) for prop in [proj])
                # Try to find this chip
                matched = chip_assignments.get((proj, mtype))
                if matched is not None and matched not in placed:
                    layout.append(ChipLayout(
                        chip_id=matched,
                        local_x_um=col_idx * pitch,
                        local_y_um=row_idx * pitch,
                        proj_key=proj,
                        matrix_type=mtype,
                    ))
                    placed.add(matched)

        # Remaining unplaced chips (fallback: sequential)
        col, row = 0, 2
        for cid in sorted(chiplet.chip_ids):
            if cid not in placed:
                layout.append(ChipLayout(
                    chip_id=cid,
                    local_x_um=col * pitch,
                    local_y_um=row * pitch,
                    proj_key="extra",
                    matrix_type="?",
                ))
                placed.add(cid)
                col += 1
                if col >= 3:
                    col = 0
                    row += 1

    else:
        # Flat: single row
        for i, cid in enumerate(sorted(chiplet.chip_ids)):
            layout.append(ChipLayout(
                chip_id=cid,
                local_x_um=i * pitch,
                local_y_um=0.0,
                proj_key=id_to_entries[cid][0].proj_key if cid in id_to_entries else "?",
                matrix_type=id_to_entries[cid][0].matrix_type if cid in id_to_entries else "?",
            ))

    return layout


# ── Interposer position assignment ────────────────────────────────────────────

def assign_interposer_positions(
    chiplets: List[ChipletSpec],
    chip_entries: List[ChipEntry],
    chip_size_um: float = 10_000.0,
    chip_gap_um: float = 500.0,
    dicing_lane_um: float = 75.0,
    chiplet_margin_um: float = 200.0,
) -> List[ChipletSpec]:
    """
    Assign 2D (x, y) positions on the interposer for each chiplet and
    compute per-chip local positions within each chiplet.

    Layout strategy:
      - Chiplets arranged in two rows: attention (top), FFN (bottom)
      - Each row: chiplets placed left to right, separated by dicing lanes
      - The interposer origin is (0, 0) at bottom-left
    """
    chip_pitch = chip_size_um + chip_gap_um

    for chiplet in chiplets:
        # Build chip layout within this chiplet
        chiplet.chip_layout = _build_chip_layout(
            chiplet, chip_entries, chip_size_um, chip_gap_um
        )

        # Compute chiplet bounding box from chip layouts
        if chiplet.chip_layout:
            max_x = max(cl.local_x_um for cl in chiplet.chip_layout) + chip_size_um
            max_y = max(cl.local_y_um for cl in chiplet.chip_layout) + chip_size_um
        else:
            # Fallback geometry
            n = chiplet.n_chips
            if chiplet.chiplet_type == "attention":
                cols, rows = 2, 2
            elif chiplet.chiplet_type == "ffn":
                cols, rows = 3, max(2, (n + 2) // 3)
            else:
                cols, rows = n, 1
            max_x = cols * chip_pitch - chip_gap_um
            max_y = rows * chip_pitch - chip_gap_um

        chiplet.width_um  = max_x + 2 * chiplet_margin_um
        chiplet.height_um = max_y + 2 * chiplet_margin_um

    # Two-row layout: attention chiplets top row, FFN bottom row
    attn_chiplets = [c for c in chiplets if c.chiplet_type == "attention"]
    ffn_chiplets  = [c for c in chiplets if c.chiplet_type == "ffn"]
    other_chiplets = [c for c in chiplets
                      if c.chiplet_type not in ("attention", "ffn")]

    # Row heights
    attn_row_h = max((c.height_um for c in attn_chiplets), default=0)
    ffn_row_h  = max((c.height_um for c in ffn_chiplets),  default=0)

    # Place attention chiplets in top row
    cursor_x = 0.0
    y_base_attn = ffn_row_h + dicing_lane_um * 4  # top row sits above FFN row
    for c in attn_chiplets:
        c.origin_x_um = cursor_x
        c.origin_y_um = y_base_attn
        cursor_x += c.width_um + dicing_lane_um * 2

    # Place FFN chiplets in bottom row
    cursor_x = 0.0
    for c in ffn_chiplets:
        c.origin_x_um = cursor_x
        c.origin_y_um = 0.0
        cursor_x += c.width_um + dicing_lane_um * 2

    # Place other chiplets to the right of both rows
    cursor_x = max(
        (c.origin_x_um + c.width_um for c in chiplets if c not in other_chiplets),
        default=0,
    ) + dicing_lane_um * 4
    cursor_y = 0.0
    for c in other_chiplets:
        c.origin_x_um = cursor_x
        c.origin_y_um = cursor_y
        cursor_y += c.height_um + dicing_lane_um * 2

    logger.info("Interposer positions assigned:")
    for c in chiplets:
        logger.info(
            f"  Chiplet {c.chiplet_id} ({c.chiplet_type}, L{c.layer_idx}): "
            f"origin=({c.origin_x_um/1000:.1f}mm, {c.origin_y_um/1000:.1f}mm)  "
            f"size=({c.width_um/1000:.1f}mm × {c.height_um/1000:.1f}mm)"
        )

    return chiplets


# ── Manifest writer ────────────────────────────────────────────────────────────

def write_chiplet_manifest(
    chiplets: List[ChipletSpec],
    phase_map_path: Path,
    output_path: Path,
    model_id: str = "unknown",
    rank: int = 64,
    strategy: str = "per_transformer_layer",
    chip_size_um: float = 10_000.0,
    chip_gap_um: float = 500.0,
) -> dict:
    """Write chiplet manifest JSON and return the manifest dict."""
    all_chips_x = [c.origin_x_um + c.width_um for c in chiplets]
    all_chips_y = [c.origin_y_um + c.height_um for c in chiplets]
    interposer_w = max(all_chips_x) if all_chips_x else 0
    interposer_h = max(all_chips_y) if all_chips_y else 0

    def _cl_dict(cl: ChipLayout) -> dict:
        return {
            "chip_id":     cl.chip_id,
            "local_x_um":  cl.local_x_um,
            "local_y_um":  cl.local_y_um,
            "proj_key":    cl.proj_key,
            "matrix_type": cl.matrix_type,
        }

    def _chiplet_dict(c: ChipletSpec) -> dict:
        return {
            "chiplet_id":   c.chiplet_id,
            "chiplet_type": c.chiplet_type,
            "layer_idx":    c.layer_idx,
            "projections":  c.projections,
            "chip_ids":     c.chip_ids,
            "n_chips":      c.n_chips,
            "origin_x_um":  c.origin_x_um,
            "origin_y_um":  c.origin_y_um,
            "width_um":     c.width_um,
            "height_um":    c.height_um,
            "chip_layout":  [_cl_dict(cl) for cl in c.chip_layout],
        }

    manifest = {
        "model_id":           model_id,
        "rank":               rank,
        "phase_map":          str(phase_map_path),
        "strategy":           strategy,
        "n_chiplets":         len(chiplets),
        "interposer_width_um":  interposer_w,
        "interposer_height_um": interposer_h,
        "chip_size_um":       chip_size_um,
        "chip_gap_um":        chip_gap_um,
        "chiplets":           [_chiplet_dict(c) for c in chiplets],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"Chiplet manifest written: {output_path}")
    logger.info(
        f"  Interposer: {interposer_w/1000:.1f}mm × {interposer_h/1000:.1f}mm  "
        f"({interposer_w*interposer_h/1e6:.0f} mm²)"
    )
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Partition PhotoMedGemma chips into chiplets for 2.5D interposer layout"
    )
    parser.add_argument("--phase-map", required=True,
                        help="Compiled phase map JSON")
    parser.add_argument("--output", default="output/chiplet_mask/chiplet_manifest.json",
                        help="Output chiplet manifest JSON path")
    parser.add_argument("--strategy", default="per_transformer_layer",
                        choices=["per_transformer_layer", "flat"],
                        help="Partitioning strategy")
    parser.add_argument("--chip-gap", type=float, default=500.0,
                        help="Gap between chips within a chiplet (μm)")
    parser.add_argument("--dicing-lane", type=float, default=75.0,
                        help="Dicing lane width between chiplets (μm)")
    args = parser.parse_args()

    phase_map_path = Path(args.phase_map)
    output_path    = Path(args.output)

    summary = load_phase_map_summary(phase_map_path)
    chiplets = partition_by_function(summary["chip_entries"], strategy=args.strategy)
    chiplets = assign_interposer_positions(
        chiplets,
        summary["chip_entries"],
        chip_gap_um=args.chip_gap,
        dicing_lane_um=args.dicing_lane,
    )
    manifest = write_chiplet_manifest(
        chiplets,
        phase_map_path,
        output_path,
        model_id=summary["model_id"],
        rank=summary["rank"],
        strategy=args.strategy,
        chip_gap_um=args.chip_gap,
    )

    # Print summary table
    print("\nChiplet Partition Summary")
    print("=" * 72)
    print(f"{'ID':>3}  {'Type':>10}  {'Layer':>5}  {'Chips':>5}  "
          f"{'Projections':<30}  {'Size (mm²)'}")
    print("-" * 72)
    for c in chiplets:
        proj_str = ", ".join(c.projections)
        size_mm2 = c.width_um * c.height_um / 1e6
        print(f"{c.chiplet_id:>3}  {c.chiplet_type:>10}  {c.layer_idx:>5}  "
              f"{c.n_chips:>5}  {proj_str:<30}  {size_mm2:.0f}")
    print("-" * 72)
    iw = manifest["interposer_width_um"] / 1000
    ih = manifest["interposer_height_um"] / 1000
    print(f"\nInterposer: {iw:.1f} mm × {ih:.1f} mm  ({iw*ih:.0f} mm²)")
    print(f"Output:     {output_path}")


if __name__ == "__main__":
    main()
