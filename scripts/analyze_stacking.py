#!/usr/bin/env python3
"""
analyze_stacking.py — 3D Stacking Feasibility Analysis for PhotoMedGemma
=========================================================================

Evaluates five stacking strategies for photonic chiplets and generates:
  1. A plain-text analysis report with all options and the recommendation
  2. A GDS cross-section schematic of the recommended 2.5D interposer approach

Recommendation: 2.5D Silicon Interposer (same as Nvidia GPU chiplets)
  - Works with top-side grating coupler optical I/O (no blocking)
  - Mature supply chain: TSMC CoWoS, ASE, Amkor
  - Electrical redistribution via fine-pitch TSVs
  - Optical I/O via fiber array pressing on top-side grating couplers

Usage
-----
    python3 scripts/analyze_stacking.py \\
        --output-dir output/stacking_analysis

    python3 scripts/analyze_stacking.py \\
        --output-dir output/stacking_analysis \\
        --n-cross-section 3
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Stacking option data model ────────────────────────────────────────────────

@dataclass
class StackingOption:
    name: str
    short_name: str
    description: str
    area_saving_factor: float       # 1.0 = no saving, 2.0 = half area per layer
    feasible: bool
    manufacturing_readiness: str    # "production" | "research" | "not_feasible"
    optical_loss_db: float          # additional optical loss per interface
    reasons_for: List[str]
    reasons_against: List[str]
    recommendation: str             # "adopt" | "partial" | "reject"


# ── Option definitions ────────────────────────────────────────────────────────

def evaluate_all_options() -> List[StackingOption]:
    """Return evaluated stacking options for the PhotoMedGemma photonic platform."""

    return [
        StackingOption(
            name="Monolithic 3D (fab-on-top)",
            short_name="Monolithic3D",
            description=(
                "Grow a second silicon photonic layer directly on top of the first "
                "using wafer bonding + CMP. Both active layers share one die."
            ),
            area_saving_factor=2.0,
            feasible=False,
            manufacturing_readiness="not_feasible",
            optical_loss_db=3.0,
            reasons_for=[
                "Theoretically 2x area reduction per stacked layer",
                "No fiber coupling between layers needed",
                "Monolithic integration means shorter optical paths",
            ],
            reasons_against=[
                "No mature SOI-on-SOI foundry process available (2025)",
                "Grating couplers on top device layer blocked by second layer deposited above",
                "Thermal crosstalk: TiN heaters on layer 1 drift layer 2 phase settings",
                "CMP planarization tolerance ±5nm — insufficient for 220nm SOI waveguide alignment",
                "Refractive index mismatch at bonding oxide interface causes mode loss",
                "Yield: two device layers must both be defect-free — probability drops as n²",
            ],
            recommendation="reject",
        ),

        StackingOption(
            name="Flip-Chip Micro-Bump Bonding",
            short_name="FlipChip",
            description=(
                "Flip one photonic die face-down onto another, connecting through "
                "50-100μm pitch C4 micro-bumps. Used in HBM memory stacking."
            ),
            area_saving_factor=1.0,
            feasible=True,
            manufacturing_readiness="production",
            optical_loss_db=99.0,   # Not applicable — optical path broken
            reasons_for=[
                "Mature process — used in HBM, Intel Foveros, AMD 3D V-Cache",
                "Fine-pitch electrical interconnects (50μm C4 pitch)",
                "Works for DAC/control signal distribution to heaters",
                "Reduces bond wire length → lower parasitic inductance for heater drivers",
            ],
            reasons_against=[
                "Photonic chips use in-plane TE mode — light travels horizontally",
                "Flipping a die face-down blocks all grating coupler access from above",
                "Micro-bump pitch (50-100μm) >> waveguide pitch (0.45μm) — no optical through-bump",
                "Underfill epoxy between dies would absorb 1310nm light at any coupling attempt",
                "Cannot connect optical paths between flipped layers",
            ],
            recommendation="partial",
        ),

        StackingOption(
            name="PCB Stacking with Fiber Ribbon Interconnects",
            short_name="PCBStack",
            description=(
                "Multiple photonic chips mounted on separate PCBs, stacked vertically "
                "and connected by multi-fiber ribbon cables (MTP/MPO connectors)."
            ),
            area_saving_factor=1.0,   # chips still side-by-side on each PCB
            feasible=True,
            manufacturing_readiness="production",
            optical_loss_db=1.0,     # per MTP connector pair
            reasons_for=[
                "Fully compatible with top-side grating couplers on all chips",
                "Standard telecom grade — 12-fiber MTP connectors, SM fiber ribbon",
                "Proven in photonic computing systems (Lightmatter, Ayar Labs)",
                "Allows cooling access from all sides of each PCB",
                "Modular: damaged chiplets on one PCB can be replaced independently",
                "Scales to 100s of PCBs with standard rack-and-backplane infrastructure",
            ],
            reasons_against=[
                "Does NOT reduce PCB footprint — chips still spread across each board",
                "MTP connector insertion loss: 0.5-1.5 dB per connection",
                "Fiber ribbon pitch (250μm) vs waveguide mode pitch (127μm) — needs V-groove array",
                "Mechanical vibration at connectors can cause phase noise in heaters",
                "Cable management complexity grows with chip count",
            ],
            recommendation="partial",
        ),

        StackingOption(
            name="Edge-Coupler Chip-to-Chip Stacking",
            short_name="EdgeCoupler",
            description=(
                "Two chips placed face-to-face at their polished edges. "
                "Light couples end-fire between chips through free space or polymer waveguide."
            ),
            area_saving_factor=1.0,  # end-to-end, not truly stacked
            feasible=False,
            manufacturing_readiness="research",
            optical_loss_db=4.0,    # mode mismatch + alignment loss
            reasons_for=[
                "No connector needed — direct chip-to-chip optical coupling",
                "Potentially zero dead-zone between chips in the optical path",
                "Compatible with in-plane TE mode propagation",
                "Allows U→Σ→Vh optical chaining without grating couplers in between",
            ],
            reasons_against=[
                "220nm SOI edge mode: 0.45μm wide × 220nm tall — extreme aspect ratio",
                "Alignment tolerance < 0.1μm — not achievable with passive die attach",
                "Active alignment systems add cost comparable to an ASIC",
                "Mode mismatch between chips: 3-6 dB insertion loss",
                "Polished chip edge requires specialized dicing (UV laser dice)",
                "Polymer waveguide bridges (OZ Optics) are hand-assembled — not scalable",
                "Not available from any standard photonics foundry as a package option",
            ],
            recommendation="reject",
        ),

        StackingOption(
            name="2.5D Silicon Interposer (RECOMMENDED)",
            short_name="Interposer2D5",
            description=(
                "All chiplets bonded side-by-side to a common passive silicon interposer. "
                "Electrical redistribution through interposer TSVs. "
                "Optical I/O via top-side fiber arrays pressing on grating couplers. "
                "Same approach as Nvidia GPU chiplet integration."
            ),
            area_saving_factor=1.05,  # slight overhead for interposer routing channels
            feasible=True,
            manufacturing_readiness="production",
            optical_loss_db=0.3,    # grating coupler coupling loss only (already in budget)
            reasons_for=[
                "Works perfectly with top-side grating couplers — light enters/exits from above",
                "Mature supply chain: TSMC CoWoS, Intel EMIB, ASE SiP, Amkor SWIFT",
                "Electrical redistribution: DAC control lines routed under interposer",
                "No additional optical loss beyond existing grating coupler design",
                "Proven at scale: Nvidia H100 (4 dies on interposer), AMD MI300X (13 chiplets)",
                "Allows mixed-chiplet integration: Si photonics + CMOS FPGA on same interposer",
                "Thermal management: interposer acts as heat spreader, TEC mounted below",
                "Modular yield: each chiplet tested before bonding → higher system yield",
                "PCB footprint reduction: 10 chips on one 54mm×22mm interposer instead of"
                " 10 separate PCB footprints",
                "Standardized bonding: C4 micro-bumps at 100μm pitch for electrical",
            ],
            reasons_against=[
                "Interposer adds ~15-20% cost to die (passive Si wafer + TSV process)",
                "Interposer fabrication lead time: 8-12 weeks (separate wafer run)",
                "Total footprint determined by sum of chiplet areas + routing space",
                "Thermal management through interposer limited: need top-side heat spreader",
                "Maximum interposer reticle size: ~800mm² (limited by stepper field)",
            ],
            recommendation="adopt",
        ),
    ]


# ── Report writer ──────────────────────────────────────────────────────────────

def generate_stacking_report(options: List[StackingOption], output_path: Path) -> str:
    """Write a plain-text analysis report. Returns the report string."""

    lines = []
    lines.append("=" * 80)
    lines.append("PhotoMedGemma — 3D/2.5D Chiplet Stacking Feasibility Analysis")
    lines.append("220nm SOI Silicon Photonics  |  1310nm  |  MZI mesh inference chip")
    lines.append("=" * 80)
    lines.append("")
    lines.append("PLATFORM CONSTRAINTS (determine what stacking options are feasible)")
    lines.append("-" * 80)
    lines.append("  • Light propagates in-plane (TE mode along chip surface)")
    lines.append("  • Optical I/O via grating couplers — require TOP-SIDE access (10° angle)")
    lines.append("  • TiN phase shifters: thermally tuned, sensitive to adjacent heat sources")
    lines.append("  • Mode pitch: 127 μm (fiber array standard)")
    lines.append("  • Waveguide width: 450 nm — alignment tolerance < 100 nm for coupling")
    lines.append("  • Each chip: 10mm × 10mm, up to 4096 MZIs per chip")
    lines.append("")

    for opt in options:
        verdict = {
            "adopt":   "[RECOMMENDED]",
            "partial": "[PARTIAL USE]",
            "reject":  "[REJECTED]   ",
        }[opt.recommendation]

        lines.append("=" * 80)
        lines.append(f"{verdict}  {opt.name}")
        lines.append("-" * 80)
        lines.append(f"  Description:       {opt.description}")
        lines.append(f"  Area saving:       {opt.area_saving_factor:.1f}x")
        lines.append(f"  Manufacturability: {opt.manufacturing_readiness}")
        if opt.optical_loss_db < 50:
            lines.append(f"  Optical loss:      {opt.optical_loss_db:.1f} dB per inter-chip interface")
        else:
            lines.append(f"  Optical loss:      N/A (optical path broken by this approach)")
        lines.append("")
        lines.append("  For:")
        for r in opt.reasons_for:
            lines.append(f"    + {r}")
        lines.append("")
        lines.append("  Against:")
        for r in opt.reasons_against:
            lines.append(f"    - {r}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("RECOMMENDATION SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append("ADOPT: 2.5D Silicon Interposer")
    lines.append("")
    lines.append("  Implementation for PhotoMedGemma:")
    lines.append("")
    lines.append("  1. Attention chiplet (Q,K,V,O chips in 2×2 grid) on interposer")
    lines.append("  2. FFN chiplet (gate,up,down chips in 3×2 grid) on interposer")
    lines.append("  3. Interposer provides electrical routing for heater DAC lines")
    lines.append("  4. Fiber array pressing on top-side grating couplers for optical I/O")
    lines.append("  5. TEC mounted under interposer for thermal stabilization")
    lines.append("  6. FPGA (softmax, LayerNorm) as additional chiplet on same interposer")
    lines.append("")
    lines.append("PARTIAL: Flip-Chip Micro-Bumps for electrical only")
    lines.append("  Use C4 micro-bumps between photonic chiplet and DAC ASIC")
    lines.append("  (no optical path through bumps, electrical only)")
    lines.append("")
    lines.append("PARTIAL: PCB Stacking for multi-board system assembly")
    lines.append("  Stack multiple interposer boards in a chassis with MTP fiber ribbons")
    lines.append("  Enables scaling to full 46-layer model (92 chiplets across ~5 boards)")
    lines.append("")
    lines.append("REJECT: Monolithic 3D and Edge-Coupler (not manufacturable at this stage)")
    lines.append("")
    lines.append("=" * 80)

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    logger.info(f"Stacking report: {output_path}")
    return report


# ── Cross-section GDS generator ───────────────────────────────────────────────

def generate_cross_section_gds(
    output_path: Path,
    n_chiplets: int = 3,
) -> Path:
    """
    Generate a schematic GDS cross-section of the recommended 2.5D interposer approach.

    Cross-section Y-axis (vertical layers, all in μm):
        0       – 200:    PCB substrate         (L_STACK_CROSS, datatype=2, dark grey)
        200     – 350:    Underfill epoxy        (L_STACK_CROSS, datatype=3, yellow)
        350     – 500:    Silicon interposer     (L_INTERPOSER=11)
        500     – 505:    Bond oxide (SiO2)      (L_STACK_CROSS, datatype=1)
        505     – 725:    Photonic chip die      (L_BOUND=5)
        725     – 730:    Top cladding SiO2      (L_STACK_CROSS, datatype=1)
        730     – 790:    Fiber array V-groove   (L_STACK_CROSS, datatype=4)

    X-axis: each chiplet = chip_w μm wide, separated by gap_um.
    """
    try:
        import gdspy
    except ImportError:
        logger.error("gdspy not installed. Run: pip install gdspy")
        return output_path

    from export_gds import (
        _rect, _label,
        L_BOUND, L_INTERPOSER,
        L_STACK_CROSS, L_STACK_LABEL,
        L_FIBER_RIBBON, L_BOND_PAD_ARRAY,
        L_CHIPLET_BOUND, L_ASSEMBLY_LABEL,
    )

    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    top = lib.new_cell("STACK_CROSS_SECTION")

    chip_w  = 800.0   # schematic chiplet width (scaled down from 10mm for readability)
    gap_um  = 120.0   # inter-chiplet gap
    margin  = 200.0   # left/right margin

    total_w = n_chiplets * chip_w + (n_chiplets - 1) * gap_um + 2 * margin

    # Y layer boundaries (μm)
    pcb_y0      = 0.0
    pcb_y1      = 200.0
    underfill_y0 = pcb_y1
    underfill_y1 = 350.0
    interp_y0   = underfill_y1
    interp_y1   = 500.0
    oxide_y0    = interp_y1
    oxide_y1    = 505.0
    die_y0      = oxide_y1
    die_y1      = 725.0
    cladding_y0 = die_y1
    cladding_y1 = 730.0
    fiber_y0    = cladding_y1
    fiber_y1    = 790.0

    # ── PCB substrate ──────────────────────────────────────────────────────────
    pcb_cell = lib.new_cell("LAYER_PCB")
    _rect(pcb_cell, 0, pcb_y0, total_w, pcb_y1, L_STACK_CROSS)
    top.add(gdspy.CellReference(pcb_cell, (0, 0)))
    top.add(gdspy.Label("PCB Substrate (FR4)", (total_w / 2, (pcb_y0 + pcb_y1) / 2),
                         "o", magnification=20, texttype=L_STACK_LABEL))

    # ── Underfill epoxy ────────────────────────────────────────────────────────
    uf_cell = lib.new_cell("LAYER_UNDERFILL")
    _rect(uf_cell, margin, underfill_y0, total_w - margin, underfill_y1, L_BOND_PAD_ARRAY)
    top.add(gdspy.CellReference(uf_cell, (0, 0)))
    top.add(gdspy.Label("Underfill epoxy", (total_w / 2, (underfill_y0 + underfill_y1) / 2),
                         "o", magnification=15, texttype=L_STACK_LABEL))

    # ── Silicon interposer ─────────────────────────────────────────────────────
    ip_cell = lib.new_cell("LAYER_INTERPOSER")
    _rect(ip_cell, margin, interp_y0, total_w - margin, interp_y1, L_INTERPOSER)
    # TSV lines through interposer
    n_tsvs = n_chiplets * 6
    for i in range(n_tsvs):
        tx = margin + (total_w - 2 * margin) * (i + 0.5) / n_tsvs
        top.add(gdspy.Rectangle((tx - 3, interp_y0), (tx + 3, interp_y1),
                                 layer=L_FIBER_RIBBON))
    top.add(gdspy.CellReference(ip_cell, (0, 0)))
    top.add(gdspy.Label("Silicon Interposer (w/ TSVs)", (total_w / 2, (interp_y0 + interp_y1) / 2),
                         "o", magnification=15, texttype=L_STACK_LABEL))

    # ── Bond oxide ──────────────────────────────────────────────────────────────
    _rect(top, margin, oxide_y0, total_w - margin, oxide_y1, L_STACK_CROSS)

    # ── Photonic chip dies + C4 micro-bumps ────────────────────────────────────
    die_cell = lib.new_cell("LAYER_PHOTONIC_DIE")
    bump_cell = lib.new_cell("LAYER_C4_BUMPS")

    for ci in range(n_chiplets):
        cx0 = margin + ci * (chip_w + gap_um)
        cx1 = cx0 + chip_w

        # Die outline
        _rect(die_cell, cx0 + 5, die_y0, cx1 - 5, die_y1, L_BOUND)

        # Si waveguide core (thin stripe inside die)
        wg_y = die_y0 + (die_y1 - die_y0) * 0.6
        _rect(die_cell, cx0 + 10, wg_y - 2, cx1 - 10, wg_y + 2, 1)  # L_SI=1

        # TiN heater stripes (schematic)
        for hi in range(4):
            hx0 = cx0 + 20 + hi * (chip_w - 40) / 4
            hx1 = hx0 + (chip_w - 40) / 5
            _rect(die_cell, hx0, wg_y + 3, hx1, wg_y + 7, 2)  # L_THETA=2

        # Die label
        chiplet_names = ["Attention\nChiplet", "FFN\nChiplet", "FPGA\nChiplet"]
        lname = chiplet_names[ci % len(chiplet_names)]
        top.add(gdspy.Label(lname,
                             (cx0 + chip_w / 2, die_y0 + (die_y1 - die_y0) * 0.25),
                             "o", magnification=18, texttype=L_STACK_LABEL))

        # C4 micro-bumps (between interposer and die)
        n_bumps = 8
        for bi in range(n_bumps):
            bx = cx0 + 20 + bi * (chip_w - 40) / n_bumps
            bump_cell.add(gdspy.Round((bx, oxide_y0), 8, layer=L_BOND_PAD_ARRAY))

    top.add(gdspy.CellReference(die_cell, (0, 0)))
    top.add(gdspy.CellReference(bump_cell, (0, 0)))

    # ── Top cladding ──────────────────────────────────────────────────────────
    _rect(top, margin, cladding_y0, total_w - margin, cladding_y1, L_STACK_CROSS)
    top.add(gdspy.Label("SiO2 cladding",
                         (margin + 50, (cladding_y0 + cladding_y1) / 2),
                         "nw", magnification=12, texttype=L_STACK_LABEL))

    # ── Fiber array V-groove assembly ─────────────────────────────────────────
    fiber_cell = lib.new_cell("LAYER_FIBER_ARRAY")
    for ci in range(n_chiplets):
        cx0 = margin + ci * (chip_w + gap_um)
        # V-groove block
        _rect(fiber_cell, cx0 + 10, fiber_y0, cx0 + chip_w - 10, fiber_y1,
              L_FIBER_RIBBON)
        # Fiber lines (angled at ~10 degrees schematically — shown as vertical stubs)
        n_fibers = 8
        for fi in range(n_fibers):
            fx = cx0 + 20 + fi * (chip_w - 40) / n_fibers
            fiber_cell.add(gdspy.Rectangle(
                (fx - 2, fiber_y1), (fx + 2, fiber_y1 + 80),
                layer=L_FIBER_RIBBON
            ))
    top.add(gdspy.CellReference(fiber_cell, (0, 0)))
    top.add(gdspy.Label("Fiber Array (V-groove, 127um pitch)",
                         (total_w / 2, (fiber_y0 + fiber_y1) / 2),
                         "o", magnification=15, texttype=L_STACK_LABEL))

    # ── Overall boundary ──────────────────────────────────────────────────────
    _rect(top, 0, 0, total_w, fiber_y1 + 100, L_CHIPLET_BOUND)

    # ── Title and scale ───────────────────────────────────────────────────────
    top.add(gdspy.Label(
        "PhotoMedGemma - 2.5D Silicon Interposer Cross-Section  (RECOMMENDED stacking strategy)",
        (total_w / 2, fiber_y1 + 60), "o", magnification=20,
        texttype=L_ASSEMBLY_LABEL
    ))
    top.add(gdspy.Label(
        "Vertical scale: 1 um = 1 um  |  Horizontal scale: schematic only",
        (total_w / 2, -60), "o", magnification=15,
        texttype=L_ASSEMBLY_LABEL
    ))

    # ── Write GDS ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(str(output_path))
    size_kb = output_path.stat().st_size // 1024
    logger.info(f"Cross-section GDS: {output_path}  ({size_kb} KB)")
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze 3D stacking options for PhotoMedGemma photonic chiplets"
    )
    parser.add_argument(
        "--output-dir", default="output/stacking_analysis",
        help="Output directory (default: output/stacking_analysis)"
    )
    parser.add_argument(
        "--n-cross-section", type=int, default=3,
        help="Number of chiplets to show in cross-section diagram (default: 3)"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip GDS cross-section, write report only"
    )
    args = parser.parse_args()

    # Add scripts dir to path for export_gds imports
    sys.path.insert(0, str(ROOT / "scripts"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    options = evaluate_all_options()

    # ── Stacking report ───────────────────────────────────────────────────────
    report_path = output_dir / "stacking_report.txt"
    report = generate_stacking_report(options, report_path)
    print(report)

    if not args.report_only:
        # ── Cross-section GDS ─────────────────────────────────────────────────
        cross_section_path = output_dir / "stack_cross_section.gds"
        generate_cross_section_gds(cross_section_path, n_chiplets=args.n_cross_section)

        # Also write KLayout properties for the cross-section
        from export_gds import write_klayout_props
        write_klayout_props(output_dir, filename="stack_cross_section.lyp")

        print()
        print("=" * 72)
        print("Stacking Analysis Complete")
        print("=" * 72)
        print(f"  Report:        {report_path}")
        print(f"  Cross-section: {cross_section_path}")
        print()
        print("  Open cross-section in KLayout:")
        print(f"    klayout {cross_section_path} \\")
        print(f"            -l {output_dir}/stack_cross_section.lyp")
        print("=" * 72)


if __name__ == "__main__":
    main()
