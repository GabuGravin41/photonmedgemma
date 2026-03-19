# PhotoMedGemma — Simulation & Compilation Outputs

This directory contains all outputs from the PhotoMedGemma compilation and
simulation pipeline. These artifacts are preserved for reproducibility and
potential publication.

## Directory Structure

```
output/
├── phase_map_demo.json      Compiled phase map (JSON) — 121,568 phase entries
├── phase_map_demo.phcfg     Compiled phase map (binary) — SPI-ready format
│
├── gds/                     GDS-II chip masks (KLayout-viewable)
│   ├── chip_000.gds         Chip 0: 64-mode attention Q projection
│   ├── chip_001.gds         Chip 1: 64-mode attention K projection
│   ├── ...
│   ├── photomedgemma.lyp    KLayout layer properties (color scheme)
│   └── gds_manifest.json    Chip dimensions and MZI counts
│
├── simulations/             Medical image inference simulations
│   ├── results_*.json       Layer-wise error statistics
│   ├── errors_*.npy         Error matrix [n_layers × n_patches]
│   ├── outputs_ref_*.npy    NumPy reference outputs
│   ├── outputs_phot_*.npy   Photonic simulation outputs
│   ├── comparison_*.svg     Visual comparison plots
│   └── input_*.svg          Input image thumbnails
│
├── tests/                   Test suite outputs
│   ├── test_results.json    Test pass/fail summary
│   └── clements_accuracy/   Clements reconstruction error vs N
│
├── netlists/                SPICE-style photonic netlists
│   ├── medgemma_demo.pntl   Full model netlist (125K lines)
│   └── manifest_demo.json   Layer manifest with MZI counts
│
└── layout/                  SVG chip layout diagrams
    ├── chip_000.svg         Per-chip MZI phase heat maps
    ├── ...
    └── layout_manifest.json Physical chip dimensions
```

## Key Numbers (Demo Compilation)

| Metric | Value |
|--------|-------|
| Transformer layers compiled | 7 (1 block: Q, K, V, O, gate, up, down) |
| Total MZIs | 120,096 |
| Total photonic chips | 10 |
| Clements reconstruction error | ~2×10⁻¹⁵ (machine precision) |
| Photonic simulation error vs SVD | < 10⁻⁶ |
| Chip size (64-mode mesh) | 21.2mm × 9.0mm |
| Chip size (256-mode mesh) | 82.6mm × 33.4mm |
| Phase entries (DAC codes) | 121,568 |
| SPI bitstream (chip 0) | 32,768 bytes |

## To Regenerate All Outputs

```bash
# 1. Compile photonic chip configuration
python3 scripts/compile_model.py --demo --output-dir output/

# 2. Generate GDS chip masks (KLayout)
python3 scripts/export_gds.py --phase-map output/phase_map_demo.json --output-dir output/gds/

# 3. Generate SVG layout diagrams
python3 scripts/generate_layout.py --phase-map output/phase_map_demo.json --svg-only

# 4. Run medical image simulation
python3 scripts/simulate_medical.py --demo --save-arrays --output-dir output/simulations/

# 5. Run test suite
python3 -m pytest tests/ -v --tb=short 2>&1 | tee output/tests/test_results.txt
```

## Viewing GDS in KLayout

```bash
# Install KLayout (free): https://www.klayout.de/build.html
klayout output/gds/chip_000.gds -l output/gds/photomedgemma.lyp

# Or via AppImage / package manager:
sudo apt install klayout   # Ubuntu/Debian
brew install klayout       # macOS
```

GDS layer color scheme:
- Blue  (#00aaff): Silicon waveguides
- Red   (#ff4444): TiN heaters (θ — main phase)
- Orange(#ff9900): TiN heaters (φ — input phase)
- Yellow(#ffdd00): Metal contacts
- Gray  (outline): Chip boundary
