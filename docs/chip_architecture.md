# Chip Architecture: Physical Design of the PhotoMedGemma Chip

## Design Philosophy

The chip architecture follows a **tile-based** design, where each tile implements one weight matrix as a photonic MZI mesh. Multiple tiles are connected in sequence to implement the full transformer forward pass.

## Platform: 220nm Silicon-on-Insulator (SOI)

We target the **220nm SOI** process available through:
- **imec** (Belgium) — iSiPP50G platform
- **AIM Photonics** (USA) — MPW run accessible
- **GlobalFoundries** — SiPh 45CLO process

Key process parameters:
```
Waveguide core: Si, 450nm wide × 220nm tall
Cladding: SiO₂
Wavelength: 1310nm (O-band, lower dispersion)
Phase shifter: Thermal (TiN heater, 2μm above waveguide)
Heater resistance: ~1 kΩ
Power for π shift: ~10–20 mW
Thermo-optic coefficient: dn/dT = 1.84×10⁻⁴ K⁻¹
Propagation loss: 2–3 dB/cm
Coupling loss (grating): 2–4 dB per coupler
Detector: Ge photodiode, >40 GHz BW, responsivity ~0.9 A/W
```

## Chip Hierarchy

```
PhotoMedGemma System
    ├── Chip Module Array (tiled across PCB)
    │   ├── Chip_0: Q projection, Layer 0–2
    │   ├── Chip_1: K projection, Layer 0–2
    │   ├── Chip_2: V projection, Layer 0–2
    │   ├── Chip_3: O projection, Layer 0–2
    │   ├── Chip_4–6: FFN, Layer 0–2
    │   └── ... (repeat for layers 3–45)
    │
    ├── Electronic Control Board
    │   ├── Microcontroller (phase DAC control)
    │   ├── ADC array (photodetector readout)
    │   ├── Softmax / LayerNorm compute (FPGA)
    │   └── Memory (KV cache, embeddings)
    │
    └── Optical I/O
        ├── Laser array (8× WDM wavelengths)
        ├── Fiber array (input coupling)
        └── Fiber array (output coupling)
```

## Single Chip Layout (10mm × 10mm)

```
┌─────────────────────────────────────────────────────────────────┐
│                      10mm × 10mm Chip                           │
│                                                                 │
│  ┌──────────┐  ┌───────────────────────────────┐  ┌─────────┐ │
│  │ Grating  │  │                               │  │ Grating │ │
│  │ Couplers │  │     Clements MZI Mesh         │  │Couplers │ │
│  │  (input) │  │                               │  │(output) │ │`
│  │          │  │  ┌─────────────────────────┐  │  │         │ │
│  │  64 I/O  │──│─▶│  Stage 0 (32 MZIs)      │  │  │ 64 I/O │ │
│  │ couplers │  │  │  Stage 1 (32 MZIs)      │  │──│▶couplers│ │
│  │          │  │  │  ...                    │  │  │         │ │
│  │          │  │  │  Stage 127 (32 MZIs)    │  │  │         │ │
│  └──────────┘  │  └─────────────────────────┘  │  └─────────┘ │
│                │                               │               │
│                │  Total: 4,096 MZIs            │               │
│                │  Footprint: 8mm × 8mm         │               │
│                └───────────────────────────────┘               │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           Heater Control Routing (metal layer)             │ │
│  │  1024 heater lines → bond pads → DAC chips                │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Capacity per chip**: 4,096 MZIs → implements a 64×64 Clements unitary (64×63/2 = 2,016 MZIs) or a rank-64 projection of a larger matrix.

## Multi-Chip Module for Full Attention Head

For a single attention head (head_dim=256, rank=64):
- V† mesh: 256×64 = requires 8 chips
- Σ stage: 1 chip (attenuators)
- U mesh: 2048×64 = requires 32 chips

Total per attention head: ~41 chips

For all 8 attention heads (WDM parallel): 41 chips (wavelength-multiplexed)
For all 7 weight matrices per layer: ~280 chips per layer
For all 46 layers: ~12,880 chips total

At $5–20 per chip in volume: **$65K–$260K total chip cost** for full MedGemma photonic compilation.

Note: This is a proof-of-concept costing. 3D integration and higher-density processes (100nm, 45nm) reduce this dramatically.

## Data Flow: Single Transformer Layer

```
Input activations (d_model=2048 floats)
    │
    ▼ [Electronic → Optical]
Optical modulator array (2048 channels)
    │
    ├─────────────────────────────────────┐
    │                                     │
    ▼ [Q projection chip array]           ▼ [Residual skip connection]
V†_Q mesh → Σ_Q → U_Q                   (fiber delay line)
    │
    ▼ [Q head output, 256 optical modes]

    ▼ [K projection chip array]
V†_K mesh → Σ_K → U_K
    │
    ▼ [K head output, 128 optical modes × 2 GQA groups]

    ▼ [V projection chip array]
V†_V mesh → Σ_V → U_V
    │
    ▼ [V head output, 128 optical modes]

    │ [Photodetector array → Electronic]
    ▼
Attention scores: Q·K^T (electronic, softmax)
    │
    ▼
Weighted V sum (electronic)
    │
    ▼ [Electronic → Optical]
    │
    ▼ [O projection chip array]
V†_O mesh → Σ_O → U_O
    │
    ▼ [O projection output, 2048 optical modes]
    │
    ▼ [Photodetector → Electronic]
    │
+ [Residual connection]
    │
LayerNorm (electronic)
    │
    ▼ [Electronic → Optical]

    ▼ [FFN gate projection]
V†_gate mesh → Σ_gate → U_gate → φ(gate_output) [GELU, electronic]
    │
    ▼ [FFN up projection]
V†_up mesh → Σ_up → U_up → up_output
    │
× [Element-wise multiply: gate ⊗ up] [Electronic]
    │
    ▼ [FFN down projection]
V†_down mesh → Σ_down → U_down
    │
    ▼ [Photodetector → Electronic]
    │
+ [Residual connection]
    │
LayerNorm (electronic)
    │
    ▼ [Output to next layer]
```

## Power Budget

### Per Transformer Layer (rank-64, 46 layers)

| Component | Power | Notes |
|-----------|-------|-------|
| Laser sources (8× WDM) | 800 mW | ~100mW per wavelength |
| Phase shifters (static) | ~0 mW | Thermal, no steady-state draw |
| Phase setting (one-time) | N/A | Done at boot, then frozen |
| Photodetectors | 50 mW | Ge PD array |
| Electronic control (FPGA) | 5 W | Softmax, LayerNorm, routing |
| Memory (KV cache) | 2 W | LPDDR5 |
| **Total** | **~8 W** | **vs ~300W for GPU inference** |

Energy per token: ~8W × (inference time in seconds per token)
At 1 token/ms: **8 mJ/token** vs ~300 mJ/token for GPU (A100)
**Improvement: ~37× better energy efficiency**

## Thermal Management

Silicon photonic phase shifters have a thermo-optic coefficient of ~1.8×10⁻⁴ K⁻¹. Ambient temperature changes shift the effective refractive index and thus the phase.

For static weight operation, we require a temperature-stabilized chip:
- On-chip temperature sensors (doped Si thermistors)
- Closed-loop PID controller adjusts global chip temperature
- Target stability: ±0.01°C
- Required TEC (thermoelectric cooler): ~2W at ΔT=10°C

This is far simpler than GPU thermal management (requiring 300W+ active cooling).

## Interface Specifications

### Optical I/O
- Input: 64-channel fiber array, single-mode fiber, FC/APC connectors
- Output: 64-channel fiber array
- Coupling: edge coupling or grating coupling (process-dependent)
- WDM: 8 wavelengths, 1310nm ± 400 GHz channel spacing (ITU-T C-band compatible)

### Electrical I/O
- Phase control: 4096 heater channels per chip, 16-bit DAC, SPI interface
- Photodetector readout: 128-channel ADC, 12-bit, 1 MHz sampling rate
- Power: 3.3V digital, 5V analog, 12V for TEC
- Interface: PCIe ×4 to host FPGA/CPU

### Form Factor
- Chip: 10mm × 10mm die, wire-bonded to ceramic package
- Module: Multi-chip on ceramic substrate, 100mm × 100mm
- System: PCIe card or standalone box (RPi-sized for single-layer demo)
