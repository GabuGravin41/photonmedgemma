# Compilation Pipeline: From MedGemma Weights to Photonic Phase Angles

## Overview

The compilation pipeline takes a trained MedGemma model and produces:
1. A set of phase angles for every MZI on the photonic chip
2. A photonic netlist (SPICE-like description of the circuit)
3. A GDS layout file for chip fabrication

```
MedGemma (HuggingFace)
    │
    ▼
[1] Model Parser          — loads weights, extracts linear layers
    │
    ▼
[2] Layer Decomposer      — SVD decomposition per weight matrix
    │
    ▼
[3] Clements Decomposer   — converts unitary matrices to MZI phase angles
    │
    ▼
[4] Phase Encoder         — quantizes phases to DAC resolution, assigns chip addresses
    │
    ▼
[5] Netlist Generator     — outputs photonic SPICE netlist (.pntl)
    │
    ▼
[6] GDS Generator         — outputs physical layout (.gds) for foundry
```

---

## Stage 1: Model Parser

**Input**: HuggingFace model identifier (`google/medgemma-4b-it`)
**Output**: Dictionary of `{layer_name: weight_tensor}`

### What We Extract

For each transformer layer, we extract the following weight matrices:
- `model.layers.{i}.self_attn.q_proj.weight`  — Query projection
- `model.layers.{i}.self_attn.k_proj.weight`  — Key projection
- `model.layers.{i}.self_attn.v_proj.weight`  — Value projection
- `model.layers.{i}.self_attn.o_proj.weight`  — Output projection
- `model.layers.{i}.mlp.gate_proj.weight`     — FFN gate (GeGLU)
- `model.layers.{i}.mlp.up_proj.weight`       — FFN up projection
- `model.layers.{i}.mlp.down_proj.weight`     — FFN down projection

For the vision encoder (SigLIP):
- `vision_tower.vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight`
- `vision_tower.vision_model.encoder.layers.{i}.mlp.fc{1,2}.weight`

### Implementation Notes
- Weights are loaded in bfloat16 and converted to float32 for SVD stability
- We apply layer-wise weight normalization tracking to preserve scale information
- RMSNorm weights are recorded separately (implemented electronically)

---

## Stage 2: Layer Decomposer (SVD)

**Input**: Weight matrix W of shape (m, n)
**Output**: U (m×R), Σ (R), V† (n×R), truncation rank R

### Algorithm

```python
def decompose_layer(W, rank=None, energy_threshold=0.99):
    # 1. Compute full SVD
    U, S, Vh = np.linalg.svd(W, full_matrices=False)

    # 2. Determine truncation rank
    if rank is None:
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        rank = np.searchsorted(cumulative_energy, energy_threshold) + 1

    # 3. Truncate
    U_r  = U[:, :rank]      # shape (m, rank)
    S_r  = S[:rank]         # shape (rank,)
    Vh_r = Vh[:rank, :]     # shape (rank, n)

    return U_r, S_r, Vh_r, rank
```

### Rank Selection Strategy

| Target | Energy Threshold | Expected Rank | MZI Reduction |
|--------|-----------------|---------------|---------------|
| High accuracy | 99.9% | ~256–512 | 16× |
| Balanced | 99% | ~64–128 | 64× |
| Aggressive | 95% | ~16–32 | 256× |
| Demo/PoC | 90% | ~8–16 | 1024× |

### Padding to Power-of-2

Clements decomposition is defined for square N×N matrices. For rectangular decomposition (m×R or n×R), we use the **rectangular Clements extension**:
- U_r of shape (m×R) is embedded into an m×m unitary by extending with an orthonormal complement
- Only the first R columns are used at the output (others are blocked by attenuators)

---

## Stage 3: Clements Decomposition

**Input**: Unitary matrix U of shape (N×N)
**Output**: List of (layer, position, θ, φ) tuples defining MZI settings

### Clements Algorithm (2016)

The Clements decomposition factors any N×N unitary into a product of 2×2 MZI unitaries:

```
U = D · M_{N-1,N}(θ,φ) · M_{N-3,N-2}(θ,φ) · ... · M_{1,2}(θ,φ) · ...
```

where M_{i,j}(θ,φ) is an MZI acting on modes i and j (identity on all other modes).

**Key property**: The decomposition uses exactly N(N-1)/2 MZIs — provably optimal.

### Algorithm Steps

```python
def clements_decompose(U):
    N = U.shape[0]
    mzis = []  # list of (row, col, theta, phi)

    U_working = U.copy()

    # Column elimination (left to right, bottom to top)
    for col in range(N - 1):
        for row in range(N - 1, col, -1):
            # Find MZI parameters to zero out U[row, col]
            theta, phi = find_nulling_mzi(U_working, row, col)

            # Apply MZI from the right (V† side)
            T = mzi_matrix(theta, phi)
            apply_mzi_right(U_working, T, row-1, row)

            mzis.append((row-1, col, theta, phi))

    # Remaining diagonal = phase screen D
    D = np.diag(U_working)
    phases = np.angle(D)

    return mzis, phases

def find_nulling_mzi(U, row, col):
    """Find (theta, phi) such that applying MZI zeros out U[row, col]."""
    a = U[row-1, col]
    b = U[row, col]

    phi = np.angle(b) - np.angle(a) + np.pi
    theta = 2 * np.arctan2(np.abs(b), np.abs(a))

    return theta, phi
```

### Output Format

```json
{
  "layer_name": "model.layers.0.self_attn.q_proj",
  "matrix_type": "U",
  "N": 2048,
  "rank": 64,
  "mzis": [
    {"mesh_row": 0, "mesh_col": 0, "theta": 1.2345, "phi": 0.5678},
    {"mesh_row": 0, "mesh_col": 1, "theta": 0.9876, "phi": 2.3456},
    ...
  ],
  "phase_screen": [0.123, 0.456, ...]
}
```

---

## Stage 4: Phase Encoder

**Input**: Floating-point phase angles θ, φ ∈ [0, 2π)
**Output**: Integer DAC codes (B-bit), chip addresses, calibration metadata

### Quantization

```python
def quantize_phase(theta, bits=12):
    """Quantize phase angle to B-bit DAC code."""
    # Normalize to [0, 1)
    theta_norm = (theta % (2 * np.pi)) / (2 * np.pi)

    # Quantize
    max_code = 2**bits - 1
    code = int(round(theta_norm * max_code))

    return code

def dequantize_phase(code, bits=12):
    """Recover phase from DAC code."""
    return (code / (2**bits)) * 2 * np.pi
```

### Phase Error from Quantization

For B-bit quantization of a phase in [0, 2π):
- LSB = 2π / 2^B
- Maximum error = π / 2^B
- RMS error ≈ π / (√3 · 2^B)

For B=12 bits: RMS phase error = 0.00077 rad ≈ 0.044°
This is well below typical fabrication errors (~0.01–0.1 rad).

### Chip Address Mapping

Each MZI on the chip has a unique address (chip_id, row, col). The phase encoder maps:
```
(layer_name, matrix_type, mesh_row, mesh_col) → (chip_id, mzi_row, mzi_col, phase_register)
```

This is stored in a `PhaseMap` data structure that serves as the interface between the compiler and the chip control electronics.

---

## Stage 5: Netlist Generator

**Input**: Phase assignments for all MZIs
**Output**: Photonic SPICE netlist (`.pntl` format)

### Photonic Netlist Format (PNTL)

We define a human-readable photonic netlist format:

```
# PhotoMedGemma Photonic Netlist v1.0
# Layer: model.layers.0.self_attn.q_proj, Matrix: U, Rank: 64

.model MZI theta=0.0 phi=0.0
.model WAVEGUIDE length=100e-6 loss_db_cm=2.0
.model ATTENUATOR value=1.0
.model PHASE_SCREEN phase=0.0

# Clements mesh - 2048 input modes, rank 64
.subckt CLEMENTS_MESH_U N=2048 R=64

# Stage 0 (even columns)
X_0_0  in0  in1   MZI theta=1.2345 phi=0.5678  out0  out1
X_0_1  in2  in3   MZI theta=0.9876 phi=2.3456  out2  out3
...

# Stage 1 (odd columns)
X_1_0  mid1  mid2  MZI theta=0.7654 phi=1.2345  out4  out5
...

# Diagonal phase screen
X_D_0  sig0  PHASE_SCREEN phase=0.123  out64_0
...

.ends CLEMENTS_MESH_U

# Sigma (attenuators)
.subckt SIGMA_STAGE N=64
X_S_0  sig0  ATTENUATOR value=123.4  out_s0
...
.ends SIGMA_STAGE

# Full layer assembly
.subckt Q_PROJ_LAYER
X_Vh  IN[0:2048]   CLEMENTS_MESH_Vh  MID[0:64]
X_SIG MID[0:64]    SIGMA_STAGE       SCALED[0:64]
X_U   SCALED[0:64] CLEMENTS_MESH_U   OUT[0:2048]
.ends Q_PROJ_LAYER
```

---

## Stage 6: GDS Generator

**Input**: Photonic netlist
**Output**: GDS II layout file for foundry submission

### Using gdsfactory

We use the [gdsfactory](https://gdsfactory.github.io/) Python library to generate GDS layouts:

```python
import gdsfactory as gf

def generate_mzi_cell(theta, phi, waveguide_spec):
    """Generate a single MZI GDS cell."""
    c = gf.Component("MZI")

    # Input directional coupler
    dc_in = c.add_ref(gf.components.coupler(gap=0.2, length=10.0))

    # Phase shifter on upper arm
    ps_upper = c.add_ref(gf.components.straight_heater_metal(
        length=100.0,
        heater_width=2.5
    ))

    # Phase shifter on lower arm
    ps_lower = c.add_ref(gf.components.straight_heater_metal(
        length=100.0,
        heater_width=2.5
    ))

    # Output directional coupler
    dc_out = c.add_ref(gf.components.coupler(gap=0.2, length=10.0))

    # Connect components
    c.connect(dc_in.ports["o3"], ps_upper.ports["o1"])
    c.connect(dc_in.ports["o4"], ps_lower.ports["o1"])
    c.connect(ps_upper.ports["o2"], dc_out.ports["o1"])
    c.connect(ps_lower.ports["o2"], dc_out.ports["o2"])

    return c
```

### Layout Floorplan

For a rank-64 Clements mesh of size 2048×2048:
- Each MZI: ~50μm × 200μm
- Mesh depth: 2048 stages
- Mesh width: 1024 MZIs per stage (alternating even/odd)
- Total footprint: ~10mm × 100mm (one axis needs wafer-scale integration)

This confirms the need for **multi-chip modules** — the mesh is physically distributed across multiple chips connected by fiber arrays or edge-coupled waveguides.

---

## Compilation Time Estimates

| Stage | Time (single layer) | Time (full model) |
|-------|--------------------|--------------------|
| Model parsing | 30s (download) + 5s | 30s + 5s |
| SVD decomposition (rank-64) | ~2s per matrix | ~10 min |
| Clements decomposition (N=2048) | ~30s per matrix | ~3 hours |
| Phase encoding | <1s | <5 min |
| Netlist generation | ~5s | ~30 min |
| GDS generation | ~60s | ~8 hours |

The Clements decomposition is the computational bottleneck. We parallelize across matrices using multiprocessing.

---

## Accuracy Validation

To validate that our compilation preserves model accuracy, we implement a **software simulation** of the compiled photonic model:

```python
def simulate_photonic_layer(x, mzi_list, phases_D, sigma, mzi_list_Vh, phases_Vh):
    """Software simulation of a compiled photonic linear layer."""
    # Apply V†
    x = apply_clements_mesh(x, mzi_list_Vh, phases_Vh)

    # Apply Σ
    x = x[:len(sigma)] * sigma  # truncate to rank R

    # Apply U
    x = apply_clements_mesh(x, mzi_list, phases_D)

    return x
```

We compare this against the original weight matrix multiplication and measure:
- Reconstruction error: ‖W·x - W_photonic·x‖₂ / ‖W·x‖₂
- Task accuracy: MedGemma accuracy on medical QA benchmarks (MedQA, PubMedQA)
