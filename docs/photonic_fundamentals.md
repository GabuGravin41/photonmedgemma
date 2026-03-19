# Photonic Fundamentals for Neural Network Engineers

This document explains the photonic building blocks used in PhotoMedGemma, written for engineers with an ML background who may not have a photonics background.

## 1. Waveguides — The "Wire" of Photonics

In electronics, a wire carries electrical current. In photonics, a **waveguide** carries light.

A silicon photonic waveguide is a rectangular strip of silicon (refractive index n≈3.47) surrounded by silicon dioxide (n≈1.44). The difference in refractive index traps light inside the silicon core via total internal reflection. A typical single-mode waveguide is 450 nm wide × 220 nm tall.

Key properties:
- **Loss**: ~2–3 dB/cm (light gets weaker as it travels)
- **Speed**: light travels at ~200,000 km/s in Si waveguide
- **Bandwidth**: can carry multiple wavelengths simultaneously (WDM)

## 2. Directional Coupler — The "Transistor" of Photonics

When two waveguides are brought very close together (~200 nm gap), light evanescently couples between them. The fraction of light that couples depends on the coupling length and gap.

A **directional coupler** with coupling ratio κ implements:
```
[E_out1]   [cos(κL)    i·sin(κL)] [E_in1]
[E_out2] = [i·sin(κL)  cos(κL)  ] [E_in2]
```

A **50:50 coupler** (κL = π/4) splits light equally — equivalent to a beamsplitter or Hadamard gate.

## 3. Phase Shifter — "Tuning" the Light

A phase shifter shifts the optical phase of light passing through a waveguide without affecting its amplitude. Two implementations:

### Thermal Phase Shifter (TPS)
A resistive heater above the waveguide changes the silicon temperature, which changes its refractive index (thermo-optic effect, dn/dT ≈ 1.8×10⁻⁴ /K).
- Phase shift: Δφ = (2π/λ) · (dn/dT) · ΔT · L
- Power: ~10–20 mW for π phase shift
- Speed: ~1–10 kHz (slow — thermal response)
- **For static weights: set once, ~0 power in steady state**

### Electro-Optic Phase Shifter (EOPS)
Uses the plasma dispersion effect in silicon (carrier injection/depletion) to change refractive index.
- Speed: ~10–40 GHz (fast)
- Power: ~1–5 mW
- Used for dynamic, high-speed weight updates

For **static compilation** (our approach), thermal phase shifters are preferred — they require zero steady-state power once the phase is set.

## 4. Mach-Zehnder Interferometer (MZI) — The Core Compute Unit

An MZI consists of:
1. An input 50:50 directional coupler (beamsplitter)
2. Two arms with independent phase shifters (θ, φ)
3. An output 50:50 directional coupler (combiner)

```
Input 1 ──┬──[θ]──┬── Output 1
           │       │
           DC      DC
           │       │
Input 2 ──┴──[φ]──┴── Output 2
```

The transfer matrix of a single MZI is:
```
T(θ, φ) = e^(iφ/2) · [ cos(θ/2)·e^(iφ)   i·sin(θ/2) ]
                       [ i·sin(θ/2)         cos(θ/2)   ]
```

By choosing θ and φ appropriately, an MZI can implement **any 2×2 unitary transformation** on two optical modes.

**This is the key**: a 2×2 matrix multiplication done by physics, at the speed of light, with no power cost during the computation itself.

## 5. MZI Mesh — Implementing Arbitrary Matrices

To implement an N×N matrix using MZIs, we use a **mesh** of N(N-1)/2 MZIs arranged in a specific topology.

### Reck Decomposition (1994)
A triangular array of MZIs. N=4 requires 6 MZIs:
```
○─[MZI]─[MZI]─[MZI]─
○────────[MZI]─[MZI]─
○─────────────[MZI]──
○────────────────────
```
Requires N(N-1)/2 MZIs and N-1 depth stages.

### Clements Decomposition (2016) ← We use this
A rectangular (balanced) array. More robust to fabrication errors, shorter depth:
```
○─[MZI]──────[MZI]──
○────────[MZI]──────
○─[MZI]──────[MZI]──
○────────[MZI]──────
```
Requires N(N-1)/2 MZIs and N depth stages.

**The Clements decomposition is key**: any N×N unitary matrix U can be decomposed into a product of 2×2 MZI rotations:
```
U = D · T_{N,N-1}(θ,φ) · ... · T_{2,1}(θ,φ) · T_{1,2}(θ,φ) · ...
```
where D is a diagonal phase matrix and T_{i,j} are individual MZI matrices.

## 6. Singular Value Decomposition — Mapping Weight Matrices

Neural network weight matrices are not unitary in general. We use SVD to decompose them:

```
W = U · Σ · V†
```

Where:
- **U** (m×m): left singular vectors — unitary → implemented as MZI mesh
- **Σ** (m×n): diagonal singular values → implemented as optical attenuators
- **V†** (n×n): right singular vectors — unitary → implemented as MZI mesh

This means **any weight matrix** can be implemented as:
1. First MZI mesh (V†) — rotates the input
2. Diagonal attenuators (Σ) — scales the modes
3. Second MZI mesh (U) — rotates into output space

The computation `y = Wx` is then performed by:
1. Light enters input waveguides encoding `x`
2. Passes through V† MZI mesh
3. Gets attenuated/amplified at Σ stage
4. Passes through U MZI mesh
5. Exits at output waveguides encoding `y = Wx`

All of this happens at the speed of light.

## 7. Optical Nonlinearity — The Hard Part

Linear operations are easy in photonics. Nonlinear operations (like GELU, sigmoid, ReLU) are harder.

### Option A: Electro-Optic (O→E→O)
- Photodetectors convert optical signal to electrical
- Electronic circuit applies nonlinearity
- Optical modulator re-encodes result as light
- Penalty: ~100–500 ps latency per nonlinearity, power cost

### Option B: Semiconductor Optical Amplifiers (SOA)
- SOAs have intrinsic nonlinear gain saturation
- Can approximate sigmoid-like functions
- All-optical: fast and low power

### Option C: Approximate Linear-Only Networks
- Use only linear layers (no activation) — approximates transformer inference
- Some works show surprisingly good accuracy with polynomial or piecewise-linear activation approximations

**Our approach**: Hybrid O→E→O for activations, with O-E-O conversion latency included in power/latency estimates.

## 8. Wavelength-Division Multiplexing (WDM)

A single waveguide can carry many wavelengths simultaneously. Using N wavelengths gives:
- N parallel matrix-vector multiplications at once
- N× throughput with no additional MZI hardware
- Only additional laser sources needed

For attention heads: each attention head can use a different wavelength, processing all heads truly in parallel. MedGemma's 8 attention heads would use 8 wavelengths.

## 9. Key Numbers for MedGemma Compilation

| Parameter | Value |
|-----------|-------|
| MedGemma hidden dim (d_model) | 2048 |
| MZIs per d_model attention projection | 2048×2047/2 ≈ 2.1M per matrix |
| Number of attention weight matrices (per layer) | 4 (Q, K, V, O) |
| Number of FFN matrices (per layer) | 3 (up, gate, down) |
| Number of transformer layers | 46 |
| Total MZI count (language model only) | ~2.7 billion |

This is the fundamental challenge: 4B parameters → ~2.7B MZIs. Current silicon photonic chips integrate ~10,000–100,000 devices. The full chip requires a multi-chip module or **3D photonic integration** — an active area of research.

**Practical path**: Start with quantized, lower-rank approximations (e.g., rank-64 SVD truncation) that dramatically reduce MZI count while preserving accuracy.

## 10. Platform Parameters (Silicon Photonics — 220nm SOI)

```
Wavelength:          1310 nm or 1550 nm
Waveguide pitch:     2–4 μm (single mode)
MZI footprint:       ~50 μm × 200 μm (0.01 mm²)
Phase shifter power: ~10 mW (thermal, π shift)
Propagation loss:    ~2 dB/cm
Coupling loss:       ~1–3 dB (fiber to chip)
Photodetector BW:    >40 GHz
DAC resolution:      8–12 bits (phase setting)
```

With a 300mm wafer and 10×10mm chip size, we can fit ~900 chips per wafer. A chip of 10mm² accommodates approximately **10,000–50,000 MZIs** depending on routing complexity — enough for several transformer attention heads at low rank.
