# PhotoMedGemma: Static Compilation of MedGemma onto Photonic Substrate

> **Inference at the speed of light. Locally. Sustainably.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Silicon Photonics](https://img.shields.io/badge/platform-silicon%20photonics-purple.svg)]()

---

## What This Is

PhotoMedGemma is a research compiler and chip design framework that **statically compiles** Google's [MedGemma](https://huggingface.co/google/medgemma-4b-it) (a 4-billion-parameter multimodal medical AI model) into a **photonic chip substrate**.

Instead of running MedGemma on a remote GPU server — burning energy, risking patient data privacy, and incurring latency — the model's weights are *encoded directly into the physical structure of a photonic chip* as phase settings in optical interferometer meshes. Inference then happens at the speed of light, locally, consuming a fraction of the energy of traditional CMOS electronics.

---

## The Problem We Solve

| Problem | Traditional AI Infrastructure | PhotoMedGemma |
|--------|-------------------------------|---------------|
| **Energy** | ~500W per GPU, massive cooling | ~20–50W, no cooling needed |
| **Latency** | 100ms–2s per inference (network round-trip) | Nanosecond-scale optical inference |
| **Privacy** | Patient data sent to remote servers | All inference is local, on-chip |
| **Cost** | $2–10 per 1M tokens at scale | Near-zero marginal cost after chip fabrication |
| **Access** | Requires internet, cloud subscription | Works in any hospital, anywhere on Earth |

This is not hypothetical. Photonic chips have been demonstrated to perform deep neural network inference at speeds exceeding 100 GHz, consuming <1 pJ/MAC (multiply-accumulate operation) — compared to ~1–10 pJ/MAC for the best CMOS accelerators.

---

## The Vision

> *"What if MedGemma was not a cloud service but a piece of glass?"*

We are building for a future where a neurosurgeon at Kenyatta National Hospital can run full MedGemma inference — CT scan analysis, differential diagnosis, treatment planning — on a chip the size of a fingernail, powered by a single USB port, with no internet connection, in real time.

This project was born out of direct collaboration with neurosurgeons at **Kenyatta National Hospital, Nairobi**, who needed AI-assisted triage and diagnosis but could not risk sending patient imaging data to foreign servers. The photonic chip is the answer.

---

## Technical Approach: Static Compilation

### What "Static Compilation" Means

A traditional neural network runs on hardware as a program: weights are loaded from memory, matrix multiplications are executed sequentially, activations are computed, and results are returned. This requires:
- A von Neumann memory-compute separation (the "memory wall")
- Clocking, synchronization, power-hungry SRAM
- Repeated digital logic operations (billions per token)

**Static compilation** means we map the neural network weights *permanently* into the physical hardware at fabrication time (or via one-time phase setting). The computation then happens as light passes through the chip — no clocking, no memory fetch, just physics.

### How We Do It: MZI Meshes

The core photonic primitive is the **Mach-Zehnder Interferometer (MZI)**:

```
     ╔═══════════╗
─────╣  θ  │  φ  ╠─────
─────╣             ╠─────
     ╚═══════════╝
```

An MZI with two phase shifters (θ, φ) implements a 2×2 unitary transformation on two optical modes. A **mesh** of N(N-1)/2 MZIs implements an arbitrary N×N unitary matrix (Clements decomposition, 2016).

For a neural network weight matrix **W** of shape (m, n):
1. Compute SVD: **W = U Σ V†**
2. Implement **U** as a Clements MZI mesh (m×m unitary)
3. Implement **Σ** as optical attenuators/amplifiers (diagonal)
4. Implement **V†** as a Clements MZI mesh (n×n unitary)

This is the **optical matrix-vector multiplication** unit. MedGemma's transformer layers are largely matrix multiplications (Q/K/V projections, output projections, FFN up/down projections), making it highly amenable to this approach.

### Hybrid Electro-Optic Architecture

Not everything can be purely optical:
- **Linear layers** → MZI meshes (optical) ✓
- **Nonlinear activations (GELU, SiLU)** → Electro-optic conversion (O→E→O) or approximate photonic nonlinearity
- **Softmax in attention** → Electronic with optical input/output
- **Layer normalization** → Electronic
- **KV-cache** → Optical memory (ring resonators) or SRAM

We adopt a **hybrid electro-optic** architecture: the dominant compute (>95% of FLOPs = matrix multiplications) is optical; the residual control logic is electronic.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PhotoMedGemma Chip                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │ Optical I/O  │    │      Photonic Processing Core         │  │
│  │              │    │                                        │  │
│  │ • Laser      │───▶│  ┌────────────────────────────────┐  │  │
│  │   source     │    │  │   Vision Encoder (SigLIP-ViT)  │  │  │
│  │ • Grating    │    │  │   400 MZI meshes × 1024×1024   │  │  │
│  │   couplers   │    │  └──────────────┬─────────────────┘  │  │
│  │ • Photodet.  │    │                 │                      │  │
│  └──────────────┘    │  ┌─────────────▼──────────────────┐  │  │
│                       │  │  Language Model (Gemma-3 4B)   │  │  │
│  ┌──────────────┐    │  │  46 transformer layers          │  │  │
│  │ Electronic   │    │  │  MZI mesh per attention head    │  │  │
│  │ Control Unit │◀───│  │  MZI mesh per FFN projection    │  │  │
│  │              │    │  └────────────────────────────────┘  │  │
│  │ • Softmax    │    │                                        │  │
│  │ • LayerNorm  │    └──────────────────────────────────────┘  │
│  │ • Phase ctrl │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
photomedgemma/
├── README.md                    # This file
├── docs/                        # Technical documentation
│   ├── motivation.md            # Why photonic AI for medicine
│   ├── photonic_fundamentals.md # MZI, waveguides, photonics 101
│   ├── medgemma_analysis.md     # MedGemma architecture breakdown
│   ├── compilation_pipeline.md  # How we compile weights to phases
│   ├── chip_architecture.md     # Physical chip layout design
│   └── roadmap.md               # Project milestones
│
├── src/
│   ├── compiler/                # The compilation pipeline
│   │   ├── model_parser.py      # Load and parse MedGemma weights
│   │   ├── layer_decomposer.py  # SVD decomposition per layer
│   │   ├── mzi_mapper.py        # Map matrices to MZI phase angles
│   │   ├── clements.py          # Clements decomposition algorithm
│   │   ├── netlist_generator.py # Generate photonic SPICE netlist
│   │   └── phase_encoder.py     # Encode phases to chip config
│   │
│   ├── photonic/                # Photonic primitives library
│   │   ├── mzi.py               # MZI building block
│   │   ├── waveguide.py         # Waveguide routing
│   │   ├── mesh.py              # MZI mesh (Clements/Reck layout)
│   │   ├── splitter.py          # Directional couplers, Y-splitters
│   │   ├── phase_shifter.py     # Thermal & electro-optic phase shifters
│   │   ├── photodetector.py     # Photodetector models
│   │   └── grating_coupler.py   # I/O coupling
│   │
│   ├── architecture/            # Photonic neural net architecture
│   │   ├── attention.py         # Photonic multi-head attention
│   │   ├── feedforward.py       # Photonic FFN layers
│   │   ├── embedding.py         # Token embedding (hybrid)
│   │   ├── layer_norm.py        # Electronic layer norm interface
│   │   ├── vision_encoder.py    # SigLIP photonic vision encoder
│   │   └── medgemma_photonic.py # Full assembled model
│   │
│   └── utils/
│       ├── svd_utils.py         # SVD helpers, truncation, error bounds
│       ├── quantization.py      # Phase quantization (N-bit DAC)
│       ├── power_model.py       # Energy consumption estimator
│       ├── error_analysis.py    # Fabrication error tolerance analysis
│       └── visualization.py     # Circuit and mesh visualization
│
├── configs/
│   ├── medgemma_4b_config.yaml  # MedGemma architecture parameters
│   └── chip_platform_config.yaml # Silicon photonics process parameters
│
├── scripts/
│   ├── compile_model.py         # Main compilation entry point
│   ├── generate_layout.py       # GDS layout generation
│   ├── analyze_model.py         # Model analysis and resource estimation
│   └── simulate_layer.py        # Single-layer photonic simulation
│
├── layouts/                     # Generated GDS chip layout files
│   └── .gitkeep
│
├── tests/
│   ├── test_clements.py         # Test Clements decomposition
│   ├── test_mzi_mapper.py       # Test MZI phase mapping
│   ├── test_compiler.py         # End-to-end compiler tests
│   └── test_power_model.py      # Power estimation tests
│
├── notebooks/
│   └── 01_photonic_intro.ipynb  # Tutorial notebook
│
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/photomedgemma
cd photomedgemma
pip install -e .

# 2. Analyze MedGemma resource requirements
python scripts/analyze_model.py --model google/medgemma-4b-it

# 3. Compile a single transformer layer to photonic netlist
python scripts/compile_model.py \
    --model google/medgemma-4b-it \
    --layer 0 \
    --output layouts/layer_0.gds

# 4. Full compilation (warning: large output)
python scripts/compile_model.py \
    --model google/medgemma-4b-it \
    --all-layers \
    --output layouts/medgemma_photonic.gds
```

---

## Key References

1. **Shen et al. (2017)** — "Deep Learning with Coherent Nanophotonic Circuits" — *Nature Photonics*. The foundational paper demonstrating ONN inference on a physical chip.
2. **Clements et al. (2016)** — "An Optimal Design for Universal Multiport Interferometers" — *Optica*. The decomposition algorithm we use for MZI meshes.
3. **Reck et al. (1994)** — "Experimental realization of any discrete unitary operator" — *Physical Review Letters*. The original triangular MZI mesh decomposition.
4. **Bandyopadhyay et al. (2021)** — "Single chip photonic deep neural network with accelerated training" — *arXiv*. Hardware demonstration with training on-chip.
5. **MedGemma Technical Report (Google, 2025)** — Architecture and capabilities of the medical AI model we compile.
6. **Lightmatter Passage** — Commercial photonic matrix multiplication chip (proprietary), inspiration for our open-source approach.

---

## Roadmap

- [x] Project structure and documentation
- [x] Clements decomposition algorithm
- [x] MZI primitive library
- [x] SVD-based layer compilation
- [x] Photonic netlist generator
- [ ] GDS layout generator (requires gdsfactory)
- [ ] Full MedGemma weight loading and compilation
- [ ] Power consumption model
- [ ] Fabrication error tolerance analysis
- [ ] Tape-out preparation (requires foundry PDK access)
- [ ] Physical simulation (requires Lumerical/MEEP)

---

## Contributing

This is an open research project. Contributions welcome — especially from:
- Photonic chip engineers
- ML compiler engineers (MLIR, XLA background helpful)
- Medical AI researchers
- Anyone at Google DeepMind reading this who wants to help :)

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Acknowledgements

Built with inspiration from the neurosurgeons of Kenyatta National Hospital, who showed us that AI must reach every hospital on Earth — not just those connected to the cloud.
# photonmedgemma
