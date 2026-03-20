# Project Roadmap

## Phase 0: Foundation
**Goal**: Complete compiler software, documentation, and simulation framework

- [x] Project architecture and documentation
- [x] Clements decomposition algorithm
- [x] MZI primitive library
- [x] SVD-based layer compilation pipeline
- [x] Photonic netlist generator
- [ ] Software simulation of compiled layers (accuracy validation)
- [ ] Power and area estimation scripts
- [ ] GDS layout generator (gdsfactory integration)
- [ ] GitHub release and community launch

## Phase 1: Single-Layer Demo Chip 
**Goal**: Fabricate and test a chip implementing one attention head, rank-64

- [ ] Finalize chip design for AIM Photonics MPW run
- [ ] Single attention head: Q projection, rank-64, head_dim=256
- [ ] 8-chip multi-chip module for V†+U meshes
- [ ] Electronic control board (FPGA + DAC + ADC)
- [ ] Chip bring-up and phase calibration
- [ ] Matrix-vector multiplication demonstration
- [ ] Accuracy benchmarking vs software simulation

## Phase 2: Full Transformer Layer
**Goal**: Full single transformer layer on multi-chip module

- [ ] All 7 weight matrices per layer (rank-128)
- [ ] Electronic attention score + softmax
- [ ] RMSNorm and GELU hybrid electronic units
- [ ] Layer-level inference throughput measurement
- [ ] Energy efficiency measurement

## Phase 3: Medical Inference System
**Goal**: End-to-end MedGemma inference on photonic hardware

- [ ] Full 46-layer language model (multi-tile system)
- [ ] SigLIP vision encoder integration
- [ ] Complete medical inference pipeline
- [ ] Accuracy benchmark: MedQA, PubMedQA, path-VQA
- [ ] Latency and energy measurement vs A100 GPU
- [ ] Pilot deployment at Kenyatta National Hospital

## Phase 4: Product and Scale
**Goal**: Manufacturable, deployable medical AI photonic chip

- [ ] 3D photonic integration (reduced chip count)
- [ ] Custom foundry PDK with optimized MZI footprint
- [ ] Manufacturing cost reduction to <$100/unit at volume
- [ ] Regulatory pathway for medical device deployment
- [ ] Partnership with African health ministries for deployment

## Key Milestones & Metrics

| Milestone | Target Date | Success Metric |
|-----------|-------------|----------------|
| Compiler v1.0 | Q1 2026 | Compiles full MedGemma to netlist |
| Phase 1 chip | Q3 2026 | Matrix multiplication within 1% error |
| Phase 2 layer | Q1 2027 | Layer inference within 2% accuracy loss |
| Full inference | Q4 2027 | MedQA accuracy within 5% of GPU baseline |
| Hospital pilot | Q2 2028 | 100 patients/day, 99.9% uptime |

## Technology Dependencies

| Dependency | Status | Alternative |
|-----------|--------|------------|
| gdsfactory (GDS generation) | Open source ✓ | klayout scripting |
| AIM/imec MPW access | Need application | University partner |
| FPGA for control (Xilinx KV260) | Commercial ✓ | Raspberry Pi + ADC |
| Lumerical FDTD (simulation) | Commercial ($$) | MEEP (open source) |
| MedGemma weights | Open on HuggingFace ✓ | — |
| 300mm SOI wafer process | Foundry partner needed | 200mm wafer (lower cost) |
