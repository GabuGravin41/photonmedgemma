# PhotoMedGemma — Papers

Five LaTeX papers presenting the PhotoMedGemma photonic chip compiler.

## Papers

| File | Title | Style | Audience |
|------|-------|-------|---------|
| `paper1_overview.tex` | PhotoMedGemma: A Photonic Chip Compiler for Deploying Medical AI in Resource-Constrained Environments | Long-form overview | Broad technical |
| `paper2_physics_for_students.tex` | Mathematical Foundations of Photonic Neural Networks: From Maxwell's Equations to Clements Decomposition | Lecture notes | PhD students / Olympiad-level |
| `paper3_photonic_components.tex` | Silicon Photonic Components for Neural Network Inference | Component reference with figure placeholders | Photonics engineers |
| `paper4_neurips_style.tex` | PhotoMedGemma: Machine-Precision Compilation of a Medical Vision-Language Model onto Photonic Hardware | NeurIPS 2026 conference | ML community |
| `paper5_ieee_ofc_style.tex` | Photonic Compilation of MedGemma-4B: A Complete SVD-Clements Pipeline | IEEE / OFC conference | Photonics community |

## Building

```bash
# Requires: pdflatex, texlive-full or equivalent
# Paper 4 requires neurips_2024.sty (download from NeurIPS website)
# Paper 5 uses IEEEtran class (install: texlive-publishers)

pdflatex paper1_overview.tex
pdflatex paper2_physics_for_students.tex
pdflatex paper3_photonic_components.tex
pdflatex paper4_neurips_style.tex    # needs neurips_2024.sty
pdflatex paper5_ieee_ofc_style.tex   # needs IEEEtran
```

For papers 4 and 5 without the style files, change the documentclass to `article`.

## Key Results (from `output/simulations/paper_results/`)

- **Chip fidelity**: ε ≈ 7×10⁻¹⁵ (machine precision) for all 4 projections
- **SVD error at rank 64**: 0.19–0.67 (design trade-off)
- **SVD error at full rank**: ε ≈ 3×10⁻¹⁵ (machine precision)
- **Quantisation error (4-bit NF4)**: < 0.22% (negligible)
- **Energy**: ~8 W vs ~300 W GPU → 37× improvement
- **MZIs at rank 64**: 2,016 per mesh / **at full rank**: 2,096,128 per mesh
