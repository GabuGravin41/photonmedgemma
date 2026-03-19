# MedGemma Architecture Analysis for Photonic Compilation

## Overview

MedGemma is Google DeepMind's family of medically-tuned vision-language models. The primary deployment target for PhotoMedGemma is **MedGemma-4B-IT** (4 billion parameters, instruction-tuned), based on the Gemma 3 architecture.

## Model Architecture

### High-Level Structure
MedGemma-4B is a multimodal model comprising:
1. **Vision Encoder**: SigLIP-So400M (400M parameters) — processes images
2. **Language Model**: Gemma-3-4B (3.6B parameters) — generates text
3. **Vision-Language Projector**: linear projection connecting encoder to LM

### Language Model (Gemma-3-4B)
Based on the Gemma 3 architecture:

```yaml
model_type: gemma3
architecture:
  hidden_size: 2048           # d_model
  num_hidden_layers: 46       # transformer blocks
  num_attention_heads: 8      # multi-head attention heads
  num_key_value_heads: 4      # grouped-query attention (GQA)
  head_dim: 256               # per-head dimension
  intermediate_size: 16384    # FFN inner dimension
  vocab_size: 262144          # vocabulary size
  max_position_embeddings: 131072

activation:
  type: geglu                 # Gated GELU
  hidden_act: gelu_pytorch_tanh

normalization:
  type: RMSNorm
  rms_norm_eps: 1.0e-6

attention:
  type: grouped_query_attention
  sliding_window: 4096        # local attention window (alternating layers)
  global_attention: full      # every other layer uses full attention

positional_encoding:
  type: RoPE                  # Rotary Position Embedding
  rope_theta: 1000000.0
  rope_local_base_freq: 10000.0
```

### Vision Encoder (SigLIP-So400M)
```yaml
model_type: siglip_vision_model
architecture:
  hidden_size: 1152           # ViT-So400M hidden dim
  num_hidden_layers: 27       # ViT transformer layers
  num_attention_heads: 16     # attention heads
  intermediate_size: 4304     # FFN inner dim
  image_size: 896             # input image resolution (896×896)
  patch_size: 14              # patch size → 64×64 = 4096 patches
  num_channels: 3
  num_patches: 4096           # (896/14)² = 4096
```

## Weight Matrix Dimensions

### Language Model Weight Matrices (per layer)

| Matrix | Shape | Parameters | SVD Rank (full) |
|--------|-------|-----------|-----------------|
| Q projection | 2048 × 2048 | 4.2M | 2048 |
| K projection | 2048 × 1024 | 2.1M | 1024 |
| V projection | 2048 × 1024 | 2.1M | 1024 |
| O projection | 2048 × 2048 | 4.2M | 2048 |
| FFN gate | 2048 × 16384 | 33.6M | 2048 |
| FFN up | 2048 × 16384 | 33.6M | 2048 |
| FFN down | 16384 × 2048 | 33.6M | 2048 |
| **Total per layer** | | **~113M** | |

Total layers: 46
Total LM parameters: ~46 × 113M ≈ **5.2B** (including embeddings, norms)

Note: GQA (num_key_value_heads=4) means K and V are shared across 2 heads each.

### Vision Encoder Weight Matrices (per layer)

| Matrix | Shape | Parameters |
|--------|-------|-----------|
| Q projection | 1152 × 1152 | 1.3M |
| K projection | 1152 × 1152 | 1.3M |
| V projection | 1152 × 1152 | 1.3M |
| O projection | 1152 × 1152 | 1.3M |
| FFN fc1 | 1152 × 4304 | 5.0M |
| FFN fc2 | 4304 × 1152 | 5.0M |
| **Total per layer** | | **~15M** |

Total ViT layers: 27
Total vision encoder parameters: ~27 × 15M ≈ **400M**

## MZI Resource Estimation

### Per Matrix (full-rank implementation)

For a matrix of shape (m, n) with m ≤ n:
- SVD: W = U(m×m) · Σ(m×n) · V†(n×n)
- U mesh: m(m-1)/2 MZIs (Clements)
- V† mesh: n(n-1)/2 MZIs (Clements)
- Σ stage: m attenuators

**Q projection (2048×2048)**:
- U mesh: 2048×2047/2 = **2,096,128 MZIs**
- V† mesh: 2048×2047/2 = **2,096,128 MZIs**
- Total: **4.2M MZIs** for Q alone

**This is the challenge**: a single attention projection requires 4.2M MZIs. With current fabrication (10,000–100,000 MZIs per chip), we need either:
1. Multi-chip modules
2. Time-multiplexing (reusing MZIs for different layers)
3. **Low-rank approximation** (practical path)

### Low-Rank Approximation (Practical Path)

By truncating SVD to rank R << d_model, we reduce MZI count dramatically:

For W ≈ U_R · Σ_R · V_R† (rank-R approximation):
- U_R mesh: m×(m-1)/2 → for rank R, only R columns needed: **m×R MZIs** (rectangular Clements)
- V_R† mesh: n×(n-1)/2 → **n×R MZIs**
- Quality: measured by retained variance = sum(σ_1²...σ_R²) / sum(all σ²)

**Rank-64 approximation of Q projection (2048×2048)**:
- U_R: 2048×64 → ~131K MZIs
- V_R†: 2048×64 → ~131K MZIs
- **Total: ~262K MZIs per matrix** vs 4.2M full-rank (16× reduction)
- Expected accuracy retention: >99% (transformers are highly compressible via SVD)

**Rank-64 full model MZI count estimate**:
- LM: 46 layers × 7 matrices × 2×262K = **175M MZIs**
- ViT: 27 layers × 6 matrices × 2×74K = **24M MZIs**
- **Total: ~200M MZIs** (still 2000× more than current single chips)

**Rank-8 approximation** (aggressive compression):
- LM: 46 × 7 × 2×(2048×8) = **21M MZIs**
- ViT: 27 × 6 × 2×(1152×8) = **3M MZIs**
- **Total: ~24M MZIs** — achievable with a multi-chip module of ~1000 chips

### Practical Phase 1 Target: Single Attention Head, Rank-64

For a single attention head (head_dim=256):
- Q head: 2048×256 → rank-64 → ~262K MZIs
- Single transformer layer, single head: ~2M MZIs

This is the target for Phase 1 fabrication demonstration.

## Accuracy Analysis

### SVD Compressibility of Transformer Weights

Empirical studies of GPT/BERT/Gemma transformer weights consistently show:
- The singular value spectrum decays rapidly (approximately exponential)
- Rank-8 captures ~70-80% of weight variance
- Rank-64 captures ~95-99% of weight variance
- Rank-256 captures >99.9% of weight variance

For medical imaging (MedGemma), we target rank-64 minimum to preserve diagnostic accuracy.

### Phase Quantization Error

Phase shifters are controlled by DAC (digital-to-analog converter). With B-bit DAC:
- Phase resolution: 2π / 2^B
- For B=8: ~1.4° resolution
- For B=12: ~0.088° resolution

Matrix reconstruction error from phase quantization scales approximately as:
- ε ≈ π / (2^B · √N) for N-mode Clements mesh

With B=8 bits and N=64: ε ≈ 0.006 — negligible for most applications.

## Compilation Strategy

Given the resource analysis above, our compilation strategy is:

### Phase 1: Proof-of-Concept (Current)
- **Target**: Single transformer attention block, rank-64
- **Chip size**: 10mm × 10mm, 220nm SOI
- **MZI count**: ~250,000 per chip, 8-chip module
- **Precision**: bfloat16 → 12-bit phase quantization

### Phase 2: Full Layer (6 months)
- **Target**: Complete transformer layer (attention + FFN), rank-128
- **Chip size**: Multi-chip wafer-scale integration
- **MZI count**: ~5M per chip, 64-chip module

### Phase 3: Full Model (2–3 years)
- **Target**: Full MedGemma-4B, rank-256
- **Technology**: 3D photonic integration, WDM parallelism
- **Fabrication**: imec or AIM Photonics MPW run

## Key Compilation Steps

1. **Load weights**: Download MedGemma from HuggingFace
2. **Extract linear layers**: Parse transformer architecture, collect weight matrices
3. **SVD decomposition**: Compute truncated SVD for each weight matrix
4. **Clements decomposition**: Convert U and V† unitaries to MZI phase angles
5. **Phase quantization**: Round phases to DAC resolution
6. **Netlist generation**: Output photonic SPICE-like netlist
7. **GDS generation**: Physical layout for foundry submission

This pipeline is fully implemented in `src/compiler/`.
