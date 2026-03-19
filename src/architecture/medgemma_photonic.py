"""
PhotoMedGemma — Full Photonic MedGemma Model
=============================================

Assembles all photonic and electronic components into a complete
inference pipeline for MedGemma-4B.

This is the top-level architecture module. It:
1. Loads compiled photonic layers (from MZIMapper output)
2. Assembles transformer blocks (photonic attention + photonic FFN)
3. Provides a forward() method for end-to-end inference simulation
4. Reports resource usage (MZI count, chips, power)

Note: This is a SIMULATION model. Actual chip inference runs directly on hardware.
This Python model simulates what the chip does, allowing accuracy verification
before fabrication.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np

from .attention import PhotonicMultiHeadAttention, GemmaAttentionConfig, PhotonicLinearProjection
from .feedforward import PhotonicFeedForward, GemmaFFNConfig
from .layer_norm import ElectronicLayerNorm
from photonic.mesh import SVDLayer
from compiler.mzi_mapper import CompiledLayer

logger = logging.getLogger(__name__)


@dataclass
class MedGemmaPhotonicConfig:
    """Full configuration for the photonic MedGemma model."""
    # Language model dimensions (Gemma-3 4B)
    d_model: int = 2048
    n_layers: int = 46
    n_heads: int = 8
    n_kv_heads: int = 4
    head_dim: int = 256
    d_ffn: int = 16384
    vocab_size: int = 262144
    max_seq_len: int = 131072
    rms_norm_eps: float = 1e-6

    # Vision encoder dimensions (SigLIP-So400M)
    vision_hidden: int = 1152
    vision_n_layers: int = 27
    vision_n_heads: int = 16
    vision_d_ffn: int = 4304
    image_size: int = 896
    patch_size: int = 14

    # Compilation settings
    svd_rank: int = 64
    dac_bits: int = 12
    wavelength_nm: float = 1310.0

    # Runtime settings
    include_photonic_loss: bool = False
    add_photonic_noise: bool = False


class PhotonicTransformerLayer:
    """
    One complete Gemma-3 transformer layer with photonic projections.

    Contains:
    - pre_attn_norm: RMSNorm (electronic)
    - self_attn: PhotonicMultiHeadAttention (photonic Q/K/V/O)
    - pre_ffn_norm: RMSNorm (electronic)
    - ffn: PhotonicFeedForward (photonic gate/up/down)
    """

    def __init__(
        self,
        layer_idx: int,
        config: MedGemmaPhotonicConfig,
        attn: Optional[PhotonicMultiHeadAttention] = None,
        ffn: Optional[PhotonicFeedForward] = None,
        pre_attn_norm: Optional[ElectronicLayerNorm] = None,
        pre_ffn_norm: Optional[ElectronicLayerNorm] = None,
    ):
        self.layer_idx = layer_idx
        self.config = config

        # Alternate between sliding and global attention
        self.is_sliding = (layer_idx % 2 == 0)  # Gemma-3 pattern

        self.pre_attn_norm = pre_attn_norm or ElectronicLayerNorm(config.d_model, config.rms_norm_eps)
        self.attn = attn
        self.pre_ffn_norm = pre_ffn_norm or ElectronicLayerNorm(config.d_model, config.rms_norm_eps)
        self.ffn = ffn

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        kv_cache: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Forward pass through one transformer layer.

        Args:
            hidden_states: (seq_len, d_model)
            attention_mask: Causal mask
            position_ids: Position indices
            kv_cache: KV cache for autoregressive generation

        Returns:
            (output, updated_kv_cache)
        """
        residual = hidden_states

        # Pre-attention RMSNorm (electronic)
        x = self.pre_attn_norm(hidden_states)

        # Multi-head attention (photonic projections + electronic attention)
        if self.attn is not None:
            x, kv_cache = self.attn.forward(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        else:
            # Fallback: identity (no compiled attention)
            x, kv_cache = x, kv_cache

        # Residual connection
        hidden_states = residual + x
        residual = hidden_states

        # Pre-FFN RMSNorm (electronic)
        x = self.pre_ffn_norm(hidden_states)

        # Feed-forward network (photonic projections + electronic gating)
        if self.ffn is not None:
            x = self.ffn.forward(x)
        else:
            x = x  # Fallback: identity

        # Residual connection
        hidden_states = residual + x

        return hidden_states, kv_cache

    def total_mzis(self) -> int:
        """Total MZI count for this transformer layer."""
        n = 0
        if self.attn:
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                proj = getattr(self.attn, proj_name, None)
                if proj and hasattr(proj, 'svd_layer'):
                    n += proj.svd_layer.total_mzis()
        if self.ffn:
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                proj = getattr(self.ffn, proj_name, None)
                if proj and hasattr(proj, 'total_mzis'):
                    n += proj.total_mzis()
        return n


class PhotonicMedGemma:
    """
    Complete photonic MedGemma inference model.

    This is the top-level simulation model representing the full
    photonic chip system. It assembles:

    1. Token embedding (electronic — too large for current chips)
    2. N transformer layers (photonic attention + photonic FFN)
    3. Final RMSNorm (electronic)
    4. LM head / output projection (electronic or photonic)

    For vision:
    1. SigLIP image patchification (electronic)
    2. Vision encoder transformer layers (photonic)
    3. Vision-language projector (photonic or electronic)
    """

    def __init__(
        self,
        config: MedGemmaPhotonicConfig,
        layers: Optional[List[PhotonicTransformerLayer]] = None,
        embed_tokens: Optional[np.ndarray] = None,
        norm_weight: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.layers = layers or []
        self.embed_tokens = embed_tokens  # vocabulary embedding table (electronic)
        self.final_norm = (
            ElectronicLayerNorm(config.d_model, config.rms_norm_eps, norm_weight)
            if norm_weight is not None
            else ElectronicLayerNorm(config.d_model, config.rms_norm_eps)
        )

    @classmethod
    def from_compiled_layers(
        cls,
        compiled_layers: List[CompiledLayer],
        config: Optional[MedGemmaPhotonicConfig] = None,
        norm_weights: Optional[Dict[str, np.ndarray]] = None,
        embed_tokens: Optional[np.ndarray] = None,
    ) -> "PhotonicMedGemma":
        """
        Assemble PhotonicMedGemma from compiled layers.

        Args:
            compiled_layers: List of CompiledLayer from MZIMapper
            config: Model configuration
            norm_weights: Dict of RMSNorm weights (layer_name → weight)
            embed_tokens: Token embedding table (optional)

        Returns:
            Assembled PhotonicMedGemma model
        """
        config = config or MedGemmaPhotonicConfig()

        # Group compiled layers by transformer layer index
        from collections import defaultdict
        layers_by_idx: Dict[int, Dict[str, CompiledLayer]] = defaultdict(dict)

        for cl in compiled_layers:
            if cl.transformer_layer_idx is not None and cl.component == "language_model":
                layers_by_idx[cl.transformer_layer_idx][cl.projection_type] = cl

        # Build transformer layers
        transformer_layers = []
        for layer_idx in sorted(layers_by_idx.keys()):
            layer_compiled = layers_by_idx[layer_idx]

            # Build attention
            attn_config = GemmaAttentionConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                layer_idx=layer_idx,
                is_sliding=(layer_idx % 2 == 0),
            )

            def make_photonic_proj(proj_type: str, name_suffix: str):
                if proj_type in layer_compiled:
                    cl = layer_compiled[proj_type]
                    svd_layer = SVDLayer(
                        U_mesh=cl.U_mesh,
                        sigma_stage=cl.sigma_stage,
                        Vh_mesh=cl.Vh_mesh,
                        layer_name=cl.layer_name,
                    )
                    return PhotonicLinearProjection(svd_layer, name_suffix)
                return None

            attn = PhotonicMultiHeadAttention(
                config=attn_config,
                q_proj=make_photonic_proj("q", "q_proj"),
                k_proj=make_photonic_proj("k", "k_proj"),
                v_proj=make_photonic_proj("v", "v_proj"),
                o_proj=make_photonic_proj("o", "o_proj"),
            )

            # Build FFN
            ffn_config = GemmaFFNConfig(
                d_model=config.d_model,
                d_ffn=config.d_ffn,
                layer_idx=layer_idx,
            )

            def make_ffn_proj(proj_type: str):
                if proj_type in layer_compiled:
                    cl = layer_compiled[proj_type]
                    return SVDLayer(
                        U_mesh=cl.U_mesh,
                        sigma_stage=cl.sigma_stage,
                        Vh_mesh=cl.Vh_mesh,
                        layer_name=cl.layer_name,
                    )
                return None

            ffn = PhotonicFeedForward(
                config=ffn_config,
                gate_proj=make_ffn_proj("gate"),
                up_proj=make_ffn_proj("up"),
                down_proj=make_ffn_proj("down"),
            )

            # RMSNorm weights (electronic)
            pre_attn_norm_w = None
            pre_ffn_norm_w = None
            if norm_weights:
                attn_key = f"model.layers.{layer_idx}.input_layernorm.weight"
                ffn_key = f"model.layers.{layer_idx}.post_feedforward_layernorm.weight"
                pre_attn_norm_w = norm_weights.get(attn_key)
                pre_ffn_norm_w = norm_weights.get(ffn_key)

            layer = PhotonicTransformerLayer(
                layer_idx=layer_idx,
                config=config,
                attn=attn,
                ffn=ffn,
                pre_attn_norm=ElectronicLayerNorm(
                    config.d_model, config.rms_norm_eps, pre_attn_norm_w
                ),
                pre_ffn_norm=ElectronicLayerNorm(
                    config.d_model, config.rms_norm_eps, pre_ffn_norm_w
                ),
            )
            transformer_layers.append(layer)

        logger.info(
            f"Assembled PhotonicMedGemma with {len(transformer_layers)} transformer layers. "
            f"Total MZIs: {sum(l.total_mzis() for l in transformer_layers):,}"
        )

        return cls(
            config=config,
            layers=transformer_layers,
            embed_tokens=embed_tokens,
        )

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        kv_cache: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Full forward pass (simulation).

        Args:
            input_ids: Token IDs, shape (seq_len,)
            attention_mask: Optional causal mask
            position_ids: Optional position indices

        Returns:
            (logits, kv_cache): Logit array, shape (seq_len, vocab_size)
        """
        seq_len = len(input_ids)

        if position_ids is None:
            position_ids = np.arange(seq_len)

        # Token embedding (electronic)
        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens[input_ids]  # (seq_len, d_model)
        else:
            # If no embeddings loaded, use random for testing
            hidden_states = np.random.randn(seq_len, self.config.d_model).astype(np.float32)

        # Build causal mask if not provided
        if attention_mask is None:
            attention_mask = np.tril(np.zeros((seq_len, seq_len)))
            attention_mask[attention_mask == 0] = -np.inf
            attention_mask[attention_mask == -np.inf] = -np.inf
            np.fill_diagonal(attention_mask, 0)

        # Transformer layers (mix of photonic and electronic operations)
        for i, layer in enumerate(self.layers):
            logger.debug(f"  Layer {i}/{len(self.layers)}")
            hidden_states, kv_cache = layer.forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # Final RMSNorm (electronic)
        hidden_states = self.final_norm(hidden_states)

        # LM head: project to vocabulary (electronic or photonic)
        # For simulation: use dot product with embedding table (tied weights)
        if self.embed_tokens is not None:
            logits = hidden_states @ self.embed_tokens.T  # (seq_len, vocab_size)
        else:
            # Random logits for testing
            logits = np.random.randn(seq_len, self.config.vocab_size).astype(np.float32)

        return logits, kv_cache

    def resource_report(self) -> dict:
        """
        Generate complete resource usage report.

        Returns:
            Dict with MZI counts, chip counts, power estimates
        """
        total_mzis = sum(l.total_mzis() for l in self.layers)
        n_layers_compiled = len(self.layers)

        report = {
            "model": "MedGemma-4B-IT (Photonic)",
            "architecture": "Gemma-3 4B",
            "n_transformer_layers": n_layers_compiled,
            "total_mzis": total_mzis,
            "chips_required": (total_mzis + 4095) // 4096,
            "svd_rank": self.config.svd_rank,
            "per_layer": []
        }

        for layer in self.layers:
            report["per_layer"].append({
                "layer_idx": layer.layer_idx,
                "n_mzis": layer.total_mzis(),
                "is_sliding": layer.is_sliding,
            })

        return report

    def __repr__(self) -> str:
        return (
            f"PhotonicMedGemma("
            f"n_layers={len(self.layers)}, "
            f"d_model={self.config.d_model}, "
            f"rank={self.config.svd_rank}, "
            f"total_mzis={sum(l.total_mzis() for l in self.layers):,})"
        )
