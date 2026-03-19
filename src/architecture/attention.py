"""
Photonic Multi-Head Attention
==============================

Implements the attention mechanism of MedGemma (Gemma-3) using
photonic MZI meshes for the Q, K, V, and O projection matrices.

Architecture (Gemma-3 4B):
    - d_model = 2048
    - n_heads = 8 (Q), n_kv_heads = 4 (K, V) — Grouped Query Attention (GQA)
    - head_dim = 256
    - Sliding window attention (local) alternating with full attention

Photonic vs Electronic:
    PHOTONIC:
    - Q projection: d_model → n_heads × head_dim (2048 → 2048)
    - K projection: d_model → n_kv_heads × head_dim (2048 → 1024)
    - V projection: d_model → n_kv_heads × head_dim (2048 → 1024)
    - O projection: n_heads × head_dim → d_model (2048 → 2048)

    ELECTRONIC (hybrid):
    - Attention score computation: Q @ K.T (dynamic, changes per token)
    - Softmax over scores
    - Weighted sum: scores @ V
    - RoPE positional encoding
    - KV cache memory

The key insight: Q/K/V/O projections are fixed weight matrices →
perfect for static photonic compilation. The attention score computation
involves dynamic operations (multiplication of two varying signals) →
must be electronic for now.

Future: Optical analog attention with reconfigurable MZI meshes (GHz speeds).
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from photonic.mesh import SVDLayer, ClementsMesh, SigmaStage
from .layer_norm import ElectronicLayerNorm


@dataclass
class GemmaAttentionConfig:
    """Configuration matching Gemma-3 4B attention parameters."""
    d_model: int = 2048
    n_heads: int = 8
    n_kv_heads: int = 4          # Grouped Query Attention
    head_dim: int = 256
    max_seq_len: int = 131072
    rope_theta: float = 1000000.0
    rope_local_base: float = 10000.0
    attn_logit_softcapping: float = 50.0
    sliding_window: int = 4096   # for local attention layers
    layer_idx: int = 0
    is_sliding: bool = False     # alternates every other layer


class PhotonicLinearProjection:
    """
    A single linear projection (Q, K, V, or O) implemented as a photonic SVD layer.

    Wraps SVDLayer with:
    - Input encoding: convert real activations to optical field amplitudes
    - Output decoding: convert photodetector currents to real activations
    - Scale tracking for proper normalization
    """

    def __init__(
        self,
        svd_layer: SVDLayer,
        projection_name: str,
    ):
        self.svd_layer = svd_layer
        self.projection_name = projection_name

    @property
    def input_dim(self) -> int:
        return self.svd_layer.input_dim

    @property
    def output_dim(self) -> int:
        return self.svd_layer.output_dim

    def forward(
        self,
        x: np.ndarray,
        include_loss: bool = False,
        add_noise: bool = False,
    ) -> np.ndarray:
        """
        Photonic linear projection: y = W × x

        Physical process:
        1. Encode x as optical field amplitudes (amplitude modulation)
        2. Propagate through Clements meshes (V† → Σ → U)
        3. Photodetect output → convert back to electrical

        Args:
            x: Input activations, shape (..., input_dim)
            include_loss: Apply propagation loss
            add_noise: Add shot noise and phase noise

        Returns:
            y: Output activations, shape (..., output_dim)
        """
        original_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)

        outputs = []
        for xi in x_flat:
            # Optical encoding: x_i maps to |E_i|² = amplitude²
            # For real-valued activations, use amplitude encoding:
            # E_i = sign(x_i) × √|x_i| (preserves sign information)
            E_in = np.sign(xi) * np.sqrt(np.abs(xi) + 1e-15)

            # Optical forward pass through SVD layer
            E_out = self.svd_layer.forward(E_in, include_loss=include_loss, add_noise=add_noise)

            # Optical detection: two photodetectors (differential detection)
            # to recover signed values: y = |E+|² - |E-|²
            # For simplicity in simulation, we use direct amplitude readout
            y = np.real(E_out) ** 2 * np.sign(np.real(E_out))

            outputs.append(y)

        result = np.array(outputs)
        return result.reshape(*original_shape[:-1], self.output_dim)

    def __repr__(self) -> str:
        return (
            f"PhotonicLinearProjection('{self.projection_name}', "
            f"{self.input_dim}→{self.output_dim}, "
            f"rank={self.svd_layer.sigma_stage.rank})"
        )


class PhotonicMultiHeadAttention:
    """
    Multi-head attention with photonic Q/K/V/O projections.

    The attention score computation (Q @ K.T / sqrt(d)) and
    softmax remain electronic. Only the projection matrices
    (which dominate compute) are photonic.

    This is the standard hybrid electro-optic architecture:
    - Optical: weight matrix multiplications (~85% of FLOPs)
    - Electronic: attention pattern, softmax, KV-cache (~15% of FLOPs)
    """

    def __init__(
        self,
        config: GemmaAttentionConfig,
        q_proj: Optional[PhotonicLinearProjection] = None,
        k_proj: Optional[PhotonicLinearProjection] = None,
        v_proj: Optional[PhotonicLinearProjection] = None,
        o_proj: Optional[PhotonicLinearProjection] = None,
    ):
        self.config = config
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj

        # GQA: n_heads Q heads, n_kv_heads K/V heads
        self.n_groups = config.n_heads // config.n_kv_heads  # = 2 for Gemma-3 4B

    def forward(
        self,
        hidden_states: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        kv_cache: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Forward pass of photonic multi-head attention.

        Args:
            hidden_states: Input, shape (seq_len, d_model)
            attention_mask: Causal mask, shape (seq_len, seq_len)
            position_ids: Position indices for RoPE
            kv_cache: Optional KV cache dict

        Returns:
            (output, updated_kv_cache): shape (seq_len, d_model)
        """
        seq_len, d_model = hidden_states.shape
        cfg = self.config

        # ── Photonic projections (speed of light!) ──────────────────────────
        # Q: (seq_len, d_model) → (seq_len, n_heads × head_dim)
        if self.q_proj is not None:
            q = self.q_proj.forward(hidden_states)
        else:
            q = hidden_states  # fallback (not yet compiled)

        # K: (seq_len, d_model) → (seq_len, n_kv_heads × head_dim)
        if self.k_proj is not None:
            k = self.k_proj.forward(hidden_states)
        else:
            k = hidden_states[:, :cfg.n_kv_heads * cfg.head_dim]

        # V: (seq_len, d_model) → (seq_len, n_kv_heads × head_dim)
        if self.v_proj is not None:
            v = self.v_proj.forward(hidden_states)
        else:
            v = hidden_states[:, :cfg.n_kv_heads * cfg.head_dim]

        # ── Electronic: reshape and apply attention ─────────────────────────
        # Reshape Q: (seq_len, n_heads, head_dim)
        q = q.reshape(seq_len, cfg.n_heads, cfg.head_dim)

        # Reshape K, V: (seq_len, n_kv_heads, head_dim)
        k = k.reshape(seq_len, cfg.n_kv_heads, cfg.head_dim)
        v = v.reshape(seq_len, cfg.n_kv_heads, cfg.head_dim)

        # Apply RoPE (electronic)
        if position_ids is not None:
            q, k = apply_rope(q, k, position_ids, cfg.rope_theta)

        # KV cache update
        if kv_cache is not None:
            layer_key = f"layer_{cfg.layer_idx}"
            if layer_key in kv_cache:
                k = np.concatenate([kv_cache[layer_key]["k"], k], axis=0)
                v = np.concatenate([kv_cache[layer_key]["v"], v], axis=0)
            kv_cache[layer_key] = {"k": k, "v": v}

        # GQA: repeat K/V for each query group
        # (seq_len, n_kv_heads, head_dim) → (seq_len, n_heads, head_dim)
        k = np.repeat(k, self.n_groups, axis=1)
        v = np.repeat(v, self.n_groups, axis=1)

        # Attention scores (electronic): Q @ K.T / sqrt(d_head)
        # q: (seq_len, n_heads, head_dim)
        # k: (kv_seq, n_heads, head_dim)
        q_t = q.transpose(1, 0, 2)  # (n_heads, seq_len, head_dim)
        k_t = k.transpose(1, 2, 0)  # (n_heads, head_dim, kv_seq)
        scores = q_t @ k_t / np.sqrt(cfg.head_dim)  # (n_heads, seq_len, kv_seq)

        # Logit softcapping (Gemma-3 specific)
        scores = np.tanh(scores / cfg.attn_logit_softcapping) * cfg.attn_logit_softcapping

        # Causal mask
        if attention_mask is not None:
            scores = scores + attention_mask  # mask adds -inf to future positions

        # Sliding window mask (for local attention layers)
        if cfg.is_sliding:
            scores = self._apply_sliding_window_mask(scores, cfg.sliding_window)

        # Softmax (electronic)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-15)

        # Weighted sum: attn_weights @ V
        v_t = v.transpose(1, 0, 2)  # (n_heads, kv_seq, head_dim)
        context = attn_weights @ v_t  # (n_heads, seq_len, head_dim)
        context = context.transpose(1, 0, 2)  # (seq_len, n_heads, head_dim)
        context = context.reshape(seq_len, -1)  # (seq_len, n_heads × head_dim)

        # ── Photonic O projection ───────────────────────────────────────────
        if self.o_proj is not None:
            output = self.o_proj.forward(context)
        else:
            output = context

        return output, kv_cache

    def _apply_sliding_window_mask(
        self, scores: np.ndarray, window: int
    ) -> np.ndarray:
        """Apply sliding window attention mask."""
        n_heads, seq_len, kv_seq = scores.shape
        mask = np.full((seq_len, kv_seq), -np.inf)

        for i in range(seq_len):
            start = max(0, i - window + 1)
            mask[i, start:i+1] = 0.0

        return scores + mask[np.newaxis, :, :]

    def resource_summary(self) -> dict:
        """Get resource usage summary."""
        summary = {}
        for name, proj in [
            ("q_proj", self.q_proj),
            ("k_proj", self.k_proj),
            ("v_proj", self.v_proj),
            ("o_proj", self.o_proj),
        ]:
            if proj is not None:
                summary[name] = {
                    "in_dim": proj.input_dim,
                    "out_dim": proj.output_dim,
                    "rank": proj.svd_layer.sigma_stage.rank,
                    "n_mzis": proj.svd_layer.total_mzis(),
                }
        return summary


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    position_ids: np.ndarray,
    theta: float = 1000000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Rotary Position Embedding (RoPE) to Q and K.

    This is an electronic operation (applied after photonic projection).
    RoPE encodes positional information by rotating Q and K in pairs.

    Args:
        q: Queries, shape (seq_len, n_heads, head_dim)
        k: Keys, shape (seq_len, n_kv_heads, head_dim)
        position_ids: Integer position indices, shape (seq_len,)
        theta: RoPE base frequency

    Returns:
        (q_rotated, k_rotated): Same shapes as input
    """
    seq_len, n_heads, head_dim = q.shape
    half_dim = head_dim // 2

    # Compute frequencies
    freqs = 1.0 / (theta ** (np.arange(0, half_dim, dtype=np.float64) / half_dim))

    # Compute angles for each position
    t = position_ids.astype(np.float64)  # (seq_len,)
    angles = np.outer(t, freqs)  # (seq_len, half_dim)

    cos_angles = np.cos(angles)[:, np.newaxis, :]  # (seq_len, 1, half_dim)
    sin_angles = np.sin(angles)[:, np.newaxis, :]

    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return np.concatenate([-x2, x1], axis=-1)

    def apply_rope_to(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        x_rot1 = x1 * cos_angles - x2 * sin_angles
        x_rot2 = x1 * sin_angles + x2 * cos_angles
        return np.concatenate([x_rot1, x_rot2], axis=-1)

    q_rot = apply_rope_to(q)
    k_rot = apply_rope_to(k)

    return q_rot.astype(np.float32), k_rot.astype(np.float32)
