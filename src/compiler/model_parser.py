"""
Model Parser — Load and Parse MedGemma Weight Matrices
=======================================================

Loads MedGemma from HuggingFace and extracts all linear weight matrices
that will be compiled to photonic MZI meshes.

MedGemma-4B-IT architecture (Gemma-3 backbone + SigLIP vision encoder):

Language Model (Gemma-3-4B):
    - 46 transformer layers
    - d_model = 2048, n_heads = 8, d_ffn = 16384
    - Per layer: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

Vision Encoder (SigLIP-So400M):
    - 27 ViT layers
    - d_model = 1152, n_heads = 16, d_ffn = 4304
    - Per layer: q_proj, k_proj, v_proj, out_proj, fc1, fc2

We compile all linear (weight matrix) operations. Non-linear operations
(LayerNorm, RMSNorm, activations, softmax) remain electronic.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LayerInfo:
    """Information about a single linear weight matrix."""
    name: str                          # full parameter name (e.g., "model.layers.0.self_attn.q_proj.weight")
    shape: Tuple[int, ...]            # weight tensor shape
    module_type: str                   # "attention" | "ffn" | "embedding" | "vision_attention" | "vision_ffn"
    transformer_layer_idx: Optional[int]  # index within the transformer (None for embeddings)
    projection_type: str               # "q", "k", "v", "o", "gate", "up", "down", "fc1", "fc2"
    component: str                     # "language_model" | "vision_encoder"
    n_params: int = 0

    def __post_init__(self):
        self.n_params = int(np.prod(self.shape))


class ModelParser:
    """
    Loads MedGemma model weights and extracts linear layer matrices.

    Supports two backends:
    1. HuggingFace transformers (requires `transformers` and `torch`)
    2. Safetensors direct loading (faster, no full model instantiation)

    For compilation, we only need the weight tensors — no tokenizer,
    no forward pass logic. We use safetensors direct loading when possible.
    """

    # Layer name patterns for MedGemma (Gemma-3 4B + SigLIP)
    LANGUAGE_MODEL_PATTERNS = {
        "q_proj":    "attention",
        "k_proj":    "attention",
        "v_proj":    "attention",
        "o_proj":    "attention",
        "gate_proj": "ffn",
        "up_proj":   "ffn",
        "down_proj": "ffn",
    }

    VISION_MODEL_PATTERNS = {
        "q_proj":   "vision_attention",
        "k_proj":   "vision_attention",
        "v_proj":   "vision_attention",
        "out_proj": "vision_attention",
        "fc1":      "vision_ffn",
        "fc2":      "vision_ffn",
    }

    def __init__(
        self,
        model_id: str = "google/medgemma-4b-it",
        cache_dir: Optional[str] = None,
        dtype: str = "float32",
    ):
        """
        Args:
            model_id: HuggingFace model ID or local path
            cache_dir: Directory to cache downloaded weights
            dtype: Weight dtype for loading ("float32" | "float16" | "bfloat16")
        """
        self.model_id = model_id
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/photomedgemma")
        self.dtype = dtype
        self._weights: Optional[Dict] = None
        self._shard_index: Optional[Dict[str, Path]] = None
        self._config: Optional[dict] = None

    def load(self, local_path: Optional[str] = None) -> "ModelParser":
        """
        Load model weights.

        Args:
            local_path: If provided, load from local directory instead of HF.

        Returns:
            self (for chaining)
        """
        if local_path:
            self._load_from_local(local_path)
        else:
            self._load_from_huggingface()

        logger.info(
            f"Indexed {len(self._weights)} tensors from {self.model_id} (lazy loading)."
        )
        return self

    def _load_from_huggingface(self):
        """Download and index model weights from HuggingFace hub."""
        try:
            from huggingface_hub import snapshot_download

            logger.info(f"Downloading model {self.model_id}...")
            local_dir = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir,
                ignore_patterns=["*.bin", "tokenizer*", "*.msgpack"],
            )
            self._load_from_local(local_dir)

        except ImportError:
            logger.warning(
                "huggingface_hub not installed. Falling back to transformers."
            )
            self._load_via_transformers()

    def _load_from_local(self, path: str):
        """
        Index model shards for lazy loading.

        Rather than loading all ~17GB of weights into RAM at once, we build a
        tensor-name → shard-file index here. Actual tensor data is fetched
        on-demand in iter_linear_layers() via safetensors.safe_open, so only
        the tensors we actually compile are ever in memory simultaneously.
        """
        path = Path(path)

        # Load model config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

        # Find all safetensors shards
        shard_files = sorted(path.glob("model*.safetensors"))
        if not shard_files:
            shard_files = sorted(path.glob("*.safetensors"))

        if not shard_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {path}. "
                f"Ensure the model is downloaded correctly."
            )

        logger.info(
            f"Indexing {len(shard_files)} safetensors shards from {path} "
            f"(lazy — weights loaded on demand)..."
        )

        # Build tensor-name → shard-path index using only metadata (no data loaded)
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors not installed. Run: pip install safetensors")

        self._shard_index: Dict[str, Path] = {}   # tensor name → shard file
        total_tensors = 0
        for shard in shard_files:
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    self._shard_index[key] = shard
                    total_tensors += 1

        logger.info(
            f"  Indexed {total_tensors} tensors across {len(shard_files)} shards. "
            f"Peak RAM usage: <1MB (lazy index only)."
        )

        # Set _weights to sentinel so .load() log works; actual data via iter_linear_layers
        self._weights = self._shard_index  # type: ignore[assignment]

    def _fetch_tensor(self, name: str) -> np.ndarray:
        """Load a single tensor from its shard file as float32 numpy array."""
        import torch
        from safetensors import safe_open

        shard = self._shard_index[name]
        with safe_open(str(shard), framework="pt") as f:
            return f.get_tensor(name).to(torch.float32).numpy()

    def _load_via_transformers(self):
        """Fallback: load via HuggingFace transformers library."""
        try:
            import torch
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "Run: pip install transformers torch"
            )

        logger.info(f"Loading model via transformers: {self.model_id}...")
        model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
        )

        self._weights = {}
        for name, param in model.named_parameters():
            self._weights[name] = param.detach().cpu().numpy().astype(np.float32)

        del model  # free memory

    def iter_linear_layers(
        self,
        include_language_model: bool = True,
        include_vision_encoder: bool = True,
        layer_range: Optional[Tuple[int, int]] = None,
    ) -> Iterator[Tuple[LayerInfo, np.ndarray]]:
        """
        Iterate over all linear weight matrices, yielding (LayerInfo, weight_matrix).

        This is the main interface for the compilation pipeline.

        Args:
            include_language_model: Include Gemma-3 LM layers
            include_vision_encoder: Include SigLIP vision encoder layers
            layer_range: Optional (start, end) tuple to limit which transformer
                         layers to include (e.g., (0, 5) for first 5 layers)

        Yields:
            (LayerInfo, weight_matrix): Info and float32 numpy array
        """
        if self._weights is None:
            raise RuntimeError("Call .load() before iterating layers.")

        # Determine the set of tensor names to fetch.
        # We use _shard_index if available (lazy path), otherwise fall back to
        # the pre-loaded weight dict (e.g. when loaded via transformers).
        index = getattr(self, "_shard_index", None) or self._weights

        for name in sorted(index.keys()):
            # Fast pre-filter on name to avoid fetching large non-compilable tensors
            # (embeddings, lm_head, norms) — these can each be 2GB+ in float32.
            if not self._is_compilable_name(name):
                continue

            # Layer-range pre-filter from name (avoids loading wrong-layer tensors)
            if layer_range is not None:
                layer_idx = self._layer_idx_from_name(name)
                if layer_idx is not None:
                    start, end = layer_range
                    if not (start <= layer_idx < end):
                        continue

            # Fetch the actual tensor (lazy: only loads this one tensor from disk)
            if self._shard_index is not None:
                weight = self._fetch_tensor(name)
                logger.debug(f"  Fetched {name}: {weight.shape}")
            else:
                weight = self._weights[name]  # type: ignore[index]

            info = self._classify_weight(name, weight.shape)
            if info is None:
                continue

            # Filter by component
            if info.component == "language_model" and not include_language_model:
                continue
            if info.component == "vision_encoder" and not include_vision_encoder:
                continue

            # Only compile 2D weight matrices
            if weight.ndim != 2:
                continue

            yield info, weight

    @staticmethod
    def _is_compilable_name(name: str) -> bool:
        """Fast string-based pre-filter: skip tensors we will never compile."""
        # Skip embedding tables and output head (huge, not photonic targets)
        if any(x in name for x in ("embed_tokens", "lm_head", "embeddings")):
            return False
        # Skip normalization scalars
        if name.endswith((".weight",)) and any(x in name for x in ("norm.", "layernorm.")):
            # Keep only weight matrices (2D); norms are usually 1D but we still pre-filter
            return False
        # Must contain a known projection keyword
        PROJ_KEYWORDS = ("q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",
                         "out_proj", "fc1", "fc2")
        return any(kw in name for kw in PROJ_KEYWORDS)

    @staticmethod
    def _layer_idx_from_name(name: str) -> Optional[int]:
        """Extract transformer layer index from a tensor name, or None."""
        import re
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            return int(m.group(1))
        return None

    def _classify_weight(
        self, name: str, shape: Tuple[int, ...]
    ) -> Optional[LayerInfo]:
        """
        Classify a weight tensor by its role in the model.

        Args:
            name: Parameter name
            shape: Tensor shape

        Returns:
            LayerInfo or None if not a compilable linear weight
        """
        # Only process 2D weight matrices
        if len(shape) != 2:
            return None

        # Skip embedding tables (too large, different structure)
        if "embed_tokens" in name or "lm_head" in name:
            return None

        # Skip normalization weights (1D)
        if "norm" in name.lower():
            return None

        # Determine component
        if "vision_tower" in name or "vision_model" in name:
            component = "vision_encoder"
            patterns = self.VISION_MODEL_PATTERNS
        elif "language_model" in name or "model.layers" in name:
            component = "language_model"
            patterns = self.LANGUAGE_MODEL_PATTERNS
        else:
            return None

        # Determine projection type
        proj_type = None
        module_type = None
        for proj, mtype in patterns.items():
            if f".{proj}." in name or name.endswith(f".{proj}"):
                proj_type = proj
                module_type = mtype
                break

        if proj_type is None:
            return None

        # Extract transformer layer index
        layer_idx = self._extract_layer_idx(name)

        return LayerInfo(
            name=name,
            shape=tuple(shape),
            module_type=module_type,
            transformer_layer_idx=layer_idx,
            projection_type=proj_type,
            component=component,
        )

    def _extract_layer_idx(self, name: str) -> Optional[int]:
        """Extract transformer layer index from parameter name."""
        import re
        # Match patterns like "layers.0.", "encoder.layers.12.", etc.
        match = re.search(r'\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return None

    def get_config(self) -> dict:
        """Return model configuration dictionary."""
        return self._config or {}

    def summary(self) -> str:
        """Return a human-readable summary of the loaded model."""
        if self._weights is None:
            return "Model not loaded. Call .load() first."

        total_params = sum(np.prod(v.shape) for v in self._weights.values())
        n_layers = 0
        n_compilable = 0

        for name, weight in self._weights.items():
            if self._classify_weight(name, weight.shape) is not None:
                n_compilable += 1

        lines = [
            f"Model: {self.model_id}",
            f"Total parameters: {total_params:,} ({total_params/1e9:.2f}B)",
            f"Weight tensors: {len(self._weights)}",
            f"Compilable linear layers: {n_compilable}",
        ]

        if self._config:
            cfg = self._config
            lines += [
                f"Architecture: {cfg.get('model_type', 'unknown')}",
                f"Hidden size: {cfg.get('hidden_size', 'unknown')}",
                f"Num layers: {cfg.get('num_hidden_layers', 'unknown')}",
                f"Num heads: {cfg.get('num_attention_heads', 'unknown')}",
            ]

        return "\n".join(lines)


def estimate_compilation_resources(
    parser: ModelParser,
    rank: int = 64,
    include_language_model: bool = True,
    include_vision_encoder: bool = True,
) -> dict:
    """
    Estimate the photonic resources (MZIs, chips, power) required to
    compile the full model.

    Args:
        parser: Loaded ModelParser
        rank: SVD truncation rank for compilation
        include_language_model: Include language model
        include_vision_encoder: Include vision encoder

    Returns:
        Dictionary of resource estimates
    """
    total_mzis = 0
    total_layers = 0
    layer_stats = {}

    for info, weight in parser.iter_linear_layers(
        include_language_model=include_language_model,
        include_vision_encoder=include_vision_encoder,
    ):
        m, n = weight.shape
        r = min(rank, m, n)

        # MZIs for U mesh (m×m) using rank-r rectangular Clements
        mzis_U = m * r  # approximate for rectangular Clements
        # MZIs for V† mesh (n×n)
        mzis_Vh = n * r

        total_mzis_layer = mzis_U + mzis_Vh
        total_mzis += total_mzis_layer
        total_layers += 1

        key = f"{info.component}.{info.projection_type}"
        if key not in layer_stats:
            layer_stats[key] = {"count": 0, "total_mzis": 0, "shape": info.shape}
        layer_stats[key]["count"] += 1
        layer_stats[key]["total_mzis"] += total_mzis_layer

    # Chip estimates (assume 4096 MZIs per 10mm×10mm chip)
    mzis_per_chip = 4096
    n_chips = (total_mzis + mzis_per_chip - 1) // mzis_per_chip

    # Power estimate (15mW per π shift, average π/2 per MZI)
    power_init_W = total_mzis * 2 * 15e-3 * 0.5  # init power (one-time)
    power_static_W = 0.0  # zero steady-state for thermal PS

    return {
        "rank": rank,
        "total_compilable_layers": total_layers,
        "total_mzis": total_mzis,
        "chips_required": n_chips,
        "chip_footprint_mm2": n_chips * 100,  # 10mm×10mm each
        "init_power_W": power_init_W,
        "static_inference_power_W": power_static_W,
        "layer_breakdown": layer_stats,
    }
