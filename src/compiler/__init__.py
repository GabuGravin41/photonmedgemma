"""
PhotoMedGemma Compiler
======================
Pipeline for compiling neural network weight matrices into
photonic MZI mesh phase settings.

Pipeline:
    ModelParser → LayerDecomposer → MZIMapper → PhaseEncoder → NetlistGenerator
"""

from compiler.model_parser import ModelParser, LayerInfo
from compiler.layer_decomposer import LayerDecomposer, DecomposedLayer
from compiler.mzi_mapper import MZIMapper, CompiledLayer
from compiler.phase_encoder import PhaseEncoder, PhaseMap
from compiler.netlist_generator import NetlistGenerator
from compiler.clements import clements_decompose, clements_simulate, ClementsResult

__all__ = [
    "ModelParser",
    "LayerInfo",
    "LayerDecomposer",
    "DecomposedLayer",
    "MZIMapper",
    "CompiledLayer",
    "PhaseEncoder",
    "PhaseMap",
    "NetlistGenerator",
    "clements_decompose",
    "clements_simulate",
    "ClementsResult",
]
