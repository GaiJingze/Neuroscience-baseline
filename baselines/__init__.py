"""
Baseline encoders for clustering/hashing tasks.
"""

from .base_encoder import BaseEncoder, DummyEncoder
from .flyhash.encoder import FlyHashEncoder
from .diehl_cook.encoder import DiehlCookEncoder
from .softhebb.encoder import SoftHebbEncoder

__all__ = ['BaseEncoder', 'DummyEncoder', 'FlyHashEncoder', 'DiehlCookEncoder', 'SoftHebbEncoder']
