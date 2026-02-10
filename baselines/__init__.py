"""
Baseline encoders for clustering/hashing tasks.
"""

from .base_encoder import BaseEncoder, DummyEncoder
from .flyhash.encoder import FlyHashEncoder
from .softhebb.encoder import SoftHebbEncoder
from .krotov.encoder import KrotovEncoder
from .biohash.encoder import BioHashEncoder
from .wta_hash.encoder import WTAHashEncoder
from .som.encoder import SOMEncoder
from .lsh.encoder import SimHashEncoder

# Diehl & Cook (optional, requires BindsNET)
try:
    from .diehl_cook.encoder import DiehlCookEncoder
    DIEHL_COOK_AVAILABLE = True
except ImportError:
    DIEHL_COOK_AVAILABLE = False
    DiehlCookEncoder = None

__all__ = [
    'BaseEncoder', 'DummyEncoder', 
    'FlyHashEncoder', 'SoftHebbEncoder', 'KrotovEncoder',
    'BioHashEncoder', 'WTAHashEncoder', 'SOMEncoder', 'SimHashEncoder',
]

if DIEHL_COOK_AVAILABLE:
    __all__.append('DiehlCookEncoder')
