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

# LM-SNN (optional, requires BindsNET)
try:
    from .lm_snn.encoder import LMSNNEncoder
    LM_SNN_AVAILABLE = True
except ImportError:
    LM_SNN_AVAILABLE = False
    LMSNNEncoder = None

# LC-SNN (optional, requires BindsNET)
try:
    from .lc_snn.encoder import LCSNNEncoder
    LC_SNN_AVAILABLE = True
except ImportError:
    LC_SNN_AVAILABLE = False
    LCSNNEncoder = None

# Deep-STDP (optional, requires BindsNET)
try:
    from .deep_stdp.encoder import DeepSTDPEncoder
    DEEP_STDP_AVAILABLE = True
except ImportError:
    DEEP_STDP_AVAILABLE = False
    DeepSTDPEncoder = None

__all__ = [
    'BaseEncoder', 'DummyEncoder',
    'FlyHashEncoder', 'SoftHebbEncoder', 'KrotovEncoder',
    'BioHashEncoder', 'WTAHashEncoder', 'SOMEncoder', 'SimHashEncoder',
]

if DIEHL_COOK_AVAILABLE:
    __all__.append('DiehlCookEncoder')
if LM_SNN_AVAILABLE:
    __all__.append('LMSNNEncoder')
if LC_SNN_AVAILABLE:
    __all__.append('LCSNNEncoder')
if DEEP_STDP_AVAILABLE:
    __all__.append('DeepSTDPEncoder')
