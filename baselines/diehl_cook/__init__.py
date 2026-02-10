"""
Diehl & Cook (2015) STDP-based SNN encoder.

This module requires BindsNET to be installed.
"""

try:
    from .encoder import DiehlCookEncoder
    __all__ = ['DiehlCookEncoder']
except ImportError:
    __all__ = []
    DiehlCookEncoder = None
