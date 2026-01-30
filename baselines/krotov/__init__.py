"""
Krotov-Hopfield Biological Learning baseline.

Based on "Unsupervised Learning by Competing Hidden Units"
by D. Krotov and J. Hopfield (PNAS, 2019)
https://doi.org/10.1073/pnas.1820458116
"""

from .encoder import KrotovEncoder

__all__ = ['KrotovEncoder']
