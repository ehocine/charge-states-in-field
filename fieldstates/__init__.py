"""fieldstates: charge states of a Coulomb impurity in gapped graphene in a perpendicular magnetic field.

Extends chargestates (paper 2) with the field term in the radial Dirac equation, valley-dependent
Furry bases, the screening of the Landau-quantized Dirac sea, and CI for two and three electrons
with arbitrary valley content.
"""
from .deps import chargestates, collapse2d  # noqa: F401

__version__ = "1.0.1"
