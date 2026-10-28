"""Stationary adaptive residual-minimisation (saddle-point) solver for scalar PDEs.

Implements the discrete-dual-norm residual minimisation of Giraldo & Calo (2023), CMAME 417,
116285 (equation numbers are quoted in each module). Importing this package does not load
dolfinx, ufl, basix or petsc4py: every heavy import happens inside the functions that need it.
"""

from adaptdg.vms.problems import Fichera3D, Problem, ReentrantCorner

__all__ = ["Fichera3D", "Problem", "ReentrantCorner"]
