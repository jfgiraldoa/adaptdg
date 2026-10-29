"""Stationary adaptive residual-minimisation (saddle-point) solver for scalar PDEs.

Implements the discrete-dual-norm residual minimisation of Giraldo & Calo (2023), CMAME 417,
116285 (equation numbers are quoted in each module). Importing this package does not load
dolfinx, ufl, basix or petsc4py: every heavy import happens inside the functions that need it.
"""

from adaptdg.vms.problems import (
    Anisotropic,
    BurgersIsotropic,
    BurgersSingle,
    ErikssonJohnson,
    ErikssonJohnson3D,
    Fichera3D,
    Heterogeneous,
    NonlinearProblem,
    PoissonSine,
    Problem,
    ReentrantCorner,
)

__all__ = ["Anisotropic", "BurgersIsotropic", "BurgersSingle", "ErikssonJohnson", "ErikssonJohnson3D", "Fichera3D",
           "Heterogeneous", "NonlinearProblem", "PoissonSine", "Problem", "ReentrantCorner"]
