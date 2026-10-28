"""Case 4 (evolving porosity, Xie B2) transport and ADAP modules (piece 9).

Importing this package imports nothing heavy; dolfinx/ufl/petsc4py/mpi4py are loaded
inside the solver constructors. Chemistry is never imported here: the transport
backends only read ``chemistry.cells``, ``chemistry.components`` and ``chemistry.inlet``,
and ``coupling.advance_case4_step`` calls ``chemistry.react`` (see ``docs/DESIGN.md``).
"""
