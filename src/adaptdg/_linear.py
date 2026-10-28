"""Constant-operator solve path for DOLFINx linear problems.

The stock ``LinearProblem.solve`` reassembles the matrix on every call.  The
transport operators used here are constant while only the previous state and
inflow on the right-hand side change.  This wrapper assembles the operator
once and reuses its PETSc KSP/preconditioner for subsequent right-hand sides.

Extracted unchanged from ASRTM-PHREEQC (src/asrtm_phreeqc/reusable_linear_problem.py). dolfinx/petsc4py are imported lazily inside the methods.
"""

from __future__ import annotations


class ReusableLinearProblem:
    """Solve new right-hand sides without reassembling a constant matrix."""

    def __init__(self, problem) -> None:
        from dolfinx.fem.petsc import assemble_matrix

        self.problem = problem
        self.matrix_assembly_count = 0
        self.solve_count = 0

        problem.A.zeroEntries()
        assemble_matrix(problem.A, problem.a, bcs=problem.bcs)
        problem.A.assemble()
        self.matrix_assembly_count += 1

        if problem.P_mat is not None:
            problem.P_mat.zeroEntries()
            assemble_matrix(
                problem.P_mat,
                problem.preconditioner,
                bcs=problem.bcs,
            )
            problem.P_mat.assemble()
        problem.solver.setOperators(problem.A, problem.P_mat)
        if hasattr(problem.solver, "setReusePreconditioner"):
            problem.solver.setReusePreconditioner(True)

    def solve(self):
        """Assemble only the RHS, solve, and update the stored Function."""

        from petsc4py import PETSc
        import dolfinx
        from dolfinx.fem.petsc import apply_lifting, assemble_vector, assign

        problem = self.problem
        dolfinx.la.petsc._zero_vector(problem.b)
        assemble_vector(problem.b, problem.L)

        if problem.bcs:
            apply_lifting(problem.b, [problem.a], bcs=[problem.bcs])
            dolfinx.la.petsc._ghost_update(
                problem.b,
                PETSc.InsertMode.ADD,
                PETSc.ScatterMode.REVERSE,
            )
            for boundary_condition in problem.bcs:
                boundary_condition.set(problem.b.array_w)
        else:
            dolfinx.la.petsc._ghost_update(
                problem.b,
                PETSc.InsertMode.ADD,
                PETSc.ScatterMode.REVERSE,
            )

        problem.solver.solve(problem.b, problem.x)
        dolfinx.la.petsc._ghost_update(
            problem.x,
            PETSc.InsertMode.INSERT,
            PETSc.ScatterMode.FORWARD,
        )
        assign(problem.x, problem.u)
        self.solve_count += 1
        return problem.u
