"""DG1 transport on a coarser grid than the chemistry (opt-in, ``--transport-cells``).

The chemistry, porosity, minerals and the Darcy flux live on ``chemistry.cells``
cells; the DG1 transport (dg_history, with history and positivity) runs on
``transport_cells`` cells, each containing r = chemistry.cells / transport_cells
chemistry cells. Purpose: refine only the chemistry grid while keeping the
transport grid fixed (e.g. transport 80, chemistry 160).

Each substep:
  1. restriction: coarse porosity = mean of the r fine porosities; coarse
     concentration = (sum of phi_f c_f) / (r phi_c), i.e. the aqueous inventory
     of the coarse cell is exactly the sum of its fine inventories;
  2. one DG1 transport substep on the coarse grid (unchanged dg_history code);
  3. prolongation: fine cell k of a coarse cell receives the mean of the coarse
     DG1 linear profile over its sub-interval, m + s xi_k (s = half-jump,
     xi_k = -1 + (2k+1)/r), plus one additive correction per coarse cell so
     that sum_k phi_f c_f = r phi_c m exactly (mass conservation with unequal
     fine porosities).
Chemistry changes between substeps reach the coarse DG1 history through the
restriction (the history keeps its slopes and absorbs the change of the means,
as in dg_history).

Provenance: copied from asrtm-phreeqc-public @ a152b91, methods/case4dev/coarse_transport.py. Numerics, public names and operation
order unchanged; imports re-pointed to adaptdg (benchmark constants in ``adaptdg.case4.constants``,
``dorfler_mark_1d`` / ``gradient_jump_indicator_1d`` / ``conservative_positive_projection_1d`` from
``adaptdg.marking`` / ``adaptdg.transport``).
No chemistry import: the chemistry object only needs ``cells``, ``components`` and ``inlet``.
"""

from types import SimpleNamespace

import numpy as np


class CoarseTransportDG:
    def __init__(self, chemistry, time_step_s, transport_cells, varying, positivity="local"):
        from .dg_history import HistoryDGTransport
        from .variable_dt import VariableDtHistoryDGTransport

        n = chemistry.cells
        if transport_cells <= 0 or n % transport_cells:
            raise ValueError("chemistry cells must be a multiple of --transport-cells")
        self.nt, self.r = int(transport_cells), n // int(transport_cells)
        self.chemistry = chemistry
        proxy = SimpleNamespace(cells=self.nt, components=chemistry.components, inlet=chemistry.inlet)
        cls = VariableDtHistoryDGTransport if varying else HistoryDGTransport
        self.inner = cls(proxy, time_step_s, history="dg1", positivity=positivity)
        self.substep_log = self.inner.substep_log
        self.xi = -1.0 + (2.0 * np.arange(self.r) + 1.0) / self.r
        from .dg_history import NOT_SIGN_CONSTRAINED
        self.constrained = [i for i, n in enumerate(chemistry.components) if n not in NOT_SIGN_CONSTRAINED]
        self.min_prolonged = np.inf              # over sign-constrained components only
        self.min_by_component = {}

    @property
    def solver(self):
        return self.inner.solver

    def set_substep_dt(self, dt_s):
        self.inner.set_substep_dt(dt_s)

    def step(self, state, flux_m_s):
        nt, r = self.nt, self.r
        phi_f = np.asarray(state.porosity, dtype=float).reshape(nt, r)
        c_f = np.asarray(state.concentrations_mol_l, dtype=float)
        ncomp = c_f.shape[0]
        phi_c = phi_f.mean(axis=1)
        c_c = (c_f.reshape(ncomp, nt, r) * phi_f[None]).sum(axis=2) / (r * phi_c[None])
        coarse = SimpleNamespace(porosity=phi_c, concentrations_mol_l=c_c)
        means, diag = self.inner.step(coarse, flux_m_s)
        nodal = self.inner.solver.nodal                       # (ncomp, nt, 2): left, right
        s = 0.5 * (nodal[:, :, 1] - nodal[:, :, 0])
        fine = means[:, :, None] + s[:, :, None] * self.xi[None, None, :]
        current = (fine * phi_f[None]).sum(axis=2)
        delta = (r * phi_c[None] * means - current) / phi_f.sum(axis=1)[None]
        fine = fine + delta[:, :, None]
        for i, name in enumerate(self.chemistry.components):
            self.min_by_component[name] = min(self.min_by_component.get(name, np.inf), float(fine[i].min()))
        self.min_prolonged = min(self.min_prolonged, float(fine[self.constrained].min()))
        return fine.reshape(ncomp, nt * r), diag
