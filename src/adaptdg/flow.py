"""Flow state handed from the application to the transport solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FlowState:
    """Water content and Darcy flux over one time step.

    Attributes
    ----------
    mesh : dolfinx mesh.
    theta_old, theta_new : water content (volumetric) at the step start/end.
    q : Darcy flux, Raviart-Thomas RT0 function.
    """

    mesh: Any
    theta_old: Any
    theta_new: Any
    q: Any
