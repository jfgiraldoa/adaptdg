"""adaptdg: adaptive DG transport on FEniCSx. Pre-alpha, interface not stable."""

from adaptdg.fields import register_field
from adaptdg.flow import FlowState
from adaptdg.loop import AdaptiveTransport

__version__ = "0.0.1"
__all__ = ["AdaptiveTransport", "FlowState", "__version__", "register_field"]
