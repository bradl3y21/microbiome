"""Human controls for the deterministic microbiome simulation.

Run the included two-genome sandbox with ``python -m microbiome.ui``. All
simulation commands pass through the D3 controller boundary.
"""

from .app import MicrobiomeUI, dispatch_command, run_demo

__all__ = ["MicrobiomeUI", "dispatch_command", "run_demo"]
