"""Surrogate safety measures.

Public API for the SSM subpackage. Individual modules (ttc, pet, drac,
event_mining) define their own result types and compute functions; the
names below are the stable public surface.
"""

from .drac import DRACResult, compute_drac_constant_velocity
from .pet import PETResult, compute_pet_from_conflict_zone
from .ttc import TTCResult, compute_ttc_constant_velocity

__all__: list[str] = [
    "DRACResult",
    "PETResult",
    "TTCResult",
    "compute_drac_constant_velocity",
    "compute_pet_from_conflict_zone",
    "compute_ttc_constant_velocity",
]
