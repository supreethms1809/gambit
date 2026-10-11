"""CDEA: contrastive decomposition by evidence allocation."""

from cdea.allocation import AllocateConfig, Allocation, allocate
from cdea.first_order import first_order_masks, unit_gradient

__all__ = [
    "AllocateConfig",
    "Allocation",
    "allocate",
    "first_order_masks",
    "unit_gradient",
]
