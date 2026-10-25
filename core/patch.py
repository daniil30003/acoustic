from dataclasses import dataclass
from .types import Vector3

@dataclass
class Patch:
    """Патч — квадратик на стене"""
    id: int
    center: Vector3
    normal: Vector3
    absorption: float  # коэффициент поглощения (0-1)