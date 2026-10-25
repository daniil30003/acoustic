from dataclasses import dataclass
from .types import Vector3
from typing import List
import numpy as np

@dataclass
class Wall:
    """Стена с информацией о сетке патчей"""
    normal: Vector3
    fixed_axis: int              # 0=X, 1=Y, 2=Z
    fixed_value: float           # значение по фиксированной оси
    live_axes: List[int]         # какие оси "живые" [a_axis, b_axis]
    min_coords: np.ndarray       # [a_min, b_min]
    max_coords: np.ndarray       # [a_max, b_max]
    patch_size: float            # размер патча
    n_a: int                     # число патчей по оси a
    n_b: int                     # число патчей по оси b
    start_patch_id: int          # ID первого патча на этой стене
    absorption: float