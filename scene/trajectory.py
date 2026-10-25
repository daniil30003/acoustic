from typing import List
from core.types import Vector3


class Trajectory:
    """Траектория движущегося источника"""

    def __init__(self, positions: List[Vector3], step_duration: float, sample_rate: int):
        self.positions = positions
        self.step_duration = step_duration
        self.sample_rate = sample_rate

    @property
    def num_steps(self) -> int:
        return len(self.positions)

    @property
    def total_duration(self) -> float:
        return self.num_steps * self.step_duration

    def get_position_at(self, time: float) -> Vector3:
        """Получить позицию в момент времени"""
        step = int(time / self.step_duration)
        step = max(0, min(step, self.num_steps - 1))
        return self.positions[step]