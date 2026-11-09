from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class RoomConfig:
    dimensions: Tuple[float, float, float]
    absorption: float
    rt60: float = field(init=False, default=None)

    def __post_init__(self):
        x, y, z = self.dimensions
        volume = x * y * z
        surface = 2 * (x * y + x * z + y * z)
        alpha = self.absorption
        if alpha > 0 and surface > 0:
            self.rt60 = 0.161 * volume / (-surface * np.log(1 - alpha))
        else:
            self.rt60 = 10.0


@dataclass
class MicrophoneArray:
    positions: np.ndarray
    name: str = "default"

    @classmethod
    def tetrahedron(cls, center: np.ndarray, radius: float = 0.05) -> "MicrophoneArray":
        vertices = np.array([
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=np.float32)
        vertices /= np.linalg.norm(vertices[0])
        positions = vertices * radius + np.asarray(center, dtype=np.float32)
        return cls(positions=positions, name="tetrahedron")

    @property
    def num_mics(self) -> int:
        return self.positions.shape[0]


@dataclass
class SourceTrajectory:
    positions: np.ndarray
    sample_rate: int
    step_duration: float

    @property
    def num_steps(self) -> int:
        return self.positions.shape[0]

    @property
    def total_duration(self) -> float:
        return self.num_steps * self.step_duration


class AcousticSimulator(ABC):
    def __init__(self, room_config: RoomConfig, mic_array: MicrophoneArray):
        self.room_config = room_config
        self.mic_array = mic_array

    @abstractmethod
    def simulate(self, trajectory: SourceTrajectory, max_order: int = 15, **kwargs) -> List[np.ndarray]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass