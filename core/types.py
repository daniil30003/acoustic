import numpy as np
from dataclasses import dataclass


@dataclass
class PathInfo:
    """Информация о пути от пары патчей до микрофона"""
    distance: float  # расстояние до микрофона (метры)
    energy: float  # энергия (с учётом поглощений)


@dataclass
class HitRecord:
    """Попадание луча в микрофон"""
    time: float  # время прихода (секунды)
    energy: float  # энергия попадания


class Vector3:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: 'Vector3') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self) -> float:
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> 'Vector3':
        length = self.length()
        if length > 0:
            return Vector3(self.x / length, self.y / length, self.z / length)
        return Vector3()

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @staticmethod
    def from_numpy(arr: np.ndarray) -> 'Vector3':
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))

    def __repr__(self):
        return f"Vector3({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"