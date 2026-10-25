import numpy as np
from typing import List, Tuple
from core.wall import Wall
from core.patch import Patch
from core.types import Vector3


class Room:
    """Комната: стены + патчи"""

    def __init__(self, size_x: float, size_y: float, size_z: float,
                 patch_size: float, absorption: float):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.patch_size = patch_size
        self.absorption = absorption

        self.walls: List[Wall] = []
        self.patches: List[Patch] = []

        self._build_walls()
        self._build_patches()

    def _build_walls(self):
        """Создаёт 6 стен куба"""
        current_patch_id = 0

        # Пол (z=0) — нормаль ВНУТРЬ комнаты (ВВЕРХ)
        n_a = int(self.size_x / self.patch_size)
        n_b = int(self.size_y / self.patch_size)
        self.walls.append(Wall(
            normal=Vector3(0, 0, 1),  # БЫЛО (0, 0, -1)
            fixed_axis=2,
            fixed_value=0.0,
            live_axes=[0, 1],
            min_coords=np.array([0.0, 0.0]),
            max_coords=np.array([self.size_x, self.size_y]),
            patch_size=self.patch_size,
            n_a=n_a,
            n_b=n_b,
            start_patch_id=current_patch_id,
            absorption=self.absorption
        ))
        current_patch_id += n_a * n_b

        # Потолок (z=size_z) — нормаль ВНУТРЬ (ВНИЗ)
        self.walls.append(Wall(
            normal=Vector3(0, 0, -1),  # БЫЛО (0, 0, 1)
            fixed_axis=2,
            fixed_value=self.size_z,
            live_axes=[0, 1],
            min_coords=np.array([0.0, 0.0]),
            max_coords=np.array([self.size_x, self.size_y]),
            patch_size=self.patch_size,
            n_a=n_a,
            n_b=n_b,
            start_patch_id=current_patch_id,
            absorption=self.absorption
        ))
        current_patch_id += n_a * n_b

        # Стена x=0 — нормаль ВНУТРЬ (+X)
        n_a = int(self.size_y / self.patch_size)
        n_b = int(self.size_z / self.patch_size)
        self.walls.append(Wall(
            normal=Vector3(1, 0, 0),  # БЫЛО (-1, 0, 0)
            fixed_axis=0,
            fixed_value=0.0,
            live_axes=[1, 2],
            min_coords=np.array([0.0, 0.0]),
            max_coords=np.array([self.size_y, self.size_z]),
            patch_size=self.patch_size,
            n_a=n_a,
            n_b=n_b,
            start_patch_id=current_patch_id,
            absorption=self.absorption
        ))
        current_patch_id += n_a * n_b

        # Стена x=size_x — нормаль ВНУТРЬ (-X)
        self.walls.append(Wall(
            normal=Vector3(-1, 0, 0),  # БЫЛО (1, 0, 0)
            fixed_axis=0,
            fixed_value=self.size_x,
            live_axes=[1, 2],
            min_coords=np.array([0.0, 0.0]),
            max_coords=np.array([self.size_y, self.size_z]),
            patch_size=self.patch_size,
            n_a=n_a,
            n_b=n_b,
            start_patch_id=current_patch_id,
            absorption=self.absorption
        ))
        current_patch_id += n_a * n_b

        # Стена y=0 — нормаль ВНУТРЬ (+Y)
        n_a = int(self.size_x / self.patch_size)
        n_b = int(self.size_z / self.patch_size)
        self.walls.append(Wall(
            normal=Vector3(0, 1, 0),  # БЫЛО (0, -1, 0)
            fixed_axis=1,
            fixed_value=0.0,
            live_axes=[0, 2],
            min_coords=np.array([0.0, 0.0]),
            max_coords=np.array([self.size_x, self.size_z]),
            patch_size=self.patch_size,
            n_a=n_a,
            n_b=n_b,
            start_patch_id=current_patch_id,
            absorption=self.absorption
        ))
        current_patch_id += n_a * n_b

        # Стена y=size_y — нормаль ВНУТРЬ (-Y)
        self.walls.append(Wall(
            normal=Vector3(0, -1, 0),  # БЫЛО (0, 1, 0)
            fixed_axis=1,
            fixed_value=self.size_y,
            live_axes=[0, 2],
            min_coords=np.array([0.0, 0.0]),
            max_coords=np.array([self.size_x, self.size_z]),
            patch_size=self.patch_size,
            n_a=n_a,
            n_b=n_b,
            start_patch_id=current_patch_id,
            absorption=self.absorption
        ))

    def _build_patches(self):
        """Создаёт патчи для всех стен"""
        for wall in self.walls:
            for a_idx in range(wall.n_a):
                for b_idx in range(wall.n_b):
                    a_coord = wall.min_coords[0] + (a_idx + 0.5) * wall.patch_size
                    b_coord = wall.min_coords[1] + (b_idx + 0.5) * wall.patch_size

                    center = np.zeros(3)
                    center[wall.fixed_axis] = wall.fixed_value
                    center[wall.live_axes[0]] = a_coord
                    center[wall.live_axes[1]] = b_coord

                    patch = Patch(
                        id=wall.start_patch_id + a_idx * wall.n_b + b_idx,
                        center=Vector3.from_numpy(center),
                        normal=wall.normal,
                        absorption=wall.absorption
                    )
                    self.patches.append(patch)

    def intersect_wall(self, origin: Vector3, direction: Vector3) -> Tuple[int, float, Vector3]:
        """Поиск пересечения луча со стенами"""
        closest_wall_idx = -1
        min_dist = float('inf')
        hit_point = None

        for wall_idx, wall in enumerate(self.walls):
            denom = wall.normal.to_numpy().dot(direction.to_numpy())

            if denom >= 0:
                continue

            if abs(denom) < 1e-6:
                continue

            wall_point = np.zeros(3)
            wall_point[wall.fixed_axis] = wall.fixed_value
            wall_point[wall.live_axes[0]] = wall.min_coords[0]
            wall_point[wall.live_axes[1]] = wall.min_coords[1]

            t = np.dot(wall_point - origin.to_numpy(), wall.normal.to_numpy()) / denom

            if t < 0.01:  # Игнорируем стены ближе 1 см
                continue

            point = origin.to_numpy() + t * direction.to_numpy()

            a_coord = point[wall.live_axes[0]]
            b_coord = point[wall.live_axes[1]]

            if (a_coord < wall.min_coords[0] or a_coord > wall.max_coords[0] or
                    b_coord < wall.min_coords[1] or b_coord > wall.max_coords[1]):
                continue

            if t < min_dist:
                min_dist = t
                closest_wall_idx = wall_idx
                hit_point = Vector3.from_numpy(point)

        # Сдвиг точки попадания внутрь комнаты
        if hit_point is not None and closest_wall_idx >= 0:
            epsilon = 0.01  # 1 см
            if abs(hit_point.x) < epsilon:
                hit_point.x = epsilon
            elif abs(hit_point.x - self.size_x) < epsilon:
                hit_point.x = self.size_x - epsilon

            if abs(hit_point.y) < epsilon:
                hit_point.y = epsilon
            elif abs(hit_point.y - self.size_y) < epsilon:
                hit_point.y = self.size_y - epsilon

            if abs(hit_point.z) < epsilon:
                hit_point.z = epsilon
            elif abs(hit_point.z - self.size_z) < epsilon:
                hit_point.z = self.size_z - epsilon

        return closest_wall_idx, min_dist, hit_point

    def point_to_patch_id(self, hit_point: Vector3, wall_idx: int) -> int:
        """Определяет ID патча по точке попадания"""
        wall = self.walls[wall_idx]

        a_coord = hit_point.to_numpy()[wall.live_axes[0]]
        b_coord = hit_point.to_numpy()[wall.live_axes[1]]

        a_idx = int((a_coord - wall.min_coords[0]) / wall.patch_size)
        b_idx = int((b_coord - wall.min_coords[1]) / wall.patch_size)

        a_idx = max(0, min(a_idx, wall.n_a - 1))
        b_idx = max(0, min(b_idx, wall.n_b - 1))

        return wall.start_patch_id + a_idx * wall.n_b + b_idx

    def get_patch_center(self, patch_id: int) -> Vector3:
        """Возвращает центр патча по ID"""
        return self.patches[patch_id].center

    def get_total_patches(self) -> int:
        """Общее число патчей"""
        return len(self.patches)