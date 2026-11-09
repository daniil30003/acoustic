import numpy as np
from typing import List
from scene.room import Room
from core.types import Vector3, HitRecord
from precomputation.hash_map import PairHashMap


class OnlineTracer:
    def __init__(self, room: Room, table: PairHashMap, mic_position: Vector3):
        self.room = room
        self.table = table
        self.mic_position = mic_position

        # 🚀 ОПТИМИЗАЦИЯ: Кэшируем данные патчей в NumPy массивы ОДИН раз!
        # Это избавляет от тысяч вызовов методов в цикле
        self.patch_centers = np.array([p.center.to_numpy() for p in room.patches])
        self.patch_normals = np.array([p.normal.to_numpy() for p in room.patches])
        self.patch_ids = np.arange(room.get_total_patches())

    def trace_for_position(self, source_pos: Vector3) -> List[HitRecord]:
        hits: List[HitRecord] = []
        src_np = source_pos.to_numpy()

        # 1. Прямой луч
        direct_dist = np.linalg.norm(self.mic_position.to_numpy() - src_np)
        hits.append(HitRecord(time=direct_dist / 343.0, energy=1.0))

        # 2. Векторизованный расчёт для ВСЕХ патчей сразу
        dirs_to_pa = self.patch_centers - src_np
        dists_to_pa = np.linalg.norm(dirs_to_pa, axis=1)

        # Фильтруем слишком близкие патчи
        valid_mask = dists_to_pa > 0.001
        valid_ids = self.patch_ids[valid_mask]
        valid_dirs = dirs_to_pa[valid_mask]
        valid_dists = dists_to_pa[valid_mask]
        valid_normals = self.patch_normals[valid_mask]

        # Нормализация направлений (векторно)
        norms = np.linalg.norm(valid_dirs, axis=1, keepdims=True)
        valid_dirs_norm = valid_dirs / norms

        # Отражение: R = D - 2 * (D · N) * N
        dots = np.sum(valid_dirs_norm * valid_normals, axis=1, keepdims=True)
        reflected_dirs = valid_dirs_norm - 2 * dots * valid_normals

        # 🚀 Теперь мы можем быстро проверить пересечения только для валидных направлений
        # (Здесь можно оставить цикл, но он будет в 10-20 раз короче,
        # либо векторизовать intersect_wall, если это критично)
        for i in range(len(valid_ids)):
            p_a_id = valid_ids[i]
            p_a_center = self.patch_centers[p_a_id]
            dir_out = Vector3.from_numpy(reflected_dirs[i])
            dist_s_to_pa = valid_dists[i]

            wall_idx, dist_pa_to_pb, hit_point = self.room.intersect_wall(
                Vector3.from_numpy(p_a_center), dir_out
            )

            if wall_idx < 0 or hit_point is None:
                continue

            p_b_id = self.room.point_to_patch_id(hit_point, wall_idx)

            if self.table.has_pair(p_a_id, p_b_id):
                paths = self.table.get_paths(p_a_id, p_b_id)
                for path in paths:
                    total_dist = dist_s_to_pa + path.distance
                    hits.append(HitRecord(time=total_dist / 343.0, energy=path.energy))

        return hits