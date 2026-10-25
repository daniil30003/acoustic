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

    def trace_for_position(self, source_pos: Vector3) -> List[HitRecord]:
        hits: List[HitRecord] = []

        # 1. Прямой луч (0 отскоков)
        direct_dist = (self.mic_position - source_pos).length()
        hits.append(HitRecord(time=direct_dist / 343.0, energy=1.0))

        # 2. Отражённые лучи
        for p_a_id in range(self.room.get_total_patches()):
            p_a_center = self.room.get_patch_center(p_a_id)

            dir_s_to_pa = p_a_center - source_pos
            dist_s_to_pa = dir_s_to_pa.length()

            if dist_s_to_pa < 0.001:
                continue

            dir_s_to_pa = dir_s_to_pa.normalized()

            # Отражение от P_A
            normal_pa = self.room.patches[p_a_id].normal
            dir_out = self._reflect(dir_s_to_pa, normal_pa)

            # Хоп 2: P_A -> P_B
            wall_idx, dist_pa_to_pb, hit_point = self.room.intersect_wall(p_a_center, dir_out)

            if wall_idx < 0 or hit_point is None:
                continue

            p_b_id = self.room.point_to_patch_id(hit_point, wall_idx)

            if not self.table.has_pair(p_a_id, p_b_id):
                continue

            paths = self.table.get_paths(p_a_id, p_b_id)

            for path in paths:
                total_dist = dist_s_to_pa + path.distance

                total_energy = path.energy

                hits.append(HitRecord(time=total_dist / 343.0, energy=total_energy))

        return hits

    @staticmethod
    def _reflect(direction: Vector3, normal: Vector3) -> Vector3:
        dot = direction.dot(normal)
        return direction - normal * (2 * dot)