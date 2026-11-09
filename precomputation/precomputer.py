import numpy as np
from scene.room import Room
from core.types import Vector3
from precomputation.hash_map import PairHashMap


class Precomputer:
    def __init__(self, room: Room, mic_position: Vector3, energy_threshold: float = 0.01):
        self.room = room
        self.mic_position = mic_position
        self.energy_threshold = energy_threshold
        self.table = PairHashMap()

    def precompute(self) -> PairHashMap:
        for target_patch_id in range(self.room.get_total_patches()):
            self._trace_ray_from_mic(target_patch_id)
        return self.table

    def _trace_ray_from_mic(self, target_patch_id: int):
        target_center = self.room.get_patch_center(target_patch_id)
        dir_mic_to_p1 = target_center - self.mic_position
        dist_mic_to_p1 = dir_mic_to_p1.length()

        if dist_mic_to_p1 < 0.05:
            return

        dir_mic_to_p1 = dir_mic_to_p1.normalized()

        current_patch_id = target_patch_id
        current_pos = target_center
        current_dir = dir_mic_to_p1

        current_energy = 1.0 * (1.0 - self.room.patches[target_patch_id].absorption)
        accumulated_dist = dist_mic_to_p1

        while current_energy > self.energy_threshold:
            normal = self.room.patches[current_patch_id].normal
            dot_prod = current_dir.dot(normal)

            if dot_prod >= 0:
                return

            if abs(dot_prod) < 1e-6:
                return

            reflected_dir = current_dir - normal * (2 * dot_prod)
            reflected_dir = reflected_dir.normalized()

            wall_idx, hit_dist, hit_point = self.room.intersect_wall(current_pos, reflected_dir)

            if wall_idx < 0 or hit_point is None:
                return

            if hit_dist < 0.05:
                return

            next_patch_id = self.room.point_to_patch_id(hit_point, wall_idx)

            accumulated_dist += hit_dist
            current_energy *= (1.0 - self.room.patches[next_patch_id].absorption)

            self.table.add_path(next_patch_id, current_patch_id,
                                accumulated_dist, current_energy)

            current_patch_id = next_patch_id
            current_pos = hit_point
            current_dir = reflected_dir