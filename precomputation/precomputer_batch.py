import numpy as np
import cupy as cp
from scene.room import Room
from core.types import Vector3
from precomputation.hash_map import PairHashMap


class PrecomputerBatch:
    """
    Batch-версия предподсчёта на GPU (CuPy).
    Все лучи трассируются одновременно, шаг за шагом.
    """

    def __init__(self, room: Room, mic_position: Vector3, energy_threshold: float = 0.01):
        self.room = room
        self.mic_position = mic_position
        self.energy_threshold = energy_threshold
        self.table = PairHashMap()

        self.Lx = float(room.size_x)
        self.Ly = float(room.size_y)
        self.Lz = float(room.size_z)
        self.absorption = room.absorption

        print("  Загрузка данных в GPU...")
        # Центры и нормали патчей
        patch_centers_np = np.array([p.center.to_numpy() for p in room.patches], dtype=np.float32)
        patch_normals_np = np.array([p.normal.to_numpy() for p in room.patches], dtype=np.float32)
        self.patch_centers = cp.asarray(patch_centers_np)  # [N, 3]
        self.patch_normals = cp.asarray(patch_normals_np)  # [N, 3] (ДОБАВЛЕНО!)
        self.num_patches = len(room.patches)

        # Данные стен
        self.wall_normals = cp.array([w.normal.to_numpy() for w in room.walls], dtype=cp.float32)
        self.wall_fixed_axis = [w.fixed_axis for w in room.walls]
        self.wall_fixed_value = [w.fixed_value for w in room.walls]
        self.wall_live_axes = [w.live_axes for w in room.walls]
        self.wall_min_coords = [w.min_coords for w in room.walls]
        self.wall_max_coords = [w.max_coords for w in room.walls]

        print(f"  ✅ Загружено {self.num_patches} патчей в GPU")

    def precompute(self) -> PairHashMap:
        print(f"Batch-предподсчёт: {self.num_patches} лучей...")

        mic_pos = cp.asarray(self.mic_position.to_numpy(), dtype=cp.float32)

        # ШАГ 0: Начальные направления от микрофона к патчам
        current_dirs = self.patch_centers - mic_pos  # [N, 3]
        dists = cp.linalg.norm(current_dirs, axis=1)  # [N]
        current_dirs = current_dirs / cp.maximum(dists[:, None], 1e-6)  # Нормализация

        # Начальное состояние
        energy = cp.ones(self.num_patches, dtype=cp.float32) * (1.0 - self.absorption)
        accumulated_dist = dists.copy()
        current_patch_ids = cp.arange(self.num_patches, dtype=cp.int32)
        origins = self.patch_centers.copy()

        alive_mask = dists > 0.01

        all_prev_ids = []
        all_curr_ids = []
        all_dists = []
        all_energies = []

        bounce_count = 0
        max_bounces = 200

        while cp.any(alive_mask) and bounce_count < max_bounces:
            alive_indices = cp.where(alive_mask)[0]
            num_alive = len(alive_indices)

            if num_alive == 0:
                break

            alive_dirs = current_dirs[alive_indices]
            alive_origins = origins[alive_indices]
            alive_energy = energy[alive_indices]
            alive_accum_dist = accumulated_dist[alive_indices]
            alive_patch_ids = current_patch_ids[alive_indices]
            alive_normals = self.patch_normals[alive_indices]

            # 🚀 1. ОТРАЖЕНИЕ от текущего патча (ЭТОГО НЕ ХВАТАЛО!)
            dots = cp.sum(alive_dirs * alive_normals, axis=1, keepdims=True)
            reflected_dirs = alive_dirs - 2 * dots * alive_normals
            reflected_dirs = reflected_dirs / cp.maximum(cp.linalg.norm(reflected_dirs, axis=1, keepdims=True), 1e-6)

            # 🚀 2. Пересечение со стенами (используем reflected_dirs!)
            t_walls = cp.full((num_alive, 6), cp.inf, dtype=cp.float32)
            hit_points_all = cp.zeros((num_alive, 6, 3), dtype=cp.float32)

            for wall_idx in range(6):
                normal = self.wall_normals[wall_idx]
                fixed_axis = self.wall_fixed_axis[wall_idx]
                fixed_value = self.wall_fixed_value[wall_idx]
                live_axes = self.wall_live_axes[wall_idx]
                min_coords = self.wall_min_coords[wall_idx]
                max_coords = self.wall_max_coords[wall_idx]

                denom = cp.sum(reflected_dirs * normal, axis=1)
                valid_denom = denom < -1e-6  # Луч должен лететь В стену

                wall_point = cp.zeros(3, dtype=cp.float32)
                wall_point[fixed_axis] = fixed_value
                wall_point[live_axes[0]] = min_coords[0]
                wall_point[live_axes[1]] = min_coords[1]

                diff = wall_point - alive_origins
                t = cp.sum(diff * normal, axis=1) / cp.where(valid_denom, denom, -1.0)

                valid_t = (t > 0.01) & valid_denom
                hit_point = alive_origins + t[:, None] * reflected_dirs

                coord_a = hit_point[:, live_axes[0]]
                coord_b = hit_point[:, live_axes[1]]
                in_bounds = (coord_a >= min_coords[0]) & (coord_a <= max_coords[0]) & \
                            (coord_b >= min_coords[1]) & (coord_b <= max_coords[1])

                is_valid = valid_t & in_bounds
                t_walls[:, wall_idx] = cp.where(is_valid, t, cp.inf)
                hit_points_all[:, wall_idx, :] = hit_point

            # 3. Ближайшее пересечение
            closest_wall = cp.argmin(t_walls, axis=1)
            hit_dist = cp.min(t_walls, axis=1)

            hit_points = cp.zeros((num_alive, 3), dtype=cp.float32)
            for i in range(num_alive):
                w_idx = int(closest_wall[i])
                if t_walls[i, w_idx] < cp.inf:
                    hit_points[i] = hit_points_all[i, w_idx]

            # 4. Обновление энергии и расстояния
            new_energy = alive_energy * (1.0 - self.absorption)
            new_accum_dist = alive_accum_dist + hit_dist

            no_hit = hit_dist == cp.inf
            low_energy = new_energy < self.energy_threshold
            stuck = hit_dist < 0.05
            survives = ~no_hit & ~low_energy & ~stuck

            # 5. Запись путей
            if cp.any(survives):
                survive_indices = cp.where(survives)[0]
                num_survives = len(survive_indices)

                next_patch_ids = cp.zeros(num_survives, dtype=cp.int32)
                for idx, global_idx in enumerate(survive_indices):
                    w_idx = int(closest_wall[global_idx])
                    hit_point_np = cp.asnumpy(hit_points[global_idx])
                    wall = self.room.walls[w_idx]

                    a_coord = hit_point_np[wall.live_axes[0]]
                    b_coord = hit_point_np[wall.live_axes[1]]
                    a_idx = int((a_coord - wall.min_coords[0]) / wall.patch_size)
                    b_idx = int((b_coord - wall.min_coords[1]) / wall.patch_size)
                    a_idx = max(0, min(a_idx, wall.n_a - 1))
                    b_idx = max(0, min(b_idx, wall.n_b - 1))

                    next_patch_ids[idx] = wall.start_patch_id + a_idx * wall.n_b + b_idx

                # Сохраняем в CPU-массивы
                all_prev_ids.append(cp.asnumpy(alive_patch_ids[survive_indices]))
                all_curr_ids.append(cp.asnumpy(next_patch_ids))
                all_dists.append(cp.asnumpy(new_accum_dist[survive_indices]))
                all_energies.append(cp.asnumpy(new_energy[survive_indices]))

                # 6. Обновление состояний для следующего шага
                survive_global_indices = alive_indices[survive_indices]

                current_dirs[survive_global_indices] = reflected_dirs[survive_indices]
                origins[survive_global_indices] = hit_points[survive_indices]
                energy[survive_global_indices] = new_energy[survive_indices]
                accumulated_dist[survive_global_indices] = new_accum_dist[survive_indices]
                current_patch_ids[survive_global_indices] = next_patch_ids

                new_alive_mask = cp.zeros(self.num_patches, dtype=cp.bool_)
                new_alive_mask[survive_global_indices] = True
                alive_mask = new_alive_mask
            else:
                alive_mask = cp.zeros(self.num_patches, dtype=cp.bool_)

            bounce_count += 1
            if bounce_count % 10 == 0:
                print(f"    Отскок {bounce_count}: {int(cp.sum(alive_mask))} живых лучей")

        # 7. Сборка результатов
        print(f"  Загрузка путей в таблицу...")
        if all_prev_ids:
            all_prev_ids = np.concatenate(all_prev_ids)
            all_curr_ids = np.concatenate(all_curr_ids)
            all_dists = np.concatenate(all_dists)
            all_energies = np.concatenate(all_energies)

            for prev_id, curr_id, dist, energy_val in zip(all_prev_ids, all_curr_ids, all_dists, all_energies):
                self.table.add_path(int(prev_id), int(curr_id), float(dist), float(energy_val))

        stats = self.table.get_stats()
        print(f"  ✅ Готово. Уникальных пар: {stats['unique_pairs']}, Всего путей: {stats['total_paths']}")

        return self.table