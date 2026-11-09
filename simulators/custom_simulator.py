import numpy as np
from typing import List
from core.acoustic_simulator import AcousticSimulator, RoomConfig, MicrophoneArray, SourceTrajectory

# Импортируем твои классы
from scene.room import Room
from precomputation.precomputer import Precomputer
from tracing.online_tracer import OnlineTracer
from core.types import Vector3, HitRecord


class CustomRayTracingSimulator(AcousticSimulator):
    """
    Собственный симулятор на основе предподсчёта + онлайн трассировки.

    Архитектура:
    1. Предподсчёт: из микрофона трассируем лучи ко всем патчам комнаты.
       Строим таблицу PairHashMap: (prev_patch, cur_patch) -> [(distance, energy), ...]
       Это означает: если луч идёт от prev_patch к cur_patch, он попадёт в микрофон
       через distance метров с энергией energy.

    2. Онлайн: для каждой позиции источника:
       - Считаем прямой путь (0 отскоков)
       - Для каждого патча p_a находим отражение -> p_b
       - Проверяем, есть ли пара (p_a, p_b) в таблице
       - Если есть — берём пути до микрофона

    3. Построение RIR: из HitRecord (time, energy) строим дискретный импульсный отклик.
    """

    def __init__(self, room_config: RoomConfig, mic_array: MicrophoneArray,
                 patch_size: float = 0.2, energy_threshold: float = 0.001):
        """
        Args:
            room_config: конфигурация комнаты (размеры, поглощение)
            mic_array: массив микрофонов (пока используем только первый)
            patch_size: размер патча в метрах (меньше = точнее, но медленнее)
            energy_threshold: порог энергии для остановки трассировки
        """
        super().__init__(room_config, mic_array)

        self.patch_size = patch_size
        self.energy_threshold = energy_threshold

        # Создаём комнату
        x, y, z = room_config.dimensions
        self.room = Room(
            size_x=float(x),
            size_y=float(y),
            size_z=float(z),
            patch_size=patch_size,
            absorption=room_config.absorption
        )

        print(f"🏗️ Room: {x}x{y}x{z}, патчи: {self.room.get_total_patches()}")

        # Берём первый микрофон (центр тетраэдра)
        mic_pos_np = mic_array.positions[0]
        self.mic_position = Vector3(float(mic_pos_np[0]), float(mic_pos_np[1]), float(mic_pos_np[2]))

        # Предподсчёт
        print(f"⏳ Предподсчёт (energy_threshold={energy_threshold})...")
        precomputer = Precomputer(
            room=self.room,
            mic_position=self.mic_position,
            energy_threshold=energy_threshold
        )
        self.table = precomputer.precompute()

        # Онлайн трассировщик
        self.tracer = OnlineTracer(self.room, self.table, self.mic_position)

        print(f"✅ Симулятор готов. Уникальных пар: {self.table.get_stats()['unique_pairs']}")

    def simulate(self, trajectory: SourceTrajectory, max_order: int = 15, **kwargs) -> List[np.ndarray]:
        """
        Для каждого шага траектории строит RIR.

        Returns:
            List[np.ndarray]: список RIR формы (num_mics, num_samples) для каждого шага
        """
        fs = trajectory.sample_rate
        rir_duration = kwargs.get('rir_duration', 1.5)  # длительность RIR в секундах
        num_samples = int(rir_duration * fs)
        num_mics = self.mic_array.num_mics  # пока 1, но интерфейс поддерживает больше

        all_rirs = []

        for step_idx, source_pos_np in enumerate(trajectory.positions):
            source_pos = Vector3(
                float(source_pos_np[0]),
                float(source_pos_np[1]),
                float(source_pos_np[2])
            )

            # Онлайн трассировка для этой позиции
            hits = self.tracer.trace_for_position(source_pos)

            # Строим RIR из hits
            rir = self._hits_to_rir(hits, fs, num_samples)

            # Расширяем до (num_mics, num_samples) — пока все микрофоны получают одинаковый RIR
            # (в будущем можно сделать предподсчёт для каждого микрофона отдельно)
            rir_multi = np.tile(rir, (num_mics, 1))

            all_rirs.append(rir_multi)

        return all_rirs

    def _hits_to_rir(self, hits: List[HitRecord], fs: int, num_samples: int) -> np.ndarray:
        """
        Преобразует список HitRecord в дискретный RIR.

        Каждый hit — это (time, energy). Мы ставим дельта-функцию в сэмпл
        time * fs с амплитудой sqrt(energy) (т.к. energy — это мощность,
        а амплитуда = sqrt(мощность)).
        """
        rir = np.zeros(num_samples, dtype=np.float32)

        for hit in hits:
            sample_idx = int(hit.time * fs)
            if 0 <= sample_idx < num_samples:
                # Амплитуда = корень из энергии (energy — это мощность)
                amplitude = np.sqrt(hit.energy)
                rir[sample_idx] += amplitude

        return rir

    def get_name(self) -> str:
        return f"Custom Ray Tracing (patch={self.patch_size}m)"