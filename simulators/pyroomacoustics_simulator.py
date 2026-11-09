import numpy as np
from typing import List
from core.acoustic_simulator import AcousticSimulator, RoomConfig, MicrophoneArray, SourceTrajectory
import pyroomacoustics as pra


class PyRoomAcousticsSimulator(AcousticSimulator):
    def __init__(self, room_config: RoomConfig, mic_array: MicrophoneArray):
        super().__init__(room_config, mic_array)
        self._last_room = None

    def _create_room(self, fs: int, max_order: int, scattering: float = 0.1):
        """Создаёт комнату с оптимальными параметрами"""
        x, y, z = self.room_config.dimensions
        absorption = self.room_config.absorption

        # Материал с поглощением и рассеянием
        material = pra.Material(
            energy_absorption=absorption,
            scattering=scattering
        )

        # Создаём комнату с ISM + ray tracing
        room = pra.ShoeBox(
            [x, y, z],
            fs=fs,
            materials=material,
            max_order=max_order,
            ray_tracing=True,
            air_absorption=True,
        )

        # Настраиваем ray tracing (без max_order - он уже в конструкторе)
        try:
            room.set_ray_tracing(
                n_rays=10000,
                energy_thres=1e-07,
                time_thres=2.0,
            )
        except Exception as e:
            print(f"   ⚠️ Ray tracing недоступен: {e}")
            room.ray_tracing = False

        return room

    def simulate(self, trajectory: SourceTrajectory, max_order: int = 15, **kwargs) -> List[np.ndarray]:
        fs = trajectory.sample_rate
        scattering = kwargs.get('scattering', 0.1)

        # Создаём комнату ОДИН раз
        room = self._create_room(fs, max_order, scattering)
        self._last_room = room

        # Добавляем микрофоны (pyroomacoustics ожидает форму (3, N))
        mic_positions = self.mic_array.positions.T
        mic_array_pra = pra.MicrophoneArray(mic_positions, fs)
        room.add_microphone_array(mic_array_pra)

        # Добавляем ВСЕ источники сразу
        for source_pos in trajectory.positions:
            room.add_source(source_pos.tolist())

        # Считаем RIR для всех источников сразу (один вызов)
        room.compute_rir()

        # Извлекаем RIR для каждого источника
        all_rirs = []
        num_mics = self.mic_array.num_mics
        num_sources = len(trajectory.positions)

        for source_idx in range(num_sources):
            max_len = max(len(room.rir[m][source_idx]) for m in range(num_mics))

            rir_for_step = np.zeros((num_mics, max_len), dtype=np.float32)
            for m in range(num_mics):
                rir_m = room.rir[m][source_idx]
                rir_for_step[m, :len(rir_m)] = rir_m

            all_rirs.append(rir_for_step)

        return all_rirs

    def get_rt60(self) -> float:
        """Возвращает RT60"""
        if self._last_room is None:
            return 0.0
        try:
            rt60 = self._last_room.rt60_theory()
            if isinstance(rt60, (list, tuple)):
                return rt60[0] if len(rt60) > 0 else 0.0
            return float(rt60)
        except Exception:
            return 0.0

    def get_name(self) -> str:
        return "pyroomacoustics (ISM + ray tracing)"