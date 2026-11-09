import os
import sys
import numpy as np
from typing import List
from core.acoustic_simulator import AcousticSimulator, RoomConfig, MicrophoneArray, SourceTrajectory

# Автоматический поиск CUDA DLL
try:
    from utils.cuda_setup import ensure_cuda_dlls

    ensure_cuda_dlls()
except ImportError:
    pass

try:
    import gpuRIR

    GPURIR_AVAILABLE = True
except (ImportError, OSError) as e:
    GPURIR_AVAILABLE = False
    print(f"⚠️ gpuRIR недоступен: {e}")


class GpuRIRSimulator(AcousticSimulator):
    """
    Улучшенный симулятор на основе gpuRIR.

    Особенности:
    - Гибридный метод: точный ISM для ранних отражений + диффузная модель для хвоста.
    - Поддержка диаграмм направленности (кардиоида и т.д.).
    - Аппаратное ускорение на GPU (CUDA).
    """

    def __init__(self, room_config: RoomConfig, mic_array: MicrophoneArray):
        super().__init__(room_config, mic_array)
        if not GPURIR_AVAILABLE:
            raise RuntimeError("gpuRIR не установлена или CUDA не найдена")

    def simulate(self, trajectory: SourceTrajectory, max_order: int = 15, **kwargs) -> List[np.ndarray]:
        fs = trajectory.sample_rate
        x, y, z = self.room_config.dimensions
        absorption = self.room_config.absorption

        # Параметры из kwargs
        Tmax = kwargs.get("tmax", 2.0)  # Увеличили до 2.0 сек для хорошего эха
        # Tdiff: время перехода к диффузной модели.
        # 0.1 сек = точные ранние отражения, остальное - быстро.
        Tdiff = kwargs.get("tdiff", 0.1)

        # Диаграммы направленности (по умолчанию всенаправленные)
        spkr_pattern = kwargs.get("spkr_pattern", "omni")
        mic_pattern = kwargs.get("mic_pattern", "omni")

        # Согласно документации gpuRIR:
        # room_sz: 3 элемента [x, y, z]
        room_sz = [float(x), float(y), float(z)]

        reflection_coefficient = 1.0 - float(absorption)
        beta = [reflection_coefficient] * 6

        # nb_img: 3 элемента (количество изображений в каждом измерении)
        nb_img = [int(max_order)] * 3

        source_positions = trajectory.positions.astype(np.float32)
        mic_positions = self.mic_array.positions.astype(np.float32)

        # Вызов симуляции
        rir_full = gpuRIR.simulateRIR(
            room_sz=room_sz,
            beta=beta,
            pos_src=source_positions,
            pos_rcv=mic_positions,
            nb_img=nb_img,
            Tmax=Tmax,
            fs=fs,
            Tdiff=Tdiff,
            spkr_pattern=spkr_pattern,
            mic_pattern=mic_pattern
        )

        rir_full = np.asarray(rir_full)

        # gpuRIR возвращает форму: (num_sources, num_receivers, num_samples)
        # Нам нужно: (num_sources, num_mics, num_samples) - это уже так!
        # Но если num_sources=1, то форма (1, num_mics, num_samples)

        # Преобразуем в список RIR для каждого шага траектории
        # rir_full[step_idx] даст (num_mics, num_samples)
        all_rirs = []
        for step_idx in range(trajectory.num_steps):
            if step_idx < rir_full.shape[0]:
                all_rirs.append(rir_full[step_idx])
            else:
                # Если источников меньше, чем шагов (не должно быть), заполняем нулями
                all_rirs.append(np.zeros((self.mic_array.num_mics, int(Tmax * fs)), dtype=np.float32))

        return all_rirs

    def get_name(self) -> str:
        return "gpuRIR (Hybrid ISM + Diffuse)"