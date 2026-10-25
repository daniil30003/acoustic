import numpy as np
import scipy.io.wavfile as wavfile
from typing import Tuple


class AudioLoader:

    @staticmethod
    def load(filename: str) -> Tuple[int, np.ndarray]:
        """Загружает WAV, возвращает (sample_rate, audio_data)"""
        sample_rate, data = wavfile.read(filename)

        if data.dtype == np.int16:
            audio = data.astype(np.float32) / 32767.0
        elif data.dtype == np.int32:
            audio = data.astype(np.float32) / 2147483647.0
        elif data.dtype == np.float32:
            audio = data.astype(np.float32)
        else:
            raise ValueError(f"Неподдерживаемый формат: {data.dtype}")

        if audio.ndim > 1:
            audio = audio[:, 0]

        return int(sample_rate), audio

    @staticmethod
    def save(filename: str, sample_rate: int, audio: np.ndarray):
        """Сохраняет float32 аудио в WAV"""
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(filename, sample_rate, audio_int16)