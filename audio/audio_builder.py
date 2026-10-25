import numpy as np
from scipy.signal import fftconvolve
from typing import List
from core.types import HitRecord
from audio.audio_loader import AudioLoader


class AudioBuilder:

    def __init__(self, sample_rate: int, step_duration: float,
                 rir_duration: float = 0.5):
        self.sample_rate = sample_rate
        self.step_duration = step_duration
        self.rir_duration = rir_duration

    def hits_to_rir(self, hits: List[HitRecord]) -> np.ndarray:
        num_samples = int(self.rir_duration * self.sample_rate)
        rir = np.zeros(num_samples, dtype=np.float32)

        if not hits:
            return rir

        min_time = min(h.time for h in hits)

        for hit in hits:
            t_rel = hit.time - min_time
            pos = t_rel * self.sample_rate

            idx_floor = int(pos)
            frac = pos - idx_floor

            amp = np.sqrt(hit.energy)

            if 0 <= idx_floor < num_samples - 1:
                rir[idx_floor] += amp * (1.0 - frac)
                rir[idx_floor + 1] += amp * frac
            elif 0 <= idx_floor < num_samples:
                rir[idx_floor] += amp * (1.0 - frac)


        return rir

    def build_wet_audio(self, dry_audio: np.ndarray,
                        all_hits: List[List[HitRecord]]) -> np.ndarray:
        window_size = int(self.step_duration * self.sample_rate)

        tail_samples = int(self.rir_duration * self.sample_rate)
        output_size = len(dry_audio) + tail_samples
        output = np.zeros(output_size, dtype=np.float32)

        for step_idx, hits in enumerate(all_hits):
            if not hits:
                continue

            rir = self.hits_to_rir(hits)

            start_sample = step_idx * window_size
            end_sample = start_sample + window_size

            if start_sample >= len(dry_audio):
                break

            actual_end = min(end_sample, len(dry_audio))
            dry_chunk = dry_audio[start_sample:actual_end]

            if len(dry_chunk) == 0:
                continue

            wet_chunk = fftconvolve(dry_chunk, rir, mode='full')

            out_start = start_sample
            out_end = out_start + len(wet_chunk)

            if out_end > len(output):
                wet_chunk = wet_chunk[:len(output) - out_start]
                out_end = len(output)

            output[out_start:out_end] += wet_chunk

        return output