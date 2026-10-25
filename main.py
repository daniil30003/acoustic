import numpy as np
from scene.room import Room
from scene.trajectory import Trajectory
from precomputation.precomputer import Precomputer
from tracing.online_tracer import OnlineTracer
from audio.audio_builder import AudioBuilder
from audio.audio_loader import AudioLoader
from core.types import Vector3


def main():
    wav_path = "input/input.wav"
    try:
        sample_rate, dry_audio = AudioLoader.load(wav_path)
    except FileNotFoundError:
        print(f"Файл {wav_path} не найден")
        return

    audio_duration = len(dry_audio) / sample_rate

    room = Room(size_x=10.0, size_y=10.0, size_z=4.0,
                patch_size=0.1, absorption=0.4)
    print(f"  Патчи: {room.get_total_patches()}")

    mic_position = Vector3(2.5, 2.5, 1.5)

    step_duration = 0.05

    num_steps = int(np.ceil(audio_duration / step_duration))
    print(f"  Шаг: {step_duration * 1000:.0f} мс")
    print(f"  Количество шагов: {num_steps}")

    positions = []
    for i in range(num_steps):
        progress = i / max(1, num_steps - 1)
        x_pos = 1.0 + progress * 3.0
        y_pos = 2.5
        z_pos = 1.5
        positions.append(Vector3(x_pos, y_pos, z_pos))

    trajectory = Trajectory(positions, step_duration=step_duration, sample_rate=sample_rate)

    precomputer = Precomputer(
        room=room,
        mic_position=mic_position,
        energy_threshold=0.001
    )
    table = precomputer.precompute()

    tracer = OnlineTracer(room, table, mic_position)

    all_hits = []
    for step_idx, source_pos in enumerate(trajectory.positions):
        hits = tracer.trace_for_position(source_pos)
        all_hits.append(hits)

    builder = AudioBuilder(
        sample_rate=sample_rate,
        step_duration=trajectory.step_duration,
        rir_duration=1.5
    )

    wet_audio = builder.build_wet_audio(dry_audio, all_hits)

    output_path = "output/wet_full.wav"
    AudioLoader.save(output_path, sample_rate, wet_audio)

if __name__ == "__main__":
    main()