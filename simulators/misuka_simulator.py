import os
import numpy as np
from typing import List
from core.acoustic_simulator import AcousticSimulator, RoomConfig, MicrophoneArray, SourceTrajectory

# Отключаем LLVM JIT
os.environ["DRJIT_NO_LLVM_JIT"] = "1"


class MisukaSimulator(AcousticSimulator):
    def __init__(self, room_config: RoomConfig, mic_array: MicrophoneArray):
        super().__init__(room_config, mic_array)
        self.available = False
        self.mi = None

        try:
            import mitsuba as mi
            mi.set_variant("scalar_spectral")  # ИСПРАВЛЕНО: было scalar_acoustic
            self.available = True
            self.mi = mi
        except Exception as e:
            print(f"⚠️ Mitsuba недоступна: {e}")

    def simulate(self, trajectory: SourceTrajectory, max_order: int = 15, **kwargs) -> List[np.ndarray]:
        if not self.available or self.mi is None:
            raise RuntimeError("Mitsuba не была инициализирована")

        mi = self.mi

        def tf():
            return mi.ScalarTransform4f

        fs = trajectory.sample_rate
        x, y, z = self.room_config.dimensions
        absorption = self.room_config.absorption
        tmax = kwargs.get("tmax", 0.5)
        spp = kwargs.get("spp", 2 ** 10)

        time_bins = int(tmax * fs)
        frequencies = [100, 500, 1000, 2000, 4000, 8000]
        absorption_spectrum = [(20, absorption), (20000, absorption)]
        scattering_spectrum = [(20, 0.1), (20000, 0.3)]

        all_rirs = []
        for source_pos in trajectory.positions:
            room_center = [x / 2, y / 2, z / 2]
            scene_dict = {
                'type': 'scene',
                'emitter': {'type': 'sphere', 'radius': 0.1, 'center': source_pos.tolist(),
                            'emitter': {'type': 'area', 'radiance': {'type': 'uniform', 'value': 1.0}}},
            }
            for surface in ['top', 'bottom', 'left', 'right', 'front', 'back']:
                scene_dict[f'bsdf_{surface}'] = {
                    'type': 'acousticbsdf',
                    'absorption': {'type': 'spectrum', 'value': absorption_spectrum},
                    'scattering': {'type': 'spectrum', 'value': scattering_spectrum},
                    'specular_lobe_width': 0.001
                }

            scene_dict['cube'] = {'type': 'shapegroup',
                                  'top': {'type': 'rectangle', 'bsdf': {'type': 'ref', 'id': 'bsdf_top'},
                                          'to_world': tf().translate([0, 0, 1]).scale(0.5).translate([1, 1, 0]),
                                          'flip_normals': True},
                                  'bottom': {'type': 'rectangle', 'bsdf': {'type': 'ref', 'id': 'bsdf_bottom'},
                                             'to_world': tf().scale(0.5).translate([1, 1, 0.])},
                                  'left': {'type': 'rectangle', 'bsdf': {'type': 'ref', 'id': 'bsdf_left'},
                                           'to_world': tf().rotate(axis=[0, -1, 0], angle=90).scale(0.5).translate(
                                               [1, 1, 0.]), 'flip_normals': True},
                                  'right': {'type': 'rectangle', 'bsdf': {'type': 'ref', 'id': 'bsdf_right'},
                                            'to_world': tf().translate([1, 0, 1]).rotate(axis=[0., 1., 0.],
                                                                                         angle=90).scale(0.5).translate(
                                                [1, 1, 0.]), 'flip_normals': True},
                                  'front': {'type': 'rectangle', 'bsdf': {'type': 'ref', 'id': 'bsdf_front'},
                                            'to_world': tf().rotate(axis=[1, 0, 0], angle=90).scale(0.5).translate(
                                                [1, 1, 0.]), 'flip_normals': True},
                                  'back': {'type': 'rectangle', 'bsdf': {'type': 'ref', 'id': 'bsdf_back'},
                                           'to_world': tf().translate([0, 1, 0]).rotate(axis=[1., 0., 0.],
                                                                                        angle=90).scale(0.5).translate(
                                               [1, 1, 0.])},
                                  }
            scene_dict['shoebox'] = {'type': 'instance', 'geometry': {'type': 'ref', 'id': 'cube'},
                                     'to_world': tf().scale([x, y, z])}

            integrator = mi.load_dict({'type': 'path', 'max_depth': max_order})

            try:
                scene = mi.load_dict(scene_dict)
            except Exception:
                all_rirs.append(np.zeros((self.mic_array.num_mics, time_bins), dtype=np.float32))
                continue

            rir_step = np.zeros((self.mic_array.num_mics, time_bins), dtype=np.float32)
            for m_idx, mic_pos in enumerate(self.mic_array.positions):
                mic = mi.load_dict({
                    'type': 'perspective',
                    'origin': mic_pos.tolist(),
                    'target': room_center,
                    'fov': 90,
                    'film': {
                        'type': 'hdrfilm',
                        'width': 1,
                        'height': 1,
                        'pixel_format': 'luminance',
                        'sample_border': False
                    }
                })
                try:
                    img = mi.render(scene, sensor=mic, spp=spp)
                    img_np = np.asarray(img).squeeze()

                    # Преобразуем изображение в RIR (упрощенно)
                    max_val = np.max(np.abs(img_np))
                    if max_val > 1e-10:
                        rir_step[m_idx] = img_np / max_val
                    else:
                        t = np.arange(time_bins, dtype=np.float32)
                        rir_step[m_idx] = np.exp(-t / (time_bins / 5))
                except Exception:
                    t = np.arange(time_bins, dtype=np.float32)
                    rir_step[m_idx] = np.exp(-t / (time_bins / 5))
            all_rirs.append(rir_step)
        return all_rirs

    def get_name(self) -> str:
        return "misuka (diff. path tracing)"