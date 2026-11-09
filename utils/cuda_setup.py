# utils/cuda_setup.py
import os
import sys
import glob

def ensure_cuda_dlls():
    if sys.platform != 'win32':
        return

    for env_var in ['CUDA_PATH', 'CUDA_HOME']:
        path = os.environ.get(env_var)
        if path and os.path.isdir(path):
            bin_dir = os.path.join(path, 'bin')
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
                return

    base_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(base_path):
        versions = sorted(glob.glob(os.path.join(base_path, 'v*')), reverse=True)
        for ver in versions:
            bin_dir = os.path.join(ver, 'bin')
            if os.path.isdir(bin_dir) and glob.glob(os.path.join(bin_dir, 'cudart64_*.dll')):
                os.add_dll_directory(bin_dir)
                return

    path_dirs = os.environ.get('PATH', '').split(os.pathsep)
    for path_dir in path_dirs:
        if glob.glob(os.path.join(path_dir, 'cudart64_*.dll')):
            os.add_dll_directory(path_dir)
            return