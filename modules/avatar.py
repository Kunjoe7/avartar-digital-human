import sys
import os
import tempfile
import argparse
import threading
import logging
import numpy as np

import config

# Add FLOAT to path
sys.path.insert(0, config.FLOAT_DIR)

import torch
from generate import InferenceAgent

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()


def _build_opt(gpu_id: int, nfe: int = config.FLOAT_NFE):
    """Build FLOAT options for a specific GPU."""
    parser = argparse.ArgumentParser()

    # Base options
    parser.add_argument('--pretrained_dir', type=str, default=os.path.join(config.FLOAT_DIR, 'checkpoints'))
    parser.add_argument('--seed', default=15, type=int)
    parser.add_argument('--fix_noise_seed', action='store_true')
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--fps', type=float, default=25.)
    parser.add_argument('--sampling_rate', type=int, default=16000)
    parser.add_argument('--audio_marcing', type=int, default=2)
    parser.add_argument('--wav2vec_sec', default=2, type=float)
    parser.add_argument('--wav2vec_model_path', default=os.path.join(config.FLOAT_DIR, 'checkpoints', 'wav2vec2-base-960h'))
    parser.add_argument('--audio2emotion_path', default=os.path.join(config.FLOAT_DIR, 'checkpoints', 'wav2vec-english-speech-emotion-recognition'))
    parser.add_argument('--attention_window', default=2, type=int)
    parser.add_argument('--only_last_features', action='store_true')
    parser.add_argument('--average_emotion', action='store_true')
    parser.add_argument('--audio_dropout_prob', default=0.1, type=float)
    parser.add_argument('--ref_dropout_prob', default=0.1, type=float)
    parser.add_argument('--emotion_dropout_prob', default=0.1, type=float)
    parser.add_argument('--style_dim', type=int, default=512)
    parser.add_argument('--dim_a', type=int, default=512)
    parser.add_argument('--dim_w', type=int, default=512)
    parser.add_argument('--dim_h', type=int, default=1024)
    parser.add_argument('--dim_m', type=int, default=20)
    parser.add_argument('--dim_e', type=int, default=7)
    parser.add_argument('--fmt_depth', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=4.0, type=float)
    parser.add_argument('--no_learned_pe', action='store_true')
    parser.add_argument('--num_prev_frames', type=int, default=10)
    parser.add_argument('--max_grad_norm', default=1, type=float)
    parser.add_argument('--ode_atol', default=1e-5, type=float)
    parser.add_argument('--ode_rtol', default=1e-5, type=float)
    parser.add_argument('--nfe', default=nfe, type=int)
    parser.add_argument('--torchdiffeq_ode_method', default='euler')
    parser.add_argument('--a_cfg_scale', default=2.0, type=float)
    parser.add_argument('--e_cfg_scale', default=1.0, type=float)
    parser.add_argument('--r_cfg_scale', default=1.0, type=float)
    parser.add_argument('--n_diff_steps', type=int, default=500)
    parser.add_argument('--diff_schedule', type=str, default='cosine')
    parser.add_argument('--diffusion_mode', type=str, default='sample')
    # Inference options
    parser.add_argument('--ckpt_path', default=config.FLOAT_CKPT, type=str)
    parser.add_argument('--ref_path', default=None, type=str)
    parser.add_argument('--aud_path', default=None, type=str)
    parser.add_argument('--emo', default=None, type=str)
    parser.add_argument('--no_crop', action='store_true')
    parser.add_argument('--res_video_path', default=None, type=str)
    parser.add_argument('--res_dir', default=os.path.join(config.TEMP_DIR, 'results'), type=str)

    opt = parser.parse_args([])
    opt.rank = gpu_id
    opt.ngpus = 1
    return opt


class FloatGPUPool:
    """Pool of FLOAT models across multiple GPUs for parallel video generation."""

    def __init__(self, gpu_ids: list[int] = config.FLOAT_GPUS):
        self.gpu_ids = gpu_ids
        self.agents = {}
        self.semaphores = {}
        self._lock = threading.Lock()

    def load_all(self):
        """Load FLOAT model on each GPU. Call at startup."""
        for gpu_id in self.gpu_ids:
            logger.info(f"Loading FLOAT on GPU {gpu_id}...")
            opt = _build_opt(gpu_id)
            agent = InferenceAgent(opt)
            self.agents[gpu_id] = agent
            self.semaphores[gpu_id] = threading.Semaphore(1)
        logger.info(f"FLOAT GPU pool ready: {list(self.agents.keys())}")

    def generate_video(self, audio_path: str, ref_image: str | None = None,
                       output_path: str | None = None, nfe: int | None = None) -> str:
        """Generate video using any available GPU from the pool.

        nfe overrides the number of FLOAT function evaluations for this clip only
        (lower = faster, slightly lower quality); defaults to config.FLOAT_NFE.
        """
        if ref_image is None:
            ref_image = config.AVATAR_IMAGE
        if output_path is None:
            output_path = tempfile.mktemp(suffix=".mp4", dir=config.TEMP_DIR)
        if nfe is None:
            nfe = config.FLOAT_NFE

        # Try to acquire any GPU
        while True:
            for gpu_id in self.gpu_ids:
                if self.semaphores[gpu_id].acquire(blocking=False):
                    try:
                        agent = self.agents[gpu_id]
                        # Pin the current CUDA device for this thread so any
                        # device-less tensor creation inside FLOAT (e.g. the bare
                        # `.cuda()` in styledecoder.py) lands on THIS gpu instead
                        # of the default cuda:0. Without this, segments dispatched
                        # to GPU 1/2 crash with "tensors on cuda:1 and cuda:0".
                        with torch.cuda.device(gpu_id):
                            result = agent.run_inference(
                                res_video_path=output_path,
                                ref_path=ref_image,
                                audio_path=audio_path,
                                no_crop=True,
                                nfe=nfe,
                                verbose=True,
                            )
                        return result
                    finally:
                        self.semaphores[gpu_id].release()
            # All GPUs busy, wait briefly
            import time
            time.sleep(0.1)

    def generate_idle_video(self, duration: float = 5.0, output_path: str | None = None) -> str:
        """Generate a seamless idle loop video.

        Generates a base clip from silence, then creates forward+reverse
        concatenation so the loop point is seamless (end frame == start frame).
        """
        if output_path is None:
            output_path = config.IDLE_VIDEO_PATH

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create silent audio
        silence_path = tempfile.mktemp(suffix=".wav", dir=config.TEMP_DIR)
        sample_rate = 16000
        silence = np.zeros(int(duration * sample_rate), dtype=np.float32)
        import scipy.io.wavfile as wavfile
        wavfile.write(silence_path, sample_rate, silence)

        # Generate base clip
        base_path = tempfile.mktemp(suffix=".mp4", dir=config.TEMP_DIR)
        self.generate_video(silence_path, output_path=base_path)

        # Create seamless loop: forward + reverse (no audio needed for idle)
        import subprocess
        reversed_path = tempfile.mktemp(suffix=".mp4", dir=config.TEMP_DIR)
        # Reverse the video
        subprocess.run([
            "ffmpeg", "-y", "-i", base_path,
            "-vf", "reverse", "-an", reversed_path
        ], capture_output=True)
        # Concatenate: forward + reversed → seamless loop
        concat_list = tempfile.mktemp(suffix=".txt", dir=config.TEMP_DIR)
        with open(concat_list, "w") as f:
            f.write(f"file '{base_path}'\n")
            f.write(f"file '{reversed_path}'\n")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c", "copy", output_path
        ], capture_output=True)

        # Cleanup temp files
        for p in [silence_path, base_path, reversed_path, concat_list]:
            if os.path.exists(p):
                os.remove(p)

        result = output_path

        return result


def get_pool() -> FloatGPUPool:
    global _pool
    # Double-checked locking: build the pool FULLY (load_all) before publishing it
    # to `_pool`. Otherwise concurrent callers during the ~10s cold load would see
    # a half-built pool with empty `agents`/`semaphores` -> KeyError on gpu_id.
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = FloatGPUPool()
                pool.load_all()
                _pool = pool
    return _pool


def generate_video(audio_path: str, ref_image: str | None = None,
                   output_path: str | None = None, nfe: int | None = None) -> str:
    """Public API - uses the GPU pool."""
    return get_pool().generate_video(audio_path, ref_image, output_path, nfe=nfe)


def generate_idle_video(duration: float = 3.0) -> str:
    """Generate idle loop video."""
    return get_pool().generate_idle_video(duration)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    pool = get_pool()
    print("FLOAT GPU pool loaded successfully.")
