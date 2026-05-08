"""Silero VAD wrapper for real-time voice activity detection."""

import torch
import numpy as np
import config


class VoiceActivityDetector:
    def __init__(
        self,
        threshold: float = config.VAD_THRESHOLD,
        silence_duration: float = config.VAD_SILENCE_DURATION,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.sample_rate = sample_rate
        # Number of silent chunks before declaring speech_end
        # Chunks are 512 samples at 16kHz = 32ms each
        self.chunk_size = 512
        self.silence_chunks_needed = int(silence_duration * sample_rate / self.chunk_size)

        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        self._reset()

    def _reset(self):
        self.model.reset_states()
        self.is_speaking = False
        self.silent_chunks = 0
        self.speech_buffer = []

    def process_chunk(self, audio_chunk: np.ndarray):
        """Process an audio chunk (int16 or float32, 16kHz).

        Returns:
            tuple: (event, audio_data)
            event is one of: "speech_start", "speech_end", None
            audio_data is the complete speech audio (float32 np array) on speech_end, else None
        """
        # Convert to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_f32 = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio_chunk.astype(np.float32)

        # Process in 512-sample sub-chunks for Silero VAD
        for i in range(0, len(audio_f32), self.chunk_size):
            sub = audio_f32[i : i + self.chunk_size]
            if len(sub) < self.chunk_size:
                sub = np.pad(sub, (0, self.chunk_size - len(sub)))

            tensor = torch.from_numpy(sub)
            prob = self.model(tensor, self.sample_rate).item()

            if prob >= self.threshold:
                self.silent_chunks = 0
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = []
                    # Don't return yet - accumulate first
                self.speech_buffer.append(sub)
            else:
                if self.is_speaking:
                    self.speech_buffer.append(sub)
                    self.silent_chunks += 1
                    if self.silent_chunks >= self.silence_chunks_needed:
                        # Speech ended
                        full_audio = np.concatenate(self.speech_buffer)
                        self._reset()
                        return ("speech_end", full_audio)

        if self.is_speaking and self.silent_chunks == 0 and len(self.speech_buffer) > 0:
            # Just started speaking or continuing
            if len(self.speech_buffer) <= len(audio_f32) // self.chunk_size + 1:
                return ("speech_start", None)

        return (None, None)

    def force_end(self):
        """Force end current speech and return buffer."""
        if self.is_speaking and self.speech_buffer:
            full_audio = np.concatenate(self.speech_buffer)
            self._reset()
            return full_audio
        self._reset()
        return None


if __name__ == "__main__":
    vad = VoiceActivityDetector()
    print("VAD loaded successfully.")
