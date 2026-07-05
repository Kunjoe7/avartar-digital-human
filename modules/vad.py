"""Silero VAD wrapper for real-time voice activity detection.

Silero detects *silence*; an optional smart-turn EOU model (see modules/eou.py)
then judges whether a detected pause is a real end-of-turn or just a think-pause,
so the avatar doesn't cut the user off mid-thought.
"""

import logging

import torch
import numpy as np

import config
from modules import eou

logger = logging.getLogger(__name__)


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

        # EOU (semantic turn detection) thresholds, in chunk counts.
        chunk_sec = self.chunk_size / sample_rate  # 0.032s
        self.use_eou = getattr(config, "USE_EOU", False)
        self.eou_threshold = getattr(config, "EOU_THRESHOLD", 0.5)
        self.eou_confirm = max(1, int(getattr(config, "EOU_CONFIRM_CONSULTS", 2)))
        self.eou_pause_chunks = max(1, round(getattr(config, "EOU_PAUSE_DURATION", 0.20) / chunk_sec))
        self.eou_recheck_chunks = max(1, round(getattr(config, "EOU_RECHECK_DURATION", 0.15) / chunk_sec))
        self.eou_max_silence_chunks = max(
            self.eou_pause_chunks + 1, round(getattr(config, "EOU_MAX_SILENCE", 2.0) / chunk_sec)
        )

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
        self.eou_complete_streak = 0

    def _should_end_turn(self) -> bool:
        """Given the current trailing silence, decide if the user's turn ended.

        EOU off / unavailable: pure silence duration (original behavior).
        EOU on: consult smart-turn at the first short pause and periodically after,
        ending when it judges the utterance complete; a hard max-silence cap forces
        the end regardless so we never hang waiting for a "complete" verdict.
        """
        if not self.use_eou:
            return self.silent_chunks >= self.silence_chunks_needed

        # Hard safety cap.
        if self.silent_chunks >= self.eou_max_silence_chunks:
            logger.info("[eou] forced end at %.2fs silence (max cap)",
                        self.silent_chunks * self.chunk_size / self.sample_rate)
            return True

        # Consult at the first pause, then once per recheck interval of continued silence.
        past_pause = self.silent_chunks - self.eou_pause_chunks
        if past_pause < 0 or (past_pause > 0 and past_pause % self.eou_recheck_chunks != 0):
            return False

        p = eou.predict_complete(np.concatenate(self.speech_buffer))
        if p is None:
            # Model unavailable at runtime -> degrade to the silence threshold.
            return self.silent_chunks >= self.silence_chunks_needed
        # Hysteresis: only end after EOU_CONFIRM_CONSULTS consecutive 'complete'
        # verdicts, so one momentary high reading at a clause boundary doesn't cut
        # the user off mid-sentence.
        if p >= self.eou_threshold:
            self.eou_complete_streak += 1
        else:
            self.eou_complete_streak = 0
        end = self.eou_complete_streak >= self.eou_confirm
        logger.info("[eou] P=%.2f silent=%.2fs streak=%d/%d -> %s", p,
                    self.silent_chunks * self.chunk_size / self.sample_rate,
                    self.eou_complete_streak, self.eou_confirm,
                    "END" if end else "hold")
        return end

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
                    if self._should_end_turn():
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
