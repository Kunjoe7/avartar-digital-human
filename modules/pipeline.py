"""Core orchestrator: streaming sentence-by-sentence processing with barge-in support."""

import threading
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future

from modules import asr, llm, tts, avatar

logger = logging.getLogger(__name__)


class Pipeline:
    """State machine: idle → listening → processing → speaking → idle"""

    def __init__(self):
        self.state = "idle"  # idle, listening, processing, speaking
        self.cancel_event = threading.Event()
        self.video_queue = queue.Queue()
        self.chat_history = []  # List of {"role": ..., "content": ...}
        self._lock = threading.Lock()
        self._processing_thread = None
        # Track which sentences were actually played
        self._pending_assistant_text = ""
        self._played_sentences = []

    def on_speech_start(self):
        """Called by audio_server when user starts speaking (barge-in)."""
        if self.state in ("processing", "speaking"):
            logger.info("Barge-in detected! Cancelling current response.")
            self.cancel_event.set()
            # Clear video queue
            while not self.video_queue.empty():
                try:
                    self.video_queue.get_nowait()
                except queue.Empty:
                    break
            # If we were mid-response, only keep what was played
            if self._pending_assistant_text and self._played_sentences:
                played = "".join(self._played_sentences)
                # Update the last assistant message to only what was played
                for i in range(len(self.chat_history) - 1, -1, -1):
                    if self.chat_history[i]["role"] == "assistant":
                        self.chat_history[i]["content"] = played
                        break
            self._pending_assistant_text = ""
            self._played_sentences = []

        self.state = "listening"

    def on_speech_end(self, audio_array):
        """Called by audio_server when user finishes speaking.
        audio_array: float32 numpy array at 16kHz.
        """
        self.state = "processing"
        self.cancel_event.clear()
        self._pending_assistant_text = ""
        self._played_sentences = []

        # Run processing in background thread
        self._processing_thread = threading.Thread(
            target=self._process_speech, args=(audio_array,), daemon=True
        )
        self._processing_thread.start()

    def on_speech_end_text(self, text):
        """Text input: skip ASR and feed text directly into pipeline."""
        self.state = "processing"
        self.cancel_event.clear()
        self._pending_assistant_text = ""
        self._played_sentences = []

        self._processing_thread = threading.Thread(
            target=self._process_text, args=(text,), daemon=True
        )
        self._processing_thread.start()

    def _process_speech(self, audio_array):
        """Full pipeline: ASR → LLM stream → TTS+FLOAT (pipelined) → video queue."""
        try:
            # Step 1: ASR
            if self.cancel_event.is_set():
                self.state = "idle"
                return

            user_text = asr.transcribe_array(audio_array, sample_rate=16000)
            logger.info(f"ASR result: {user_text}")

            if not user_text.strip():
                self.state = "idle"
                return

            self.chat_history.append({"role": "user", "content": user_text})

            # Step 2: LLM streaming → pipelined TTS+FLOAT
            if self.cancel_event.is_set():
                self.state = "idle"
                return

            self.state = "processing"
            self._run_pipelined_synthesis(user_text)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.state = "idle"

    def _process_text(self, user_text):
        """Text-only pipeline: skip ASR, go straight to LLM → TTS+FLOAT (pipelined)."""
        try:
            self.chat_history.append({"role": "user", "content": user_text})

            if self.cancel_event.is_set():
                self.state = "idle"
                return

            self.state = "processing"
            self._run_pipelined_synthesis(user_text)

        except Exception as e:
            logger.error(f"Pipeline error (text): {e}", exc_info=True)
            self.state = "idle"

    def _run_pipelined_synthesis(self, user_text):
        """Pipelined LLM → TTS → FLOAT: overlap TTS(N+1) with FLOAT(N)."""
        full_response = ""
        pending_tts_future = None  # Future for TTS of next sentence
        pending_tts_sentence = None
        assistant_entry_added = False

        with ThreadPoolExecutor(max_workers=1) as tts_executor:
            for sentence in llm.chat_stream(user_text, cancel_event=self.cancel_event):
                if self.cancel_event.is_set():
                    break

                full_response += sentence
                logger.info(f"LLM sentence: {sentence}")

                # Progressively update chat history so frontend sees partial response
                if not assistant_entry_added:
                    self.chat_history.append({"role": "assistant", "content": full_response})
                    assistant_entry_added = True
                else:
                    # Update the last assistant entry in-place
                    for i in range(len(self.chat_history) - 1, -1, -1):
                        if self.chat_history[i]["role"] == "assistant":
                            self.chat_history[i]["content"] = full_response
                            break

                if self.cancel_event.is_set():
                    break

                # If there's a pending TTS result from prev iteration, render its FLOAT now
                # while we kick off TTS for current sentence in parallel
                if pending_tts_future is not None:
                    # Start TTS for current sentence in background
                    current_tts_future = tts_executor.submit(
                        tts.synthesize, sentence, None, self.cancel_event
                    )
                    current_tts_sentence = sentence

                    # Wait for previous TTS and render FLOAT
                    prev_tts_path = pending_tts_future.result()
                    if prev_tts_path is None or self.cancel_event.is_set():
                        pending_tts_future = None
                        break

                    video_path = avatar.generate_video(prev_tts_path)
                    if video_path is None or self.cancel_event.is_set():
                        pending_tts_future = None
                        break

                    self.video_queue.put({
                        "video": video_path,
                        "sentence": pending_tts_sentence
                    })
                    self.state = "speaking"

                    # Current sentence's TTS is now our pending
                    pending_tts_future = current_tts_future
                    pending_tts_sentence = current_tts_sentence
                else:
                    # First sentence: just start TTS, no FLOAT to overlap with
                    pending_tts_future = tts_executor.submit(
                        tts.synthesize, sentence, None, self.cancel_event
                    )
                    pending_tts_sentence = sentence

            # Process the last pending TTS result
            if pending_tts_future is not None and not self.cancel_event.is_set():
                tts_path = pending_tts_future.result()
                if tts_path is not None and not self.cancel_event.is_set():
                    video_path = avatar.generate_video(tts_path)
                    if video_path is not None and not self.cancel_event.is_set():
                        self.video_queue.put({
                            "video": video_path,
                            "sentence": pending_tts_sentence
                        })
                        self.state = "speaking"

        # Finalize chat history (already added progressively, just ensure it's complete)
        if full_response and not self.cancel_event.is_set():
            self._pending_assistant_text = full_response
            # Update final content in case last sentence wasn't captured
            if assistant_entry_added:
                for i in range(len(self.chat_history) - 1, -1, -1):
                    if self.chat_history[i]["role"] == "assistant":
                        self.chat_history[i]["content"] = full_response
                        break
            else:
                self.chat_history.append({"role": "assistant", "content": full_response})
            self.video_queue.put(None)
        elif full_response and not assistant_entry_added:
            self.chat_history.append({"role": "assistant", "content": full_response})

        if not self.cancel_event.is_set():
            self.state = "speaking"

    def get_next_video(self):
        """Non-blocking: get next video from queue.

        Returns:
            dict with "video" and "sentence" keys, or None if queue empty,
            or False if response is complete.
        """
        try:
            item = self.video_queue.get_nowait()
            if item is None:
                # End of response
                self.state = "idle"
                return False
            self._played_sentences.append(item["sentence"])
            return item
        except queue.Empty:
            return None

    def mark_playback_done(self):
        """Called when frontend finishes playing all videos."""
        if self.video_queue.empty() and self.state == "speaking":
            self.state = "idle"

    def get_chat_history(self):
        """Return current chat history for display."""
        return list(self.chat_history)

    def reset(self):
        """Reset everything."""
        self.cancel_event.set()
        time.sleep(0.1)
        self.cancel_event.clear()
        self.chat_history.clear()
        self._pending_assistant_text = ""
        self._played_sentences = []
        while not self.video_queue.empty():
            try:
                self.video_queue.get_nowait()
            except queue.Empty:
                break
        self.state = "idle"
        llm.reset_conversation()
