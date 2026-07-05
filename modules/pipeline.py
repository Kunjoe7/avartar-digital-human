"""Core orchestrator: streaming sentence-by-sentence processing with barge-in support."""

import os
import threading
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future

import config
from modules import asr, llm, tts, avatar

logger = logging.getLogger(__name__)


class Pipeline:
    """State machine: idle → listening → processing → speaking → idle"""

    def __init__(self):
        self.state = "idle"  # idle, listening, processing, speaking
        self.cancel_event = threading.Event()
        self.video_queue = queue.Queue()
        self.chat_history = []  # List of {"role": ..., "content": ...} (frontend display)
        self.llm_history = []   # Conversation sent to the LLM API (this session only)
        self._lock = threading.Lock()
        self._processing_thread = None
        # Track which sentences were actually played
        self._pending_assistant_text = ""
        self._played_sentences = []
        # Monotonic turn id: each new utterance bumps it. A response only touches
        # shared state (enqueue video) while it still owns the current turn, so a
        # barged-in response can't leak stale segments into the next turn even
        # after cancel_event is cleared for the new turn.
        self._turn = 0
        # perf_counter() at the moment the user stopped speaking — the T0 for the
        # latency waterfall logged through the rest of the turn.
        self._t0 = 0.0

    def _aborted(self, turn):
        """True if this response was cancelled or superseded by a newer turn."""
        return self.cancel_event.is_set() or turn != self._turn

    def on_speech_start(self):
        """Called by audio_server when user starts speaking (barge-in)."""
        if self.state in ("processing", "speaking"):
            logger.info("Barge-in detected! Cancelling current response.")
            self.cancel_event.set()
            self._turn += 1  # invalidate the in-flight response immediately
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
        self._t0 = time.perf_counter()
        self.state = "processing"
        self.cancel_event.clear()
        self._pending_assistant_text = ""
        self._played_sentences = []
        self._turn += 1
        turn = self._turn

        # Run processing in background thread
        self._processing_thread = threading.Thread(
            target=self._process_speech, args=(audio_array, turn), daemon=True
        )
        self._processing_thread.start()

    def on_speech_end_text(self, text):
        """Text input: skip ASR and feed text directly into pipeline."""
        self._t0 = time.perf_counter()
        self.state = "processing"
        self.cancel_event.clear()
        self._pending_assistant_text = ""
        self._played_sentences = []
        self._turn += 1
        turn = self._turn

        self._processing_thread = threading.Thread(
            target=self._process_text, args=(text, turn), daemon=True
        )
        self._processing_thread.start()

    def _process_speech(self, audio_array, turn):
        """Full pipeline: ASR → LLM stream → TTS+FLOAT (pipelined) → video queue."""
        try:
            # Step 1: ASR
            if self._aborted(turn):
                self.state = "idle"
                return

            user_text = asr.transcribe_array(audio_array, sample_rate=16000)
            logger.info(f"[latency] ASR done at +{time.perf_counter() - self._t0:.2f}s "
                        f"-> {user_text!r}")

            if not user_text.strip():
                self.state = "idle"
                return

            if self._aborted(turn):
                self.state = "idle"
                return

            self.chat_history.append({"role": "user", "content": user_text})

            # Step 2: LLM streaming → pipelined TTS+FLOAT
            self.state = "processing"
            self._run_synthesis(user_text, turn)

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.state = "idle"

    def _process_text(self, user_text, turn):
        """Text-only pipeline: skip ASR, go straight to LLM → TTS+FLOAT (pipelined)."""
        try:
            if self._aborted(turn):
                self.state = "idle"
                return

            self.chat_history.append({"role": "user", "content": user_text})
            self.state = "processing"
            self._run_synthesis(user_text, turn)

        except Exception as e:
            logger.error(f"Pipeline error (text): {e}", exc_info=True)
            self.state = "idle"

    def _run_synthesis(self, user_text, turn):
        """Dispatch to the configured synthesis mode."""
        mode = getattr(config, "SYNTHESIS_MODE", "stream_parallel")
        if mode == "batch":
            self._run_batch_synthesis(user_text, turn)
        elif mode == "stream":
            self._run_pipelined_synthesis(user_text, turn)
        elif mode == "hybrid":
            self._run_hybrid_synthesis(user_text, turn)
        else:
            self._run_streaming_parallel_synthesis(user_text, turn)

    def _run_streaming_parallel_synthesis(self, user_text, turn):
        """Fastest mode: stream sentences from the LLM (chat text appears live in
        <1s) AND render each sentence's TTS+FLOAT concurrently across the whole
        GPU pool, while enqueuing strictly in sentence order.

        A producer thread pulls sentences off the LLM stream and submits each as a
        TTS->FLOAT job to a pool sized to len(FLOAT_GPUS); this consumer reads the
        resulting futures IN ORDER and enqueues the finished clips. So sentence 1
        starts playing after just 1 TTS + 1 FLOAT, while sentences 2..N are already
        rendering on the other GPUs -> no stall between segments.
        """
        futures_q = queue.Queue()
        SENTINEL = object()
        n_gpus = max(1, len(config.FLOAT_GPUS))

        def _render(sentence, idx):
            if self._aborted(turn):
                return None
            # First sentence renders at a lower NFE to get the avatar talking sooner.
            nfe = config.FLOAT_NFE_FIRST if idx == 0 else config.FLOAT_NFE
            t_tts0 = time.perf_counter()
            tts_path = tts.synthesize(sentence, None, self.cancel_event)
            if tts_path is None or self._aborted(turn):
                return None
            t_float0 = time.perf_counter()
            video_path = avatar.generate_video(tts_path, nfe=nfe)
            # The wav is consumed by FLOAT; drop it now so tmp/ doesn't fill up.
            try:
                os.remove(tts_path)
            except OSError:
                pass
            logger.info("[latency] seg %d rendered: tts=%.2fs float=%.2fs (nfe=%d)",
                        idx, t_float0 - t_tts0, time.perf_counter() - t_float0, nfe)
            return video_path

        def _producer(executor):
            """Pull sentences off the LLM stream, submit renders, update chat live."""
            assistant_added = False
            full_response = ""
            try:
                for idx, sentence in enumerate(
                    llm.chat_stream(user_text, self.llm_history, cancel_event=self.cancel_event)
                ):
                    if self._aborted(turn):
                        break
                    if idx == 0:
                        logger.info("[latency] LLM first sentence at +%.2fs",
                                    time.perf_counter() - self._t0)
                    full_response += sentence
                    logger.info(f"LLM sentence: {sentence}")

                    # Live chat update so the frontend shows text as it streams in.
                    if not assistant_added:
                        self.chat_history.append({"role": "assistant", "content": full_response})
                        assistant_added = True
                    else:
                        for i in range(len(self.chat_history) - 1, -1, -1):
                            if self.chat_history[i]["role"] == "assistant":
                                self.chat_history[i]["content"] = full_response
                                break

                    futures_q.put((executor.submit(_render, sentence, idx), sentence))
                self._pending_assistant_text = full_response
            except Exception:
                logger.exception("streaming producer failed")
            finally:
                futures_q.put(SENTINEL)

        first_seg = True
        with ThreadPoolExecutor(max_workers=n_gpus) as executor:
            producer = threading.Thread(
                target=_producer, args=(executor,), name="llm-producer", daemon=True
            )
            producer.start()

            # Consume render futures strictly in order (sentences i+1.. render in
            # parallel while we wait on sentence i).
            while True:
                item = futures_q.get()
                if item is SENTINEL:
                    break
                fut, sentence = item
                if self._aborted(turn):
                    fut.cancel()
                    continue  # keep draining to SENTINEL so the producer finishes
                video_path = fut.result()
                if video_path is None or self._aborted(turn):
                    continue
                self.video_queue.put({
                    "video": video_path,
                    "sentence": sentence,
                    "_t_enqueue": time.perf_counter(),
                })
                self.state = "speaking"
                if first_seg:
                    first_seg = False
                    logger.info("[latency] FIRST segment enqueued at +%.2fs",
                                time.perf_counter() - self._t0)

            producer.join(timeout=1.0)

        if not self._aborted(turn):
            self.video_queue.put(None)
            self.state = "speaking"

    @staticmethod
    def _split_sentences(text):
        """Split a full response into chunks (same boundaries as the LLM streamer):
        hard sentence-final punctuation always splits; commas/semicolons split too
        once the pending chunk is at least MIN_CHUNK_CHARS long."""
        hard = {"。", "！", "？", ".", "!", "?", "\n"}
        soft = ({",", "，", "、", "；", ";"}
                if getattr(config, "SPLIT_ON_COMMA", False) else set())
        min_chars = getattr(config, "MIN_CHUNK_CHARS", 0)
        sentences, buf = [], ""
        for ch in text:
            buf += ch
            if ch in hard or (ch in soft and len(buf.strip()) >= min_chars):
                s = buf.strip()
                if s:
                    sentences.append(s)
                buf = ""
        if buf.strip():
            sentences.append(buf.strip())
        return sentences

    def _run_hybrid_synthesis(self, user_text, turn):
        """Hybrid: get the FULL LLM answer first so the chat shows the complete,
        stable text at once (no live updates), then render per-sentence TTS+FLOAT
        (pipelined) so the avatar starts speaking quickly. The displayed answer
        text never changes while it plays.
        """
        if self._aborted(turn):
            self.state = "idle"
            return

        # 1. Full answer in one shot (no live, sentence-by-sentence chat updates).
        full_response = (llm.chat(user_text, self.llm_history) or "").strip()
        logger.info(f"LLM full response: {full_response}")
        if not full_response or self._aborted(turn):
            self.state = "idle"
            return

        # 2. Show the complete answer at once as one stable chat entry.
        self.chat_history.append({"role": "assistant", "content": full_response})
        self._pending_assistant_text = full_response

        # 3. Render sentences CONCURRENTLY across the FLOAT GPU pool, but enqueue
        #    them strictly in order so the avatar still speaks them in sequence.
        #    avatar.generate_video() is thread-safe and hands each call a free GPU
        #    (0/1/2), so up to len(FLOAT_GPUS) sentences render in parallel,
        #    cutting total render time from ~N to ~ceil(N/len(GPUs)) clips.
        #    Note: first-segment latency is unchanged (still one TTS + one render);
        #    parallelism speeds up sentences 2..N so playback doesn't stall.
        sentences = self._split_sentences(full_response)

        def _render(sentence):
            if self._aborted(turn):
                return None
            tts_path = tts.synthesize(sentence, None, self.cancel_event)
            if tts_path is None or self._aborted(turn):
                return None
            return avatar.generate_video(tts_path)

        with ThreadPoolExecutor(max_workers=max(1, len(config.FLOAT_GPUS))) as ex:
            futures = [ex.submit(_render, s) for s in sentences]
            for i, fut in enumerate(futures):
                if self._aborted(turn):
                    for g in futures[i:]:
                        g.cancel()
                    break
                # Wait for sentence i in order (sentences i+1.. render in parallel).
                video_path = fut.result()
                if video_path is None or self._aborted(turn):
                    for g in futures[i:]:
                        g.cancel()
                    break
                self.video_queue.put({"video": video_path, "sentence": sentences[i]})
                self.state = "speaking"

        if not self._aborted(turn):
            self.video_queue.put(None)
            self.state = "speaking"

    def _run_batch_synthesis(self, user_text, turn):
        """Batch mode: get the FULL LLM answer first, show it at once, then do a
        SINGLE TTS + FLOAT render. The chat no longer updates live and the avatar
        plays one continuous clip instead of changing per sentence.
        """
        if self._aborted(turn):
            self.state = "idle"
            return

        # 1. Full answer in one shot (no live, sentence-by-sentence updates).
        full_response = (llm.chat(user_text, self.llm_history) or "").strip()
        logger.info(f"LLM full response: {full_response}")
        if not full_response or self._aborted(turn):
            self.state = "idle"
            return

        # 2. Show the complete answer immediately as one stable chat entry.
        self.chat_history.append({"role": "assistant", "content": full_response})
        self._pending_assistant_text = full_response

        # 3. One TTS for the whole answer.
        tts_path = tts.synthesize(full_response, None, self.cancel_event)
        if tts_path is None or self._aborted(turn):
            self.state = "idle"
            return

        # 4. One FLOAT video for the whole answer.
        video_path = avatar.generate_video(tts_path)
        if video_path is None or self._aborted(turn):
            self.state = "idle"
            return

        # 5. Play it as a single segment, then signal completion.
        self.video_queue.put({"video": video_path, "sentence": full_response})
        self.state = "speaking"
        self.video_queue.put(None)

    def _run_pipelined_synthesis(self, user_text, turn):
        """Pipelined LLM → TTS → FLOAT: overlap TTS(N+1) with FLOAT(N)."""
        full_response = ""
        pending_tts_future = None  # Future for TTS of next sentence
        pending_tts_sentence = None
        assistant_entry_added = False

        with ThreadPoolExecutor(max_workers=1) as tts_executor:
            for sentence in llm.chat_stream(user_text, self.llm_history, cancel_event=self.cancel_event):
                if self._aborted(turn):
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

                if self._aborted(turn):
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
                    if prev_tts_path is None or self._aborted(turn):
                        pending_tts_future = None
                        break

                    video_path = avatar.generate_video(prev_tts_path)
                    if video_path is None or self._aborted(turn):
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
            if pending_tts_future is not None and not self._aborted(turn):
                tts_path = pending_tts_future.result()
                if tts_path is not None and not self._aborted(turn):
                    video_path = avatar.generate_video(tts_path)
                    if video_path is not None and not self._aborted(turn):
                        self.video_queue.put({
                            "video": video_path,
                            "sentence": pending_tts_sentence
                        })
                        self.state = "speaking"

        # Finalize chat history (already added progressively, just ensure it's complete)
        if full_response and not self._aborted(turn):
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

        if not self._aborted(turn):
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
        self._turn += 1  # invalidate any in-flight response
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
        self.llm_history.clear()
