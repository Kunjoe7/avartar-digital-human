"""WebSocket server for receiving browser audio stream."""

import asyncio
import numpy as np
import threading
import logging
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket

import config
from modules.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class AudioServer:
    def __init__(self, on_speech_start=None, on_speech_end=None):
        """
        Args:
            on_speech_start: callback() called when user starts speaking (for barge-in)
            on_speech_end: callback(audio_array) called with complete speech audio (float32, 16kHz)
        """
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.vad = VoiceActivityDetector()
        self.enabled = True
        self._speech_started_notified = False
        self._app = None
        self._server_thread = None

    async def _ws_endpoint(self, websocket: WebSocket):
        await websocket.accept()
        logger.info("Audio WebSocket client connected")
        self._speech_started_notified = False

        try:
            while True:
                data = await websocket.receive_bytes()
                if not self.enabled:
                    continue

                # Incoming data is 16kHz PCM int16
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                event, audio_data = self.vad.process_chunk(audio_chunk)

                if event == "speech_start" and not self._speech_started_notified:
                    self._speech_started_notified = True
                    if self.on_speech_start:
                        self.on_speech_start()

                elif event == "speech_end":
                    self._speech_started_notified = False
                    if self.on_speech_end:
                        self.on_speech_end(audio_data)

        except Exception as e:
            logger.info(f"WebSocket disconnected: {e}")

    def _create_app(self):
        self._app = Starlette(
            routes=[WebSocketRoute("/ws", self._ws_endpoint)],
        )
        return self._app

    def start(self, port: int = config.WS_PORT):
        """Start the WebSocket server in a background thread."""
        import uvicorn

        app = self._create_app()

        def _run():
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        logger.info(f"Audio WebSocket server started on ws://0.0.0.0:{port}/ws")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def on_start():
        print("Speech started!")

    def on_end(audio):
        print(f"Speech ended! Audio length: {len(audio) / 16000:.2f}s")

    server = AudioServer(on_speech_start=on_start, on_speech_end=on_end)
    server.start()
    print(f"Server running on ws://0.0.0.0:{config.WS_PORT}/ws")
    import time
    while True:
        time.sleep(1)
