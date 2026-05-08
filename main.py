import os
import sys
import json
import asyncio
import logging
import numpy as np

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.makedirs(config.TEMP_DIR, exist_ok=True)

from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute, Mount
from starlette.responses import FileResponse, JSONResponse
from starlette.requests import Request
from starlette.websockets import WebSocket
from starlette.staticfiles import StaticFiles

from modules.pipeline import Pipeline
from modules.vad import VoiceActivityDetector

# --------------- globals ---------------
pipeline = Pipeline()
vad = VoiceActivityDetector()
mic_enabled = False
_speech_started_notified = False

# Connected state WebSocket clients
state_clients: set[WebSocket] = set()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------- HTTP routes ---------------

async def index(request: Request):
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


async def serve_video(request: Request):
    """Serve video files from tmp/ and assets/."""
    subdir = request.path_params["subdir"]
    filename = request.path_params["filename"]

    # Only allow tmp and assets subdirectories
    if subdir not in ("tmp", "assets"):
        return JSONResponse({"error": "not found"}, status_code=404)

    filepath = os.path.join(BASE_DIR, subdir, filename)
    if not os.path.isfile(filepath):
        return JSONResponse({"error": "not found"}, status_code=404)

    return FileResponse(filepath, media_type="video/mp4")


async def api_toggle(request: Request):
    global mic_enabled
    mic_enabled = not mic_enabled
    status = "on" if mic_enabled else "off"
    logger.info(f"Mic toggled: {status}")
    return JSONResponse({"mic": status})


async def api_reset(request: Request):
    global mic_enabled
    pipeline.reset()
    mic_enabled = False
    await broadcast_state({"type": "reset"})
    return JSONResponse({"status": "ok"})


async def api_text(request: Request):
    """Handle text input (fallback/testing)."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)

    # Run pipeline in background thread (blocking operations)
    import threading
    def _process():
        pipeline.on_speech_end_text(text)
    threading.Thread(target=_process, daemon=True).start()
    return JSONResponse({"status": "processing"})


async def api_test_asr(request: Request):
    """Diagnostic: receive raw int16 PCM audio, run ASR, return result."""
    body = await request.body()
    audio_i16 = np.frombuffer(body, dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    duration = len(audio_f32) / 16000
    rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
    max_val = float(np.max(np.abs(audio_f32)))

    logger.info(f"[TestASR] Received {len(audio_i16)} samples, "
                f"duration={duration:.2f}s, rms={rms:.4f}, max={max_val:.4f}")

    # Save for inspection
    import scipy.io.wavfile as wavfile
    debug_path = os.path.join(BASE_DIR, "tmp", "test_asr_input.wav")
    wavfile.write(debug_path, 16000, audio_i16)
    logger.info(f"[TestASR] Saved to {debug_path}")

    # Run ASR
    from modules import asr
    text = asr.transcribe_array(audio_f32, sample_rate=16000)
    logger.info(f"[TestASR] Result: {text}")

    return JSONResponse({
        "text": text,
        "duration": f"{duration:.2f}",
        "rms": f"{rms:.4f}",
        "max": f"{max_val:.4f}",
    })


# --------------- Audio WebSocket ---------------

async def ws_audio(websocket: WebSocket):
    global _speech_started_notified
    await websocket.accept()
    logger.info("Audio WebSocket client connected")
    _speech_started_notified = False
    _chunk_count = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            if not mic_enabled:
                continue

            audio_chunk = np.frombuffer(data, dtype=np.int16)
            _chunk_count += 1
            if _chunk_count <= 3:
                logger.info(f"[AudioDebug] chunk #{_chunk_count}: len={len(audio_chunk)}, "
                           f"max={np.max(np.abs(audio_chunk))}, "
                           f"rms={np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)):.1f}")

            event, audio_data = vad.process_chunk(audio_chunk)

            if event == "speech_start" and not _speech_started_notified:
                _speech_started_notified = True
                pipeline.on_speech_start()

            elif event == "speech_end":
                _speech_started_notified = False
                # Debug: log audio stats and save sample
                duration = len(audio_data) / 16000
                rms = np.sqrt(np.mean(audio_data**2))
                logger.info(f"[AudioDebug] speech_end: duration={duration:.2f}s, "
                           f"rms={rms:.4f}, max={np.max(np.abs(audio_data)):.4f}")
                # Save debug wav
                import scipy.io.wavfile as wavfile
                debug_path = os.path.join(BASE_DIR, "tmp", "debug_speech.wav")
                wavfile.write(debug_path, 16000, (audio_data * 32767).astype(np.int16))
                logger.info(f"[AudioDebug] Saved debug audio to {debug_path}")
                pipeline.on_speech_end(audio_data)

    except Exception as e:
        logger.info(f"Audio WebSocket disconnected: {e}")


# --------------- State WebSocket ---------------

async def ws_state(websocket: WebSocket):
    await websocket.accept()
    state_clients.add(websocket)
    logger.info(f"State WebSocket client connected (total: {len(state_clients)})")
    try:
        # Keep connection alive; client doesn't send data
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        state_clients.discard(websocket)
        logger.info(f"State WebSocket client disconnected (total: {len(state_clients)})")


async def broadcast_state(msg: dict):
    """Send JSON message to all connected state clients."""
    if not state_clients:
        return
    text = json.dumps(msg)
    disconnected = set()
    for ws in state_clients:
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.add(ws)
    state_clients.difference_update(disconnected)


# --------------- Background poller ---------------

async def state_poller():
    """Background task: poll pipeline for new videos and state changes, broadcast to clients."""
    last_state = None
    while True:
        await asyncio.sleep(0.2)

        if not state_clients:
            continue

        # Check for new video
        item = pipeline.get_next_video()

        if item is False:
            # Response complete
            await broadcast_state({
                "type": "video_end",
                "chat_history": pipeline.get_chat_history(),
                "state": pipeline.state,
            })
        elif item is not None:
            # New video segment
            video_path = item["video"]
            # Convert absolute path to URL
            if video_path.startswith(os.path.join(BASE_DIR, "tmp")):
                video_url = "/video/tmp/" + os.path.basename(video_path)
            elif video_path.startswith(os.path.join(BASE_DIR, "assets")):
                video_url = "/video/assets/" + os.path.basename(video_path)
            else:
                video_url = "/video/tmp/" + os.path.basename(video_path)

            await broadcast_state({
                "type": "video",
                "video_url": video_url,
                "subtitle": item["sentence"],
                "chat_history": pipeline.get_chat_history(),
                "state": "speaking",
            })

        # Broadcast state changes
        current_state = pipeline.state
        if current_state != last_state:
            last_state = current_state
            await broadcast_state({
                "type": "state",
                "state": current_state,
                "chat_history": pipeline.get_chat_history(),
            })


# --------------- App setup ---------------

async def on_startup():
    asyncio.create_task(state_poller())


# Ensure static directory exists
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)

app = Starlette(
    routes=[
        Route("/", index),
        Route("/video/{subdir}/{filename}", serve_video),
        Route("/api/toggle", api_toggle, methods=["POST"]),
        Route("/api/reset", api_reset, methods=["POST"]),
        Route("/api/text", api_text, methods=["POST"]),
        Route("/api/test_asr", api_test_asr, methods=["POST"]),
        WebSocketRoute("/ws/audio", ws_audio),
        WebSocketRoute("/ws/state", ws_state),
        Mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static"),
    ],
    on_startup=[on_startup],
)


if __name__ == "__main__":
    import uvicorn

    # Generate idle video if it doesn't exist
    if not os.path.exists(config.IDLE_VIDEO_PATH):
        logger.info("Generating idle loop video (first-time setup)...")
        from modules import avatar
        try:
            avatar.generate_idle_video(duration=config.IDLE_VIDEO_DURATION)
            logger.info(f"Idle video saved to {config.IDLE_VIDEO_PATH}")
        except Exception as e:
            logger.warning(f"Could not generate idle video: {e}")

    uvicorn.run(app, host="0.0.0.0", port=7861, log_level="info")
