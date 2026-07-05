import os
import sys
import time
import json
import asyncio
import logging
import threading
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------- Per-session state ---------------
# Each browser (identified by a client-generated `sid`) gets a fully isolated
# Session: its own Pipeline (chat + LLM history + video queue + state machine),
# its own VAD stream state, its own mic toggle, and its own set of state-WS
# clients. Nothing is shared between sessions, so two users never cross-talk.

class Session:
    def __init__(self):
        self.pipeline = Pipeline()
        self.vad = VoiceActivityDetector()
        self.mic_enabled = False
        self.speech_started_notified = False
        self.state_clients: set[WebSocket] = set()
        # state_poller bookkeeping (per session)
        self.last_state = None
        self.last_chat_sig = None
        self.empty_since = None  # wall-clock when state_clients last dropped to 0


sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(sid: str) -> Session:
    """Get the session for `sid`, creating it (incl. a fresh VAD) on first use.
    Heavy (loads Silero VAD) — call via run_in_executor from async handlers."""
    with _sessions_lock:
        s = sessions.get(sid)
        if s is None:
            s = Session()
            sessions[sid] = s
            logger.info("New session %s (total sessions: %d)", sid, len(sessions))
        return s


async def session_for(scope) -> Session:
    """Resolve the Session for a WebSocket/Request from its `?sid=` query param
    (falls back to a shared 'default' session for old clients without one)."""
    sid = scope.query_params.get("sid") or "default"
    return await asyncio.get_running_loop().run_in_executor(None, get_or_create_session, sid)


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
    session = await session_for(request)
    session.mic_enabled = not session.mic_enabled
    status = "on" if session.mic_enabled else "off"
    logger.info(f"Mic toggled: {status}")
    return JSONResponse({"mic": status})


async def api_reset(request: Request):
    session = await session_for(request)
    session.pipeline.reset()
    session.mic_enabled = False
    await broadcast(session.state_clients, {"type": "reset"})
    return JSONResponse({"status": "ok"})


async def api_text(request: Request):
    """Handle text input (fallback/testing)."""
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "empty"}, status_code=400)

    session = await session_for(request)
    # Run pipeline in background thread (blocking operations)
    threading.Thread(target=session.pipeline.on_speech_end_text, args=(text,), daemon=True).start()
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
    await websocket.accept()
    session = await session_for(websocket)
    logger.info("Audio WebSocket client connected")
    session.speech_started_notified = False
    _chunk_count = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            if not session.mic_enabled:
                continue

            audio_chunk = np.frombuffer(data, dtype=np.int16)
            _chunk_count += 1
            if _chunk_count <= 3:
                logger.info(f"[AudioDebug] chunk #{_chunk_count}: len={len(audio_chunk)}, "
                           f"max={np.max(np.abs(audio_chunk))}, "
                           f"rms={np.sqrt(np.mean(audio_chunk.astype(np.float32)**2)):.1f}")

            # Run Silero VAD in a worker thread so its inference doesn't block the
            # asyncio event loop (which also pushes video to the frontend — VAD on
            # the loop made the two contend and made video delivery stutter).
            event, audio_data = await asyncio.get_running_loop().run_in_executor(
                None, session.vad.process_chunk, audio_chunk
            )

            if event == "speech_start" and not session.speech_started_notified:
                session.speech_started_notified = True
                session.pipeline.on_speech_start()

            elif event == "speech_end":
                session.speech_started_notified = False
                # Debug: log audio stats
                duration = len(audio_data) / 16000
                rms = np.sqrt(np.mean(audio_data**2))
                logger.info(f"[AudioDebug] speech_end: duration={duration:.2f}s, "
                           f"rms={rms:.4f}, max={np.max(np.abs(audio_data)):.4f}")
                # Optional debug wav — OFF by default. The synchronous disk write
                # stalled the audio loop on every utterance. Enable: DEBUG_SAVE_AUDIO=1
                if config.DEBUG_SAVE_AUDIO:
                    import scipy.io.wavfile as wavfile
                    debug_path = os.path.join(BASE_DIR, "tmp", "debug_speech.wav")
                    wavfile.write(debug_path, 16000, (audio_data * 32767).astype(np.int16))
                    logger.info(f"[AudioDebug] Saved debug audio to {debug_path}")
                session.pipeline.on_speech_end(audio_data)

    except Exception as e:
        logger.info(f"Audio WebSocket disconnected: {e}")


# --------------- State WebSocket ---------------

async def ws_state(websocket: WebSocket):
    await websocket.accept()
    session = await session_for(websocket)
    session.state_clients.add(websocket)
    session.empty_since = None
    logger.info(f"State WebSocket client connected (session clients: {len(session.state_clients)})")
    try:
        # Keep connection alive; client doesn't send data
        while True:
            await websocket.receive_text()
    except Exception:
        pass
    finally:
        session.state_clients.discard(websocket)
        if not session.state_clients:
            session.empty_since = time.time()
        logger.info(f"State WebSocket client disconnected (session clients: {len(session.state_clients)})")


async def broadcast(clients: set, msg: dict):
    """Send a JSON message to the given set of state clients."""
    if not clients:
        return
    text = json.dumps(msg)
    disconnected = set()
    # Iterate over a snapshot: `await send_text` yields control, during which a
    # client may connect/disconnect and mutate the set. Iterating the live set
    # then raises "Set changed size during iteration", which would kill the
    # state_poller task and freeze all video delivery mid-response.
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.add(ws)
    clients.difference_update(disconnected)


# --------------- Background poller ---------------

async def _poll_session(session: Session):
    """Push one session's ready videos + state/chat changes to its own clients."""
    pipeline = session.pipeline
    clients = session.state_clients

    # Drain ALL segments ready this tick (not just one) so finished clips reach
    # the browser promptly instead of one-per-poll-interval.
    while True:
        item = pipeline.get_next_video()
        if item is None:
            break  # queue empty for now
        if item is False:
            await broadcast(clients, {
                "type": "video_end",
                "chat_history": pipeline.get_chat_history(),
                "state": pipeline.state,
            })
            break
        # New video segment
        video_path = item["video"]
        if video_path.startswith(os.path.join(BASE_DIR, "tmp")):
            video_url = "/video/tmp/" + os.path.basename(video_path)
        elif video_path.startswith(os.path.join(BASE_DIR, "assets")):
            video_url = "/video/assets/" + os.path.basename(video_path)
        else:
            video_url = "/video/tmp/" + os.path.basename(video_path)

        # Latency: how long the finished clip sat in the queue before delivery.
        t_enq = item.get("_t_enqueue")
        if t_enq is not None:
            logger.info("[latency] deliver %s after %.2fs in queue",
                        os.path.basename(video_path), time.perf_counter() - t_enq)

        await broadcast(clients, {
            "type": "video",
            "video_url": video_url,
            "subtitle": item["sentence"],
            "chat_history": pipeline.get_chat_history(),
            "state": "speaking",
        })

    # Broadcast state changes
    current_state = pipeline.state
    if current_state != session.last_state:
        session.last_state = current_state
        chat = pipeline.get_chat_history()
        session.last_chat_sig = (len(chat), sum(len(m.get("content", "")) for m in chat))
        await broadcast(clients, {
            "type": "state",
            "state": current_state,
            "chat_history": chat,
        })

    # Push streamed text the moment it changes — independent of video. In
    # stream_parallel mode the assistant's sentences land in chat_history as the
    # LLM streams them, well before the first clip finishes rendering, so this
    # makes the reply TEXT appear in ~1s instead of waiting for video.
    chat = pipeline.get_chat_history()
    sig = (len(chat), sum(len(m.get("content", "")) for m in chat))
    if sig != session.last_chat_sig:
        session.last_chat_sig = sig
        await broadcast(clients, {"type": "chat", "chat_history": chat})


async def state_poller():
    """Background task: poll every session's pipeline and push to its own clients."""
    while True:
        await asyncio.sleep(config.STATE_POLL_INTERVAL)
        # Never let a transient error kill this task: it is the ONLY producer of
        # video/state pushes to every frontend, so if it dies all avatars freeze
        # mid-response and stay broken until restart.
        try:
            for session in list(sessions.values()):
                if not session.state_clients:
                    continue
                await _poll_session(session)
        except Exception:
            logger.exception("state_poller iteration failed; continuing")


# --------------- App setup ---------------

def _warmup_models():
    """Load the heavy models at startup so the FIRST user interaction is fast.
    Otherwise the first request pays a one-time ~10s+ cold load of the FLOAT GPU
    pool (and the ASR model). Runs in a daemon thread so the server starts
    listening immediately; get_pool()/get_model() are lock-guarded, so a user
    request arriving mid-warmup just waits on the same load (no double load).
    """
    try:
        logger.info("Pre-warming models (FLOAT pool on GPUs %s + ASR on GPU %s)...",
                    config.FLOAT_GPUS, config.ASR_GPU)
        from modules import avatar, asr
        avatar.get_pool()
        asr.get_model()
        if getattr(config, "USE_EOU", False):
            from modules import eou
            eou.get_model()  # self-handles failure -> falls back to silence VAD
        logger.info("Model pre-warm complete; first response will be fast.")
    except Exception as e:
        logger.warning("Model pre-warm failed (will lazy-load on demand): %s", e)


def _reap_idle_sessions():
    """Drop sessions whose clients have all been gone (idle) for a grace period,
    so distinct browsers over time don't leak Pipeline/VAD objects forever. The
    shared 'default' session is never reaped."""
    grace = getattr(config, "SESSION_IDLE_TTL_SEC", 600)
    now = time.time()
    with _sessions_lock:
        for sid in list(sessions.keys()):
            if sid == "default":
                continue
            s = sessions[sid]
            if (not s.state_clients and s.pipeline.state == "idle"
                    and s.empty_since and now - s.empty_since > grace):
                del sessions[sid]
                logger.info("Reaped idle session %s (total sessions: %d)", sid, len(sessions))


def _temp_janitor():
    """Delete stale per-sentence clips from TEMP_DIR so long sessions don't fill
    the disk. Each utterance produces a .wav + .mp4 that are only needed until the
    browser has fetched and played them; anything older than the TTL is safe to
    remove. Runs forever in a daemon thread. Also reaps idle sessions.
    """
    ttl = getattr(config, "TEMP_FILE_TTL_SEC", 180)
    interval = getattr(config, "TEMP_CLEAN_INTERVAL_SEC", 30)
    exts = (".mp4", ".wav", ".txt", ".mp3")
    while True:
        try:
            now = time.time()
            removed = 0
            for name in os.listdir(config.TEMP_DIR):
                if not name.endswith(exts):
                    continue
                path = os.path.join(config.TEMP_DIR, name)
                try:
                    if os.path.isfile(path) and now - os.path.getmtime(path) > ttl:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass
            if removed:
                logger.info("[janitor] removed %d stale temp files", removed)
            _reap_idle_sessions()
        except Exception:
            logger.exception("temp janitor iteration failed; continuing")
        time.sleep(interval)


async def on_startup():
    asyncio.create_task(state_poller())
    threading.Thread(target=_warmup_models, name="warmup", daemon=True).start()
    threading.Thread(target=_temp_janitor, name="janitor", daemon=True).start()


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

    # Enable HTTPS for public access (browser mic requires a secure context
    # on any non-localhost origin). Falls back to plain HTTP if certs are missing.
    ssl_kwargs = {}
    have_certs = os.path.exists(config.SSL_CERT_FILE) and os.path.exists(config.SSL_KEY_FILE)
    if config.ENABLE_HTTPS and have_certs:
        ssl_kwargs = {
            "ssl_certfile": config.SSL_CERT_FILE,
            "ssl_keyfile": config.SSL_KEY_FILE,
        }
        scheme = "https"
    else:
        scheme = "http"
        if config.ENABLE_HTTPS and not have_certs:
            logger.warning(
                "ENABLE_HTTPS is set but certs not found at %s / %s — serving plain HTTP. "
                "Microphone will only work via localhost.",
                config.SSL_CERT_FILE, config.SSL_KEY_FILE,
            )

    logger.info(
        "Serving on %s://%s:%d  (open %s://<your-host>:%d/ from the public internet)",
        scheme, config.SERVER_HOST, config.SERVER_PORT, scheme, config.SERVER_PORT,
    )
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info",
        **ssl_kwargs,
    )
