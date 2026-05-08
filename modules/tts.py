import asyncio
import tempfile
import threading
import edge_tts
import config


async def _synthesize(text: str, voice: str, output_path: str) -> str:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


def synthesize(text: str, output_path: str | None = None,
               cancel_event: threading.Event | None = None) -> str | None:
    """Synthesize text to speech. Returns path to wav file, or None if cancelled."""
    if cancel_event and cancel_event.is_set():
        return None

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav", dir=config.TEMP_DIR)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            pool.submit(asyncio.run, _synthesize(text, config.TTS_VOICE, output_path)).result()
    else:
        asyncio.run(_synthesize(text, config.TTS_VOICE, output_path))
    return output_path


if __name__ == "__main__":
    import os
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    path = synthesize("Hello, I am your AI assistant, nice to meet you!")
    print(f"TTS output saved to: {path}")
