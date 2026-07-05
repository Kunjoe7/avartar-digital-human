import threading
from openai import OpenAI
import config

client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)


def _trim_history(history: list[dict]):
    """Sliding window: keep only the most recent N messages so long sessions
    don't grow the prompt without bound (which slows first-token latency)."""
    max_msgs = getattr(config, "LLM_HISTORY_MAX_MESSAGES", 0)
    if max_msgs and len(history) > max_msgs:
        del history[:-max_msgs]


def chat(user_message: str, history: list[dict]) -> str:
    """Non-streaming completion. `history` is the caller-owned conversation list
    (one per session) — mutated in place with the user + assistant turns."""
    history.append({"role": "user", "content": user_message})
    _trim_history(history)

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + history

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
    )

    assistant_message = response.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_message})
    return assistant_message


def chat_stream(user_message: str, history: list[dict],
                cancel_event: threading.Event | None = None):
    """Yields complete sentences as they arrive from the LLM stream.

    `history` is the caller-owned conversation list (one per session), mutated in
    place. Properly splits on all sentence boundaries; checks cancel_event between
    chunks.
    """
    history.append({"role": "user", "content": user_message})
    _trim_history(history)

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + history

    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        stream=True,
    )

    full_response = ""
    buffer = ""
    # Hard sentence-final punctuation always splits; soft (comma/semicolon) breaks
    # split too — but only once the pending chunk is long enough (avoids tiny
    # fragments) — so the first speakable chunk is shorter and starts sooner.
    hard_endings = {"。", "！", "？", ".", "!", "?", "\n"}
    soft_endings = ({",", "，", "、", "；", ";"}
                    if getattr(config, "SPLIT_ON_COMMA", False) else set())
    min_chars = getattr(config, "MIN_CHUNK_CHARS", 0)

    for chunk in stream:
        if cancel_event and cancel_event.is_set():
            break

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            buffer += delta

            # Extract ALL complete chunks from buffer
            while True:
                split_pos = -1
                for i, ch in enumerate(buffer):
                    if ch in hard_endings:
                        split_pos = i
                        break
                    if ch in soft_endings and len(buffer[: i + 1].strip()) >= min_chars:
                        split_pos = i
                        break

                if split_pos == -1:
                    break

                sentence = buffer[: split_pos + 1].strip()
                buffer = buffer[split_pos + 1 :]
                if sentence:
                    yield sentence

    # Flush remaining buffer
    if buffer.strip() and not (cancel_event and cancel_event.is_set()):
        yield buffer.strip()

    history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    h = []
    reply = chat("Hello, please briefly introduce yourself", h)
    print(f"LLM reply: {reply}")
