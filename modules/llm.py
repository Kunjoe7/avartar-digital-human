import threading
from openai import OpenAI
import config

client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)

conversation_history: list[dict] = []


def reset_conversation():
    conversation_history.clear()


def chat(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + conversation_history

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})
    return assistant_message


def chat_stream(user_message: str, cancel_event: threading.Event | None = None):
    """Yields complete sentences as they arrive from LLM stream.

    Properly splits on all sentence boundaries (not just the first one).
    Checks cancel_event between chunks.
    """
    conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": config.SYSTEM_PROMPT}] + conversation_history

    stream = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=messages,
        stream=True,
    )

    full_response = ""
    buffer = ""
    sentence_endings = {"。", "！", "？", ".", "!", "?", "\n"}

    for chunk in stream:
        if cancel_event and cancel_event.is_set():
            break

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            full_response += delta
            buffer += delta

            # Extract ALL complete sentences from buffer
            while True:
                split_pos = -1
                for i, ch in enumerate(buffer):
                    if ch in sentence_endings:
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

    conversation_history.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    reply = chat("Hello, please briefly introduce yourself")
    print(f"LLM reply: {reply}")
