import re
import numpy as np
from funasr import AutoModel
import config

_model = None


def get_model():
    global _model
    if _model is None:
        _model = AutoModel(
            model=config.ASR_MODEL,
            device=f"cuda:{config.ASR_GPU}",
            trust_remote_code=True,
        )
    return _model


def _clean_text(text: str) -> str:
    """Remove SenseVoice tags like <|en|><|EMO_UNKNOWN|><|Speech|><|woitn|>."""
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def transcribe(audio_path: str) -> str:
    model = get_model()
    result = model.generate(input=audio_path, language="en")
    if result and len(result) > 0:
        return _clean_text(result[0]["text"])
    return ""


def transcribe_array(audio_array: np.ndarray, sample_rate: int = 16000) -> str:
    model = get_model()
    result = model.generate(input=audio_array, fs=sample_rate, language="en")
    if result and len(result) > 0:
        return _clean_text(result[0]["text"])
    return ""


if __name__ == "__main__":
    print("ASR module loaded. Call transcribe(audio_path) to use.")
    model = get_model()
    print("Model loaded successfully.")
