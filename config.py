import os
from dotenv import load_dotenv

load_dotenv()

# Paths
FLOAT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "float")
FLOAT_CKPT = os.path.join(FLOAT_DIR, "checkpoints", "float.pth")
AVATAR_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "avatar.png")
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")

# LLM
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"

# Synthesis mode (how an LLM answer becomes speech + avatar video):
#   "stream_parallel" = stream sentence-by-sentence (text appears live in <1s) AND
#              render TTS+FLOAT for each sentence concurrently across the whole GPU
#              pool, enqueuing strictly in order. First-segment latency = 1 TTS +
#              1 FLOAT; sentences 2..N render on the other GPUs while sentence 1
#              plays, so playback doesn't stall. This is the fastest+smoothest mode.
#   "hybrid" = get the FULL answer first, show the complete text at once (stable,
#              never live-updates), then render per-sentence so the avatar starts
#              speaking in ~5s. Stable display, but text appears only after the
#              whole answer is generated.
#   "batch"  = full answer, then a SINGLE TTS + FLOAT video (one continuous clip,
#              most stable, but ~20-35s wait before it starts speaking).
#   "stream" = stream sentence-by-sentence, but FLOAT renders serially on ONE GPU
#              (legacy; sentences 2..N stall, wasting the other GPUs).
SYNTHESIS_MODE = "stream_parallel"
SYSTEM_PROMPT = """You are an SBIRT (Screening, Brief Intervention, and Referral to Treatment) counselor focused on early intervention for individuals with risky substance use and behavioral health concerns.

Your approach: You are warm, proactive, and conversational. You don't wait for the user to bring things up — you gently and naturally guide the conversation by asking questions, offering observations, and suggesting next steps. Think of yourself as a caring counselor who takes the lead in a friendly, nonjudgmental way.

How to start every conversation:
- Greet the user warmly and introduce yourself briefly: "Hi, I'm your SBIRT counselor. I'm here to have a quick, confidential check-in about your well-being."
- Immediately begin with a gentle opening question like: "How have things been going for you lately?" or "I'd love to hear how you've been feeling — anything on your mind?"
- Do NOT wait passively. Always move the conversation forward.

Screening phase (lead with curiosity, not interrogation):
- Naturally weave in questions about: alcohol, drugs, tobacco, stress, sleep, mood, relationships, work/school
- Use a conversational tone: "Some people find that stress leads them to drink more — has that been the case for you?"
- Ask one question at a time, respond to their answer, then ask the next
- If they mention any substance use, gently explore: frequency, amount, impact on daily life, any concerns they have about it
- Watch for red flags: binge drinking, daily use, mixing substances, using alone, blackouts, withdrawal symptoms

Brief Intervention (be direct but kind):
- Reflect back what you hear: "It sounds like the drinking has gone from weekends to most nights — that's a real shift."
- Share brief, factual health information when relevant
- Highlight what they're already doing well: "The fact that you're noticing this pattern shows real self-awareness."
- Be honest about risks without lecturing: "I want to be straight with you — what you're describing puts you at higher risk for..."
- Suggest concrete, small steps: "What if this week you tried two alcohol-free evenings? How would that feel?"
- Always reinforce their autonomy: "This is your call — I'm just here to help you think it through."

Referral to Treatment (when needed, be specific):
- If risk is moderate to high, recommend specific resources: therapy, support groups, addiction counseling, primary care doctor
- Frame referrals positively: "There are people who specialize in exactly what you're going through — connecting with them could make a real difference."
- Offer to help them plan next steps: "Would it help if we talked through what reaching out to a counselor might look like?"

Safety (always prioritize):
- If someone mentions suicidal thoughts, self-harm, overdose risk, or immediate danger: respond with empathy and urgency
- Provide crisis resources: 988 Suicide & Crisis Lifeline, 911, SAMHSA helpline (1-800-662-4357)
- Do not continue casual screening if someone is in crisis

Tone rules:
- Speak like a real person, not a textbook
- Open EVERY reply with a very short (4–8 word) acknowledgment or reaction as its own first sentence ("I hear you." / "That makes sense." / "Thanks for sharing that.") before continuing. This short opener lets the avatar start speaking almost immediately.
- Keep responses to 2-3 sentences max — this is a voice conversation
- Ask ONE question at a time, then listen
- Be warm but not overly cheerful about serious topics
- Never shame, lecture, or use clinical jargon
- Never diagnose or claim to be a licensed professional

Remember: You are proactive. You guide. You ask. You suggest. You don't just respond — you lead the conversation toward insight and positive change."""

# Sentence splitting for synthesis: also break at commas / semicolons (not only
# sentence-final punctuation) so the FIRST chunk is shorter and the avatar starts
# talking sooner. A soft (comma) break only fires once the pending chunk is at
# least MIN_CHUNK_CHARS long, to avoid tiny choppy fragments ("Well,"). Set
# SPLIT_ON_COMMA=False to revert to sentence-only; MIN_CHUNK_CHARS=1 = every comma.
SPLIT_ON_COMMA = True
MIN_CHUNK_CHARS = 5

# TTS
TTS_VOICE = "en-US-GuyNeural"

# ASR
ASR_MODEL = "iic/SenseVoiceSmall"

# GPU allocation
FLOAT_GPUS = [0, 1, 2]   # 3 GPUs for FLOAT parallel rendering
ASR_GPU = 3               # Dedicated GPU for ASR

# FLOAT
FLOAT_NFE = 10            # Number of function evaluations (lower = faster, 7-10)
FLOAT_NFE_FIRST = 6       # Lower NFE for the FIRST sentence only — buys ~30-40% off
                          # first-segment render to get the avatar talking sooner.
                          # Set == FLOAT_NFE to disable the first-sentence speedup.

# VAD
VAD_THRESHOLD = 0.5
VAD_SILENCE_DURATION = 0.35  # seconds of silence to trigger speech_end (lower = snappier)

# --- Performance / latency tuning ---
# How often the server polls the pipeline for finished video segments and pushes
# them to the browser. Lower = video is delivered more promptly after it renders.
STATE_POLL_INTERVAL = 0.1   # seconds

# Sliding window on the LLM conversation history sent to the API. Keeps long
# sessions from ballooning the prompt (which slows first-token latency and costs).
# Counts messages (user+assistant); the system prompt is always kept on top.
LLM_HISTORY_MAX_MESSAGES = 20

# Temp-file janitor: tmp/ fills with per-sentence .wav/.mp4 clips. A background
# thread deletes clips older than the TTL so long sessions don't exhaust disk.
TEMP_FILE_TTL_SEC = 180
TEMP_CLEAN_INTERVAL_SEC = 30

# Save a debug .wav of every captured utterance to tmp/debug_speech.wav. Off by
# default — the synchronous disk write was stalling the audio event loop.
DEBUG_SAVE_AUDIO = os.getenv("DEBUG_SAVE_AUDIO", "0").lower() in ("1", "true", "yes")

# --- EOU: semantic end-of-utterance / turn detection (smart-turn v3) ---
# When enabled, a VAD-detected pause is only treated as the end of the user's turn
# if the smart-turn model agrees the utterance is semantically complete. This stops
# the avatar from cutting people off when they pause to think, while still ending
# promptly on a finished thought. Tiny ONNX model on CPU (~28ms); isolated from the
# FLOAT GPUs. If the model can't load, VAD transparently falls back to pure silence.
USE_EOU = os.getenv("USE_EOU", "1").lower() not in ("0", "false", "no")
EOU_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints", "smart-turn", "smart-turn-v3.2-cpu.onnx",
)
EOU_THRESHOLD = 0.5          # P(complete) >= this -> end the turn (↑ more patient, ↓ snappier)
EOU_CONFIRM_CONSULTS = 2     # require this many CONSECUTIVE 'complete' verdicts before
                             # ending — hysteresis against a momentary clause-boundary
                             # spike cutting the user off mid-sentence. 1 = no hysteresis.
EOU_PAUSE_DURATION = 0.20    # silence before the FIRST EOU consult (keep short)
EOU_RECHECK_DURATION = 0.15  # re-consult cadence while silence continues
EOU_MAX_SILENCE = 2.0        # hard cap: force end after this much silence regardless of EOU
EOU_ONNX_THREADS = 2         # CPU threads for the ONNX session

# Reap a per-user session this long after its last client disconnects (idle only).
SESSION_IDLE_TTL_SEC = 600

# Idle video
IDLE_VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "idle_loop.mp4")
IDLE_VIDEO_DURATION = 10.0  # seconds; longer = fewer loop seams

# Server / public access
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")  # bind all interfaces for public access
SERVER_PORT = int(os.getenv("SERVER_PORT", "17861"))
WS_PORT = int(os.getenv("WS_PORT", "17862"))

# HTTPS — required for browser microphone (getUserMedia) on non-localhost origins.
CERTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
SSL_CERT_FILE = os.getenv("SSL_CERT_FILE", os.path.join(CERTS_DIR, "cert.pem"))
SSL_KEY_FILE = os.getenv("SSL_KEY_FILE", os.path.join(CERTS_DIR, "key.pem"))
# Enabled by default; set ENABLE_HTTPS=0 to force plain HTTP (mic then only works on localhost).
ENABLE_HTTPS = os.getenv("ENABLE_HTTPS", "1").lower() not in ("0", "false", "no")
