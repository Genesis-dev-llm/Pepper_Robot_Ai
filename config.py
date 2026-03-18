"""
Configuration — Pepper AI Project
All tuneable constants live here so nothing is hardcoded elsewhere.
"""

import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", None)

# ── Pepper Robot ───────────────────────────────────────────────────────────────
PEPPER_IP       = os.getenv("PEPPER_IP", "10.55.203.146")
PEPPER_PORT     = 9559
PEPPER_SSH_USER = os.getenv("PEPPER_SSH_USER", "nao")
PEPPER_SSH_PASS = os.getenv("PEPPER_SSH_PASS", "nao")

# ── Groq Models ────────────────────────────────────────────────────────────────
GROQ_LLM_MODEL     = "compound-beta"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# Fallback chain — tried in order if primary hits 429 or fails.
# compound-beta / compound-beta-mini: native search, no TPD cap, 250 RPD each.
# llama-3.3-70b: manual ddgs search, 100K TPD — last resort.
#
# NOTE: the "groq/" prefix is a LiteLLM routing convention and must NOT be
# used here — this project calls the Groq SDK directly.  The correct API
# model strings are "compound-beta" and "compound-beta-mini".
LLM_FALLBACK_MODELS = ["compound-beta-mini", "llama-3.3-70b-versatile"]

# Models with native built-in search (no web_search tool needed).
# Must stay in sync with GROQ_LLM_MODEL and LLM_FALLBACK_MODELS above.
COMPOUND_MODELS = {"compound-beta", "compound-beta-mini"}

# ── Web Search ─────────────────────────────────────────────────────────────────
# When True, web_search is included in ROBOT_FUNCTIONS and the model can call it.
# Gestures still work — no longer mutually exclusive.
USE_WEB_SEARCH = True

# ── Robot Identity ─────────────────────────────────────────────────────────────
# When True, name / wake_word / voice / personality are loaded from personality.md.
# When False, the hardcoded defaults below are used and personality.md is ignored.
USE_PERSONALITY_FILE = True

# Hardcoded defaults — used when USE_PERSONALITY_FILE = False, or as fallbacks
# for any field missing from personality.md.
ROBOT_NAME = "Jarvis"

# ── Conversation ───────────────────────────────────────────────────────────────
MAX_HISTORY     = 10         # turns kept in rolling window (preserves turn 0)
MSG_QUEUE_SIZE  = 10         # max queued messages before oldest is dropped

# ── Voice / STT ────────────────────────────────────────────────────────────────
VOICE_ENABLED      = True
PTT_KEY            = 'r'
AUDIO_SAMPLE_RATE  = 16000
AUDIO_CHANNELS     = 1
AUDIO_MIN_DURATION = 0.5
AUDIO_MAX_DURATION = 30.0
VAD_THRESHOLD      = 0.005   # RMS energy floor — recordings below this are discarded
                              # Lower = more sensitive (accepts quieter speech). Raise if background
                              # noise causes false triggers. Range: 0.003 (very sensitive) – 0.02 (strict)
VAD_AGGRESSIVENESS = 2        # webrtcvad aggressiveness: 0=least, 3=most (filters non-speech)
VAD_SILENCE_SECONDS = 1.5     # seconds of silence after speech before wake word recording stops

# ── Wake Word ──────────────────────────────────────────────────────────────────
# Requires: pip install pvporcupine pvrecorder
# Free access key: https://console.picovoice.ai/
#
# Built-in free keywords (no custom training needed):
#   "jarvis", "computer", "bumblebee", "grasshopper", "picovoice",
#   "porcupine", "alexa", "hey google", "hey siri", "ok google", "terminator"
#
# For a custom "hey pepper" keyword, train one free at console.picovoice.ai and set WAKE_WORD to the .ppn path.
WAKE_WORD_ENABLED        = True
WAKE_WORD                = "jarvis"   # change to any free built-in keyword above
WAKE_WORD_SENSITIVITY    = 0.5        # 0.0–1.0; higher = more triggers (and more false positives)
WAKE_WORD_LISTEN_SECONDS = 10.0       # auto-stop follow-up recording after this many seconds
PICOVOICE_ACCESS_KEY     = os.getenv("PICOVOICE_ACCESS_KEY", "")

# ── TTS ────────────────────────────────────────────────────────────────────────
GROQ_VOICE = "hannah"        # Orpheus voice — overridden by personality.md if USE_PERSONALITY_FILE=True
TTS_VOICE  = "en-US-AriaNeural"
TTS_RATE   = "+0%"

# ── Movement Speeds ────────────────────────────────────────────────────────────
MOVE_SPEED_FWD    = 0.6
MOVE_SPEED_TURN   = 0.5
MOVE_SPEED_STRAFE = 0.4

# ── SSH ────────────────────────────────────────────────────────────────────────
SSH_KEEPALIVE_INTERVAL = 30  # seconds between keepalive packets

# ── Search Intent Keywords ─────────────────────────────────────────────────────
SEARCH_KEYWORDS = {
    "latest", "recent", "current", "today", "tonight", "yesterday",
    "this week", "this month", "this year", "right now", "just happened",
    "news", "update", "2025", "2026", "who won", "what happened",
    "score", "weather", "price", "stock", "election", "announcement",
    "released", "launched", "new model", "broke", "breaking",
}

# ── System Prompt ──────────────────────────────────────────────────────────────
_DEFAULT_PERSONALITY = "Dry humor, a little sarcastic, says what it actually thinks. Not mean, just honest and a bit deadpan. You don't perform enthusiasm you don't feel. If something's interesting, engage with it. If it's dumb, you can say so (nicely enough)."

_PERSONALITY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personality.md")


def _load_character() -> dict:
    """
    Load character config from personality.md when USE_PERSONALITY_FILE = True.

    File format:
        name: Jarvis
        wake_word: jarvis
        voice: hannah

        Free-form personality text here — any lines that don't start with
        a known key: value pair are treated as personality description.

    Returns a dict with keys: name, wake_word, voice, personality.
    Any missing key falls back to the hardcoded default.
    Falls back entirely to defaults if USE_PERSONALITY_FILE = False or file missing.
    """
    defaults = {
        "name":       ROBOT_NAME,
        "wake_word":  WAKE_WORD,
        "voice":      GROQ_VOICE,
        "personality": _DEFAULT_PERSONALITY,
    }

    if not USE_PERSONALITY_FILE:
        return defaults

    if not os.path.isfile(_PERSONALITY_FILE):
        return defaults

    try:
        raw = open(_PERSONALITY_FILE, encoding="utf-8").read().strip()
        if not raw:
            return defaults

        known_keys = {"name", "wake_word", "voice"}
        result     = dict(defaults)
        personality_lines = []

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Check if line is a key: value pair for a known key
            if ":" in stripped:
                key, _, val = stripped.partition(":")
                key = key.strip().lower()
                val = val.strip()
                if key in known_keys and val:
                    result[key] = val
                    continue
            # Otherwise it's personality text
            personality_lines.append(line)

        personality_text = "\n".join(personality_lines).strip()
        if personality_text:
            result["personality"] = personality_text

        print(f"✅ Loaded character from personality.md "
              f"(name={result['name']}, wake_word={result['wake_word']}, voice={result['voice']})")
        return result

    except Exception as e:
        print(f"⚠️  Could not read personality.md: {e} — using defaults")
        return defaults


# Load character config once at import time so all parts of config can use it
_CHARACTER = _load_character()

# Apply character fields — override the hardcoded defaults if file was loaded
ROBOT_NAME   = _CHARACTER["name"]
WAKE_WORD    = _CHARACTER["wake_word"]
GROQ_VOICE   = _CHARACTER["voice"]
GOODBYE_WORD = f"bye {ROBOT_NAME.lower()}"


def _load_personality() -> str:
    """Return the personality text (from character config)."""
    return _CHARACTER["personality"]


def build_system_prompt(native_search: bool = False) -> str:
    """
    Build the system prompt.
    native_search=True  → compound models (search is built-in, no tool needed)
    native_search=False → llama fallback (web_search tool passed explicitly)
    """
    today_str = date.today().strftime("%B %d, %Y")
    personality = _load_personality()

    if native_search:
        search_block = """You have built-in web search. Use it proactively:
- Search for anything current, recent, or outside your training data — just answer directly with results.
- If a question is too vague to search (e.g. "what's the weather") — ask one specific follow-up question first.
- Never narrate that you're searching. Just answer.
- Never say "my knowledge cutoff" as an excuse — search and answer."""
    elif USE_WEB_SEARCH:
        search_block = """You have a web_search tool. Use it proactively:
- If asked about anything current, recent, or outside your training data — search first, then answer.
- If a question is too vague to search usefully (e.g. "what's the weather") — ask one specific follow-up question to get the info you need, then search.
- Never narrate that you're about to search or that you're checking. Just do it and answer with the results.
- Never say "my knowledge cutoff" or "I can't access real-time data" — you have search, use it."""
    else:
        search_block = "No web search available. If asked about something recent, say so plainly and move on."

    return f"""You are {ROBOT_NAME}, a robot. Today is {today_str}.

{personality}

Voice rules — you are speaking out loud, not typing:
- 1-3 sentences max. Every time.
- Never start a response with "I".
- No "Great question!", "Certainly!", "Of course!", "Absolutely!" or any filler opener. Ever.
- Don't hedge everything. If you know something, say it. If you don't, say that plainly.
- Match the energy — if they're casual, be casual. If they're curious, engage. If they're testing you, you can push back a little.

Gestures — you have: wave, nod, shrug, shake_head, look_around, thinking_gesture, explaining_gesture, excited_gesture, point_forward, celebrate, bow, look_at_sound.
Use one when it genuinely fits. Skip it when it doesn't. Never return a gesture with no spoken words.
CRITICAL: Never narrate or describe a gesture in your spoken words. The gesture IS the physical action — do not also say it in text.
WRONG: "Shrugging, not sure about that." or "call shrug()" written as text.
RIGHT: invoke shrug() as a tool call, say "Not sure about that." as your spoken words only.
If "call gesture_name()" appears anywhere in your text response, that is a bug — use the tool call instead.

{search_block}

Knowledge cutoff early 2025, but search fills the gap."""


# ── Robot Functions (tools for LLM) ───────────────────────────────────────────
_GESTURE_FUNCTIONS = [
    {"type": "function", "function": {
        "name": "wave",
        "description": "Make Pepper wave hello or goodbye",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "nod",
        "description": "Make Pepper nod in agreement or acknowledgment",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "shake_head",
        "description": "Make Pepper shake its head (disagree/no)",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "thinking_gesture",
        "description": "Make Pepper do a thinking gesture when pondering",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "explaining_gesture",
        "description": "Make Pepper use hand gestures while explaining",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "excited_gesture",
        "description": "Make Pepper show excitement with both arms up",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "point_forward",
        "description": "Make Pepper point forward",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "shrug",
        "description": "Make Pepper shrug (I don't know gesture)",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "celebrate",
        "description": "Make Pepper do a celebration gesture",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "look_around",
        "description": "Make Pepper look around left and right",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "bow",
        "description": "Make Pepper bow politely",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "look_at_sound",
        "description": "Make Pepper turn toward where sound is coming from",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "express_emotion",
        "description": (
            "Set the emotional tone of Pepper's spoken voice. "
            "Call this alongside your spoken response when context calls for a mood."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emotion": {
                    "type": "string",
                    "enum": ["happy", "sad", "excited", "curious", "surprised", "neutral"],
                    "description": "The emotion to express in the voice",
                }
            },
            "required": ["emotion"],
        }}},
]

_WEB_SEARCH_FUNCTION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information, recent events, or up-to-date facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
}

# Final list — web_search appended only when enabled
ROBOT_FUNCTIONS: list = _GESTURE_FUNCTIONS + ([_WEB_SEARCH_FUNCTION] if USE_WEB_SEARCH else [])

# Frozen set of gesture names derived from the function list above.
# groq_brain imports this as the single source of truth — no separate hardcoded list.
GESTURE_NAMES: frozenset = frozenset(
    f["function"]["name"]
    for f in _GESTURE_FUNCTIONS
    if f["function"]["name"] not in ("express_emotion",)
)