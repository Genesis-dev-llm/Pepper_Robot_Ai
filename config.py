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
GROQ_LLM_MODEL     = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"

# ── Web Search ─────────────────────────────────────────────────────────────────
# When True, web_search is included in ROBOT_FUNCTIONS and the model can call it.
# Gestures still work — no longer mutually exclusive.
USE_WEB_SEARCH = False

# ── Conversation ───────────────────────────────────────────────────────────────
ACTIVE_TIMEOUT  = 120        # seconds of inactivity before auto-idle (0 = disabled)
MAX_HISTORY     = 10         # turns kept in rolling window (preserves turn 0)
MSG_QUEUE_SIZE  = 10         # max queued messages before oldest is dropped
GOODBYE_WORD    = "bye pepper"

# ── Voice / STT ────────────────────────────────────────────────────────────────
VOICE_ENABLED      = True
PTT_KEY            = 'r'
AUDIO_SAMPLE_RATE  = 16000
AUDIO_CHANNELS     = 1
AUDIO_MIN_DURATION = 0.5
AUDIO_MAX_DURATION = 30.0
VAD_THRESHOLD      = 0.01    # RMS energy floor — recordings below this are discarded

# ── TTS ────────────────────────────────────────────────────────────────────────
GROQ_VOICE = "hannah"        # Orpheus voice name
TTS_VOICE  = "en-US-AriaNeural"
TTS_RATE   = "+0%"

# ── Movement Speeds ────────────────────────────────────────────────────────────
MOVE_SPEED_FWD    = 0.6
MOVE_SPEED_TURN   = 0.5
MOVE_SPEED_STRAFE = 0.4

# ── SSH ────────────────────────────────────────────────────────────────────────
SSH_KEEPALIVE_INTERVAL = 30  # seconds between keepalive packets

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_TO_FILE = True
LOG_DIR     = "logs"

# ── Search Intent Keywords ─────────────────────────────────────────────────────
SEARCH_KEYWORDS = {
    "latest", "recent", "current", "today", "tonight", "yesterday",
    "this week", "this month", "this year", "right now", "just happened",
    "news", "update", "2025", "2026", "who won", "what happened",
    "score", "weather", "price", "stock", "election", "announcement",
    "released", "launched", "new model", "broke", "breaking",
}

# ── System Prompt ──────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    today_str = date.today().strftime("%B %d, %Y")
    web_line = (
        "You can search the web — use it when someone asks about recent events, "
        "prices, news, anything current."
        if USE_WEB_SEARCH else
        "You don't have web search right now, so if someone asks about something "
        "recent just say you don't know, don't stall."
    )
    return f"""You are Pepper, a robot with a personality. You're in a classroom/lab, today is {today_str}.

Talk like a real person. Match the vibe — if someone's casual, be casual. If they're testing you, you can test back a little. Short responses only, you're speaking out loud so 1-3 sentences max.

Don't say "Great question!", "Certainly!", or any of that. Don't start with "I". Don't hedge everything to death. If you know something, say it. If you don't, say that.

You've got gestures — wave, nod, shrug, shake_head, look_around, thinking_gesture, explaining_gesture, excited_gesture, point_forward, celebrate, bow, look_at_sound. Use one when it actually fits, skip it when it doesn't. Never write out what you're doing physically, never return a gesture with no spoken words.

{web_line}

Knowledge cutoff is early 2025. No financial, medical, or legal advice — say so once and move on. Make it interesting."""


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