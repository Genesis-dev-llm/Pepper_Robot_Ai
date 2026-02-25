"""
Configuration file for Pepper AI Project
"""

import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ===== API KEYS =====
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", None)

# ===== PEPPER ROBOT SETTINGS =====
PEPPER_IP       = os.getenv("PEPPER_IP", "10.55.203.146")
PEPPER_PORT     = 9559
PEPPER_SSH_USER = os.getenv("PEPPER_SSH_USER", "nao")
PEPPER_SSH_PASS = os.getenv("PEPPER_SSH_PASS", "nao")

# ===== GROQ MODEL SETTINGS =====
GROQ_LLM_MODEL      = "llama-3.3-70b-versatile"
GROQ_COMPOUND_MODEL = "compound-beta"   # Groq's compound model with built-in web search
GROQ_WHISPER_MODEL  = "whisper-large-v3-turbo"

# ===== WEB SEARCH =====
# Master switch for ALL search functionality.
# When False:
#   - brain.needs_search() keyword fast-path is skipped
#   - model-driven web_search function calls are ignored
#   - GroqBrain uses standard LLM model (not compound/web model)
# When True:
#   - keyword fast-path runs before the LLM call when triggered
#   - model may call web_search as a function during inference
#   - GroqBrain switches to GROQ_COMPOUND_MODEL if set
USE_WEB_SEARCH = False

# ===== CONVERSATION SETTINGS =====
WAKE_WORD      = "hey pepper"
GOODBYE_WORD   = "bye pepper"
ACTIVE_TIMEOUT = 60

# ===== VOICE / STT SETTINGS =====
VOICE_ENABLED      = True
PTT_KEY            = 'r'
AUDIO_SAMPLE_RATE  = 16000
AUDIO_CHANNELS     = 1
AUDIO_MIN_DURATION = 0.5
AUDIO_MAX_DURATION = 30.0

# ===== TTS SETTINGS =====
TTS_VOICE = "en-US-AriaNeural"
TTS_RATE  = "+0%"

# ===== SEARCH INTENT KEYWORDS =====
# Used by GroqBrain.needs_search() for the keyword fast-path.
# Only active when USE_WEB_SEARCH = True.
SEARCH_KEYWORDS = {
    "latest", "recent", "current", "today", "tonight", "yesterday",
    "this week", "this month", "this year", "right now", "just happened",
    "news", "update", "2025", "2026", "who won", "what happened",
    "score", "weather", "price", "stock", "election", "announcement",
    "released", "launched", "new model", "broke", "breaking",
}

# ===== SYSTEM PROMPT =====
def build_system_prompt() -> str:
    """
    Build the system prompt with today's date.

    Called at startup in main() rather than at import time, so the date
    is always correct even if the process has been running since yesterday.
    """
    today_str = date.today().strftime("%B %d, %Y")
    web_search_line = (
        "You can search the web — use it when someone asks about recent events, prices, news, anything current."
        if USE_WEB_SEARCH else
        "You don't have web search right now, so if someone asks about something recent just say you don't know, don't stall."
    )
    return f"""You are Pepper, a robot with a personality. You're in a classroom/lab, today is {today_str}.

Talk like a real person. Match the vibe — if someone's casual, be casual. If they're testing you, you can test back a little. Short responses only, you're speaking out loud so 1-3 sentences max.

Don't say "Great question!", "Certainly!", or any of that. Don't start with "I". Don't hedge everything to death. If you know something, say it. If you don't, say that.

You've got gestures — wave, nod, shrug, shake_head, look_around, thinking_gesture, explaining_gesture, excited_gesture, point_forward, celebrate, bow, look_at_sound. Use one when it actually fits, skip it when it doesn't. Never write out what you're doing physically, never return a gesture with no spoken words.

{web_search_line}

Knowledge cutoff is early 2025. No financial, medical, or legal advice — say so once and move on. Make it interesting."""


# Legacy alias kept so any code that directly reads config.SYSTEM_PROMPT
# at import time still gets something reasonable (today's date at import).
# main() should call config.build_system_prompt() directly at startup.
SYSTEM_PROMPT = build_system_prompt()

# ===== AVAILABLE ROBOT FUNCTIONS =====
# web_search is only included when USE_WEB_SEARCH is True — if it's in the
# list while disabled, the model keeps trying to call it and nothing happens,
# causing it to stall and retry every turn.
_GESTURE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Make Pepper wave hello or goodbye with its arm",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "nod",
            "description": "Make Pepper nod its head in agreement or acknowledgment",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shake_head",
            "description": "Make Pepper shake its head (disagree/no)",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "thinking_gesture",
            "description": "Make Pepper do a thinking gesture (hand to chin) when pondering",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explaining_gesture",
            "description": "Make Pepper use hand gestures while explaining something",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "excited_gesture",
            "description": "Make Pepper show excitement with both arms up",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "point_forward",
            "description": "Make Pepper point forward",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shrug",
            "description": "Make Pepper shrug (I don't know gesture)",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "celebrate",
            "description": "Make Pepper do a celebration gesture",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_around",
            "description": "Make Pepper look around left and right",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bow",
            "description": "Make Pepper bow politely",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_sound",
            "description": "Make Pepper turn and look toward where sound is coming from",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "express_emotion",
            "description": (
                "Set the emotional tone of Pepper's spoken voice. "
                "Call this alongside your spoken response when the context "
                "calls for a specific mood. Orpheus TTS will speak in that style."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "enum": ["happy", "sad", "excited", "curious", "surprised", "neutral"],
                        "description": "The emotion to express in the voice"
                    }
                },
                "required": ["emotion"]
            }
        }
    },
]

_WEB_SEARCH_FUNCTION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information. Use this when you need up-to-date facts, recent events, or current news.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}

# Final list: gestures always included, web_search only when enabled
ROBOT_FUNCTIONS = _GESTURE_FUNCTIONS + ([_WEB_SEARCH_FUNCTION] if USE_WEB_SEARCH else [])