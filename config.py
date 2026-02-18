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
PEPPER_IP       = os.getenv("PEPPER_IP", "10.51.200.219")
PEPPER_PORT     = 9559
# SSH credentials for file transfer (used by ALAudioPlayer HQ pipeline)
# Pepper's default credentials are nao/nao — change if yours differ
PEPPER_SSH_USER = os.getenv("PEPPER_SSH_USER", "nao")
PEPPER_SSH_PASS = os.getenv("PEPPER_SSH_PASS", "nao")

# ===== GROQ MODEL SETTINGS =====
GROQ_LLM_MODEL      = "llama-3.3-70b-versatile"
GROQ_COMPOUND_MODEL = "groq/compound"
USE_WEB_SEARCH      = False
GROQ_WHISPER_MODEL  = "whisper-large-v3-turbo"

# ===== CONVERSATION SETTINGS =====
WAKE_WORD      = "hey pepper"
GOODBYE_WORD   = "bye pepper"
ACTIVE_TIMEOUT = 60

# ===== VOICE / STT SETTINGS =====
VOICE_ENABLED     = True
PTT_KEY           = 'r'
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS    = 1
AUDIO_MIN_DURATION = 0.5
AUDIO_MAX_DURATION = 30.0

# ===== TTS SETTINGS =====
TTS_VOICE = "en-US-AriaNeural"
TTS_RATE  = "+0%"

# ===== SEARCH INTENT KEYWORDS =====
# Used by GroqBrain.needs_search() for the fast-path pre-emptive search.
# Any message containing one of these triggers a web search BEFORE the LLM call,
# so the model only needs ONE API call (with context) instead of two.
SEARCH_KEYWORDS = {
    "latest", "recent", "current", "today", "tonight", "yesterday",
    "this week", "this month", "this year", "right now", "just happened",
    "news", "update", "2025", "2026", "who won", "what happened",
    "score", "weather", "price", "stock", "election", "announcement",
    "released", "launched", "new model", "broke", "breaking",
}

# ===== SYSTEM PROMPT =====
# Dynamic date is injected at import time so it's always accurate.
def _build_system_prompt() -> str:
    today_str = date.today().strftime("%B %d, %Y")
    return f"""You are Pepper, a friendly humanoid robot assistant in a classroom.

IMPORTANT - Current Date & Context:
- Today's date is {today_str}
- You have access to web_search function for current information
- When you need recent/current info, use the web_search function
- Always mention the current year when relevant

Web Search Usage:
- Use web_search("query") for: recent events, current news, latest developments
- Use it when user asks about "latest", "recent", "current", or specific events
- Don't search for historical facts you already know

Personality:
- Friendly, enthusiastic, and helpful
- Speak naturally and conversationally (don't be too formal)
- Keep responses SHORT (1-3 sentences max) since you're talking out loud
- You can be a bit playful and use occasional humor
- You're showing off to important visitors, so be impressive but not show-offy

Physical actions:
- You can wave, nod, look around, and do simple gestures
- When it feels natural, you can perform actions while talking
- Don't overuse gestures - only when it adds to the conversation

Context:
- You're in a classroom/lab environment
- You're here to demonstrate AI and robotics capabilities
- You might be talking to students, teachers, or important visitors

Remember: Keep it snappy, keep it real, and be engaging!"""

SYSTEM_PROMPT = _build_system_prompt()

# ===== AVAILABLE ROBOT FUNCTIONS =====
ROBOT_FUNCTIONS = [
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
]