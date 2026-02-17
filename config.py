"""
Configuration file for Pepper AI Project
Store your API keys and robot settings here

SECURITY NOTE:
For better security, set environment variables instead of hardcoding:
  export GROQ_API_KEY="your_key_here"
  export PEPPER_IP="192.168.1.100"
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# ===== API KEYS =====
# Try environment variable first, fall back to hardcoded value
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")  # Get from https://console.groq.com/keys

# OPTIONAL: ElevenLabs for higher quality TTS (free tier: 10k chars/month)
# If not provided, system will skip ElevenLabs and fall back to Edge TTS
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", None)  # Get from https://elevenlabs.io

# ===== PEPPER ROBOT SETTINGS =====
PEPPER_IP = os.getenv("PEPPER_IP", "192.168.1.100")  # Change this to your Pepper's IP address
PEPPER_PORT = 9559

# ===== GROQ MODEL SETTINGS =====
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"  # Fast, supports gestures + custom search
GROQ_COMPOUND_MODEL = "groq/compound"  # With built-in web search (no custom functions)
USE_WEB_SEARCH = False  # False = use custom DuckDuckGo search function (better!)
GROQ_WHISPER_MODEL = "whisper-large-v3-turbo"  # Fast STT

# ===== CONVERSATION SETTINGS =====
WAKE_WORD = "hey pepper"  # What activates the robot
GOODBYE_WORD = "bye pepper"  # What deactivates it
ACTIVE_TIMEOUT = 60  # Seconds before auto-deactivating after last interaction

# ===== VOICE / STT SETTINGS =====
VOICE_ENABLED      = True          # Master switch for voice input
PTT_KEY            = 'r'           # Hold this key to record (Push-To-Talk)
AUDIO_SAMPLE_RATE  = 16000         # Hz — 16 kHz is optimal for Whisper
AUDIO_CHANNELS     = 1             # Mono is fine for speech
AUDIO_MIN_DURATION = 0.5           # Ignore clips shorter than this (seconds)
AUDIO_MAX_DURATION = 30.0          # Auto-stop after this many seconds

# ===== TTS SETTINGS =====
TTS_VOICE = "en-US-AriaNeural"  # Microsoft Edge TTS voice (female, clear)
# Other good options: "en-US-GuyNeural" (male), "en-GB-SoniaNeural" (British)
TTS_RATE = "+0%"  # Speed: -50% to +100%

# ===== SYSTEM PROMPT =====
SYSTEM_PROMPT = """You are Pepper, a friendly humanoid robot assistant in a classroom.

IMPORTANT - Current Date & Context:
- Today's date is February 13, 2026
- You have access to web_search function for current information
- When you need recent/current info, use the web_search function
- Always mention the current year (2026) when relevant

Web Search Usage:
- Use web_search("query") for: recent events, current news, latest developments
- Use it when user asks about "latest", "recent", "current", or specific 2025-2026 events
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

# ===== AVAILABLE ROBOT FUNCTIONS =====
# These will be passed to Groq's function calling
ROBOT_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "wave",
            "description": "Make Pepper wave hello or goodbye with its arm",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "nod",
            "description": "Make Pepper nod its head in agreement or acknowledgment",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shake_head",
            "description": "Make Pepper shake its head (disagree/no)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "thinking_gesture",
            "description": "Make Pepper do a thinking gesture (hand to chin) when pondering or considering something",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explaining_gesture",
            "description": "Make Pepper use hand gestures while explaining something, makes explanations more dynamic and engaging",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "excited_gesture",
            "description": "Make Pepper show excitement with both arms up, use when enthusiastic or celebrating",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "point_forward",
            "description": "Make Pepper point forward, useful when directing attention or indicating direction",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "shrug",
            "description": "Make Pepper shrug (I don't know gesture), use when uncertain or indicating lack of knowledge",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "celebrate",
            "description": "Make Pepper do a celebration gesture with arm waves, use for achievements or good news",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_around",
            "description": "Make Pepper look around left and right, useful when 'searching' or showing curiosity",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bow",
            "description": "Make Pepper bow politely, use for greetings or showing respect",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "look_at_sound",
            "description": "Make Pepper turn and look toward where sound is coming from",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use this when you need up-to-date facts, recent events, current news, or information that may have changed since your knowledge cutoff. Always use this for questions about 'recent', 'latest', 'current', or specific dates/events in 2025-2026.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find information about (e.g. 'latest AI news', 'who won super bowl 2026', 'current president of france')"
                    }
                },
                "required": ["query"]
            }
        }
    }
]