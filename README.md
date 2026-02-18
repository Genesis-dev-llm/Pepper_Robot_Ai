# 🤖 Pepper AI Robot - Phase 2 (Voice + Safety)

AI-powered Pepper robot with conversation, gestures, web search, and voice interaction. Now with high-quality audio synchronization and safety watchdogs.

## ✅ What Works (Phase 1 & 2)

- ✅ **Natural Voice interaction**: Push-to-Talk (Hold R) → Groq Whisper STT.
- ✅ **High-Quality Audio**: Premium voices (Groq/ElevenLabs) played via Pepper's speakers.
- ✅ **Groq AI Brain**: Real-time conversation with context and web search.
- ✅ **Safety Watchdog**: Movement auto-stops after 1s of inactivity (Lag protection).
- ✅ **Speech Lock**: Prevention of overlapping audio for stable conversation.
- ✅ **Keyboard Controls**: Responsive WASD movement and manual gestures.
- ✅ **AI Function Calling**: Robot performs gestures automatically based on speech.
- ✅ **Automated Config**: Automatic `.env` loading and dependency validation.

## 🚀 Setup Instructions

### 1. Install Dependencies

Ensure you are in your virtual environment (if using one):

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your **Groq API Key** and **Pepper's IP Address**.
3. (Optional) Add your **ElevenLabs API Key** for premium voice quality.

### 3. Run the Robot

The system automatically validates your dependencies (like PortAudio) at startup.

```bash
python3 main.py
```

## 🎮 Controls

### Movement (WASD + Q/E)
- `W/S` - Forward/Backward
- `A/D` - Turn Left/Right
- `Q/E` - Strafe Left/Right
- **Safety**: Releasing keys or losing connection auto-stops movement after 1s.

### Voice & Interaction
- **Hold `R`** - Speak to Pepper. Release to transcribe and send.
- **`SPACE`** - Toggle Active/Idle (Eyes blue = Active, Eyes white = Idle).
- **GUI Chat** - Type messages directly in the window.

### Manual Gestures
- `1` Wave | `2` Nod | `3` Shake | `4` Think
- `8` Explain | `9` Excited | `0` Point
- `5-7` Change Eye LEDs (Blue/Green/Red)

## 💬 How It Works

1. **Wake Up**: Press `SPACE` in the terminal to activate.
2. **Interact**: Hold `R` to talk or type in the GUI.
3. **Research**: If you ask about current events, Pepper will automatically search the web.
4. **HQ Audio**: Pepper generates a premium voice file and plays it through her head speakers.
5. **Safety**: The system ensures she doesn't drive away if the network lags.

## 📝 Project Structure

- `main.py` - Core orchestrator and safety loops.
- `pepper_interface.py` - Hardware control (NAOqi) and SpeechLock.
- `groq_brain.py` - AI logic, Whisper STT, and function calling.
- `voice_handler.py` - Microphone capture and PTT logic.
- `hybrid_tts_handler.py` - 3-tier HQ voice system (Groq/ElevenLabs/Edge).
- `pepper_gui.py` - Modern GPU-accelerated dashboard.
- `web_search_handler.py` - DuckDuckGo integration.

---

## 🤝 Need Help?
- Groq docs: https://console.groq.com/docs
- Pepper SDK: http://doc.aldebaran.com/
- **PortAudio Error?** Run `sudo apt-get install libportaudio2`.
- **Pepper Won't Connect?** Ping her IP to check the network.
