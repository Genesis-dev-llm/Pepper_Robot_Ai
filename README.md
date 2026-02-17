# 🤖 Pepper AI Robot - Phase 1

AI-powered Pepper robot with conversation, gestures, and keyboard controls.

## ✅ What Works (Phase 1)

- ✅ Connect to Pepper robot via qi
- ✅ Groq AI conversation (LLM)
- ✅ Text-to-Speech (Microsoft Edge TTS)
- ✅ Keyboard controls for movement
- ✅ Manual gesture controls
- ✅ AI function calling (robot gestures during conversation)
- ✅ LED eye color control
- ⏳ Audio capture (placeholder - using text input for now)

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
cd pepper_project
pip install -r requirements.txt --break-system-packages
```

**Note for Ubuntu:** Use `--break-system-packages` flag as mentioned.

### 2. Get Your Groq API Key

1. Go to https://console.groq.com/keys
2. Sign up (it's free, no credit card)
3. Create a new API key
4. Copy the key

### 3. Find Your Pepper's IP Address

On Pepper's tablet:
1. Tap Settings (gear icon)
2. Go to Network
3. Note the IP address (e.g., `192.168.1.100`)

### 4. Configure

**Option 1: Environment Variables (Recommended for security)**

```bash
# Copy example file
cp .env.example .env

# Edit .env with your values
nano .env

# Then export them
source .env
```

**Option 2: Direct in config.py (Easy for testing)**

Edit `config.py`:

```python
GROQ_API_KEY = "your_actual_api_key_here"
PEPPER_IP = "192.168.1.100"  # Your Pepper's IP
```

**⚠️ Security Note:** Never commit your API keys to git! Add `.env` to `.gitignore`.

### 5. Test Connection

```bash
python main.py
```

## 🎮 Controls

### Movement (WASD)
- `W` - Move forward
- `S` - Move backward
- `A` - Turn left
- `D` - Turn right
- `Q` - Strafe left
- `E` - Strafe right

### Gestures
- `1` - Wave
- `2` - Nod head
- `3` - Shake head
- `4` - Look at sound source

### LED Colors
- `5` - Blue eyes
- `6` - Green eyes  
- `7` - Red eyes

### Conversation
- `SPACE` - Toggle Active/Idle (wake Pepper)
- Type message + Enter - Simulate speech (for testing)

### System
- `X` - Quit program

## 💬 How It Works

1. Press `SPACE` to activate Pepper (eyes turn blue)
2. Type your message and press Enter (simulating speech for now)
3. Pepper thinks with Groq AI
4. Pepper may perform gestures (wave, nod, etc.) automatically
5. Pepper responds with speech (TTS)
6. Say "bye pepper" to deactivate

## 🔧 Troubleshooting

### Pepper Won't Connect
- Check IP address in `config.py`
- Make sure Pepper is powered on and awake
- Ping the IP: `ping 192.168.1.100`
- Ensure you're on the same network

### Groq API Errors
- Check API key in `config.py`
- Verify API key works: `python -c "from groq_brain import test_groq_connection; import config; test_groq_connection(config.GROQ_API_KEY)"`
- Check rate limits: https://console.groq.com/settings/limits

### No Audio Playback
Install an audio player:
```bash
sudo apt install mpg123  # Recommended
# OR
sudo apt install ffmpeg  # Includes ffplay
```

### ImportError: qi
Make sure qi is installed:
```bash
pip show qi
# If not installed:
pip install qi --break-system-packages
```

## 🎯 Next Steps (Future Phases)

### Phase 2 - Real Audio
- [ ] Add microphone capture from Pepper
- [ ] Implement Groq Whisper for real-time STT
- [ ] Add wake word detection (pvporcupine or vosk)
- [ ] Proper audio streaming

### Phase 3 - Vision
- [ ] Add YOLO for object detection
- [ ] Face recognition
- [ ] Look at person speaking
- [ ] Pass vision context to AI

### Phase 4 - Polish
- [ ] Better gesture timing
- [ ] LED animations during speech
- [ ] Smoother movements
- [ ] System prompt improvements

## 📊 Groq Free Tier Limits

- **Requests per day:** 1,000 (for llama-3.3-70b)
- **Requests per minute:** 30
- **Tokens per minute:** 12,000
- **Tokens per day:** 100,000

For classroom demos, this is **plenty**. You can do 100+ full conversations per day.

## 🐛 Known Issues

- Audio capture not yet implemented (using text input for testing)
- Movement is basic (can add smoother animations later)
- No proper wake word detection yet (using spacebar toggle)

## 💡 Tips

- Keep Pepper plugged in during long demos
- The AI sometimes calls functions - watch for automatic gestures!
- Test your setup before showing to visitors
- Keep responses short (already configured in system prompt)
- If latency is high, try the faster llama-3.1-8b-instant model

## 📝 File Structure

```
pepper_project/
├── config.py                # Configuration and settings
├── pepper_interface.py      # Pepper robot control (qi)
├── groq_brain.py           # AI brain (LLM + STT)
├── tts_handler.py          # Text-to-Speech (Edge TTS)
├── groq_tts_handler.py     # OPTIONAL: Groq Orpheus TTS
├── main.py                 # Main program with controls
├── test_setup.py           # Pre-flight checks
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── TTS_OPTIONS.md         # TTS comparison guide
└── Q_AND_A.md            # Common questions answered
```

## 🤝 Need Help?

- Groq docs: https://console.groq.com/docs
- Pepper SDK: http://doc.aldebaran.com/

---

Made with 🔥 by Claude & You