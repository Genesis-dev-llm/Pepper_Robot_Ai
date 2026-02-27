#!/usr/bin/env python3
"""
Quick test script to verify all components before running main program
"""

import sys
import os

# Resolve project root relative to this file — no hardcoded absolute paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    print("🧪 Testing imports...")
    packages = {
        "qi":                "Pepper NAOqi framework",
        "groq":              "Groq API (LLM + Whisper STT)",
        "dearpygui":         "DearPyGUI (GPU-accelerated GUI)",
        "duckduckgo_search": "DuckDuckGo web search",
        "sounddevice":       "Audio recording (Push-to-Talk)",
        "soundfile":         "WAV file I/O",
        "numpy":             "Audio array manipulation",
        "edge_tts":          "Microsoft Edge TTS fallback",
        "pynput":            "Keyboard input handler",
        "paramiko":          "SSH/SFTP for HQ audio transfer",
    }
    failed = []
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:20s} - {description}")
        except ImportError:
            print(f"  ❌ {package:20s} - {description} (NOT INSTALLED)")
            failed.append(package)

    try:
        __import__("elevenlabs")
        print(f"  ✅ {'elevenlabs':20s} - ElevenLabs TTS (optional)")
    except ImportError:
        print(f"  ⚠️  {'elevenlabs':20s} - ElevenLabs TTS (optional, not installed — OK)")

    if failed:
        print(f"\n❌ Missing: {', '.join(failed)}")
        print(f"  pip install {' '.join(failed)} --break-system-packages")
        return False
    print("\n✅ All required packages installed!")
    return True


def test_config():
    print("\n🔧 Testing configuration...")
    try:
        import config

        if config.GROQ_API_KEY == "your_groq_api_key_here":
            print("  ⚠️  Groq API key not set — get one from https://console.groq.com/keys")
            return False
        print("  ✅ Groq API key configured")

        # Check against the actual default in config.py (not a stale old default)
        default_ip = "10.55.203.146"
        if config.PEPPER_IP == default_ip:
            print(f"  ⚠️  Pepper IP is the default ({default_ip}) — update PEPPER_IP in .env if needed")
        else:
            print(f"  ✅ Pepper IP: {config.PEPPER_IP}")

        return True
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False


def test_groq_api():
    print("\n🌐 Testing Groq API connection...")
    try:
        import config
        from groq_brain import test_groq_connection
        if test_groq_connection(config.GROQ_API_KEY):
            print("  ✅ Groq API is working!")
            return True
        print("  ❌ Groq API test failed")
        return False
    except Exception as e:
        print(f"  ❌ Groq API error: {e}")
        return False


def test_pepper_connection():
    print("\n🤖 Testing Pepper connection...")
    try:
        import config
        from pepper_interface import PepperRobot
        pepper = PepperRobot(config.PEPPER_IP, config.PEPPER_PORT)
        if pepper.connect():
            print("  ✅ Connected to Pepper!")
            pepper.disconnect()
            return True
        print(f"  ❌ Could not connect — check IP: {config.PEPPER_IP}")
        return False
    except Exception as e:
        print(f"  ❌ Pepper connection error: {e}")
        return False


def test_voice():
    print("\n🎙️ Testing microphone (Push-to-Talk)...")
    try:
        import sounddevice as sd
        import config

        if not config.VOICE_ENABLED:
            print("  ⚠️  VOICE_ENABLED = False — skipping mic test")
            return True

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if not input_devices:
            print("  ❌ No microphone found")
            return False

        default_idx = sd.default.device[0]
        default_dev = devices[default_idx] if default_idx is not None else None
        if default_dev:
            print(f"  ✅ Default mic: {default_dev['name']}")
        else:
            print(f"  ✅ {len(input_devices)} microphone(s) available")
        print(f"  ✅ PTT key: '{config.PTT_KEY.upper()}' | VAD threshold: {config.VAD_THRESHOLD}")
        return True
    except Exception as e:
        print(f"  ❌ Mic test error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print(" PEPPER AI ROBOT - PRE-FLIGHT CHECK")
    print("="*60 + "\n")

    tests = [
        ("Imports",           test_imports),
        ("Configuration",     test_config),
        ("Microphone",        test_voice),
        ("Groq API",          test_groq_api),
        ("Pepper Connection", test_pepper_connection),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False

    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    for name, passed in results.items():
        print(f"  {'✅ PASS' if passed else '❌ FAIL':8s} - {name}")
    print("="*60)

    if all(results.values()):
        print("\n🎉 All tests passed! Run: source .env && python main.py\n")
    else:
        print("\n⚠️ Some tests failed. Fix issues above before running main.py")
        sys.exit(1)


if __name__ == "__main__":
    main()