#!/usr/bin/env python3
"""
Quick test script to verify all components before running main program
"""

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("🧪 Testing imports...")
    
    packages = {
        "qi":                 "Pepper NAOqi framework",
        "groq":               "Groq API (LLM + Whisper STT)",
        "dearpygui":          "DearPyGUI (GPU-accelerated GUI)",
        "duckduckgo_search":  "DuckDuckGo web search",
        "sounddevice":        "Audio recording (Push-to-Talk)",
        "soundfile":          "WAV file I/O",
        "numpy":              "Audio array manipulation",
        "edge_tts":           "Microsoft Edge TTS fallback",
        "pynput":             "Keyboard input handler",
    }
    
    failed = []
    
    for package, description in packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package:20s} - {description}")
        except ImportError:
            print(f"  ❌ {package:20s} - {description} (NOT INSTALLED)")
            failed.append(package)
    
    # elevenlabs is optional — warn but don't fail
    try:
        __import__("elevenlabs")
        print(f"  ✅ {'elevenlabs':20s} - ElevenLabs TTS (optional)")
    except ImportError:
        print(f"  ⚠️  {'elevenlabs':20s} - ElevenLabs TTS (optional, not installed — that's fine)")

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(failed)} --break-system-packages")
        return False
    
    print("\n✅ All required packages installed!")
    return True


def test_config():
    """Test if config is properly set up"""
    print("\n🔧 Testing configuration...")
    
    try:
        import config
        
        # Check API key
        if config.GROQ_API_KEY == "your_groq_api_key_here":
            print("  ⚠️  Groq API key not set in config.py")
            print("     Get your key from: https://console.groq.com/keys")
            return False
        else:
            print(f"  ✅ Groq API key configured")
        
        # Check Pepper IP
        if config.PEPPER_IP == "192.168.1.100":
            print("  ⚠️  Pepper IP is set to default (192.168.1.100)")
            print("     Update PEPPER_IP in config.py if this is not your Pepper's IP")
        else:
            print(f"  ✅ Pepper IP: {config.PEPPER_IP}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config error: {e}")
        return False


def test_groq_api():
    """Test Groq API connection"""
    print("\n🌐 Testing Groq API connection...")
    
    try:
        import config
        from groq_brain import test_groq_connection
        
        if test_groq_connection(config.GROQ_API_KEY):
            print("  ✅ Groq API is working!")
            return True
        else:
            print("  ❌ Groq API test failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Groq API error: {e}")
        return False


def test_pepper_connection():
    """Test Pepper robot connection"""
    print("\n🤖 Testing Pepper connection...")
    print("   (This might take a few seconds...)")
    
    try:
        import config
        from pepper_interface import PepperRobot
        
        pepper = PepperRobot(config.PEPPER_IP, config.PEPPER_PORT)
        
        if pepper.connect():
            print("  ✅ Connected to Pepper!")
            pepper.disconnect()
            return True
        else:
            print("  ❌ Could not connect to Pepper")
            print(f"     Check if Pepper is on and IP is correct: {config.PEPPER_IP}")
            return False
            
    except Exception as e:
        print(f"  ❌ Pepper connection error: {e}")
        return False


def test_voice():
    """Test if microphone is accessible for Push-to-Talk"""
    print("\n🎙️ Testing microphone (Push-to-Talk)...")

    try:
        import sounddevice as sd
        import config

        if not config.VOICE_ENABLED:
            print("  ⚠️  VOICE_ENABLED = False in config — skipping mic test")
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

        print(f"  ✅ PTT key: '{config.PTT_KEY.upper()}' (hold to record)")
        return True

    except Exception as e:
        print(f"  ❌ Mic test error: {e}")
        return False


def main():
    """Run all tests"""
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
            print("\n\n⚠️ Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print(" TEST SUMMARY")
    print("="*60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status:8s} - {name}")
    
    print("="*60)
    
    if all(results.values()):
        print("\n🎉 All tests passed! You're ready to run:")
        print("\n    source .env && python main.py\n")
    else:
        print("\n⚠️ Some tests failed. Fix the issues above before running main.py")
        sys.exit(1)


if __name__ == "__main__":
    main()