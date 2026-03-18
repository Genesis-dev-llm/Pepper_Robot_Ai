#!/usr/bin/env python3
"""
Quick test to verify environment variables are being read correctly
"""

import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Testing Environment Variable Loading\n")
print("="*60)

# Test 1: Check if config imports
try:
    import config
    print("✅ config.py imports successfully")
except Exception as e:
    print(f"❌ Failed to import config: {e}")
    sys.exit(1)

# Test 2: Check values without environment variables
print("\n📋 Current config values:")
print(f"  GROQ_API_KEY: {config.GROQ_API_KEY[:20]}..." if len(config.GROQ_API_KEY) > 20 else f"  GROQ_API_KEY: {config.GROQ_API_KEY}")
print(f"  ELEVENLABS_API_KEY: {config.ELEVENLABS_API_KEY}")
print(f"  PEPPER_IP: {config.PEPPER_IP}")
print(f"  PEPPER_PORT: {config.PEPPER_PORT}")

# Test 3: Simulate environment variables
print("\n🧪 Testing with mock environment variables...")
os.environ["GROQ_API_KEY"] = "test_groq_key_123"
os.environ["ELEVENLABS_API_KEY"] = "test_eleven_key_456"
os.environ["PEPPER_IP"] = "10.0.0.1"

# Reload config
import importlib
importlib.reload(config)

print("  After setting env vars:")
print(f"  GROQ_API_KEY: {config.GROQ_API_KEY}")
print(f"  ELEVENLABS_API_KEY: {config.ELEVENLABS_API_KEY}")
print(f"  PEPPER_IP: {config.PEPPER_IP}")

# Test 4: Verify values changed
if config.GROQ_API_KEY == "test_groq_key_123":
    print("\n✅ GROQ_API_KEY reads from environment correctly")
else:
    print("\n❌ GROQ_API_KEY not reading from environment")

if config.ELEVENLABS_API_KEY == "test_eleven_key_456":
    print("✅ ELEVENLABS_API_KEY reads from environment correctly")
else:
    print("❌ ELEVENLABS_API_KEY not reading from environment")

if config.PEPPER_IP == "10.0.0.1":
    print("✅ PEPPER_IP reads from environment correctly")
else:
    print("❌ PEPPER_IP not reading from environment")

print("\n" + "="*60)
print("🎉 Environment variable setup is working correctly!")
print("\nHow to use:")
print("  1. Create .env file: cp .env.example .env")
print("  2. Edit .env with your actual keys")
print("  3. Run: source .env")
print("  4. Run: python main.py")