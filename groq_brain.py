"""
Groq Brain - LLM and Speech-to-Text
Handles all AI interactions via Groq API
"""

import os
from groq import Groq
from typing import List, Dict, Optional, Tuple
import json

class GroqBrain:
    def __init__(self, api_key: str, llm_model: str, whisper_model: str, 
                 system_prompt: str, functions: List[Dict], use_web_search: bool = False,
                 compound_model: Optional[str] = None):
        """Initialize Groq client and configuration"""
        self.client = Groq(api_key=api_key)
        self.use_web_search = use_web_search
        self.llm_model = compound_model if use_web_search and compound_model else llm_model
        self.whisper_model = whisper_model
        self.system_prompt = system_prompt
        self.functions = functions
        
        # Conversation history
        self.conversation_history: List[Dict] = []
        self.max_history = 10  # Keep last 10 exchanges
        
        search_status = "with WEB SEARCH" if use_web_search else "without web search"
        print(f"🧠 Groq Brain initialized with {self.llm_model} ({search_status})")
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio file using Groq Whisper
        
        Args:
            audio_file_path: Path to audio file (wav, mp3, etc.)
        
        Returns:
            Transcribed text or None if error
        """
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.whisper_model,
                    response_format="text",
                    language="en"  # Force English for faster processing
                )
            
            text = transcription.strip()
            if text:
                print(f"🎤 Transcribed: '{text}'")
                return text
            return None
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def chat(self, user_message: str) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Send message to LLM and get response with optional function calls
        
        Args:
            user_message: User's text input
        
        Returns:
            Tuple of (response_text, function_calls)
            function_calls format: [{"name": "wave", "arguments": {}}]
        """
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Prepare messages for API call
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history
            
            # Compound models use built-in tools (web search), no custom functions
            if self.use_web_search or "compound" in self.llm_model.lower():
                print("🌐 Using web search model (no gesture support)")
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
            else:
                # Regular model - supports custom gesture functions
                print("🤖 Using standard model (with gesture support)")
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    tools=self.functions,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=150
                )
            
            message = response.choices[0].message
            
            # Extract response text
            response_text = message.content if message.content else ""
            
            # Extract function calls if any (only for non-compound models)
            function_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    function_calls.append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    })
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            if response_text:
                print(f"💬 AI: {response_text[:100]}..." if len(response_text) > 100 else f"💬 AI: {response_text}")
            if function_calls:
                print(f"🎬 Gestures: {[f['name'] for f in function_calls]}")
            
            return response_text, function_calls if function_calls else None
            
        except Exception as e:
            print(f"❌ Chat error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🔄 Conversation history cleared")

    def chat_with_context(self, user_message: str, context: str) -> tuple:
        """
        One-shot call: inject extra context (e.g. search results) for this
        turn only — the context is NOT stored in conversation history.

        Use this instead of chat() when you have tool/search results to
        fold in, so the raw data blobs don't accumulate in future turns.

        Args:
            user_message: The original user question (already in history)
            context:      Extra context to inject (search results, etc.)

        Returns:
            Tuple of (response_text, function_calls)
        """
        try:
            # Build messages: history + one-shot system context + re-state question
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + self.conversation_history[:-1]   # history minus the pending user msg
                + [{"role": "system",
                    "content": f"[Tool result / search context — do not repeat verbatim]:\n{context}"}]
                + [{"role": "user", "content": user_message}]
            )

            response = self.client.chat.completions.create(
                model      = self.llm_model,
                messages   = messages,
                tools      = self.functions,
                tool_choice = "auto",
                temperature = 0.7,
                max_tokens  = 150,
            )

            message       = response.choices[0].message
            response_text = message.content if message.content else ""

            function_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    function_calls.append({
                        "name":      tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {},
                    })

            # Store only the final assistant answer — not the raw context blob
            self.conversation_history.append({
                "role":    "assistant",
                "content": response_text,
            })

            # Trim if needed
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[-self.max_history * 2:]

            if response_text:
                print(f"💬 AI (ctx): {response_text[:100]}..." if len(response_text) > 100 else f"💬 AI (ctx): {response_text}")

            return response_text, function_calls if function_calls else None

        except Exception as e:
            print(f"❌ chat_with_context error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def add_context(self, context: str):
        """Add context to conversation (e.g., vision info)"""
        # Add as system message so it doesn't count toward history limit
        self.conversation_history.insert(0, {
            "role": "system",
            "content": f"Current context: {context}"
        })


def test_groq_connection(api_key: str):
    """Quick test to verify Groq API is working"""
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'hello' in one word"}],
            max_tokens=10
        )
        print(f"✅ Groq API test successful: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Groq API test failed: {e}")
        return False