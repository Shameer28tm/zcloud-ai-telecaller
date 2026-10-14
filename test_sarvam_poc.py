"""Isolated Sarvam AI Tamil STT/TTS Proof-of-Concept Test.

Tests:
1. Bulbul v4 Flash Text-to-Speech (native Tamil & Thanglish)
2. Saaras v4 Speech-to-Text (audio transcription)
3. Latency, audio format, transcript accuracy, quota/error reporting.

Uses the official `sarvamai` Python SDK.
Does NOT modify production LiveKit agent or telephony.
"""

from __future__ import annotations

import base64
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

# Path to .env in current repository
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


def resolve_sarvam_api_key() -> Optional[str]:
    """Resolves SARVAM_API_KEY from environment or .env without exposing its value."""
    key = os.environ.get("SARVAM_API_KEY")
    if key and key.strip():
        return key.strip()

    # Fallback: check if .env has api-subscription-key or alternate naming
    if ENV_PATH.exists():
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("SARVAM_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
                elif line.startswith("api-subscription-key:"):
                    val = line.split(":", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
                elif line.startswith("api-subscription-key="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        return val
    return None


def run_sarvam_poc() -> None:
    print("=" * 65)
    print("      SARVAM AI TAMIL STT / TTS PROOF-OF-CONCEPT TEST")
    print("=" * 65)

    api_key = resolve_sarvam_api_key()
    if not api_key:
        print("\n[ERROR] SARVAM_API_KEY could not be found in local environment or .env.")
        print("Please ensure SARVAM_API_KEY is configured in your environment.")
        return

    # Never print or expose the key
    masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
    print(f"\n[OK] SARVAM_API_KEY detected (format: {masked}, length: {len(api_key)} chars)")

    try:
        from sarvamai import SarvamAI
    except ImportError:
        print("\n[ERROR] 'sarvamai' package is not installed. Please install with: uv pip install sarvamai")
        return

    client = SarvamAI(api_subscription_key=api_key)

    # Temporary test output directory
    output_dir = Path(__file__).resolve().parent / "test_output_sarvam"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Temporary output directory ready: {output_dir}")

    # =========================================================================
    # PART 1: Bulbul v4 Flash Text-to-Speech (Tamil & Thanglish)
    # =========================================================================
    print("\n" + "-" * 65)
    print("PART 1: BULBUL v4 FLASH TEXT-TO-SPEECH (TTS)")
    print("-" * 65)

    tamil_test_text = "வணக்கம்! அப்பல்லோ பல் மருத்துவமனைக்கு உங்களை வரவேற்கிறோம். உங்களுக்கு என்ன சிகிச்சைக்கான அப்பாயின்ட்மென்ட் வேண்டும்?"
    thanglish_test_text = "Vanakkam! Apollo dental clinic-ku ungalai welcome panrom. Ungalukku dental cleaning appointment book pannanuma?"

    tamil_wav_path = output_dir / "bulbul_v4_flash_tamil.wav"
    thanglish_wav_path = output_dir / "bulbul_v4_flash_thanglish.wav"

    # Test 1A: Native Tamil sentence
    print(f"\n[Test 1A] Synthesizing Native Tamil Speech (Model: bulbul:v4-flash)...")
    print(f"  Input text: \"{tamil_test_text}\"")
    start_t = time.perf_counter()
    try:
        resp_tamil = client.text_to_speech.convert(
            text=tamil_test_text,
            language_code="ta-IN",
            model="bulbul:v4-flash",
            speaker="gokul_ta_narration",
        )
        latency_tamil = (time.perf_counter() - start_t) * 1000.0

        audios = getattr(resp_tamil, "audios", []) or []
        if not audios:
            print("  [FAIL] Response received but 'audios' list is empty.")
        else:
            raw_audio = base64.b64decode(audios[0])
            with open(tamil_wav_path, "wb") as f:
                f.write(raw_audio)

            # Inspect audio format
            with wave.open(str(tamil_wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.getnframes()
                duration = frames / float(sample_rate)

            print(f"  [SUCCESS] API Call Succeeded!")
            print(f"  Latency: {latency_tamil:.1f} ms")
            print(f"  Audio Format: WAV PCM ({channels} ch, {sample_rate} Hz, 16-bit)")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  File Size: {len(raw_audio):,} bytes")
            print(f"  Saved to: {tamil_wav_path.name}")
    except Exception as exc:
        latency_tamil = (time.perf_counter() - start_t) * 1000.0
        print(f"  [FAIL] Bulbul v4 Flash Tamil TTS error after {latency_tamil:.1f}ms: {exc}")

    # Test 1B: Conversational Thanglish sentence
    print(f"\n[Test 1B] Synthesizing Conversational Thanglish Speech (Model: bulbul:v4-flash)...")
    print(f"  Input text: \"{thanglish_test_text}\"")
    start_t = time.perf_counter()
    try:
        resp_thanglish = client.text_to_speech.convert(
            text=thanglish_test_text,
            language_code="ta-IN",
            model="bulbul:v4-flash",
            speaker="gokul_ta_narration",
        )
        latency_thanglish = (time.perf_counter() - start_t) * 1000.0

        audios = getattr(resp_thanglish, "audios", []) or []
        if not audios:
            print("  [FAIL] Response received but 'audios' list is empty.")
        else:
            raw_audio = base64.b64decode(audios[0])
            with open(thanglish_wav_path, "wb") as f:
                f.write(raw_audio)

            with wave.open(str(thanglish_wav_path), "rb") as wf:
                sample_rate = wf.getframerate()
                channels = wf.getnchannels()
                frames = wf.getnframes()
                duration = frames / float(sample_rate)

            print(f"  [SUCCESS] API Call Succeeded!")
            print(f"  Latency: {latency_thanglish:.1f} ms")
            print(f"  Audio Format: WAV PCM ({channels} ch, {sample_rate} Hz, 16-bit)")
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  File Size: {len(raw_audio):,} bytes")
            print(f"  Saved to: {thanglish_wav_path.name}")
    except Exception as exc:
        latency_thanglish = (time.perf_counter() - start_t) * 1000.0
        print(f"  [FAIL] Bulbul v4 Flash Thanglish TTS error after {latency_thanglish:.1f}ms: {exc}")

    # =========================================================================
    # PART 2: Saaras v4 Speech-to-Text (STT)
    # =========================================================================
    print("\n" + "-" * 65)
    print("PART 2: SAARAS v4 SPEECH-TO-TEXT (STT)")
    print("-" * 65)

    # Test 2A: Transcribe generated Tamil audio
    if tamil_wav_path.exists():
        print(f"\n[Test 2A] Transcribing Tamil Audio (Model: saaras:v4, Language: ta-IN)...")
        print(f"  Ground Truth: \"{tamil_test_text}\"")
        start_t = time.perf_counter()
        try:
            with open(tamil_wav_path, "rb") as audio_file:
                stt_tamil_resp = client.speech_to_text.transcribe(
                    file=audio_file,
                    model="saaras:v4",
                    language_code="ta-IN",
                    mode="transcribe",
                )
            latency_stt_ta = (time.perf_counter() - start_t) * 1000.0
            transcript_ta = getattr(stt_tamil_resp, "transcript", "")
            print(f"  [SUCCESS] API Call Succeeded!")
            print(f"  Latency: {latency_stt_ta:.1f} ms")
            print(f"  Transcript: \"{transcript_ta}\"")
        except Exception as exc:
            latency_stt_ta = (time.perf_counter() - start_t) * 1000.0
            print(f"  [FAIL] Saaras v4 Tamil STT error after {latency_stt_ta:.1f}ms: {exc}")
    else:
        print("  [SKIPPED] Tamil audio file not available for STT test.")

    # Test 2B: Transcribe generated Thanglish audio
    if thanglish_wav_path.exists():
        print(f"\n[Test 2B] Transcribing Thanglish Audio (Model: saaras:v4, Language: ta-IN, codemix)...")
        print(f"  Ground Truth: \"{thanglish_test_text}\"")
        start_t = time.perf_counter()
        try:
            with open(thanglish_wav_path, "rb") as audio_file:
                stt_thanglish_resp = client.speech_to_text.transcribe(
                    file=audio_file,
                    model="saaras:v4",
                    language_code="ta-IN",
                    mode="codemix",
                )
            latency_stt_th = (time.perf_counter() - start_t) * 1000.0
            transcript_th = getattr(stt_thanglish_resp, "transcript", "")
            print(f"  [SUCCESS] API Call Succeeded!")
            print(f"  Latency: {latency_stt_th:.1f} ms")
            print(f"  Transcript: \"{transcript_th}\"")
        except Exception as exc:
            latency_stt_th = (time.perf_counter() - start_t) * 1000.0
            print(f"  [FAIL] Saaras v4 Thanglish STT error after {latency_stt_th:.1f}ms: {exc}")
    else:
        print("  [SKIPPED] Thanglish audio file not available for STT test.")

    print("\n" + "=" * 65)
    print("                      POC TEST COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    run_sarvam_poc()
