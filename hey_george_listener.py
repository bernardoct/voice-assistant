#!/usr/bin/env python3
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests

from openwakeword.model import Model

from ha_client import play_on_sonos, set_volume

# --- STT BIAS/PROMPT ---
DEFAULT_BIAS_WORDS = [
    "Tolomeo",
    "Artemide",
    "Noguchi",
]

# --- CONFIG ---
SAMPLE_RATE = 16000
CHUNK = 1280  # ~80 ms at 16kHz
TRIGGER_THRESHOLD = 0.75
REARM_THRESHOLD = 0.35
RECORD_SECONDS = 4.0
COOLDOWN_SECONDS = 1.0
REARM_SILENCE_FRAMES = 5

# Choose a built-in wake word model to get working immediately.
# Later you will replace this with a custom "hey_george" model.
WAKEWORD_NAME = "hey_jarvis"

# If your mic is not default, set device index here (None = default)
INPUT_DEVICE = None
DEVICE_INDEX = 1      # <-- PulseAudio device

sd.default.device = (DEVICE_INDEX, None)


def record_wav(seconds: float) -> str:
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        # device=INPUT_DEVICE,
    )
    sd.wait()
    audio = np.squeeze(audio)
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, audio, SAMPLE_RATE)
    return temp_file.name


def build_initial_prompt(bias_words: list[str]) -> str:
    if not bias_words:
        return ""
    words = ", ".join(bias_words)
    return (
        "The audio is always either in English, Portuguese, or Croatian. Look for:\n",
        "* home assistant actions (e.g., turn on, turn off, brightness, etc.), ",
        f"followed by \n* a home device name/noun, including {words}."
    )


def transcribe(stt_url: str, wav_path: str) -> str:
    bias_words = list(DEFAULT_BIAS_WORDS)
    prompt = build_initial_prompt(bias_words)
    params = {}
    if bias_words:
        params["bias"] = bias_words
    if prompt:
        params["prompt"] = prompt
    with open(wav_path, "rb") as audio:
        response = requests.post(
            stt_url,
            params=params,
            files={"audio": ("audio.wav", audio, "audio/wav")},
            timeout=120,
        )
    response.raise_for_status()
    return response.json().get("text", "").strip()


def play_error_sound_on_sonos(settings):
    url = f"{settings.ha_url}/local/capture_error.wav"
    play_on_sonos(settings, "media_player.living_room", url)
    

def speak_reply_on_sonos(settings, message: str) -> None:
    message = message.strip()
    if not message:
        return
    try:
        import pyttsx3
    except Exception as exc:
        print(f"TTS unavailable (pyttsx3 import failed): {exc}")
        return
    output_dir = settings.tts_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(output_dir))
    tmp_path = tmp.name
    tmp.close()
    try:
        engine = pyttsx3.init()
        engine.save_to_file(message, tmp_path)
        engine.runAndWait()
    except Exception as exc:
        print(f"TTS failed: {exc}")
        return
    if not settings.tts_base_url:
        print("TTS audio generated but TTS_BASE_URL is not set; skipping Sonos playback.")
        return
    url = f"{settings.ha_url}/local/{Path(tmp_path).name}"
    play_on_sonos(settings, "media_player.living_room", url)


def run_listener(settings, on_text) -> None:
    print(
        f"Listening for wake word: {WAKEWORD_NAME} (temporary). Threshold={TRIGGER_THRESHOLD}"
    )
    oww = Model(wakeword_models=[WAKEWORD_NAME])

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=CHUNK,
        # device=INPUT_DEVICE,
    )
    stream.start()
    armed = True
    silence_frames = 0

    try:
        while True:
            data, _ = stream.read(CHUNK)
            pcm16 = np.squeeze(data)

            preds = oww.predict(pcm16)
            score = float(preds.get(WAKEWORD_NAME, 0.0))

            if armed and score >= TRIGGER_THRESHOLD:
                print("Wake word detected. Recording command...")

                set_volume(settings, "media_player.living_room", 0.1)
                play_on_sonos(
                    settings,
                    "media_player.living_room",
                    f"{settings.ha_url}/local/ready_for_capture.wav",
                )

                time.sleep(0.15)  # tiny pause to avoid clipping first syllable

                wav = record_wav(RECORD_SECONDS)
                play_on_sonos(
                    settings,
                    "media_player.living_room",
                    f"{settings.ha_url}/local/capture_ended.wav",
                )
                try:
                    t0 = time.time()
                    text = transcribe(settings.stt_url, wav)
                    dt = time.time() - t0
                    print(f"Transcription took {dt:.2f} seconds.")
                    print("Heard:", text)
                    if text:
                        on_text(text)
                finally:
                    try:
                        os.remove(wav)
                    except OSError:
                        pass

                time.sleep(COOLDOWN_SECONDS)
                armed = False
                silence_frames = 0
            elif not armed:
                if score < REARM_THRESHOLD:
                    silence_frames += 1
                else:
                    silence_frames = 0
                if silence_frames >= REARM_SILENCE_FRAMES:
                    armed = True
    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    from voice_assistant import main

    main()
