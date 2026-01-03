#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import time
import json
import subprocess
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests

from openwakeword.model import Model

# --- CONFIG ---
SAMPLE_RATE = 16000
CHUNK = 1280                 # ~80 ms at 16kHz
TRIGGER_THRESHOLD = 0.75
REARM_THRESHOLD = 0.35
RECORD_SECONDS = 4.0
COOLDOWN_SECONDS = 1.0
REARM_SILENCE_FRAMES = 5

# Your laptop STT endpoint
# STT_URL = "http://192.168.1.116:8008/stt"
STT_URL = "http://192.168.1.117:8008/stt"

# Your routing script (the one that reads registry + calls HA)
ROUTER = "/home/bernardoct/voiceassistant/voice_route.py"

# Choose a built-in wake word model to get working immediately.
# Later you will replace this with a custom "hey_george" model.
WAKEWORD_NAME = "hey_jarvis"

# If your mic is not default, set device index here (None = default)
INPUT_DEVICE = None

envfile = Path.home() / ".ha_env"
if envfile.exists():
    for line in envfile.read_text().splitlines():
        if line.startswith("export "):
            k, v = line[len("export "):].split("=", 1)
            os.environ.setdefault(k, v.strip('"'))

HA_URL = os.environ["HA_URL"]
HA_TOKEN = os.environ["HA_TOKEN"]
# ----------------

def record_wav(seconds: float) -> str:
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1,
                   dtype="float32", device=INPUT_DEVICE)
    sd.wait()
    audio = np.squeeze(audio)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(f.name, audio, SAMPLE_RATE)
    return f.name

def transcribe(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        r = requests.post(STT_URL, files={"audio": ("audio.wav", f, "audio/wav")}, timeout=120)
    r.raise_for_status()
    return r.json().get("text", "").strip()

def route_text(text: str) -> None:
    # Pass HA_TOKEN/HA_URL via environment already set in your shell/systemd
    subprocess.run([sys.executable, ROUTER, text], check=False)

def play_on_sonos(entity_id: str, url: str):
    r = requests.post(
        f"{HA_URL}/api/services/media_player/play_media",
        headers={
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "entity_id": entity_id,
            "media_content_id": url,
            "media_content_type": "music",
        },
        timeout=10,
    )
    r.raise_for_status()

def set_volume(entity_id: str, level: float):
    requests.post(
        f"{HA_URL}/api/services/media_player/volume_set",
        headers={
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "entity_id": entity_id,
            "volume_level": level,
        },
        timeout=10,
    ).raise_for_status()

def main():
    print(f"Listening for wake word: {WAKEWORD_NAME} (temporary). Threshold={TRIGGER_THRESHOLD}")
    oww = Model(wakeword_models=[WAKEWORD_NAME])

    stream = sd.InputStream(
        channels=1,
        samplerate=SAMPLE_RATE,
        dtype="int16",
        blocksize=CHUNK,
        device=INPUT_DEVICE,
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
                
                set_volume("media_player.living_room", 0.1)
                play_on_sonos(
                    "media_player.living_room",
                    "http://192.168.1.203:8123/local/ready_for_capture.wav",
                )
                
                time.sleep(0.15)  # tiny pause to avoid clipping first syllable

                wav = record_wav(RECORD_SECONDS)
                try:
                    text = transcribe(wav)
                    print("Heard:", text)
                    if text:
                        route_text(text)
                finally:
                    try:
                        os.remove(wav)
                    except OSError:
                        pass

                # cooldown to avoid double-triggers
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
    main()
