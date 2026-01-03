import time
from faster_whisper import WhisperModel

audio = "output.wav"  # pick a representative clip

for ct in ["int8_float32", "int8"]:
    model = WhisperModel("base.en", device="cpu", compute_type=ct)
    t0 = time.time()
    segments, info = model.transcribe(audio, beam_size=1)  # beam_size=1 for speed test
    _ = list(segments)
    dt = time.time() - t0
    print(ct, "seconds:", dt, "duration:", info.duration, "RTF:", dt / info.duration)
