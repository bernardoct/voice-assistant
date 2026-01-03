# Deprecated in favor of running a container on the Jetson Nano

import time

from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import shutil

app = FastAPI()

# Pick: "tiny" / "base" / "small"
# If you have NVIDIA GPU + CUDA, set device="cuda", compute_type="float16"
model = WhisperModel("base.en", device="cpu", compute_type="int8")

@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    info, text = parse_audio(tmp_path)
    return {"text": text, "language": info.language, "duration": info.duration}

def parse_audio(tmp_path):
    t0 = time.time()
    segments, info = model.transcribe(tmp_path, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    print(f"Time: {time.time() - t0} s")
    return info,text

# if __name__ == "__main__":
#     print(parse_audio("output_noguchi.wav")[1])
