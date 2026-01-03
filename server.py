from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile, shutil, time, os

app = FastAPI()

# Default: fast
model_fast = WhisperModel("small", device="cuda", compute_type="int8_float16")

# Fallback: more robust
model_robust = WhisperModel("medium", device="cuda", compute_type="float16")

# VAD tuned for short assistant queries (faster endpointing, fewer trailing silences)
VAD_PARAMS = {
    "min_silence_duration_ms": 180,  # lower = stop sooner (latency down)
    "speech_pad_ms": 80,             # keep a little context
}

# Decoding tuned for low latency
TRANSCRIBE_KW = dict(
    vad_filter=True,
    vad_parameters=VAD_PARAMS,
    beam_size=2,
    best_of=1,
    temperature=0.0,
    without_timestamps=True,
    condition_on_previous_text=False,  # helps short utterances; avoids drift
)

# Simple fallback policy thresholds
LANG_PROB_FALLBACK = 0.60
MIN_CHARS_FALLBACK = 3

@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    # Save upload to temp file (fastest easy path; memory path is possible but more work)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    try:
        info, text, used = transcribe_with_fallback(tmp_path)
        return {
            "text": text,
            "language": info.language,
            "language_probability": getattr(info, "language_probability", None),
            "duration": info.duration,
            "model": used,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def transcribe_once(model, path):
    t0 = time.time()
    segments, info = model.transcribe(path, **TRANSCRIBE_KW)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    dt = time.time() - t0

    duration = info.duration or 0.0
    rtf = dt / duration if duration > 0 else float("inf")

    return info, text, dt, rtf

def transcribe_with_fallback(path):
    info, text, dt, rtf = transcribe_once(model_fast, path)

    lang_prob = getattr(info, "language_probability", 1.0)
    need_fallback = (len(text) < MIN_CHARS_FALLBACK) or (lang_prob < LANG_PROB_FALLBACK)

    print(
        f"[fast] "
        f"time={dt:.3f}s "
        f"audio={info.duration:.2f}s "
        f"RTF={rtf:.2f} "
        f"lang={info.language} "
        f"p={lang_prob:.2f} "
        f"text='{text}'"
    )

    if need_fallback:
        info2, text2, dt2, rtf2 = transcribe_once(model_robust, path)
        lang_prob2 = getattr(info2, "language_probability", 1.0)

        print(
            f"[fallback] "
            f"time={dt2:.3f}s "
            f"audio={info2.duration:.2f}s "
            f"RTF={rtf2:.2f} "
            f"lang={info2.language} "
            f"p={lang_prob2:.2f} "
            f"text='{text2}'"
        )

        return info2, text2, "medium(float16)"
    else:
        return info, text, "small(int8_float16)"