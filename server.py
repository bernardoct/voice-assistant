from fastapi import FastAPI, UploadFile, File, Query
from faster_whisper import WhisperModel
import tempfile, shutil, time, os
import inspect
from typing import List, Optional

app = FastAPI()

# Default: fast
model_fast = WhisperModel("small", device="cuda", compute_type="int8_float16")

# Fallback: more robust
model_robust = WhisperModel("medium", device="cuda", compute_type="float16")

VAD_PARAMS = {
    "min_silence_duration_ms": 180,
    "speech_pad_ms": 80,
}

TRANSCRIBE_KW = dict(
    vad_filter=True,
    vad_parameters=VAD_PARAMS,
    beam_size=2,
    best_of=1,
    temperature=0.0,
    without_timestamps=True,
    condition_on_previous_text=False,
)

LANG_PROB_FALLBACK = 0.60
MIN_CHARS_FALLBACK = 3

def add_hotwords_if_supported(model: WhisperModel, kw: dict, bias_words: List[str]) -> dict:
    """
    Some faster-whisper versions support hotwords in WhisperModel.transcribe.
    Detect it safely and add if available.
    """
    if not bias_words:
        return kw

    try:
        sig = inspect.signature(model.transcribe)
        if "hotwords" in sig.parameters:
            # faster-whisper typically expects a string like "word1 word2"
            kw = dict(kw)
            kw["hotwords"] = " ".join(bias_words)
    except Exception:
        pass

    return kw

@app.post("/stt")
async def stt(
    audio: UploadFile = File(...),
    bias: Optional[List[str]] = Query(default=None, description="Words/phrases to bias toward"),
    prompt: Optional[str] = Query(default=None, description="Initial prompt to bias transcription"),
):
    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    bias_words = []
    if bias:
        bias_words.extend([b.strip() for b in bias if b and b.strip()])
    prompt_text = (prompt or "").strip()

    # de-dupe while preserving order
    seen = set()
    bias_words = [w for w in bias_words if not (w.lower() in seen or seen.add(w.lower()))]

    try:
        info, text, used = transcribe_with_fallback(
            tmp_path,
            bias_words=bias_words,
            prompt=prompt_text,
        )
        return {
            "text": text,
            "language": info.language,
            "language_probability": getattr(info, "language_probability", None),
            "duration": info.duration,
            "model": used,
            "bias_words": bias_words,
            "prompt": prompt_text or None,
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def transcribe_once(model, path, bias_words: List[str], prompt: str):
    t0 = time.time()

    kw = dict(TRANSCRIBE_KW)
    if prompt:
        kw["initial_prompt"] = prompt

    # ---- NEW: hotwords if supported ----
    kw = add_hotwords_if_supported(model, kw, bias_words)

    segments, info = model.transcribe(path, **kw)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    dt = time.time() - t0

    duration = info.duration or 0.0
    rtf = dt / duration if duration > 0 else float("inf")

    return info, text, dt, rtf

def transcribe_with_fallback(path, bias_words: List[str], prompt: str):
    info, text, dt, rtf = transcribe_once(model_fast, path, bias_words=bias_words, prompt=prompt)

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
        info2, text2, dt2, rtf2 = transcribe_once(
            model_robust,
            path,
            bias_words=bias_words,
            prompt=prompt,
        )
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
