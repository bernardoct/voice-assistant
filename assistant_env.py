from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


ENV_PATH = Path.home() / ".ha_env"


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if line.startswith("export "):
            key, value = line[len("export "):].split("=", 1)
            os.environ.setdefault(key, value.strip('"'))


@dataclass(frozen=True)
class Settings:
    ha_url: str
    ha_token: str | None
    registry_path: Path
    stt_url: str
    llm_url: str
    llm_model: str
    llm_api_key: str
    tts_domain: str
    tts_service: str
    tts_target_field: str


def load_settings() -> Settings:
    load_env_file()
    ha_url = os.environ.get("HA_URL", "http://192.168.1.203:8123")
    ha_token = os.environ.get("HA_TOKEN")
    registry_path = Path(
        os.environ.get(
            "HA_REGISTRY_PATH", str(Path.home() / ".cache" / "ha_entities.json")
        )
    )
    stt_url = os.environ.get("STT_URL", "http://192.168.1.117:8008/stt")
    llm_url = os.environ.get("LLM_URL") or os.environ.get(
        "OPENAI_URL", "http://192.168.1.117:8000/v1/chat/completions"
    )
    llm_model = os.environ.get("LLM_MODEL", "local-model")
    llm_api_key = os.environ.get("LLM_API_KEY", "local-anything")
    tts_domain = os.environ.get("TTS_DOMAIN", "tts")
    tts_service = os.environ.get("TTS_SERVICE", "speak")
    tts_target_field = os.environ.get("TTS_TARGET_FIELD", "media_player_entity_id")

    return Settings(
        ha_url=ha_url,
        ha_token=ha_token,
        registry_path=registry_path,
        stt_url=stt_url,
        llm_url=llm_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        tts_domain=tts_domain,
        tts_service=tts_service,
        tts_target_field=tts_target_field,
    )
